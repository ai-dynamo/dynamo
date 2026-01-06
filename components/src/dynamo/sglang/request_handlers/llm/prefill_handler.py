# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    """Handler for prefill workers in disaggregated serving mode."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
    ) -> None:
        """Initialize prefill worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
        """
        self.engine = engine
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(self.engine)
        super().__init__(component, engine, config, publisher)
        self._consume_tasks = set()
        logging.info(
            f"Prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    def cleanup(self) -> None:
        """Shutdown the prefill engine and cleanup resources."""
        # Cancel all pending consume tasks
        for task in self._consume_tasks:
            if not task.done():
                task.cancel()
        self._consume_tasks.clear()

        self.engine.shutdown()
        logging.info("Prefill engine shutdown")
        super().cleanup()

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate prefill output and provide bootstrap info for decode worker.

        Args:
            request: Request dict with 'request', 'sampling_params', and possibly 'bootstrap_room' keys.
            context: Context object for cancellation handling.

        Yields:
            Bootstrap info dict with host, port, and room for decode worker connection.
        """
        logging.debug(f"New Request ID: {context.id()}")
        trace_id = context.trace_id

        if "request" in request:
            # DisaggPreprocessedRequest format
            inner_request = request["request"]
            sampling_params = request.get("sampling_params", {})
        else:
            inner_request = request
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            sampling_params = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
            }
            sampling_params = {
                k: v for k, v in sampling_params.items() if v is not None
            }

        # Use provided bootstrap_room from bootstrap_info if available, otherwise generate one
        bootstrap_room = None
        bootstrap_info_from_req = inner_request.get("bootstrap_info")
        if isinstance(bootstrap_info_from_req, dict):
            bootstrap_room = bootstrap_info_from_req.get("bootstrap_room")
            if bootstrap_room is not None:
                logging.debug(f"Using router-provided bootstrap_room: {bootstrap_room}")

        if bootstrap_room is None:
            bootstrap_room = self._generate_bootstrap_room()
            logging.debug(f"Generated bootstrap_room locally: {bootstrap_room}")

        bootstrap_info = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

        # Yield bootstrap_info for PrefillRouter - required for async generator contract
        # and Rust-side expects disaggregated_params in first output
        yield {
            "token_ids": [],
            "text": None,
            "finish_reason": None,
            "disaggregated_params": bootstrap_info,
        }

        input_param = self._get_input_param(inner_request)

        # Propagate trace context to SGLang
        if self.enable_trace:
            self._propagate_trace_context_to_sglang(context, bootstrap_room)

        results = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=bootstrap_room,
            rid=trace_id,
        )

        task = asyncio.create_task(self._consume_results(results, context))
        self._consume_tasks.add(task)
        task.add_done_callback(self._consume_tasks.discard)

        await task

    async def _consume_results(
        self, results: AsyncGenerator[Any, None], context: Context
    ) -> None:
        """Consume async generator results without processing.

        Args:
            results: Async generator from engine.async_generate.
            context: Context object for cancellation handling.
        """
        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in results:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logging.debug(f"New Prefill Request ID: {sglang_request_id}")

                # Note: No explicit cancellation checks needed here.
                # When abort_request is called by the cancellation monitor,
                # SGLang will terminate this async generator automatically.
