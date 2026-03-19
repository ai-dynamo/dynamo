# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        shutdown_controller=None,
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
        super().__init__(
            component,
            engine,
            config,
            publisher,
            shutdown_controller=shutdown_controller,
        )
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
            request: Request dict with 'request' and 'sampling_params' keys.
            context: Context object for cancellation handling.

        Yields:
            Bootstrap info dict with host, port, and room for decode worker connection.
        """
        logging.debug(f"New Request ID: {context.id()}")
        bootstrap_room = self._generate_bootstrap_room()

        bootstrap_info = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

        yield bootstrap_info
        task = asyncio.create_task(
            self._run_request(request, context, bootstrap_room)
        )
        self._consume_tasks.add(task)
        task.add_done_callback(self._consume_tasks.discard)

    async def _run_request(
        self, request: Dict[str, Any], context: Context, bootstrap_room: int
    ) -> None:
        async with self._track_active_request():
            input_param = self._get_input_param(request["request"])
            results = await self.engine.async_generate(
                **input_param,
                sampling_params=request["sampling_params"],
                stream=True,
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=bootstrap_room,
            )
            await self._consume_results(results, context)

    async def _consume_results(
        self, results: AsyncGenerator[Any, None], context: Context
    ) -> None:
        """Consume async generator results without processing.

        Args:
            results: Async generator from engine.async_generate.
            context: Context object for cancellation handling.
        """
        request_id_future = asyncio.Future()
        try:
            async with self._cancellation_monitor(
                request_id_future, context
            ) as forced_shutdown_event:
                async for res in results:
                    if hasattr(res, "is_error") and callable(getattr(res, "is_error")):
                        try:
                            if res.is_error():
                                err = None
                                err_attr = getattr(res, "error_message", None)
                                if callable(err_attr):
                                    err = err_attr()
                                elif err_attr is not None:
                                    err = err_attr
                                raise GeneratorExit(
                                    f"SGLang prefill engine returned error: {err or 'incomplete stream'}"
                                )
                        except AttributeError:
                            pass

                    # Extract SGLang request ID from the first response and set the future
                    if not request_id_future.done():
                        meta_info = res.get("meta_info", {})
                        sglang_request_id = meta_info.get("id")
                        if sglang_request_id:
                            request_id_future.set_result(sglang_request_id)
                            logging.debug(
                                f"New Prefill Request ID: {sglang_request_id}"
                            )

                if forced_shutdown_event.is_set():
                    raise GeneratorExit(
                        "SGLang prefill request was aborted during forced worker shutdown."
                    )
        except asyncio.CancelledError:
            raise GeneratorExit(
                "SGLang prefill engine was shut down during token generation."
            ) from None
