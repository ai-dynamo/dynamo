# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl
from opentelemetry import propagate, trace
from sglang.srt.tracing import trace as sglang_trace

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

    def _propagate_trace_context_to_sglang(self, rid: str):
        """Propagate current OTEL trace context to SGLang.

        Lightweight: Returns immediately if tracing is disabled or no valid span exists.

        Args:
            rid: Request ID to associate with the trace context.
        """
        # CRITICAL PATH OPTIMIZATION: Early exit if tracing is disabled
        if not sglang_trace.tracing_enabled or not rid:
            return

        # Fast path: Check if we have a valid span before doing any work
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context() if current_span else None
        if not span_context or not span_context.is_valid():
            return

        # Only do expensive operations if tracing is enabled and span is valid
        carrier: dict[str, str] = {}
        propagate.inject(carrier)

        # Build trace context in SGLang's expected format
        trace_context = {
            "root_span": carrier,
            "prev_span": {
                "span_id": span_context.span_id,
                "trace_id": span_context.trace_id,
            }
        }

        # Propagate to SGLang (has internal guard but we've already checked)
        sglang_trace.trace_set_proc_propagate_context(rid, trace_context)

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
        trace_id = context.trace_id
        bootstrap_room = self._generate_bootstrap_room()

        bootstrap_info = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

        yield bootstrap_info

        input_param = self._get_input_param(request["request"])

        # Propagate trace context to SGLang
        self._propagate_trace_context_to_sglang(trace_id)

        results = await self.engine.async_generate(
            **input_param,
            sampling_params=request["sampling_params"],
            stream=True,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=bootstrap_room,
            rid=trace_id,
        )

        task = asyncio.create_task(self._consume_results(results, context))
        self._consume_tasks.add(task)
        task.add_done_callback(self._consume_tasks.discard)

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
