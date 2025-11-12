# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl
from opentelemetry import propagate, trace
from sglang.srt.tracing import trace as sglang_trace
from sglang.srt.tracing.trace import SglangTracePropagateContext

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

    def _propagate_trace_context_to_sglang(self, rid: str, context: Context, bootstrap_room: int):
        """Propagate Dynamo's trace context to SGLang via remote_trace_contexts.

        Lightweight: Returns immediately if tracing is disabled or no trace context exists.

        Args:
            rid: Request ID to associate with the trace context.
            context: Dynamo Context object containing trace information from Rust.
            bootstrap_room: Bootstrap room ID for disaggregated mode.
        """
        logging.info(f"[TRACE PROPAGATION] Called for rid={rid}, bootstrap_room={bootstrap_room}, sglang_tracing_enabled={sglang_trace.tracing_enabled}")

        # CRITICAL PATH OPTIMIZATION: Early exit if tracing is disabled
        if not sglang_trace.tracing_enabled or not rid:
            logging.info(f"[TRACE PROPAGATION] Skipping - tracing_enabled={sglang_trace.tracing_enabled}, rid={rid}")
            return

        # Get trace context from Dynamo (Rust side)
        trace_id = context.trace_id
        span_id = context.span_id

        logging.info(f"[TRACE PROPAGATION] Dynamo context: trace_id={trace_id}, span_id={span_id}, parent_span_id={context.parent_span_id}")

        if not trace_id or not span_id:
            logging.warning(f"[TRACE PROPAGATION] Missing trace context - trace_id={trace_id}, span_id={span_id}")
            return

        # Build W3C traceparent: version-trace_id-parent_span_id-flags
        # Use Dynamo's current span as the parent for SGLang
        traceparent = f"00-{trace_id}-{span_id}-01"
        logging.info(f"[TRACE PROPAGATION] Built traceparent: {traceparent}")

        # Build trace context in SGLang's expected format (for remote propagation)
        carrier = {"traceparent": traceparent}

        # Extract OTEL context from the carrier
        otel_context = propagate.extract(carrier)

        # Build the propagate context for this bootstrap room
        trace_context = {
            str(bootstrap_room): {
                "root_span": carrier,
                "prev_span": {
                    "span_id": int(span_id, 16),  # Convert hex to int
                    "trace_id": int(trace_id, 16),
                }
            }
        }

        # Encode as base64 (like HTTP headers do)
        import json
        import base64
        json_str = json.dumps(trace_context, ensure_ascii=False)
        base64_context = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

        logging.info(f"[TRACE PROPAGATION] Calling trace_set_remote_propagate_context with bootstrap_room={bootstrap_room}")
        # Propagate to SGLang's remote_trace_contexts (like HTTP headers)
        sglang_trace.trace_set_remote_propagate_context(base64_context)
        logging.info(f"[TRACE PROPAGATION] Successfully propagated trace context for rid={rid}, bootstrap_room={bootstrap_room}")

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

        # Propagate trace context to SGLang with bootstrap_room
        self._propagate_trace_context_to_sglang(trace_id, context, bootstrap_room)

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
