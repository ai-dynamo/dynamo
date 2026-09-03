# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, AsyncGenerator, Dict, Optional

import sglang as sgl

from dynamo._core import Context
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang._compat import require_reasoning_kwargs
from dynamo.sglang.args import Config
from dynamo.sglang.engine_generate import (
    build_native_generate_request,
    native_generate_payload,
    native_generate_stream,
)
from dynamo.sglang.parallel_sampling import (
    BOOTSTRAP_ROOMS_KEY,
    choice_request_ids,
    requested_parallel_samples,
    resolve_prefill_bootstrap_rooms,
)
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.request_handlers.llm.decode_handler import _sampling_option_params
from dynamo.sglang.request_handlers.llm.mm_disagg_utils import (
    build_disagg_mm_kwargs,
    raise_if_unextracted_multimodal,
)

# Sentinel value matching u32::MAX from the C/Go prefill-routing ABI.
# This remains as a compatibility fallback for older callers that still encode
# an unresolved data-parallel rank in-band instead of omitting the field.
_DP_RANK_UNSET = 2**32 - 1


class PrefillWorkerHandler(BaseWorkerHandler):
    """Handler for prefill workers in disaggregated serving mode."""

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize prefill worker handler.

        Args:
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
        """
        self.engine = engine
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(self.engine)
        super().__init__(engine, config, publisher, generate_endpoint, shutdown_event)
        self._consume_tasks: set[asyncio.Task[Any]] = set()
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

        super().cleanup()
        self.engine.shutdown()
        logging.info("Prefill engine shutdown")

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
                "n": sampling_opts.get("n"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                **_sampling_option_params(sampling_opts),
                **self._get_guided_decoding_params(
                    sampling_opts.get("guided_decoding")
                ),
            }
            sampling_params = {
                k: v for k, v in sampling_params.items() if v is not None
            }
        native_payload = native_generate_payload(inner_request)
        # n > 1 runs as n single-sample sub-requests, one room each (see
        # dynamo.sglang.parallel_sampling). Native Generate is always n == 1.
        num_choices = 1
        if native_payload is None:
            num_choices = requested_parallel_samples(sampling_params)
            sampling_params["n"] = 1
            sampling_params["max_new_tokens"] = 1

        # Use provided bootstrap_info if available (e.g., for health checks with FAKE_BOOTSTRAP_HOST)
        # Otherwise use real bootstrap host/port from engine and generate room locally
        bootstrap_host = self.bootstrap_host
        bootstrap_port = self.bootstrap_port

        bootstrap_info_from_req = inner_request.get("bootstrap_info")
        if not isinstance(bootstrap_info_from_req, dict):
            bootstrap_info_from_req = None
        if bootstrap_info_from_req is not None:
            # Allow overriding bootstrap_host for fake-transfer mode (health checks)
            if "bootstrap_host" in bootstrap_info_from_req:
                bootstrap_host = bootstrap_info_from_req["bootstrap_host"]
                logging.debug(
                    f"Using request-provided bootstrap_host: {bootstrap_host}"
                )
            if "bootstrap_port" in bootstrap_info_from_req:
                bootstrap_port = bootstrap_info_from_req["bootstrap_port"]
                logging.debug(
                    f"Using request-provided bootstrap_port: {bootstrap_port}"
                )

        # One room per choice: router-drawn when available, else drawn here.
        bootstrap_rooms = resolve_prefill_bootstrap_rooms(
            bootstrap_info_from_req, num_choices, self._generate_bootstrap_room
        )
        logging.debug(f"Using bootstrap rooms: {bootstrap_rooms}")

        bootstrap_info: Dict[str, Any] = {
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_rooms[0],
        }
        if num_choices > 1:
            bootstrap_info[BOOTSTRAP_ROOMS_KEY] = bootstrap_rooms

        input_param = self._get_input_param(inner_request)

        # Prefill encodes the media so the KV it transfers carries the vision
        # context; decode extracts the same URLs to match the token layout.
        raise_if_unextracted_multimodal(inner_request)
        mm_kwargs = build_disagg_mm_kwargs(inner_request)

        routing = inner_request.get("routing") or {}
        priority = routing.get("priority")
        dp_rank = routing.get("dp_rank")

        if dp_rank is not None and dp_rank == _DP_RANK_UNSET:
            dp_rank = None

        trace_header = context.trace_headers() if self.enable_trace else None

        lora_path = self._resolve_lora(inner_request)
        if lora_path:
            logging.debug(
                f"Prefill request {context.id()} will use LoRA adapter: {lora_path}"
            )

        priority_kwargs = self._priority_kwargs(priority)
        results: list[AsyncIterator[Any]] = []
        if native_payload is not None:
            input_ids = input_param.get("input_ids")
            if not isinstance(input_ids, list):
                raise ValueError("native SGLang Generate requires token input")
            native_request = build_native_generate_request(
                native_payload,
                input_ids=input_ids,
                fallback_rid=trace_id or context.id(),
                priority=priority_kwargs.get("priority"),
                sampling_overrides={"n": 1, "max_new_tokens": 1},
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_rooms[0],
                external_trace_header=trace_header,
                routed_dp_rank=dp_rank,
                lora_path=lora_path,
            )
            results.append(native_generate_stream(self.engine, native_request))
        else:
            reasoning_kwargs = require_reasoning_kwargs(self.engine, inner_request)
            rids = choice_request_ids(trace_id, context.id(), num_choices)
            for bootstrap_room, rid in zip(bootstrap_rooms, rids):
                results.append(
                    await self.engine.async_generate(
                        **input_param,
                        **mm_kwargs,
                        sampling_params=sampling_params,
                        stream=True,
                        **reasoning_kwargs,
                        bootstrap_host=bootstrap_host,
                        bootstrap_port=bootstrap_port,
                        bootstrap_room=bootstrap_room,
                        external_trace_header=trace_header,
                        rid=rid,
                        data_parallel_rank=dp_rank,
                        lora_path=lora_path,
                        **priority_kwargs,
                    )
                )
        if inner_request.get(HEALTH_CHECK_KEY):
            # Canary: stream engine output so the Rust canary sees scheduler output.
            # No _cancellation_monitor — probe is bounded (max_tokens=1, FAKE_BOOTSTRAP_HOST).
            async for res in results[0]:
                yield res
            return

        # Yield bootstrap_info for PrefillRouter - required for async generator
        # contract and Rust-side expects disaggregated_params in first output.
        yield {
            "token_ids": [],
            "text": None,
            "finish_reason": None,
            "disaggregated_params": bootstrap_info,
        }

        # Every sub-request must be iterated for SGLang to submit it, so each
        # gets its own consumer; the handoff completes once all of them have.
        # A failing consumer leaves its siblings running, bounded by
        # max_new_tokens=1 and SGLang's bootstrap timeout.
        tasks = []
        for choice_results in results:
            task = asyncio.create_task(self._consume_results(choice_results, context))
            self._consume_tasks.add(task)
            task.add_done_callback(self._consume_tasks.discard)
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _consume_results(
        self, results: AsyncIterator[Any], context: Context
    ) -> None:
        """Consume async generator results without processing.

        Args:
            results: Async generator from engine.async_generate.
            context: Context object for cancellation handling.
        """
        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future: asyncio.Future[str] = asyncio.Future()
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
