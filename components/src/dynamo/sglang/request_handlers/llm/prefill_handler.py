# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import sys
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

    _REQUEST_REGISTRATION_TIMEOUT_SECONDS = 5.0

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

    async def cleanup_async(self) -> None:
        """Await pending prefill consumers before shutting down the engine."""
        tasks = list(self._consume_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logging.error(
                        "Prefill consumer failed during handler cleanup",
                        exc_info=(type(result), result, result.__traceback__),
                    )
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
        sglang_request_id = trace_id or context.id()

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
        if native_payload is None:
            sampling_params["n"] = 1
            sampling_params["max_new_tokens"] = 1

        # Use provided bootstrap_info if available (e.g., for health checks with FAKE_BOOTSTRAP_HOST)
        # Otherwise use real bootstrap host/port from engine and generate room locally
        bootstrap_host = self.bootstrap_host
        bootstrap_port = self.bootstrap_port
        bootstrap_room = None

        bootstrap_info_from_req = inner_request.get("bootstrap_info")
        if isinstance(bootstrap_info_from_req, dict):
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
            bootstrap_room = bootstrap_info_from_req.get("bootstrap_room")
            if bootstrap_room is not None:
                logging.debug(f"Using router-provided bootstrap_room: {bootstrap_room}")

        if bootstrap_room is None:
            bootstrap_room = self._generate_bootstrap_room()
            logging.debug(f"Generated bootstrap_room locally: {bootstrap_room}")

        bootstrap_info = {
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

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
        if native_payload is not None:
            input_ids = input_param.get("input_ids")
            if not isinstance(input_ids, list):
                raise ValueError("native SGLang Generate requires token input")
            native_request = build_native_generate_request(
                native_payload,
                input_ids=input_ids,
                fallback_rid=sglang_request_id,
                priority=priority_kwargs.get("priority"),
                sampling_overrides={"n": 1, "max_new_tokens": 1},
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
                external_trace_header=trace_header,
                routed_dp_rank=dp_rank,
                lora_path=lora_path,
            )
            if not isinstance(native_request.rid, str):
                raise ValueError("SGLang prefill requires a single request ID")
            sglang_request_id = native_request.rid
            results = native_generate_stream(self.engine, native_request)
        else:
            results = await self.engine.async_generate(
                **input_param,
                **mm_kwargs,
                sampling_params=sampling_params,
                stream=True,
                **require_reasoning_kwargs(self.engine, inner_request),
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
                external_trace_header=trace_header,
                rid=sglang_request_id,
                data_parallel_rank=dp_rank,
                lora_path=lora_path,
                **priority_kwargs,
            )
        if inner_request.get(HEALTH_CHECK_KEY):
            # Canary: stream engine output so the Rust canary sees scheduler output.
            # No _cancellation_monitor — probe is bounded (max_tokens=1, FAKE_BOOTSTRAP_HOST).
            async for res in results:
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

        task = asyncio.create_task(
            self._consume_results(results, sglang_request_id, context)
        )
        self._consume_tasks.add(task)
        task.add_done_callback(self._consume_tasks.discard)

        await task

    async def _wait_for_request_registration(self, rid: str) -> None:
        """Wait until SGLang owns the request ID, without waiting for output."""
        tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
        rid_to_state = getattr(tokenizer_manager, "rid_to_state", None)
        if rid_to_state is None:
            raise RuntimeError("SGLang tokenizer manager has no request registry")

        async def poll_registry() -> None:
            while rid not in rid_to_state:
                await asyncio.sleep(0.001)

        try:
            await asyncio.wait_for(
                poll_registry(),
                timeout=self._REQUEST_REGISTRATION_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"SGLang did not register prefill request {rid} within "
                f"{self._REQUEST_REGISTRATION_TIMEOUT_SECONDS:g}s"
            ) from error

    async def _consume_results(
        self,
        results: AsyncIterator[Any],
        sglang_request_id: str,
        context: Context,
    ) -> None:
        """Consume async generator results without processing.

        Args:
            results: Async generator from engine.async_generate.
            sglang_request_id: Request ID submitted to SGLang.
            context: Context object for cancellation handling.
        """
        request_id_future: asyncio.Future[str] = asyncio.Future()
        registration_task = asyncio.create_task(
            self._wait_for_request_registration(sglang_request_id)
        )
        first_result_task: asyncio.Task[Any] | None = asyncio.create_task(
            anext(results)
        )
        pre_registration_cancellation = context.async_killed_or_stopped()
        first_result_ready = False

        try:
            # Advancing the lazy iterator registers the request. A disconnect
            # observed before registration remains sticky; keep the iterator
            # alive until the exact RID can be aborted safely.
            while not registration_task.done():
                wait_for: set[asyncio.Future[Any]] = {registration_task}
                if first_result_task is not None and not first_result_task.done():
                    wait_for.add(first_result_task)
                if not pre_registration_cancellation.done():
                    wait_for.add(pre_registration_cancellation)

                done, _ = await asyncio.wait(
                    wait_for,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if first_result_task is not None and first_result_task in done:
                    try:
                        await first_result_task
                    except StopAsyncIteration as error:
                        raise RuntimeError(
                            "SGLang prefill stream ended before request registration"
                        ) from error
                    first_result_ready = True

            await registration_task
            request_id_future.set_result(sglang_request_id)
            logging.debug(f"Registered Prefill Request ID: {sglang_request_id}")

            if not pre_registration_cancellation.done():
                pre_registration_cancellation.cancel()
                try:
                    await pre_registration_cancellation
                except asyncio.CancelledError:
                    pass

            async with self._cancellation_monitor(request_id_future, context):
                if not first_result_ready:
                    assert first_result_task is not None
                    try:
                        await first_result_task
                    except StopAsyncIteration:
                        return
                first_result_task = None

                # Keep draining after abort so accepted KV-transfer work can
                # release its resources before the consumer exits.
                async for _ in results:
                    pass
        finally:
            pending_exception = sys.exc_info()[1]
            cleanup_tasks: list[tuple[asyncio.Future[Any], bool]] = [
                (registration_task, False),
                (pre_registration_cancellation, False),
            ]
            if first_result_task is not None:
                cleanup_tasks.append((first_result_task, True))

            for task, _ in cleanup_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(
                *(task for task, _ in cleanup_tasks),
                return_exceptions=True,
            )
            for (_, allow_stop_iteration), result in zip(
                cleanup_tasks, results, strict=True
            ):
                if isinstance(result, asyncio.CancelledError) or (
                    allow_stop_iteration and isinstance(result, StopAsyncIteration)
                ):
                    continue
                if not isinstance(result, BaseException):
                    continue
                if pending_exception is None:
                    raise result
                if result is not pending_exception:
                    logging.error(
                        "SGLang prefill task failed during cleanup",
                        exc_info=(type(result), result, result.__traceback__),
                    )
