#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use SGLang for input and output processing
#

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as _futures_wait
from dataclasses import dataclass
from typing import Any

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from dynamo._core import Client
from dynamo._internal import ModelDeploymentCard
from dynamo.frontend.frontend_args import FrontendConfig
from dynamo.llm import (
    KvRouter,
    ModelCardInstanceId,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    fetch_model,
)
from dynamo.llm.exceptions import InvalidArgument, Unknown
from dynamo.runtime import DistributedRuntime

from .sglang_prepost import (
    SglangStreamingPostProcessor,
    ToolCallParserType,
    _get_history_tool_calls_count,
    convert_tools,
    create_parsers,
    preprocess_chat_request,
)
from .utils import PreprocessError, extract_mm_urls, random_uuid, worker_warmup

logger = logging.getLogger(__name__)


def _runtime_config_parser_name(
    mdc: ModelDeploymentCard,
    key: str,
) -> str | None:
    runtime_config = mdc.runtime_config()
    if not isinstance(runtime_config, dict):
        return None
    value = runtime_config.get(key)
    return value if isinstance(value, str) and value else None


def _unsupported_n_message(n: int) -> str:
    return f"Unsupported value: 'n={n}'. " "This endpoint currently supports only n=1."


def _engine_error_message(
    engine_response: Any,
    request_id: str,
) -> str:
    """Extract a human-readable message from an invalid engine response."""
    if isinstance(engine_response, dict) and engine_response.get("status") == "error":
        backend_msg = engine_response.get("message") or "unknown backend error"
        return f"Backend error for request {request_id}: {backend_msg}"
    return f"Invalid engine response for request {request_id}"


_FINISH_REASON_MAP: dict[str, str] = {
    "eos": "stop",
    "stop": "stop",
    "length": "length",
    "error": "error",
    "abort": "stop",
    "cancelled": "stop",
    "content_filter": "stop",
}


def _map_finish_reason(raw: str | None) -> str | None:
    """Map Dynamo router finish reasons to OpenAI finish reasons.

    Exact matches use the dict.  Prefixed variants (``error:timeout``,
    ``abort:cancelled``) are handled by ``startswith`` fallbacks.
    """
    if raw is None:
        return None
    mapped = _FINISH_REASON_MAP.get(raw)
    if mapped is not None:
        return mapped
    if raw.startswith("error"):
        return "error"
    if raw.startswith("abort"):
        return "stop"
    return raw


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per process by _init_worker)
# ---------------------------------------------------------------------------
_w_tokenizer: Any = None
_w_tool_call_parser_name: str | None = None
_w_reasoning_parser_name: str | None = None
_w_exclude_tools_when_tool_choice_none: bool = True


@dataclass
class SglangPreprocessWorkerResult:
    """Picklable return value from the SGLang preprocess worker."""

    prompt_token_ids: list[int]
    dynamo_preproc: dict[str, Any]
    request: dict[str, Any]


def _init_worker(
    model_path: str,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
    exclude_tools_when_tool_choice_none: bool = True,
    trust_remote_code: bool = False,
) -> None:
    """Initialize a worker process with its own tokenizer."""
    global _w_tokenizer, _w_tool_call_parser_name, _w_reasoning_parser_name
    global _w_exclude_tools_when_tool_choice_none
    _w_tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code)
    _w_tool_call_parser_name = tool_call_parser_name
    _w_reasoning_parser_name = reasoning_parser_name
    _w_exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none


def _preprocess_worker(
    request: dict[str, Any],
    model_name: str,
    eos_token_id: int | None,
) -> SglangPreprocessWorkerResult:
    """Preprocess a request in a worker process and return a picklable result."""
    pre = preprocess_chat_request(
        request,
        tokenizer=_w_tokenizer,
        tool_call_parser_name=_w_tool_call_parser_name,
        reasoning_parser_name=_w_reasoning_parser_name,
        exclude_tools_when_tool_choice_none=_w_exclude_tools_when_tool_choice_none,
    )

    n = request.get("n", 1)
    if n != 1:
        raise PreprocessError(_unsupported_n_message(n))

    dynamo_preproc = _build_dynamo_preproc(
        request,
        pre.prompt_token_ids,
        model_name,
        eos_token_id,
        pre.guided_decoding,
        pre.tool_call_parser,
    )

    return SglangPreprocessWorkerResult(
        prompt_token_ids=pre.prompt_token_ids,
        dynamo_preproc=dynamo_preproc,
        request=request,
    )


def _build_dynamo_preproc(
    request: dict[str, Any],
    prompt_token_ids: list[int],
    model_name: str,
    eos_token_id: int | None,
    guided_decoding: dict[str, Any] | None = None,
    tool_call_parser: ToolCallParserType | None = None,
) -> dict[str, Any]:
    """Build the Dynamo preprocessed request dict from request fields."""
    max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")

    stop = request.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif stop is None:
        stop = []

    stop_token_ids = request.get("stop_token_ids", [])

    # Handle logprobs
    logprobs_val = None
    logprobs = request.get("logprobs")
    top_logprobs = request.get("top_logprobs")
    if logprobs is True:
        logprobs_val = top_logprobs or 1
    elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
        logprobs_val = logprobs
    elif top_logprobs not in (None, 0):
        logprobs_val = top_logprobs

    preproc = {
        "model": model_name,
        "token_ids": prompt_token_ids,
        "stop_conditions": {
            "max_tokens": max_tokens,
            "stop": stop,
            "stop_token_ids": stop_token_ids,
            "min_tokens": request.get("min_tokens", 0),
            "ignore_eos": request.get("ignore_eos", False),
        },
        "sampling_options": {
            "n": request.get("n", 1),
            "presence_penalty": request.get("presence_penalty", 0.0),
            "frequency_penalty": request.get("frequency_penalty", 0.0),
            "repetition_penalty": request.get("repetition_penalty", 1.0),
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 1.0),
            # SGLang uses -1 for "disabled", OpenAI/vLLM use 0
            "top_k": request.get("top_k", 0) or -1,
            "min_p": request.get("min_p", 0.0),
            "seed": request.get("seed"),
            "guided_decoding": guided_decoding,
        },
        "output_options": {
            "logprobs": logprobs_val,
            "prompt_logprobs": None,
            # Preserve special tokens only when a tool-call parser is
            # actually active — the parser needs delimiter tokens
            # (e.g. <|tool_call|>) to detect calls. Mirrors the
            # post-processor's _skip_special_tokens logic.
            "skip_special_tokens": tool_call_parser is None,
        },
        "eos_token_ids": [eos_token_id] if eos_token_id is not None else [],
        "annotations": [],
        "routing": request.get("routing"),
    }

    # Forward multimodal URLs so the backend handler can load the media.
    mm_data = extract_mm_urls(request.get("messages", []))
    if mm_data:
        preproc["multi_modal_data"] = mm_data

    return preproc


class SglangProcessor:
    def __init__(
        self,
        tokenizer,
        router,  # Client or KvRouter
        tool_call_parser_name: str | None,
        reasoning_parser_name: str | None,
        eos_token_id: int | None,
        debug_perf: bool = False,
        preprocess_pool: ProcessPoolExecutor | None = None,
        preprocess_workers: int = 0,
        stream_interval: int = 1,
    ):
        self.tokenizer = tokenizer
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.tool_call_parser_name = tool_call_parser_name
        self.reasoning_parser_name = reasoning_parser_name
        self.exclude_tools_when_tool_choice_none = True
        self.eos_token_id = eos_token_id
        self.debug_perf = debug_perf
        self.stream_interval = stream_interval
        self.preprocess_pool = preprocess_pool
        if preprocess_pool is not None:
            self._worker_semaphore: asyncio.Semaphore | None = asyncio.Semaphore(
                preprocess_workers + 2
            )
        else:
            self._worker_semaphore = None

    async def generator(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Main entry point: preprocess, route, post-process a chat request."""
        if self.debug_perf:
            from .perf_instrumentation import (  # type: ignore[import-not-found, import-untyped]
                enter_generator,
                exit_generator,
            )

            active = enter_generator()
            t_start = time.monotonic()
            logger.info("[perf] sglang generator enter: active_requests=%d", active)

        try:
            if self.preprocess_pool is None:
                async for item in self._generator_inner(request):
                    yield item
            else:
                async for item in self._generator_inner_pool(request):
                    yield item
        finally:
            if self.debug_perf:
                active = exit_generator()
                elapsed_ms = (time.monotonic() - t_start) * 1000.0
                logger.info(
                    "[perf] sglang generator exit: total=%.2fms active_requests=%d",
                    elapsed_ms,
                    active,
                )

    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Single-process path: preprocess, dispatch, stream post-process."""
        request_id = random_uuid()

        try:
            if self.debug_perf:
                t0 = time.monotonic()

            pre = preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                tool_call_parser_name=self.tool_call_parser_name,
                reasoning_parser_name=self.reasoning_parser_name,
                exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
            )

            if self.debug_perf:
                t1 = time.monotonic()
                logger.info(
                    "[perf] sglang preprocess: %.2fms (request=%s)",
                    (t1 - t0) * 1000.0,
                    request_id,
                )

            tokens = pre.prompt_token_ids

            n = request.get("n", 1)
            if n != 1:
                logger.error("Unsupported n=%d, only n=1 is supported", n)
                raise InvalidArgument(_unsupported_n_message(n))

            dynamo_preproc = _build_dynamo_preproc(
                request,
                tokens,
                request["model"],
                self.eos_token_id,
                pre.guided_decoding,
                pre.tool_call_parser,
            )
        except InvalidArgument:
            raise
        except Exception as exc:
            logger.exception("SGLang preprocessing failed for request %s", request_id)
            raise Unknown(f"Preprocessing error: {exc}") from exc

        post = SglangStreamingPostProcessor(
            tokenizer=self.tokenizer,
            tool_call_parser=pre.tool_call_parser,
            reasoning_parser=pre.reasoning_parser,
            history_tool_calls_count=_get_history_tool_calls_count(
                request.get("messages", [])
            ),
            sglang_tools=convert_tools(request.get("tools")),
            tool_call_parser_name=self.tool_call_parser_name,
        )

        async for item in self._generate_and_stream(
            request_id, request, dynamo_preproc, tokens, post
        ):
            yield item

    async def _generator_inner_pool(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Pool path: preprocess in worker, stream in main process."""
        request_id = random_uuid()

        # --- Phase 1: Preprocess (semaphore held) ---
        assert self._worker_semaphore is not None
        assert self.preprocess_pool is not None
        try:
            async with self._worker_semaphore:
                future = self.preprocess_pool.submit(
                    _preprocess_worker,
                    request,
                    request["model"],
                    self.eos_token_id,
                )
                preproc_result: SglangPreprocessWorkerResult = (
                    await asyncio.wrap_future(future)
                )
        except PreprocessError as exc:
            raise InvalidArgument(str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "SGLang worker preprocessing failed for request %s", request_id
            )
            raise Unknown(f"Worker error: {exc}") from exc

        # --- Phase 2: Recreate parsers in main process (not picklable) ---
        tool_call_parser, reasoning_parser = create_parsers(
            request,
            tool_call_parser_name=self.tool_call_parser_name,
            reasoning_parser_name=self.reasoning_parser_name,
        )

        post = SglangStreamingPostProcessor(
            tokenizer=self.tokenizer,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            history_tool_calls_count=_get_history_tool_calls_count(
                request.get("messages", [])
            ),
            sglang_tools=convert_tools(request.get("tools")),
            tool_call_parser_name=self.tool_call_parser_name,
        )

        async for item in self._generate_and_stream(
            request_id,
            request,
            preproc_result.dynamo_preproc,
            preproc_result.prompt_token_ids,
            post,
        ):
            yield item

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        post: SglangStreamingPostProcessor,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Shared streaming logic for both single-process and pool paths."""
        token_count = 0
        post_proc_total_ms = 0.0
        created_ts = int(time.time())
        stream_interval = self.stream_interval

        try:
            if self.is_kv_router:
                dynamo_stream = await self.router.generate(
                    token_ids=tokens,
                    model=dynamo_preproc["model"],
                    stop_conditions=dynamo_preproc["stop_conditions"],
                    sampling_options=dynamo_preproc["sampling_options"],
                    output_options=dynamo_preproc["output_options"],
                    multi_modal_data=dynamo_preproc.get("multi_modal_data"),
                )
            else:
                dynamo_stream = await self.router.generate(
                    dynamo_preproc, annotated=False
                )

            # Accumulate tokens for batched detokenization when
            # stream_interval > 1.  Flush every N tokens or on
            # finish_reason.  Use si=1 for the first chunk to minimize
            # TTFT, then switch to the configured interval.
            pending_token_ids: list[int] = []
            pending_usage: dict[str, Any] | None = None
            first_chunk = True

            async for dynamo_response in dynamo_stream:
                if self.is_kv_router:
                    engine_response = dynamo_response
                elif hasattr(dynamo_response, "data"):
                    engine_response = dynamo_response.data()
                else:
                    engine_response = dynamo_response

                if (
                    not isinstance(engine_response, dict)
                    or "token_ids" not in engine_response
                ):
                    msg = _engine_error_message(engine_response, request_id)
                    logger.error(
                        "Engine returned an invalid response for request %s: %s",
                        request_id,
                        engine_response,
                    )
                    raise Unknown(msg)

                new_ids = engine_response["token_ids"]
                raw_finish = engine_response.get("finish_reason")
                finish_reason = _map_finish_reason(raw_finish)

                if usage := engine_response.get("completion_usage"):
                    pending_usage = usage

                pending_token_ids.extend(new_ids)

                # Flush on finish or when we've accumulated enough tokens.
                # First chunk flushes immediately (si=1) to minimize TTFT.
                flush_threshold = 1 if first_chunk else stream_interval
                if finish_reason or len(pending_token_ids) >= flush_threshold:
                    mapped_response = {
                        "token_ids": pending_token_ids,
                        "finish_reason": finish_reason,
                    }

                    if self.debug_perf:
                        t_pp0 = time.monotonic()

                    choice = post.process_output(mapped_response)

                    if self.debug_perf:
                        t_pp1 = time.monotonic()
                        post_proc_total_ms += (t_pp1 - t_pp0) * 1000.0
                        token_count += len(pending_token_ids)

                    if choice:
                        dynamo_out: dict[str, Any] = {
                            "id": request_id,
                            "choices": [choice],
                            "created": created_ts,
                            "model": request["model"],
                            "object": "chat.completion.chunk",
                        }
                        if pending_usage:
                            dynamo_out["usage"] = pending_usage

                        yield dynamo_out

                    pending_token_ids = []
                    pending_usage = None
                    first_chunk = False
        except Unknown:
            raise
        except Exception as e:
            logger.exception("Error generating response for request %s", request_id)
            raise Unknown(
                f"Error generating response for request {request_id}: {e}"
            ) from e
        finally:
            if self.debug_perf and token_count > 0:
                logger.info(
                    "[perf] sglang stream done: request=%s tokens=%d "
                    "post_processor_total=%.2fms (%.3fms/tok)",
                    request_id,
                    token_count,
                    post_proc_total_ms,
                    post_proc_total_ms / token_count,
                )


class SglangEngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        config: FrontendConfig,
        debug_perf: bool = False,
        tool_call_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
    ):
        self.runtime = runtime
        self.router_config = router_config
        self.config = config
        self.debug_perf = debug_perf
        self.tool_call_parser_name = tool_call_parser_name
        self.reasoning_parser_name = reasoning_parser_name

        self.trust_remote_code = config.trust_remote_code
        self.stream_interval = 20
        raw_stream_interval = os.getenv("DYN_SGLANG_STREAM_INTERVAL")
        if raw_stream_interval:
            try:
                self.stream_interval = max(1, int(raw_stream_interval))
            except ValueError:
                logger.warning(
                    "Invalid DYN_SGLANG_STREAM_INTERVAL=%r, using default=%d",
                    raw_stream_interval,
                    self.stream_interval,
                )

    async def chat_engine_factory(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
    ) -> PythonAsyncEngine:
        """Called by Rust when a model is discovered."""
        model_type = mdc.model_type()
        if not model_type.supports_chat():
            raise RuntimeError(
                f"model type {model_type} is not supported by this factory"
            )
        loop = asyncio.get_running_loop()

        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_model(source_path, ignore_weights=True)

        logger.info("Loading SGLang tokenizer from %s", source_path)
        tokenizer = get_tokenizer(source_path, trust_remote_code=self.trust_remote_code)

        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        tool_call_parser_name = (
            self.tool_call_parser_name
            or _runtime_config_parser_name(mdc, "tool_call_parser")
        )
        reasoning_parser_name = (
            self.reasoning_parser_name
            or _runtime_config_parser_name(mdc, "reasoning_parser")
        )

        if tool_call_parser_name:
            logger.info("SGLang tool call parser: %s", tool_call_parser_name)
        if reasoning_parser_name:
            logger.info("SGLang reasoning parser: %s", reasoning_parser_name)

        (namespace_name, component_name, endpoint_name) = instance_id.triple()
        generate_endpoint = self.runtime.endpoint(
            f"{namespace_name}.{component_name}.{endpoint_name}"
        )
        router: Client | KvRouter
        if self.router_config.router_mode == RouterMode.KV:
            router = KvRouter(
                endpoint=generate_endpoint,
                block_size=self.config.kv_cache_block_size or 16,
                kv_router_config=self.router_config.kv_router_config,
            )
        else:
            router = await generate_endpoint.client(
                router_mode=self.router_config.router_mode
            )

        preprocess_pool = None
        preprocess_workers = self.config.preprocess_workers
        if preprocess_workers > 0:
            logger.info(
                "Creating SGLang preprocess worker pool with %d workers for %s",
                preprocess_workers,
                source_path,
            )
            preprocess_pool = ProcessPoolExecutor(
                max_workers=preprocess_workers,
                initializer=_init_worker,
                initargs=(
                    source_path,
                    tool_call_parser_name,
                    reasoning_parser_name,
                    self.config.exclude_tools_when_tool_choice_none,
                    self.trust_remote_code,
                ),
            )
            futures = [
                preprocess_pool.submit(worker_warmup) for _ in range(preprocess_workers)
            ]
            done, not_done = _futures_wait(futures, timeout=120)
            if not_done:
                for f in not_done:
                    f.cancel()
                preprocess_pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    "Timed out waiting for SGLang preprocess worker pool warmup"
                )
            try:
                for f in done:
                    f.result()
            except Exception:
                preprocess_pool.shutdown(wait=False, cancel_futures=True)
                raise
            logger.info(
                "SGLang preprocess worker pool ready (%d workers)", preprocess_workers
            )

        logger.info("SGLang processor stream_interval=%d", self.stream_interval)

        gen = SglangProcessor(
            tokenizer,
            router,
            tool_call_parser_name,
            reasoning_parser_name,
            eos_token_id,
            debug_perf=self.debug_perf,
            preprocess_pool=preprocess_pool,
            preprocess_workers=preprocess_workers,
            stream_interval=self.stream_interval,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
