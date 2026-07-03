#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use SGLang for input and output processing
#

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as _futures_wait
from dataclasses import dataclass
from typing import Any

from sglang.srt.parser.conversation import chat_template_exists
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from dynamo._internal import ModelDeploymentCard
from dynamo.frontend.frontend_args import FrontendConfig
from dynamo.llm import ModelCardInstanceId, PythonAsyncEngine, RoutedEngine
from dynamo.llm.exceptions import InvalidArgument, Unknown

from .sglang_prepost import (
    SglangStreamingPostProcessor,
    ToolCallParserType,
    _client_wants_separate_reasoning,
    _get_history_tool_calls_count,
    convert_tools,
    create_parsers,
    detect_force_reasoning_from_template,
    preprocess_chat_request,
)
from .utils import (
    PreprocessError,
    extract_mm_urls,
    handle_engine_error,
    make_internal_error,
    nvext_extra_field_requested,
    random_uuid,
    read_jinja_chat_template,
    resolve_chat_template,
    worker_warmup,
)

logger = logging.getLogger(__name__)


def _cached_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        return None
    cached_tokens = prompt_details.get("cached_tokens")
    return cached_tokens if isinstance(cached_tokens, int) else None


def _normalize_eos_token_ids(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, (list, tuple, set)):
        token_ids: list[int] = []
        seen: set[int] = set()
        for token_id in value:
            if isinstance(token_id, int) and not isinstance(token_id, bool):
                if token_id not in seen:
                    token_ids.append(token_id)
                    seen.add(token_id)
        return token_ids
    return []


def _tokenizer_eos_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = _normalize_eos_token_ids(getattr(tokenizer, "eos_token_ids", None))
    if eos_token_ids:
        return eos_token_ids
    return _normalize_eos_token_ids(getattr(tokenizer, "eos_token_id", None))


def _load_tokenizer(source_path: str, trust_remote_code: bool):
    """Load the SGLang tokenizer, falling back to an on-disk chat template
    (e.g. chat_template.json) when the tokenizer defines none."""
    tokenizer = get_tokenizer(source_path, trust_remote_code=trust_remote_code)
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = resolve_chat_template(source_path, backend="sglang")
    return tokenizer


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
_w_template_force_reasoning: bool = False


def _load_chat_template(chat_template: str | None) -> str | None:
    """Load a chat template override from a Jinja template file."""
    if not chat_template:
        return None
    if chat_template_exists(chat_template):
        raise ValueError(
            "SGLang built-in chat template names are not supported by "
            "Dynamo's SGLang chat processor; pass a .jinja template file path."
        )
    expanded_template = os.path.expanduser(os.path.expandvars(chat_template))
    if not os.path.exists(expanded_template):
        raise FileNotFoundError(f"Chat template file not found: {expanded_template}")
    if not os.path.isfile(expanded_template):
        raise FileNotFoundError(
            f"Chat template path is not a file: {expanded_template}"
        )
    if not expanded_template.endswith(".jinja"):
        raise ValueError(
            "Dynamo's SGLang chat processor supports only .jinja chat template "
            f"files, got: {expanded_template}"
        )
    return read_jinja_chat_template(expanded_template, backend="sglang")


@dataclass
class SglangPreprocessWorkerResult:
    """Picklable return value from the SGLang preprocess worker."""

    prompt_token_ids: list[int]
    dynamo_preproc: dict[str, Any]
    request: dict[str, Any]
    force_reasoning: bool = False
    # ``effective_reasoning_parser_name`` is None when the request opted out
    # via ``separate_reasoning=False``; the main process must skip creating
    # a reasoning parser in that case so the pool path matches the inline
    # path byte-for-byte.
    effective_reasoning_parser_name: str | None = None


def _init_worker(
    model_path: str,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
    exclude_tools_when_tool_choice_none: bool = True,
    trust_remote_code: bool = False,
    template_force_reasoning: bool = False,
    chat_template: str | None = None,
) -> None:
    """Initialize a worker process with its own tokenizer."""
    global _w_tokenizer, _w_tool_call_parser_name, _w_reasoning_parser_name
    global _w_exclude_tools_when_tool_choice_none, _w_template_force_reasoning
    _w_tokenizer = _load_tokenizer(model_path, trust_remote_code)
    if chat_template is not None:
        _w_tokenizer.chat_template = chat_template
    _w_tool_call_parser_name = tool_call_parser_name
    _w_reasoning_parser_name = reasoning_parser_name
    _w_exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
    _w_template_force_reasoning = template_force_reasoning


def _preprocess_worker(
    request: dict[str, Any],
    model_name: str,
    eos_token_ids: list[int] | None,
) -> SglangPreprocessWorkerResult:
    """Preprocess a request in a worker process and return a picklable result."""
    pre = preprocess_chat_request(
        request,
        tokenizer=_w_tokenizer,
        tool_call_parser_name=_w_tool_call_parser_name,
        reasoning_parser_name=_w_reasoning_parser_name,
        exclude_tools_when_tool_choice_none=_w_exclude_tools_when_tool_choice_none,
        template_force_reasoning=_w_template_force_reasoning,
    )

    n = request.get("n", 1)
    if n != 1:
        raise PreprocessError(_unsupported_n_message(n))

    dynamo_preproc = _build_dynamo_preproc(
        request,
        pre.prompt_token_ids,
        model_name,
        eos_token_ids,
        pre.guided_decoding,
        pre.tool_call_parser,
    )

    effective_reasoning_parser_name = (
        _w_reasoning_parser_name if _client_wants_separate_reasoning(request) else None
    )

    return SglangPreprocessWorkerResult(
        prompt_token_ids=pre.prompt_token_ids,
        dynamo_preproc=dynamo_preproc,
        request=request,
        force_reasoning=pre.force_reasoning,
        effective_reasoning_parser_name=effective_reasoning_parser_name,
    )


def _build_dynamo_preproc(
    request: dict[str, Any],
    prompt_token_ids: list[int],
    model_name: str,
    eos_token_ids: int | list[int] | None,
    guided_decoding: dict[str, Any] | None = None,
    tool_call_parser: ToolCallParserType | None = None,
) -> dict[str, Any]:
    """Build the Dynamo preprocessed request dict from request fields."""
    max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")

    stop = request.get("stop")
    stop_token_ids = request.get("stop_token_ids", [])
    if isinstance(stop, str):
        stop = [stop]
    elif isinstance(stop, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in stop
    ):
        stop_token_ids = [*stop_token_ids, *stop]
        stop = []
    elif stop is None:
        stop = []

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
            "return_tokens_as_token_ids": request.get("return_tokens_as_token_ids"),
        },
        "eos_token_ids": _normalize_eos_token_ids(eos_token_ids),
        "annotations": [],
        "routing": request.get("routing"),
    }

    # Forward multimodal URLs so the backend handler can load the media.
    mm_data = extract_mm_urls(request.get("messages", []))
    if mm_data:
        preproc["multi_modal_data"] = mm_data

    nvext = request.get("nvext") or {}
    nvext_passthrough = {
        key: nvext[key] for key in ("metadata_upload", "extra_fields") if key in nvext
    }
    if nvext_passthrough:
        preproc["extra_args"] = {"nvext": nvext_passthrough}

    return preproc


class SglangProcessor:
    def __init__(
        self,
        tokenizer,
        routed_engine: RoutedEngine,
        tool_call_parser_name: str | None,
        reasoning_parser_name: str | None,
        eos_token_ids: list[int] | None,
        debug_perf: bool = False,
        preprocess_pool: ProcessPoolExecutor | None = None,
        preprocess_workers: int = 0,
        stream_interval: int = 1,
    ):
        self.tokenizer = tokenizer
        # Detect force_reasoning once from the chat template, matching
        # sglang's template_manager. Per-request overrides still apply
        # (see resolve_request_force_reasoning).
        self.template_force_reasoning = detect_force_reasoning_from_template(
            getattr(tokenizer, "chat_template", None)
        )
        if self.template_force_reasoning:
            logger.info(
                "Detected force-reasoning pattern in chat template; "
                "thinking tokens will route to delta.reasoning_content by "
                "default (clients can opt out via "
                "separate_reasoning=false or "
                "chat_template_kwargs.enable_thinking=false)."
            )
        self.routed_engine = routed_engine
        self.tool_call_parser_name = tool_call_parser_name
        self.reasoning_parser_name = reasoning_parser_name
        self.exclude_tools_when_tool_choice_none = True
        self.eos_token_ids = _normalize_eos_token_ids(eos_token_ids)
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
        self, request: dict[str, Any], context: Any | None = None
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
                async for item in self._generator_inner(request, context=context):
                    yield item
            else:
                async for item in self._generator_inner_pool(request, context=context):
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
        self, request: dict[str, Any], context: Any | None = None
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
                template_force_reasoning=self.template_force_reasoning,
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
                self.eos_token_ids,
                pre.guided_decoding,
                pre.tool_call_parser,
            )
        except PreprocessError as exc:
            raise InvalidArgument(str(exc)) from exc
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
            eos_token_ids=self.eos_token_ids,
        )

        async for item in self._generate_and_stream(
            request_id, request, dynamo_preproc, tokens, post, context=context
        ):
            yield item

    async def _generator_inner_pool(
        self, request: dict[str, Any], context: Any | None = None
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
                    self.eos_token_ids,
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
        # The worker already decided effective_reasoning_parser_name based on
        # the request's separate_reasoning flag and computed force_reasoning;
        # we mirror those choices to keep pool- and inline-path outputs
        # identical.
        tool_call_parser, reasoning_parser = create_parsers(
            request,
            tool_call_parser_name=self.tool_call_parser_name,
            reasoning_parser_name=preproc_result.effective_reasoning_parser_name,
            force_reasoning=preproc_result.force_reasoning,
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
            eos_token_ids=self.eos_token_ids,
        )

        async for item in self._generate_and_stream(
            request_id,
            request,
            preproc_result.dynamo_preproc,
            preproc_result.prompt_token_ids,
            post,
            context=context,
        ):
            yield item

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        post: SglangStreamingPostProcessor,
        context: Any | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Shared streaming logic for both single-process and pool paths."""
        token_count = 0
        post_proc_total_ms = 0.0
        created_ts = int(time.time())
        stream_interval = self.stream_interval

        try:
            dynamo_stream = await self.routed_engine.generate(
                dynamo_preproc, context=context
            )

            # Accumulate tokens for batched detokenization when
            # stream_interval > 1.  Flush every N tokens or on
            # finish_reason.  Use si=1 for the first chunk to minimize
            # TTFT, then switch to the configured interval.
            pending_token_ids: list[int] = []
            pending_usage: dict[str, Any] | None = None
            first_chunk = True
            input_tokens = len(tokens)
            cumulative_output_tokens = 0

            async for dynamo_response in dynamo_stream:
                if dynamo_response.is_error():
                    comments = dynamo_response.comments() or []
                    message = "; ".join(comments) or "unknown routed_engine error"
                    logger.error(
                        "routed_engine error for request %s: %s",
                        request_id,
                        message,
                    )
                    yield make_internal_error(request_id, message)
                    break
                engine_response = dynamo_response.data()

                if engine_response is None:
                    # No data or error fields, means we may have a comment or other kind of event.
                    # Skip for now.
                    continue

                if (
                    not isinstance(engine_response, dict)
                    or "token_ids" not in engine_response
                ):
                    yield handle_engine_error(engine_response, request_id, logger)
                    break

                new_ids = engine_response["token_ids"]
                chunk_tokens = len(new_ids)
                cumulative_output_tokens += chunk_tokens
                raw_finish = engine_response.get("finish_reason")
                finish_reason = _map_finish_reason(raw_finish)
                stop_reason = engine_response.get("stop_reason")

                if usage := engine_response.get("completion_usage"):
                    pending_usage = usage
                engine_data = engine_response.get("engine_data")

                pending_token_ids.extend(new_ids)

                # Flush on finish or when we've accumulated enough tokens.
                # First chunk flushes immediately (si=1) to minimize TTFT.
                flush_threshold = 1 if first_chunk else stream_interval
                if finish_reason or len(pending_token_ids) >= flush_threshold:
                    usage_for_metrics = pending_usage
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

                    envelope: dict[str, Any] = {"_dynamo_annotated": True}
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
                            pending_usage = None
                        response_nvext: dict[str, Any] = {}
                        if stop_reason is not None and nvext_extra_field_requested(
                            request, "stop_reason"
                        ):
                            response_nvext["stop_reason"] = stop_reason
                        if engine_data is not None and (
                            nvext_extra_field_requested(request, "engine_data")
                        ):
                            response_nvext["engine_data"] = engine_data
                        if response_nvext:
                            dynamo_out["nvext"] = response_nvext

                        envelope["data"] = dynamo_out

                    metrics: dict[str, Any] = {
                        "input_tokens": input_tokens,
                        "output_tokens": cumulative_output_tokens,
                        "chunk_tokens": len(pending_token_ids),
                    }
                    cached_tokens = _cached_tokens_from_usage(usage_for_metrics)
                    if cached_tokens is not None:
                        metrics["cached_tokens"] = cached_tokens
                    envelope["event"] = "llm_metrics"
                    envelope["comment"] = [json.dumps(metrics)]

                    yield envelope

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
        config: FrontendConfig,
        debug_perf: bool = False,
        tool_call_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        chat_template: str | None = None,
    ):
        self.config = config
        self.debug_perf = debug_perf
        self.tool_call_parser_name = tool_call_parser_name
        self.reasoning_parser_name = reasoning_parser_name
        self.chat_template = chat_template

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
        routed_engine: RoutedEngine,
    ) -> PythonAsyncEngine:
        """Called by Rust when a model is discovered."""
        model_type = mdc.model_type()
        if not model_type.supports_chat():
            raise RuntimeError(
                f"model type {model_type} is not supported by this factory"
            )
        loop = asyncio.get_running_loop()

        local_dir = mdc.local_dir()
        if not os.path.isdir(local_dir):
            raise RuntimeError(
                f"MDC local_dir {local_dir!r} not populated for model {mdc.name()!r}; "
                f"download_config must run before the engine factory."
            )

        logger.info("Loading SGLang tokenizer from %s", local_dir)
        tokenizer = _load_tokenizer(local_dir, self.trust_remote_code)
        chat_template = _load_chat_template(self.chat_template)
        if chat_template is not None:
            logger.info("Using custom chat template override")
            tokenizer.chat_template = chat_template

        eos_token_ids = _tokenizer_eos_token_ids(tokenizer)

        # Static reasoning-template scan (mirrors sglang's template_manager).
        # Shared with worker-pool processes via initargs so they compute the
        # same per-request force_reasoning flag as the main process.
        template_force_reasoning = detect_force_reasoning_from_template(
            getattr(tokenizer, "chat_template", None)
        )

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

        preprocess_pool = None
        preprocess_workers = self.config.preprocess_workers
        if preprocess_workers > 0:
            logger.info(
                "Creating SGLang preprocess worker pool with %d workers for %s",
                preprocess_workers,
                local_dir,
            )
            preprocess_pool = ProcessPoolExecutor(
                max_workers=preprocess_workers,
                initializer=_init_worker,
                initargs=(
                    local_dir,
                    tool_call_parser_name,
                    reasoning_parser_name,
                    self.config.exclude_tools_when_tool_choice_none,
                    self.trust_remote_code,
                    template_force_reasoning,
                    chat_template,
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
            routed_engine,
            tool_call_parser_name,
            reasoning_parser_name,
            eos_token_ids,
            debug_perf=self.debug_perf,
            preprocess_pool=preprocess_pool,
            preprocess_workers=preprocess_workers,
            stream_interval=self.stream_interval,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
