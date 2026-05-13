# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

import pybase64
import sglang as sgl
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.sglang._compat import filter_supported_async_generate_kwargs
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

# Escape hatch: set to "1" (or any truthy value) to allow top_logprobs_num >= 1.
# Default-off because SGLang's tokenizer manager detokenizes top-k tokens
# per-position serially (O(N) per generated token), causing severe latency
# degradation. Flip once upstream lands batched top-logprob detokenization:
# https://github.com/sgl-project/sglang/pull/24447
_ALLOW_TOP_LOGPROBS_ENV = "DYN_SGL_ALLOW_TOP_LOGPROBS"

_TOP_LOGPROBS_UNSUPPORTED_MSG = (
    "Dynamo's SGLang backend does not currently support logprobs >= 1 due to "
    "an O(N) per-position detokenization in the upstream sglang tokenizer "
    "manager. Use logprobs=0 for chosen-token logprobs, or set "
    "DYN_SGL_ALLOW_TOP_LOGPROBS=1 to override at your own risk. "
    "Track the upstream fix at https://github.com/sgl-project/sglang/pull/24447."
)


def _top_logprobs_allowed() -> bool:
    """Return True if the DYN_SGL_ALLOW_TOP_LOGPROBS escape hatch is enabled."""
    return os.environ.get(_ALLOW_TOP_LOGPROBS_ENV, "").lower() not in ("", "0", "false")


def _extract_media_urls(mm_data: Dict[str, Any], media_key: str) -> list[str] | None:
    """Normalize multimodal URL items from the frontend wire format."""

    items = mm_data.get(media_key)
    if not items:
        return None

    urls: list[str] = []
    for item in items:
        if isinstance(item, str):
            urls.append(item)
            continue

        if isinstance(item, dict):
            url = item.get("Url")
            if isinstance(url, str):
                urls.append(url)

    return urls or None


def _nvext_extra_field_requested(request: Dict[str, Any], field: str) -> bool:
    nvext = request.get("nvext")
    if not isinstance(nvext, dict):
        return False
    extra_fields = nvext.get("extra_fields")
    if not isinstance(extra_fields, list):
        return False
    return field in extra_fields


def _user_stop_token_ids(request: Dict[str, Any]) -> set[int]:
    stop_conditions = request.get("stop_conditions")
    if isinstance(stop_conditions, dict):
        return {
            token_id
            for token_id in (stop_conditions.get("stop_token_ids") or [])
            if isinstance(token_id, int) and not isinstance(token_id, bool)
        }

    stop = request.get("stop")
    if isinstance(stop, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in stop
    ):
        return set(stop)

    return {
        token_id
        for token_id in (request.get("stop_token_ids") or [])
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    }


def _openai_stop_sampling_params(request: Dict[str, Any]) -> Dict[str, Any]:
    stop = request.get("stop")
    if isinstance(stop, str):
        return {"stop": stop}
    if isinstance(stop, list):
        if stop and all(
            isinstance(item, int) and not isinstance(item, bool) for item in stop
        ):
            return {"stop_token_ids": stop}
        if stop and all(isinstance(item, str) for item in stop):
            return {"stop": stop}

    stop_token_ids = [
        token_id
        for token_id in (request.get("stop_token_ids") or [])
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    ]
    if stop_token_ids:
        return {"stop_token_ids": stop_token_ids}
    return {}


def _extract_sglang_stop_reason(
    finish_reason: Dict[str, Any] | None,
    user_stop_token_ids: set[int] | None = None,
) -> Any | None:
    """Extract SGLang's matched stop value for Dynamo's stop_reason field."""

    if not finish_reason:
        return None

    matched = finish_reason.get("matched")
    if isinstance(matched, bool):
        return None
    if isinstance(matched, str):
        return matched
    if isinstance(matched, int):
        if user_stop_token_ids is not None and matched not in user_stop_token_ids:
            return None
        return matched
    if isinstance(matched, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in matched
    ):
        if user_stop_token_ids is not None and any(
            item not in user_stop_token_ids for item in matched
        ):
            return None
        return matched

    return None


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            shutdown_event: Optional event to signal shutdown.
            generate_endpoint: The endpoint handle for discovery registration.
        """
        super().__init__(
            engine,
            config,
            publisher,
            generate_endpoint,
            shutdown_event,
        )
        # Resolve the optional return_routed_experts kwarg once. Gating on the
        # opt-in flag avoids sending the kwarg on sglang builds whose
        # Engine.async_generate does not declare it (notably the deepseek_v4
        # branch). Doing this at init keeps the per-request hot path free of
        # signature inspection.
        self._routed_experts_kwargs: Dict[
            str, Any
        ] = self._resolve_routed_experts_kwargs(self.engine, self.config.server_args)
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            logging.info("Decode worker handler initialized (aggregated mode)")
        self._sglang_openai_chat = OpenAIServingChat(
            self.engine.tokenizer_manager, self.engine.template_manager
        )

    @staticmethod
    def _resolve_routed_experts_kwargs(engine: Any, server_args: Any) -> Dict[str, Any]:
        """Resolve the return_routed_experts kwarg for this engine.

        Returns ``{"return_routed_experts": True}`` only when the user opted in
        via ``enable_return_routed_experts=True`` AND the engine's
        ``async_generate`` signature declares the kwarg. Returns ``{}`` for the
        default-off path and for sglang builds that do not declare the kwarg
        (e.g. the ``deepseek_v4`` branch).
        """
        if not getattr(server_args, "enable_return_routed_experts", False):
            return {}
        return filter_supported_async_generate_kwargs(
            engine, {"return_routed_experts": True}
        )

    def cleanup(self) -> None:
        """Shutdown the engine and cleanup resources."""
        super().cleanup()
        self.engine.shutdown()
        logging.info("Engine shutdown")

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format.

        Args:
            request: Request dict in either token-based or OpenAI format.

        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        use_sglang_preprocessor = getattr(
            self, "use_sglang_preprocessor", self.use_sglang_tokenizer
        )
        if not use_sglang_preprocessor:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            output_options = request.get("output_options", {})

            _hidden = stop_conditions.get("stop_token_ids_hidden") or []
            _plain = stop_conditions.get("stop_token_ids") or []
            _merged = list(set(_hidden).union(_plain))
            stop_token_ids = _merged if _merged else None

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "n": sampling_opts.get("n"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
                "skip_special_tokens": output_options.get("skip_special_tokens"),
                "stop_token_ids": stop_token_ids,
                **self._get_guided_decoding_params(
                    sampling_opts.get("guided_decoding")
                ),
            }
        else:
            # OpenAI request format
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "n": request.get("n"),
                "max_new_tokens": request.get("max_tokens"),
                **_openai_stop_sampling_params(request),
                **self._get_guided_decoding_params(request.get("guided_decoding")),
            }

        # Keep max_new_tokens even when None — SGLang treats None as "generate
        # until EOS/context-length" whereas omitting it triggers a default of 128.
        keep_if_none = {"max_new_tokens"}
        return {
            k: v for k, v in param_mapping.items() if v is not None or k in keep_if_none
        }

    def _build_sglang_chat_generation_inputs(
        self, request: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Build SGLang-native prompt/media/sampling inputs for chat requests."""

        sglang_request = self._build_sglang_chat_request(request)
        reasoning_effort = (
            sglang_request.chat_template_kwargs.pop("reasoning_effort", None)
            if sglang_request.chat_template_kwargs
            else None
        )
        if reasoning_effort is not None:
            sglang_request.reasoning_effort = reasoning_effort

        is_multimodal = self.engine.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._sglang_openai_chat._process_messages(
            sglang_request, is_multimodal
        )
        sampling_params = sglang_request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self._sglang_openai_chat.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        if is_multimodal or isinstance(processed_messages.prompt_ids, str):
            input_param = {"prompt": processed_messages.prompt}
        else:
            input_param = {"input_ids": processed_messages.prompt_ids}

        media_kwargs = {
            "image_data": processed_messages.image_data,
            "video_data": processed_messages.video_data,
            "audio_data": processed_messages.audio_data,
        }
        return input_param, sampling_params, media_kwargs

    @staticmethod
    def _build_logprob_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
        """Build logprob kwargs for SGLang async_generate from output_options.

        Maps the Dynamo output_options format (shared with vLLM/TRT-LLM) to
        SGLang's async_generate keyword arguments:

          - return_logprob (bool): enables logprob computation
          - top_logprobs_num (int): number of top-k logprobs per token
          - logprob_start_len (int): absolute position in the sequence where
            logprob computation begins. SGLang defaults this to -1, which
            means len(prompt) - 1 (i.e. output tokens only). Setting it to 0
            computes logprobs from the start of the prompt — this is how we
            implement prompt_logprobs. We don't expose logprob_start_len
            directly; it's an SGLang-internal detail derived from whether the
            user requested prompt_logprobs.

        Args:
            request: Request dict containing optional output_options.

        Returns:
            Dict of logprob-related kwargs for engine.async_generate().
        """
        kwargs: Dict[str, Any] = {}
        output_options = request.get("output_options", {})
        if not output_options:
            return kwargs

        allow_top = _top_logprobs_allowed()

        def _parse(name: str, value: Any) -> Optional[int]:
            try:
                parsed = int(value)
            except (ValueError, TypeError):
                logging.warning(
                    f"Invalid {name} value: {value} (must be integer), ignoring"
                )
                return None
            if parsed < 0:
                logging.warning(
                    f"Invalid {name} value: {value} (must be non-negative), ignoring"
                )
                return None
            if parsed >= 1 and not allow_top:
                raise ValueError(_TOP_LOGPROBS_UNSUPPORTED_MSG)
            return parsed

        logprobs_value = output_options.get("logprobs")
        if logprobs_value is not None:
            parsed = _parse("logprobs", logprobs_value)
            if parsed is not None:
                kwargs["return_logprob"] = True
                kwargs["top_logprobs_num"] = parsed

        prompt_logprobs_value = output_options.get("prompt_logprobs")
        if prompt_logprobs_value is not None:
            parsed = _parse("prompt_logprobs", prompt_logprobs_value)
            if parsed is not None:
                kwargs["return_logprob"] = True
                # SGLang has a single top_logprobs_num for both prompt
                # and output tokens, so take the max of the two.
                kwargs["top_logprobs_num"] = max(
                    kwargs.get("top_logprobs_num", 0), parsed
                )
                # logprob_start_len=0 computes from prompt start;
                # omitting it (or -1) computes output tokens only.
                kwargs["logprob_start_len"] = 0

        # Belt-and-suspenders: if return_logprob was requested and the gate is
        # not open, pin top_logprobs_num=0 so no future code path can flip it on.
        if kwargs.get("return_logprob") and not allow_top:
            kwargs["top_logprobs_num"] = 0

        return kwargs

    @staticmethod
    def _extract_logprobs(
        meta_info: Dict[str, Any],
        num_output_logprobs_so_far: int,
        return_tokens_as_token_ids: bool = False,
    ) -> tuple:
        """Extract logprobs from SGLang meta_info for new tokens.

        While Dynamo forces stream_output=True (args.py) so that output_ids
        are disjoint per chunk, SGLang's output_token_logprobs and
        output_top_logprobs in meta_info are always cumulative. We track an
        offset to slice out only the new entries each chunk.

        Args:
            meta_info: SGLang response meta_info dict.
            num_output_logprobs_so_far: Number of logprob entries already
                processed in previous chunks.

        Returns:
            Tuple of (log_probs, top_logprobs, new_total):
            - log_probs: List of floats (selected token logprob per position)
            - top_logprobs: List of lists of dicts with rank/token_id/token/logprob
            - new_total: Updated count of logprob entries processed so far
        """
        output_token_logprobs = meta_info.get("output_token_logprobs")
        if not output_token_logprobs:
            return None, None, num_output_logprobs_so_far

        new_logprobs = output_token_logprobs[num_output_logprobs_so_far:]
        if not new_logprobs:
            return None, None, num_output_logprobs_so_far

        # Extract selected-token logprobs: each entry is (logprob, token_id, text_or_None)
        log_probs = [float(entry[0]) for entry in new_logprobs]

        # Extract top logprobs if available
        top_logprobs: list[list[dict[str, Any]]] | None = None
        output_top = meta_info.get("output_top_logprobs")
        if output_top:
            new_top = output_top[num_output_logprobs_so_far:]
            if new_top:
                top_logprobs = []
                for position_entries in new_top:
                    if position_entries is None:
                        top_logprobs.append([])
                        continue
                    position_list = []
                    for rank_idx, entry in enumerate(position_entries):
                        tok_id = entry[1]
                        token_str = (
                            f"token_id:{tok_id}"
                            if return_tokens_as_token_ids
                            else entry[2]
                        )
                        position_list.append(
                            {
                                "rank": rank_idx + 1,
                                "token_id": tok_id,
                                "token": token_str,
                                "logprob": float(entry[0]),
                            }
                        )
                    top_logprobs.append(position_list)

        new_total = len(output_token_logprobs)
        return log_probs, top_logprobs, new_total

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response in aggregated or disaggregated mode.

        Args:
            request: Request dict with input and sampling parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.

        Raises:
            RuntimeError: If no bootstrap info received from prefill worker.
        """
        logging.debug(f"New Request ID: {context.id()}")
        trace_id = context.trace_id
        use_sglang_preprocessor = getattr(
            self, "use_sglang_preprocessor", self.use_sglang_tokenizer
        )
        if use_sglang_preprocessor and "messages" in request:
            input_param, sampling_params, media_kwargs = (
                self._build_sglang_chat_generation_inputs(request)
            )
        else:
            sampling_params = self._build_sampling_params(request)
            input_param = self._get_input_param(request)
            media_kwargs = {}
        priority = (request.get("routing") or {}).get("priority")
        logprob_kwargs = self._build_logprob_kwargs(request)

        output_options = request.get("output_options", {})
        return_tokens_as_token_ids = bool(
            output_options.get("return_tokens_as_token_ids")
        )
        user_stop_token_ids = _user_stop_token_ids(request)

        lora_path = self._resolve_lora(request)
        if lora_path:
            logging.debug(f"Request {context.id()} will use LoRA adapter: {lora_path}")

        if self.serving_mode == DisaggregationMode.DECODE:
            # Check if bootstrap_info is pre-computed in the request (from frontend)
            bootstrap_info = request.get("bootstrap_info")

            if not bootstrap_info:
                raise RuntimeError(
                    "bootstrap_info is required for disaggregated decode but was not provided"
                )

            logging.debug(
                f"Using bootstrap_info: "
                f"host={bootstrap_info['bootstrap_host']}, "
                f"port={bootstrap_info['bootstrap_port']}, "
                f"room={bootstrap_info['bootstrap_room']}"
            )

            trace_header = build_trace_headers(context) if self.enable_trace else None

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            decode = await self.engine.async_generate(
                **input_param,
                **media_kwargs,
                sampling_params=sampling_params,
                stream=True,
                **self._routed_experts_kwargs,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
                external_trace_header=trace_header,
                rid=trace_id,
                data_parallel_rank=dp_rank,
                **self._session_kwargs(request),
                lora_path=lora_path,
                **logprob_kwargs,
                **self._priority_kwargs(priority),
            )

            use_sglang_postprocessor = getattr(
                self, "use_sglang_postprocessor", self.use_sglang_tokenizer
            )
            if not use_sglang_postprocessor:
                async for out in self._process_token_stream(
                    decode,
                    context,
                    return_tokens_as_token_ids,
                    user_stop_token_ids=user_stop_token_ids,
                ):
                    yield out
            else:
                async for out in self._process_text_stream(
                    decode,
                    context,
                    request=request,
                    user_stop_token_ids=user_stop_token_ids,
                ):
                    yield out
        else:
            if not media_kwargs:
                # Extract image/video URLs for multimodal requests. SGLang's
                # mm_data_processor handles loading/preprocessing, and the
                # scheduler does vision encoding.
                mm_data = request.get("multi_modal_data", {})
                image_data = _extract_media_urls(mm_data, "image_url")
                video_data = _extract_media_urls(mm_data, "video_url")
                media_kwargs = {"image_data": image_data, "video_data": video_data}

            trace_header = build_trace_headers(context) if self.enable_trace else None

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            agg = await self.engine.async_generate(
                **input_param,
                **media_kwargs,
                sampling_params=sampling_params,
                stream=True,
                **self._routed_experts_kwargs,
                external_trace_header=trace_header,
                rid=trace_id,
                data_parallel_rank=dp_rank,
                **self._session_kwargs(request),
                lora_path=lora_path,
                **logprob_kwargs,
                **self._priority_kwargs(priority),
            )
            use_sglang_postprocessor = getattr(
                self, "use_sglang_postprocessor", self.use_sglang_tokenizer
            )
            if not use_sglang_postprocessor:
                async for out in self._process_token_stream(
                    agg,
                    context,
                    return_tokens_as_token_ids,
                    user_stop_token_ids=user_stop_token_ids,
                ):
                    yield out
            else:
                async for out in self._process_text_stream(
                    agg,
                    context,
                    request=request,
                    user_stop_token_ids=user_stop_token_ids,
                ):
                    yield out

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        return_tokens_as_token_ids: bool = False,
        user_stop_token_ids: set[int] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.

        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            Dict with token_ids and optional finish_reason.
        """
        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future: asyncio.Future[str] = asyncio.Future()
        # SGLang's token stream is asymmetric: output_ids are disjoint deltas
        # when stream_output=True, but meta_info output logprobs are cumulative.
        # With n>1, chunks for different choices are interleaved, so track the
        # cumulative-logprob cursor per choice index instead of globally.
        output_logprobs_per_choice: dict[int, int] = {}
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.
                # SGLang omits index for non-n/legacy chunks; treat those as
                # choice 0 while preserving explicit indices for n>1.
                output_idx = res.get("index") or 0
                out: dict[str, Any] = {"index": output_idx}
                finish_reason = res["meta_info"]["finish_reason"]
                if finish_reason:
                    out["finish_reason"] = normalize_finish_reason(
                        finish_reason["type"]
                    )
                    stop_reason = _extract_sglang_stop_reason(
                        finish_reason, user_stop_token_ids
                    )
                    if stop_reason is not None:
                        out["stop_reason"] = stop_reason

                # With stream_output=True, output_ids contains only new tokens (disjoint)
                output_ids = res.get("output_ids", [])
                # Empty, non-final chunks can happen during scheduler idle ticks.
                # Keep waiting for the next chunk unless cancellation was requested.
                if not output_ids and not finish_reason:
                    if context.is_stopped():
                        break
                    continue

                # Pass through disjoint token segments directly
                out["token_ids"] = output_ids

                # Extract logprobs for new tokens if available
                (
                    log_probs,
                    top_logprobs,
                    next_logprobs_total,
                ) = self._extract_logprobs(
                    res["meta_info"],
                    output_logprobs_per_choice.get(output_idx, 0),
                    return_tokens_as_token_ids=return_tokens_as_token_ids,
                )
                output_logprobs_per_choice[output_idx] = next_logprobs_total
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

                routed_experts = res["meta_info"].get("routed_experts")
                if routed_experts is not None:
                    # Base64-encode tensor bytes to match sglang's output format.
                    routed_experts = pybase64.b64encode(
                        routed_experts.numpy().tobytes()
                    ).decode("utf-8")
                    # Internal transport field consumed by frontend nvext mapping.
                    out["disaggregated_params"] = {"routed_experts": routed_experts}
                if finish_reason:
                    input_tokens = res["meta_info"]["prompt_tokens"]
                    completion_tokens = res["meta_info"]["completion_tokens"]
                    cached_tokens = res["meta_info"]["cached_tokens"]
                    prefill_prompt_tokens_details = None
                    if cached_tokens is not None and cached_tokens > 0:
                        prefill_prompt_tokens_details = {"cached_tokens": cached_tokens}
                    out["completion_usage"] = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_tokens + completion_tokens,
                        "prompt_tokens_details": prefill_prompt_tokens_details,
                    }
                if not context.is_stopped():
                    yield out

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        request: Dict[str, Any] | None = None,
        user_stop_token_ids: set[int] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        request = request or {}
        sglang_request = self._build_sglang_chat_request(request)
        sglang_request.stream = True

        parser_dict: dict[int, Any] = {}
        reasoning_parser_dict: dict[int, Any] = {}
        is_firsts: dict[int, bool] = {}
        stream_offsets: dict[int, int] = {}
        n_prev_tokens: dict[int, int] = {}
        has_tool_calls: dict[int, bool] = {}
        finish_reasons: dict[int, Dict[str, Any]] = {}
        prompt_tokens: dict[int, int] = {}
        reasoning_tokens: dict[int, int] = {}
        completion_tokens: dict[int, int] = {}
        cached_tokens: dict[int, int] = {}
        hidden_states: dict[int, Any] = {}
        response_ids: dict[int, str] = {}

        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future: asyncio.Future[str] = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.

                # Same defaulting as token mode: non-n chunks are choice 0.
                index = res.get("index") or 0
                meta_info = res["meta_info"]
                response_ids[index] = meta_info["id"]
                prompt_tokens[index] = meta_info.get("prompt_tokens", 0)
                reasoning_tokens[index] = meta_info.get("reasoning_tokens", 0)
                completion_tokens[index] = meta_info.get("completion_tokens", 0)
                cached_tokens[index] = meta_info.get("cached_tokens", 0) or 0
                hidden_states[index] = meta_info.get("hidden_states", None)

                choice_logprobs = None
                if sglang_request.logprobs:
                    choice_logprobs = (
                        self._sglang_openai_chat._process_streaming_logprobs(
                            res, n_prev_tokens.get(index, 0)
                        )
                    )
                    n_prev_tokens[index] = len(
                        meta_info.get("output_token_logprobs") or []
                    )

                finish_reason = meta_info["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None
                if finish_reason_type:
                    finish_reasons[index] = finish_reason

                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    chunk = ChatCompletionStreamResponse(
                        id=meta_info["id"],
                        created=int(time.time()),
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(role="assistant", content=""),
                                finish_reason=None,
                                logprobs=None,
                            )
                        ],
                        model=sglang_request.model,
                    )
                    if not context.is_stopped():
                        yield self._dump_sglang_model(chunk)

                text = res.get("text", "")
                if getattr(
                    self.config.server_args, "incremental_streaming_output", True
                ):
                    delta = text
                else:
                    offset = stream_offsets.get(index, 0)
                    delta = text[offset:]
                    stream_offsets[index] = len(text)

                if self._sglang_openai_chat.reasoning_parser and (
                    sglang_request.separate_reasoning
                ):
                    reasoning_text, delta = (
                        self._sglang_openai_chat._process_reasoning_stream(
                            index,
                            delta,
                            reasoning_parser_dict,
                            res,
                            sglang_request,
                        )
                    )
                    if reasoning_text:
                        chunk = ChatCompletionStreamResponse(
                            id=meta_info["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(reasoning_content=reasoning_text),
                                    finish_reason=None,
                                )
                            ],
                            model=sglang_request.model,
                        )
                        if (
                            sglang_request.stream_options
                            and sglang_request.stream_options.continuous_usage_stats
                        ):
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )
                        if not context.is_stopped():
                            yield self._dump_sglang_model(chunk)

                if (
                    sglang_request.tool_choice != "none"
                    and sglang_request.tools
                    and self._sglang_openai_chat.tool_call_parser
                ):
                    async for raw_chunk in self._sglang_openai_chat._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        res,
                        sglang_request,
                        has_tool_calls,
                    ):
                        chunk_dict = self._sse_chunk_to_dict(raw_chunk)
                        if chunk_dict and not context.is_stopped():
                            yield chunk_dict

                    if finish_reason_type is not None and index in parser_dict:
                        remaining_chunk = (
                            self._sglang_openai_chat._check_for_unstreamed_tool_args(
                                parser_dict[index],
                                res,
                                sglang_request,
                                index,
                            )
                        )
                        chunk_dict = self._sse_chunk_to_dict(remaining_chunk)
                        if chunk_dict and not context.is_stopped():
                            yield chunk_dict
                elif delta:
                    chunk = ChatCompletionStreamResponse(
                        id=meta_info["id"],
                        created=int(time.time()),
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(content=delta),
                                finish_reason=None,
                                matched_stop=None,
                                logprobs=choice_logprobs,
                            )
                        ],
                        model=sglang_request.model,
                    )
                    if (
                        sglang_request.stream_options
                        and sglang_request.stream_options.continuous_usage_stats
                    ):
                        chunk.usage = UsageProcessor.calculate_token_usage(
                            prompt_tokens=prompt_tokens.get(index, 0),
                            reasoning_tokens=reasoning_tokens.get(index, 0),
                            completion_tokens=completion_tokens.get(index, 0),
                        )
                    if not context.is_stopped():
                        yield self._dump_sglang_model(chunk)

            for index, finish_reason in finish_reasons.items():
                finish_reason_type = finish_reason["type"]
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(index, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"
                finish_chunk = ChatCompletionStreamResponse(
                    id=response_ids.get(index, ""),
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason["matched"]
                                if "matched" in finish_reason
                                else None
                            ),
                        )
                    ],
                    model=sglang_request.model,
                    usage=None,
                )
                if not context.is_stopped():
                    yield self._dump_sglang_model(finish_chunk)

            if sglang_request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=response_ids.get(index, ""),
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,
                                )
                            ],
                            model=sglang_request.model,
                        )
                        if not context.is_stopped():
                            yield self._dump_sglang_model(hidden_states_chunk)

            if (
                sglang_request.stream_options
                and sglang_request.stream_options.include_usage
                and prompt_tokens
            ):
                request_id = request_id_future.result() if request_id_future.done() else ""
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=sglang_request.n,
                    enable_cache_report=getattr(
                        self.config.server_args, "enable_cache_report", False
                    ),
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    choices=[],
                    model=sglang_request.model,
                    usage=usage,
                )
                if not context.is_stopped():
                    yield self._dump_sglang_model(usage_chunk)

    def _build_sglang_chat_request(self, request: Dict[str, Any]) -> ChatCompletionRequest:
        raw_request = None
        extra_args = request.get("extra_args")
        if isinstance(extra_args, dict):
            raw_request = extra_args.get("openai_request")
        if raw_request is None:
            raw_request = request

        if isinstance(raw_request, ChatCompletionRequest):
            chat_request = raw_request
        else:
            if not isinstance(raw_request, dict):
                raw_request = {}
            raw_request = dict(raw_request)
            chat_template_args = raw_request.pop("chat_template_args", None)
            if (
                chat_template_args is not None
                and "chat_template_kwargs" not in raw_request
            ):
                raw_request["chat_template_kwargs"] = chat_template_args
            if "messages" not in raw_request:
                raw_request = {
                    **raw_request,
                    "messages": [{"role": "user", "content": ""}],
                }
            raw_request.setdefault(
                "model", getattr(self.config.server_args, "served_model_name", "unknown")
            )
            chat_request = ChatCompletionRequest.model_validate(raw_request)
        if not chat_request.model:
            chat_request.model = self.config.server_args.served_model_name
        return chat_request

    @staticmethod
    def _dump_sglang_model(model: Any) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        return dict(model)

    @staticmethod
    def _sse_chunk_to_dict(raw_chunk: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw_chunk:
            return None
        payload = raw_chunk.strip()
        if payload.startswith("data:"):
            payload = payload[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            return None
        return json.loads(payload)
