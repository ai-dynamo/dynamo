# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

import pybase64
import sglang as sgl

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

# Escape hatch: set to "1" (or any truthy value) to allow top_logprobs_num >= 1.
# Default-off because SGLang's tokenizer manager detokenizes top-k tokens
# per-position serially (O(N) per generated token), causing severe latency
# degradation. Flip once upstream batches detokenize_top_logprobs_tokens.
_ALLOW_TOP_LOGPROBS_ENV = "DYN_SGL_ALLOW_TOP_LOGPROBS"

_TOP_LOGPROBS_UNSUPPORTED_MSG = (
    "Dynamo's SGLang backend does not currently support logprobs >= 1 due to "
    "an O(N) per-position detokenization in the upstream sglang tokenizer "
    "manager. Use logprobs=0 for chosen-token logprobs, or set "
    "DYN_SGL_ALLOW_TOP_LOGPROBS=1 to override at your own risk. "
    "Track the upstream fix at https://github.com/sgl-project/sglang/issues/."
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
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            logging.info("Decode worker handler initialized (aggregated mode)")

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
        if not self.use_sglang_tokenizer:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            _hidden = stop_conditions.get("stop_token_ids_hidden") or []
            _plain = stop_conditions.get("stop_token_ids") or []
            _merged = list(set(_hidden).union(_plain))
            stop_token_ids = _merged if _merged else None

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
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
                "max_new_tokens": request.get("max_tokens"),
                **self._get_guided_decoding_params(request.get("guided_decoding")),
            }

        return {k: v for k, v in param_mapping.items() if v is not None}

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
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)
        return_routed_experts = getattr(
            self.config.server_args, "enable_return_routed_experts", False
        )
        priority = (request.get("routing") or {}).get("priority")
        logprob_kwargs = self._build_logprob_kwargs(request)

        output_options = request.get("output_options", {})
        return_tokens_as_token_ids = bool(
            output_options.get("return_tokens_as_token_ids")
        )

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
                sampling_params=sampling_params,
                stream=True,
                return_routed_experts=return_routed_experts,
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

            if not self.use_sglang_tokenizer:
                async for out in self._process_token_stream(
                    decode, context, return_tokens_as_token_ids
                ):
                    yield out
            else:
                async for out in self._process_text_stream(decode, context):
                    yield out
        else:
            # Extract image/video URLs for multimodal requests. SGLang's mm_data_processor
            # handles loading/preprocessing, and the scheduler does vision encoding.
            mm_data = request.get("multi_modal_data", {})
            image_data = _extract_media_urls(mm_data, "image_url")
            video_data = _extract_media_urls(mm_data, "video_url")

            trace_header = build_trace_headers(context) if self.enable_trace else None

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            agg = await self.engine.async_generate(
                **input_param,
                image_data=image_data,
                video_data=video_data,
                sampling_params=sampling_params,
                stream=True,
                return_routed_experts=return_routed_experts,
                external_trace_header=trace_header,
                rid=trace_id,
                data_parallel_rank=dp_rank,
                **self._session_kwargs(request),
                lora_path=lora_path,
                **logprob_kwargs,
                **self._priority_kwargs(priority),
            )
            if not self.use_sglang_tokenizer:
                async for out in self._process_token_stream(
                    agg, context, return_tokens_as_token_ids
                ):
                    yield out
            else:
                async for out in self._process_text_stream(agg, context):
                    yield out

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        return_tokens_as_token_ids: bool = False,
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
        # Logprob offset: output_ids are disjoint (stream_output=True) but
        # meta_info logprobs are cumulative — track how many we've emitted.
        num_output_logprobs_so_far = 0
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
                out: dict[str, Any] = {}
                finish_reason = res["meta_info"]["finish_reason"]
                if finish_reason:
                    out["finish_reason"] = normalize_finish_reason(
                        finish_reason["type"]
                    )

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
                    num_output_logprobs_so_far,
                ) = self._extract_logprobs(
                    res["meta_info"],
                    num_output_logprobs_so_far,
                    return_tokens_as_token_ids=return_tokens_as_token_ids,
                )
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0

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

                index = res.get("index", 0)
                text = res.get("text", "")

                finish_reason = res["meta_info"]["finish_reason"]
                finish_reason_type = (
                    normalize_finish_reason(finish_reason["type"])
                    if finish_reason
                    else None
                )
                next_count = len(text)
                delta = text[count:]

                choice_data = {
                    "index": index,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": finish_reason_type,
                }

                response = {
                    "id": res["meta_info"]["id"],
                    "created": int(time.time()),
                    "choices": [choice_data],
                    "model": self.config.server_args.served_model_name,
                    "object": "chat.completion.chunk",
                }
                routed_experts = res["meta_info"].get("routed_experts")
                if routed_experts is not None:
                    # Base64-encode tensor bytes to match sglang's output format.
                    routed_experts = pybase64.b64encode(
                        routed_experts.numpy().tobytes()
                    ).decode("utf-8")
                    response["nvext"] = {"routed_experts": routed_experts}
                if not context.is_stopped():
                    yield response
                count = next_count
