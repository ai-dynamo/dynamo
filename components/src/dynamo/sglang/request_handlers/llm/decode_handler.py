# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import sglang as sgl
from PIL.Image import Image as PILImage

from dynamo._core import Context
from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.constants import DisaggregationMode
from dynamo.common.metadata_upload import MetadataUploader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.sglang._compat import filter_supported_async_generate_kwargs
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.request_handlers.llm.mm_disagg_utils import (
    IMAGE_URL_KEY,
    VIDEO_URL_KEY,
    build_disagg_mm_kwargs,
    extract_media_urls,
)

_SAMPLING_OPTION_FIELDS = (
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
)


def _nvext_extra_field_requested(request: Dict[str, Any], field: str) -> bool:
    nvext = request.get("nvext")
    extra_args = request.get("extra_args") or {}
    extra_nvext = extra_args.get("nvext") if isinstance(extra_args, dict) else None

    for source in (nvext, extra_nvext):
        if not isinstance(source, dict):
            continue
        extra_fields = source.get("extra_fields")
        if isinstance(extra_fields, list) and field in extra_fields:
            return True
    return False


def _sampling_option_params(values: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sampling options that SGLang accepts as sampling params."""
    params = {field: values.get(field) for field in _SAMPLING_OPTION_FIELDS}
    if values.get("seed") is not None:
        params["sampling_seed"] = values.get("seed")
    return params


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
        enable_frontend_decoding: bool = False,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            shutdown_event: Optional event to signal shutdown.
            generate_endpoint: The endpoint handle for discovery registration.
            enable_frontend_decoding: If True, multimodal images arrive as
                ``Decoded`` variants over NIXL RDMA from the Rust frontend
                and must be read+converted to PIL before passing to SGLang.
                Off by default; the worker keeps the URL-string fast path.
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
        self._enable_frontend_decoding = enable_frontend_decoding
        self._image_loader: Optional[ImageLoader] = None
        if self._enable_frontend_decoding:
            # Lazy-inits a NIXL connector internally for Decoded variants.
            self._image_loader = ImageLoader(enable_frontend_decoding=True)
        self._mm_hashes_supported: bool = self._resolve_mm_hashes_supported(self.engine)
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            mode = "frontend-decoded" if self._enable_frontend_decoding else "standard"
            logging.info(f"Decode worker handler initialized (aggregated mode, {mode})")

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

    @staticmethod
    def _resolve_mm_hashes_supported(engine: Any) -> bool:
        """Probe whether engine.async_generate accepts ``mm_hashes``.

        SGLang accepted the kwarg starting with the upstream interop PR; older
        builds (and forks lacking the patch) raise TypeError if we pass it.
        Probing the signature once at init keeps the request hot path free of
        repeated inspection. Returns ``False`` when the kwarg is absent — the
        request still completes, MM-aware routing just falls back to the
        text-prefix overlap signal.
        """
        probe = filter_supported_async_generate_kwargs(engine, {"mm_hashes": None})
        return "mm_hashes" in probe

    @staticmethod
    def _extract_mm_hashes(request: Dict[str, Any]) -> Optional[List[str]]:
        """Pull the per-image hashes the Rust frontend forwards via extra_args.

        Returns ``None`` when the field is absent or malformed; SGLang then
        recomputes the hash internally via ``hash_feature()``.
        """
        extra_args = request.get("extra_args")
        if not isinstance(extra_args, dict):
            return None
        mm_hashes = extra_args.get("mm_hashes")
        if not mm_hashes:
            return None
        if not isinstance(mm_hashes, list):
            return None
        # Fail closed if a non-string slipped into the list — downstream
        # SGLang treats mm_hashes as List[str] and a bad element would
        # crash the worker mid-request. Routing falls back to text-prefix.
        if not all(isinstance(h, str) for h in mm_hashes):
            logging.warning(
                "extra_args.mm_hashes contained non-str entries; "
                "ignoring routing-side hashes and letting SGLang recompute"
            )
            return None
        return mm_hashes

    def _metadata_uploader_from_request(
        self, request: Dict[str, Any]
    ) -> MetadataUploader | None:
        if not getattr(getattr(self.config, "dynamo_args", None), "enable_rl", False):
            return None
        return MetadataUploader.from_backend_request(request)

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
                "n": sampling_opts.get("n"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
                "stop_token_ids": stop_token_ids,
                **_sampling_option_params(sampling_opts),
                **self._get_guided_decoding_params(
                    sampling_opts.get("guided_decoding")
                ),
            }
        else:
            # OpenAI request format
            param_mapping = {
                "n": request.get("n"),
                "max_new_tokens": request.get("max_tokens"),
                **_sampling_option_params(request),
                **_openai_stop_sampling_params(request),
                **self._get_guided_decoding_params(request.get("guided_decoding")),
            }

        # Keep max_new_tokens even when None — SGLang treats None as "generate
        # until EOS/context-length" whereas omitting it triggers a default of 128.
        keep_if_none = {"max_new_tokens"}
        return {
            k: v for k, v in param_mapping.items() if v is not None or k in keep_if_none
        }

    @staticmethod
    def _build_logprob_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
        return _shared_logprobs.build_sglang_logprob_kwargs(
            request.get("output_options", {}) or {},
            allow_top_logprobs=_shared_logprobs.sglang_top_logprobs_allowed(),
        )

    @staticmethod
    def _extract_logprobs(
        meta_info: Dict[str, Any],
        num_output_logprobs_so_far: int,
        return_tokens_as_token_ids: bool = False,
    ) -> tuple:
        return _shared_logprobs.extract_from_sglang_meta(
            meta_info,
            num_output_logprobs_so_far,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

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
        priority = (request.get("routing") or {}).get("priority")
        logprob_kwargs = self._build_logprob_kwargs(request)
        metadata_uploader = self._metadata_uploader_from_request(request)

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

            trace_header = context.trace_headers() if self.enable_trace else None

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            # Decode re-extracts the media so its token layout matches prefill's
            # and the transferred KV lines up.
            decode_mm_kwargs = build_disagg_mm_kwargs(request)

            decode = await self.engine.async_generate(
                **input_param,
                **decode_mm_kwargs,
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

            if not self.use_sglang_tokenizer:
                async for out in self._process_token_stream(
                    decode,
                    context,
                    return_tokens_as_token_ids,
                    user_stop_token_ids=user_stop_token_ids,
                    metadata_uploader=metadata_uploader,
                ):
                    yield out
            else:
                async for out in self._process_text_stream(
                    decode,
                    context,
                    request=request,
                    user_stop_token_ids=user_stop_token_ids,
                    metadata_uploader=metadata_uploader,
                ):
                    yield out
        else:
            # Extract image/video URLs for multimodal requests. SGLang's mm_data_processor
            # handles loading/preprocessing, and the scheduler does vision encoding.
            mm_data = request.get("multi_modal_data", {})
            video_data = extract_media_urls(mm_data, VIDEO_URL_KEY)

            image_data: list[str] | list[PILImage] | None
            if self._enable_frontend_decoding:
                # Invariant from __init__: _image_loader is non-None iff
                # _enable_frontend_decoding is True. Assert narrows the
                # Optional for the type checker without runtime branching.
                assert self._image_loader is not None
                image_items = mm_data.get(IMAGE_URL_KEY) or []
                if image_items:
                    image_data = await self._image_loader.load_image_batch(image_items)
                else:
                    image_data = None
            else:
                image_data = extract_media_urls(mm_data, IMAGE_URL_KEY)

            trace_header = context.trace_headers() if self.enable_trace else None

            # Extract dp_rank from routing info (set by KV router)
            routing = request.get("routing") or {}
            dp_rank = routing.get("dp_rank")

            mm_hashes_kwargs: Dict[str, Any] = {}
            if self._mm_hashes_supported:
                forwarded = self._extract_mm_hashes(request)
                if forwarded is not None:
                    mm_hashes_kwargs["mm_hashes"] = forwarded

            agg = await self.engine.async_generate(
                **input_param,
                image_data=image_data,
                video_data=video_data,
                sampling_params=sampling_params,
                stream=True,
                **self._routed_experts_kwargs,
                **mm_hashes_kwargs,
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
                    agg,
                    context,
                    return_tokens_as_token_ids,
                    user_stop_token_ids=user_stop_token_ids,
                    metadata_uploader=metadata_uploader,
                ):
                    yield out
            else:
                async for out in self._process_text_stream(
                    agg,
                    context,
                    request=request,
                    user_stop_token_ids=user_stop_token_ids,
                    metadata_uploader=metadata_uploader,
                ):
                    yield out

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        return_tokens_as_token_ids: bool = False,
        user_stop_token_ids: set[int] | None = None,
        metadata_uploader: MetadataUploader | None = None,
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
                meta_info = res.get("meta_info", {})
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
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
                finish_reason = meta_info["finish_reason"]
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

                if metadata_uploader is None:
                    # Extract logprobs for new tokens if available
                    (
                        log_probs,
                        top_logprobs,
                        next_logprobs_total,
                    ) = self._extract_logprobs(
                        meta_info,
                        output_logprobs_per_choice.get(output_idx, 0),
                        return_tokens_as_token_ids=return_tokens_as_token_ids,
                    )
                    output_logprobs_per_choice[output_idx] = next_logprobs_total
                    if log_probs is not None:
                        out["log_probs"] = log_probs
                    if top_logprobs is not None:
                        out["top_logprobs"] = top_logprobs

                routed_experts = meta_info.get("routed_experts")
                if routed_experts is not None and metadata_uploader is None:
                    # sglang >= 0.5.11 base64-encodes routed_experts upstream. It rides
                    # the engine's opaque engine_data passthrough (surfaced by the frontend
                    # as nvext.routed_experts); disaggregated_params stays KV-transfer only.
                    out["engine_data"] = {"routed_experts": routed_experts}
                if finish_reason:
                    input_tokens = meta_info["prompt_tokens"]
                    completion_tokens = meta_info["completion_tokens"]
                    cached_tokens = meta_info["cached_tokens"]
                    prefill_prompt_tokens_details = None
                    if cached_tokens is not None and cached_tokens > 0:
                        prefill_prompt_tokens_details = {"cached_tokens": cached_tokens}
                    out["completion_usage"] = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_tokens + completion_tokens,
                        "prompt_tokens_details": prefill_prompt_tokens_details,
                    }
                    if metadata_uploader is not None:
                        try:
                            await metadata_uploader.upload_choice(output_idx, meta_info)
                        finally:
                            meta_info.clear()
                elif metadata_uploader is not None:
                    meta_info.clear()
                if not context.is_stopped():
                    yield out

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        request: Dict[str, Any] | None = None,
        user_stop_token_ids: set[int] | None = None,
        metadata_uploader: MetadataUploader | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        request = request or {}
        # SGLang text chunks are cumulative per choice. Keep independent text
        # offsets so interleaved n>1 choices do not compute deltas from each
        # other's previous text.
        text_counts_per_choice: dict[int, int] = {}

        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future: asyncio.Future[str] = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                meta_info = res.get("meta_info", {})
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
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

                text = res.get("text", "")

                finish_reason = meta_info["finish_reason"]
                finish_reason_type = (
                    normalize_finish_reason(finish_reason["type"])
                    if finish_reason
                    else None
                )
                next_count = len(text)
                count = text_counts_per_choice.get(index, 0)
                delta = text[count:]

                choice_data = {
                    "index": index,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": finish_reason_type,
                }
                stop_reason = _extract_sglang_stop_reason(
                    finish_reason, user_stop_token_ids
                )

                response = {
                    "id": meta_info["id"],
                    "created": int(time.time()),
                    "choices": [choice_data],
                    "model": self.config.server_args.served_model_name,
                    "object": "chat.completion.chunk",
                }
                response_nvext: dict[str, Any] = {}
                if stop_reason is not None and _nvext_extra_field_requested(
                    request, "stop_reason"
                ):
                    response_nvext["stop_reason"] = stop_reason
                routed_experts = meta_info.get("routed_experts")
                if routed_experts is not None and metadata_uploader is None:
                    # sglang >= 0.5.11 base64-encodes routed_experts upstream.
                    response_nvext["routed_experts"] = routed_experts
                if finish_reason and metadata_uploader is not None:
                    try:
                        await metadata_uploader.upload_choice(index, meta_info)
                    finally:
                        meta_info.clear()
                elif metadata_uploader is not None:
                    meta_info.clear()
                if response_nvext:
                    response["nvext"] = response_nvext
                if not context.is_stopped():
                    yield response
                text_counts_per_choice[index] = next_count
