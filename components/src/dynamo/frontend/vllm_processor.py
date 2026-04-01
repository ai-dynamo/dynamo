#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import base64
import concurrent.futures
import functools
import logging
import os
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.inputs.data import TokensPrompt
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import GENERATION_TASKS
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from dynamo._internal import ModelDeploymentCard
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.frontend.frontend_args import FrontendConfig
from dynamo.llm import (
    KvRouter,
    ModelCardInstanceId,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    fetch_model,
)
from dynamo.runtime import Client, DistributedRuntime
from dynamo.vllm.multimodal_utils.hash_utils import (
    build_block_mm_infos,
    compute_mm_uuids_from_images,
    find_image_token_ranges,
    unwrap_pil_image,
)

from .prepost import StreamingPostProcessor, preprocess_chat_request
from .utils import random_uuid

logger = logging.getLogger(__name__)

_FORMAT_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "eos": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
    "cancelled": FinishReason.ABORT,
    "content_filter": FinishReason.STOP,
}


def map_finish_reason(raw_reason: str | None) -> FinishReason | None:
    if raw_reason is None:
        return None
    if raw_reason.startswith("error"):
        return FinishReason.ERROR
    if raw_reason.startswith("abort"):
        return FinishReason.ABORT
    if raw_reason.startswith("content_filter"):
        logger.info("Router finish_reason indicates content filtering: %s", raw_reason)
        raw_reason = "content_filter"
    mapped = _FINISH_REASON_MAP.get(raw_reason)
    if mapped is None:
        logger.warning("Unknown finish_reason from router: %s", raw_reason)
    return mapped


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        router: Any,  # Client or KvRouter
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        block_size: int = 16,
        mm_image_token_id: int | None = None,
        mm_image_processor: Any = None,
        image_loader: ImageLoader | None = None,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        # Thread pool for offloading CPU-bound process_inputs() so it doesn't
        # block the asyncio event loop under concurrent requests.
        self._preproc_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=64, thread_name_prefix="vllm_preproc"
        )
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self.block_size = block_size
        self.mm_image_token_id = mm_image_token_id
        self._mm_image_processor = mm_image_processor
        self._image_loader = image_loader
        self.exclude_tools_when_tool_choice_none = True

    def _get_eos_token_ids(self) -> list[int]:
        """Return EOS token ids using tokenizer metadata.

        vLLM 0.17.0 removed EngineCoreRequest.eos_token_id, so Dynamo can no
        longer read EOS ids from the preprocessed request object.
        """
        eos_token_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_token_ids is not None and not isinstance(eos_token_ids, int):
            return list(eos_token_ids)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return []
        return [eos_token_id]

    async def _prefetch_and_replace_image_urls(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Pre-fetch HTTP image URLs via ImageLoader, replace with data URIs.

        Returns the (modified request, original_http_urls) tuple.
        original_http_urls preserves the original HTTP URLs in order so the
        backend receives short URLs via multi_modal_data, not base64 blobs.

        On cache hit: vLLM's renderer decodes base64 instead of making an HTTP
        request, eliminating redundant downloads for repeated image URLs.
        On cache miss: ImageLoader fetches and caches for future requests.
        """
        assert self._image_loader is not None

        messages = request.get("messages", [])
        http_urls: list[str] = []
        for msg in messages:
            for part in (
                msg.get("content") if isinstance(msg.get("content"), list) else []
            ):
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith(("http://", "https://")):
                        http_urls.append(url)

        if not http_urls:
            return request, []

        # Fetch in parallel — cache misses download, hits return immediately.
        await asyncio.gather(
            *[self._image_loader.load_image(url) for url in http_urls],
            return_exceptions=True,
        )

        # Build URL → data URI map from cached raw bytes.
        url_to_data_uri: dict[str, str] = {}
        for url in http_urls:
            cached = self._image_loader.get_cached_raw_bytes(url)
            if cached is not None:
                raw_bytes, fmt = cached
                mime = _FORMAT_TO_MIME.get(fmt, "image/jpeg")
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                url_to_data_uri[url] = f"data:{mime};base64,{b64}"

        if not url_to_data_uri:
            return request, http_urls

        # Replace HTTP URLs in-place so vLLM's renderer gets data URIs.
        for msg in messages:
            for part in (
                msg.get("content") if isinstance(msg.get("content"), list) else []
            ):
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url in url_to_data_uri:
                        part["image_url"]["url"] = url_to_data_uri[url]

        # Return original HTTP URLs — callers must use these for backend
        # multi_modal_data, not the data URIs now in request["messages"].
        return request, http_urls

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a backend using the router.
        """

        async for item in self._generator_inner(request):
            yield item

    @_nvtx.range_decorator("frontend:_generator_inner", color="green")
    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        # Pre-fetch HTTP image URLs via ImageLoader so vLLM's renderer gets
        # data URIs instead of re-downloading.  Capture original HTTP URLs
        # before replacement — they are sent to the backend as RawUrl so the
        # backend fetches a short URL, not a base64 blob over NATS.
        original_image_urls: list[str] = []
        if self._image_loader is not None:
            request, original_image_urls = await self._prefetch_and_replace_image_urls(
                request
            )

        # vLLM's Pydantic model requires image_url.detail to be 'auto'/'low'/'high'.
        # Normalize missing/null detail before passing to preprocess_chat_request.
        for msg in request.get("messages", []):
            for part in (
                msg.get("content") if isinstance(msg.get("content"), list) else []
            ):
                if part.get("type") == "image_url":
                    img_url = part.setdefault("image_url", {})
                    if img_url.get("detail") is None:
                        img_url["detail"] = "auto"

        _t_pre0 = time.perf_counter()
        with _nvtx.annotate("frontend:preprocess_chat_request", color="blue"):
            pre = await preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                renderer=self.input_processor.renderer,
                tool_parser_class=self.tool_parser_class,
                exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
            )
        logger.info(
            "[timing] preprocess_chat_request took %.1f ms",
            (time.perf_counter() - _t_pre0) * 1000,
        )

        request_for_sampling = pre.request_for_sampling
        tool_parser = pre.tool_parser
        chat_template_kwargs = pre.chat_template_kwargs
        engine_prompt = pre.engine_prompt
        tokens = pre.prompt_token_ids

        if request_for_sampling.max_completion_tokens is not None:
            max_tokens = request_for_sampling.max_completion_tokens
        elif request_for_sampling.max_tokens is not None:
            max_tokens = request_for_sampling.max_tokens
        else:
            # This should mean model max - prompt len.
            max_tokens = None

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        # generation_config.json
        # Skip eos_token_id: vLLM 0.17.0 made SamplingParams.eos_token_id a
        # read-only property; eos tokens are handled via eos_token_ids below.
        for k, v in self.input_processor.generation_config_fields.items():
            if k == "eos_token_id":
                continue
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

        # User request: copy fields supported by both request schema and
        # SamplingParams, excluding fields handled separately below.
        sampling_fields = (
            set(getattr(SamplingParams, "__annotations__", ()))
            & set(type(request_for_sampling).model_fields)
        ) - {"max_tokens", "logprobs", "output_kind"}
        for k in sorted(sampling_fields):
            v = getattr(request_for_sampling, k, None)
            if v is not None:
                setattr(sampling_params, k, v)
        logprobs = request_for_sampling.logprobs
        top_logprobs = request_for_sampling.top_logprobs
        if logprobs is True:
            sampling_params.logprobs = top_logprobs or 1
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params.logprobs = logprobs
        elif top_logprobs not in (None, 0):
            sampling_params.logprobs = top_logprobs
        if sampling_params.logprobs is not None and sampling_params.logprobs > 0:
            logger.warning(
                "Logprobs requested but not supported in distributed inference mode"
            )

        # Fast-path MM token expansion: compute expanded token IDs from image
        # dimensions instead of running the full HF image processor inside
        # process_inputs (~15ms).  When this succeeds we skip passing
        # multi_modal_data to process_inputs, then patch prompt_token_ids
        # afterward.  Text-only requests are unaffected (no multi_modal_data).
        fast_expanded_tokens: list[int] | None = None
        if (
            self._mm_image_processor is not None
            and self.mm_image_token_id is not None
            and "multi_modal_data" in engine_prompt
        ):
            try:
                _t_fast = time.perf_counter()
                with _nvtx.annotate("frontend:fast_mm_expand", color="orange"):
                    imgs = engine_prompt["multi_modal_data"].get("image") or []
                    imgs = imgs if isinstance(imgs, list) else [imgs]
                    pil_imgs = [unwrap_pil_image(img) for img in imgs]
                    get_num_patches = (
                        self._mm_image_processor.get_number_of_image_patches
                    )
                    merge_size = self._mm_image_processor.merge_size
                    tokens_per_image = [
                        int(get_num_patches(img.height, img.width, {}))
                        // (merge_size**2)
                        for img in pil_imgs
                    ]
                    expanded: list[int] = []
                    img_idx = 0
                    for t in tokens:
                        if t == self.mm_image_token_id and img_idx < len(
                            tokens_per_image
                        ):
                            expanded.extend(
                                [self.mm_image_token_id] * tokens_per_image[img_idx]
                            )
                            img_idx += 1
                        else:
                            expanded.append(t)
                    fast_expanded_tokens = expanded
                logger.info(
                    "[timing] fast MM expansion took %.2f ms, tokens %d -> %d",
                    (time.perf_counter() - _t_fast) * 1000,
                    len(tokens),
                    len(expanded),
                )
            except Exception as exc:
                logger.info(
                    "Fast MM expansion failed (%s), falling back to process_inputs",
                    exc,
                )

        # This calls update_from_generation_config and update_from_tokenizer on SamplingParams
        prompt_inputs = TokensPrompt(prompt_token_ids=tokens)
        if fast_expanded_tokens is None:
            # Fast path unavailable: pass images to process_inputs for expansion.
            if "multi_modal_data" in engine_prompt:
                prompt_inputs["multi_modal_data"] = engine_prompt["multi_modal_data"]
            if "multi_modal_uuids" in engine_prompt:
                prompt_inputs["multi_modal_uuids"] = engine_prompt["multi_modal_uuids"]
        if request_for_sampling.cache_salt is not None:
            prompt_inputs["cache_salt"] = request_for_sampling.cache_salt
        if request_for_sampling.mm_processor_kwargs is not None:
            prompt_inputs[
                "mm_processor_kwargs"
            ] = request_for_sampling.mm_processor_kwargs

        loop = asyncio.get_running_loop()
        _t0 = time.perf_counter()
        with _nvtx.annotate("frontend:process_inputs", color="yellow"):
            vllm_preproc: EngineCoreRequest = await loop.run_in_executor(
                self._preproc_executor,
                functools.partial(
                    self.input_processor.process_inputs,
                    request_id,
                    prompt_inputs,
                    sampling_params,
                    GENERATION_TASKS,  # vLLM 0.17.0: required supported_tasks arg
                ),
            )
        logger.info(
            "[timing] process_inputs took %.1f ms", (time.perf_counter() - _t0) * 1000
        )

        InputProcessor.assign_request_id(vllm_preproc)

        # Patch prompt_token_ids with fast-path expanded tokens.
        # output_processor.add_request uses this for prompt length accounting.
        if fast_expanded_tokens is not None:
            vllm_preproc.prompt_token_ids = fast_expanded_tokens

        # vLLM 0.17.0 removed EngineCoreRequest.eos_token_id. Dynamo now uses
        # tokenizer metadata for EOS ids when constructing the router payload.

        # Convert to a Python object that has fields that match our PreprocessedRequest
        sp = vllm_preproc.sampling_params
        if sp.n != 1:
            logger.error("Unsupported SamplingParams.n=%d, only n=1 is supported", sp.n)
            yield {
                "error": {
                    "message": (
                        f"Unsupported value: 'n={sp.n}'. "
                        "This endpoint currently supports only n=1."
                    ),
                    "type": "invalid_request_error",
                    "param": "n",
                    "code": "unsupported_value",
                }
            }
            return

        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "stop_token_ids": sp.stop_token_ids,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
            },
            "sampling_options": {
                "n": sp.n,
                "presence_penalty": sp.presence_penalty,
                "frequency_penalty": sp.frequency_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "seed": sp.seed,
            },
            "output_options": {
                "logprobs": sp.logprobs,
                "prompt_logprobs": sp.prompt_logprobs,
                "skip_special_tokens": sp.skip_special_tokens,
            },
            "eos_token_ids": self._get_eos_token_ids(),
            "annotations": [],
        }

        post = StreamingPostProcessor(
            tokenizer=self.tokenizer,
            request_for_sampling=request_for_sampling,
            sampling_params=sampling_params,
            prompt_token_ids=tokens,
            tool_parser=tool_parser,
            reasoning_parser_class=self.reasoning_parser_class,
            chat_template_kwargs=chat_template_kwargs,
        )

        # --- MM-aware routing info ---
        mm_routing_info: dict | None = None
        expanded_tokens = list(vllm_preproc.prompt_token_ids)

        # Use original HTTP URLs for backend multi_modal_data.
        # If _image_loader replaced them with data URIs, original_image_urls
        # holds the originals; otherwise scan messages directly.
        if original_image_urls:
            image_urls = original_image_urls
        else:
            image_urls = [
                part.get("image_url", {}).get("url")
                for msg in request.get("messages", [])
                for part in (
                    msg.get("content") if isinstance(msg.get("content"), list) else []
                )
                if part.get("type") == "image_url"
            ]
            image_urls = [u for u in image_urls if u]

        # Get PIL images already downloaded during preprocess_chat_request.
        # Unwrap MediaWithBytes so we hash pixel data (not JPEG bytes).
        pil_images: list = []
        mm_raw = engine_prompt.get("multi_modal_data")
        if mm_raw:
            imgs = mm_raw.get("image")
            if imgs is not None:
                raw_list = imgs if isinstance(imgs, list) else [imgs]
                pil_images = [unwrap_pil_image(img) for img in raw_list]

        logger.debug(
            "[mm-routing] placeholder_tokens=%d vllm_expanded_tokens=%d "
            "image_urls=%d pil_images=%d image_token_id=%s",
            len(tokens),
            len(expanded_tokens),
            len(image_urls),
            len(pil_images),
            self.mm_image_token_id,
        )

        if self.is_kv_router and pil_images and self.mm_image_token_id is not None:
            # Must match the backend's hash path (handlers.py _compute_mm_uuids):
            # hash unwrapped PIL img.tobytes(), NOT mm_features.mm_hash which may
            # be derived from MediaWithBytes.original_bytes (raw JPEG).
            with _nvtx.annotate("frontend:compute_mm_uuids", color="cyan"):
                mm_hashes = [
                    int(u[:16], 16) for u in compute_mm_uuids_from_images(pil_images)
                ]
            image_ranges = find_image_token_ranges(
                expanded_tokens, self.mm_image_token_id
            )
            if image_ranges and len(image_ranges) == len(mm_hashes):
                block_mm_infos = build_block_mm_infos(
                    len(expanded_tokens), self.block_size, mm_hashes, image_ranges
                )
                mm_routing_info = {
                    "routing_token_ids": expanded_tokens,
                    "block_mm_infos": block_mm_infos,
                }
                logger.debug(
                    "[mm-routing] built mm_routing_info: routing_tokens=%d blocks=%d",
                    len(expanded_tokens),
                    len(block_mm_infos),
                )
            else:
                logger.warning(
                    "[mm-routing] image_ranges count (%d) != mm_hashes count (%d); "
                    "falling back to text-only routing",
                    len(image_ranges),
                    len(mm_hashes),
                )

        async for item in self._generate_and_stream(
            request_id,
            request,
            dynamo_preproc,
            tokens,
            vllm_preproc,
            post,
            mm_routing_info=mm_routing_info,
            image_urls=image_urls,
        ):
            yield item

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        vllm_preproc: EngineCoreRequest,
        post: StreamingPostProcessor,
        mm_routing_info: dict | None = None,
        image_urls: list[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.output_processor.add_request(vllm_preproc, None)

        try:
            if self.is_kv_router:
                extra_args = {"messages": request.get("messages", [])}
                # Image URLs let the backend download & process images itself.
                multi_modal_data = (
                    {"image_url": [{"RawUrl": u} for u in image_urls]}
                    if image_urls
                    else None
                )
                dynamo_stream = await self.router.generate(
                    token_ids=tokens,
                    model=dynamo_preproc["model"],
                    stop_conditions=dynamo_preproc["stop_conditions"],
                    sampling_options=dynamo_preproc["sampling_options"],
                    output_options=dynamo_preproc["output_options"],
                    extra_args=extra_args,
                    multi_modal_data=multi_modal_data,
                    mm_routing_info=mm_routing_info,
                )
            else:
                dynamo_stream = await self.router.generate(
                    dynamo_preproc, annotated=False
                )

            async for dynamo_response in dynamo_stream:
                if self.is_kv_router:
                    engine_response = dynamo_response
                elif hasattr(dynamo_response, "data"):
                    engine_response = dynamo_response.data()
                else:
                    engine_response = dynamo_response

                if engine_response is None or "token_ids" not in engine_response:
                    logger.error("No outputs from engine for request %s", request_id)
                    yield {
                        "error": {
                            "message": f"Invalid engine response for request {request_id}",
                            "type": "internal_error",
                        }
                    }
                    break

                raw_finish_reason = engine_response.get("finish_reason")
                finish_reason = map_finish_reason(raw_finish_reason)
                stop_reason = engine_response.get("stop_reason")

                vllm_response = EngineCoreOutput(
                    request_id=vllm_preproc.request_id,
                    new_token_ids=engine_response["token_ids"],
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )

                vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                    [vllm_response]
                )

                if vllm_out.reqs_to_abort:
                    pass

                choices = []
                if not vllm_out.request_outputs:
                    continue
                for output in vllm_out.request_outputs[0].outputs:
                    choice = post.process_output(output)
                    if choice:
                        choices.append(choice)

                if choices:
                    dynamo_out = {
                        "id": request_id,
                        "choices": choices,
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
                    }
                    if usage := engine_response.get("completion_usage"):
                        dynamo_out["usage"] = usage

                    yield dynamo_out
        finally:
            if vllm_preproc.request_id in self.output_processor.request_states:
                self.output_processor.abort_requests(
                    [vllm_preproc.request_id], internal=True
                )


class EngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        config: FrontendConfig,
        flags: Namespace,
    ):
        if config.preprocess_workers != 0:
            raise RuntimeError(
                "preprocess_workers > 0 is not supported by vllm preprocessor"
            )

        self.runtime = runtime
        self.router_config = router_config
        self.config = config
        self.flags = flags
        self.stream_interval = 20
        raw_stream_interval = os.getenv("DYN_VLLM_STREAM_INTERVAL")
        if raw_stream_interval:
            try:
                self.stream_interval = max(1, int(raw_stream_interval))
            except ValueError:
                logger.warning(
                    "Invalid DYN_VLLM_STREAM_INTERVAL=%r, using default=%d",
                    raw_stream_interval,
                    self.stream_interval,
                )

    async def chat_engine_factory(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
    ) -> PythonAsyncEngine:
        """
        Called by Rust when a model is discovered.
        """
        model_type = mdc.model_type()
        if not model_type.supports_chat():
            raise RuntimeError(
                f"model type {model_type} is not supported by this factory"
            )
        loop = asyncio.get_running_loop()

        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_model(source_path, ignore_weights=True)

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"

        model_config = ModelConfig(
            model=source_path,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format=load_format),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        input_processor = InputProcessor(vllm_config)
        tokenizer = input_processor.get_tokenizer()
        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=self.stream_interval,
        )
        logger.info("vLLM OutputProcessor stream_interval=%d", self.stream_interval)

        tool_parser_name = self.flags.tool_call_parser or mdc.runtime_config().get(
            "tool_call_parser"
        )
        if tool_parser_name:
            tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
        else:
            tool_parser_class = None

        reasoning_parser_name = self.flags.reasoning_parser or mdc.runtime_config().get(
            "reasoning_parser"
        )
        if reasoning_parser_name:
            reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(
                reasoning_parser_name
            )
        else:
            reasoning_parser_class = None

        namespace_name, component_name, endpoint_name = instance_id.triple()
        generate_endpoint = self.runtime.endpoint(
            f"{namespace_name}.{component_name}.{endpoint_name}"
        )
        router: Client | KvRouter
        block_size = self.config.kv_cache_block_size or 16
        if self.router_config.router_mode == RouterMode.KV:
            router = KvRouter(
                endpoint=generate_endpoint,
                block_size=block_size,
                kv_router_config=self.router_config.kv_router_config,
            )
        else:
            router = await generate_endpoint.client(
                router_mode=self.router_config.router_mode
            )

        # Discover image_token_id for MM-aware KV routing.
        # Used to find image token ranges in vllm_preproc.prompt_token_ids.
        # vllm processor only loads the tokenizer, not the full AutoProcessor so cannot get the image_token_id directly.
        mm_image_token_id: int | None = None
        try:
            inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
            mm_image_token_id = getattr(inner_tok, "image_token_id", None)
            if mm_image_token_id is None:
                unk_id = getattr(inner_tok, "unk_token_id", None)
                for tok_str in ["<|image_pad|>", "<image>", "<IMG>"]:
                    tid = inner_tok.convert_tokens_to_ids(tok_str)
                    if tid is not None and tid != unk_id:
                        mm_image_token_id = tid
                        break
        except Exception as exc:
            logger.debug("Could not determine mm_image_token_id: %s", exc)
        logger.info(
            "MM image_token_id=%s block_size=%d (None=MM routing disabled)",
            mm_image_token_id,
            block_size,
        )

        # Load AutoProcessor to enable fast MM token expansion from image dims.
        # This avoids running the full HF image processor inside process_inputs
        # (~15ms) for every multimodal request.  Only config files are needed —
        # no model weights (load_format="dummy" is already set above).
        mm_image_processor = None
        if mm_image_token_id is not None:
            try:
                from transformers import AutoProcessor

                auto_proc = AutoProcessor.from_pretrained(source_path)
                ip = getattr(auto_proc, "image_processor", None)
                if ip is not None and hasattr(ip, "get_number_of_image_patches"):
                    mm_image_processor = ip
                    logger.info("MM fast expansion enabled via %s", type(ip).__name__)
                else:
                    logger.info(
                        "AutoProcessor loaded but image_processor lacks "
                        "get_number_of_image_patches; fast MM expansion disabled"
                    )
            except Exception as exc:
                logger.info(
                    "AutoProcessor load failed, fast MM expansion disabled: %s", exc
                )

        image_loader = ImageLoader() if mm_image_token_id is not None else None

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            block_size=block_size,
            mm_image_token_id=mm_image_token_id,
            mm_image_processor=mm_image_processor,
            image_loader=image_loader,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
