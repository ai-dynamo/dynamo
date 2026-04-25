#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import logging
import os
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import GENERATION_TASKS
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from dynamo._internal import ModelDeploymentCard
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlSender,
    MmKwargsSender,
    MmKwargsShmSender,
)
from dynamo.common.multimodal.routing_utils import build_mm_routing_info_from_features
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

from .prepost import StreamingPostProcessor, preprocess_chat_request
from .utils import extract_mm_urls, random_uuid

logger = logging.getLogger(__name__)


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
        enable_auto_tool_choice: bool = False,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self.exclude_tools_when_tool_choice_none = True
        self.block_size = block_size
        self.enable_auto_tool_choice = enable_auto_tool_choice
        # Sender for mm_kwargs transfer — instantiated lazily on first MM request.
        # MmKwargsShmSender for same-node transfers (default), MmKwargsNixlSender
        # for cross-node RDMA. Controlled by DYNAMO_MM_TRANSFER env var.
        self._sender: MmKwargsSender | None = None
        # Set DYNAMO_DISABLE_NIXL_MM=1 to disable mm_kwargs transfer entirely.
        # Set DYNAMO_MM_TRANSFER to choose transfer mode:
        #   shm (default): shared memory. Same-node only (~2ms). If the
        #     backend can't read the segment (cross-node), it falls back to
        #     normal processing (backend runs HF processor).
        #   nixl: NIXL RDMA. Works cross-node via IB.
        self.nixl_mm_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
        transfer_mode = os.environ.get("DYNAMO_MM_TRANSFER", "shm").lower()
        self.use_shm_transfer = transfer_mode == "shm"
        logger.info("[mm-routing] Transfer mode: %s", transfer_mode)

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

    async def _prepare_mm_routing(
        self,
        vllm_preproc: EngineCoreRequest,
        dynamo_preproc: dict[str, Any],
    ) -> tuple[dict | None, list, bool]:
        """Extract MM routing info and prepare mm_kwargs transfer.

        Returns:
            (mm_routing_info, cleanup_items, transferred)
            cleanup_items: passed to self._sender.cleanup() after streaming.
            transferred: True when all features with data were sent successfully.
        """
        mm_routing_info = None
        cleanup_items: list = []
        nixl_transferred = False

        rng_routing = _nvtx.start_range("mm_frontend:build_routing_info", color="cyan")
        if self.is_kv_router and vllm_preproc.mm_features:
            mm_routing_info = build_mm_routing_info_from_features(
                vllm_preproc.mm_features,
                prompt_token_ids=list(vllm_preproc.prompt_token_ids),
                block_size=self.block_size,
            )
            # Forward mm_hashes to backend for hash consistency — the backend
            # will use these directly instead of recomputing.
            mm_hashes_list = [f.mm_hash for f in vllm_preproc.mm_features]
            mm_placeholders_list = [
                (f.mm_position.offset, f.mm_position.length)
                for f in vllm_preproc.mm_features
            ]
            # Transport mm_hashes and mm_placeholders to backend via extra_args.
            if "extra_args" not in dynamo_preproc:
                dynamo_preproc["extra_args"] = {}
            dynamo_preproc["extra_args"]["mm_hashes"] = mm_hashes_list
            dynamo_preproc["extra_args"]["mm_placeholders"] = mm_placeholders_list
            # Forward the expanded prompt_token_ids (with image placeholders)
            # so the backend can use them in the pre-rendered MultiModalInput.
            dynamo_preproc["extra_args"]["expanded_token_ids"] = list(
                vllm_preproc.prompt_token_ids
            )

            n_blocks = len(mm_routing_info["block_mm_infos"]) if mm_routing_info else 0
            n_mm_blocks = sum(
                1 for b in (mm_routing_info or {}).get("block_mm_infos", []) if b
            )
            logger.debug(
                "[mm-routing] Built mm_routing_info: %d mm_features, "
                "%d hashes, %d total blocks, %d blocks with MM content, "
                "block_size=%d",
                len(vllm_preproc.mm_features),
                len(mm_hashes_list),
                n_blocks,
                n_mm_blocks,
                self.block_size,
            )
            if logger.isEnabledFor(logging.DEBUG):
                for i, f in enumerate(vllm_preproc.mm_features):
                    logger.debug(
                        "[mm-routing]   feature[%d]: modality=%s, hash=%s..., "
                        "offset=%d, length=%d",
                        i,
                        f.modality,
                        f.mm_hash[:16] if f.mm_hash else "None",
                        f.mm_position.offset,
                        f.mm_position.length,
                    )

            # Transfer pre-processed mm_kwargs to the backend so it can skip
            # the HF processor.  Strategy:
            #   - shm (default): shared memory, same-node only (~2ms).
            #     Cross-node backends fail gracefully and fall back to
            #     normal processing.
            #   - nixl: NIXL RDMA (works cross-node via IB).
            if not self.nixl_mm_enabled:
                logger.debug(
                    "[mm-routing] mm_kwargs transfer disabled via DYNAMO_DISABLE_NIXL_MM"
                )
            else:
                try:
                    if self._sender is None:
                        self._sender = (
                            MmKwargsShmSender()
                            if self.use_shm_transfer
                            else MmKwargsNixlSender()
                        )
                    # NVTX annotation is owned by MmKwargsSender.prepare via
                    # the subclass's _nvtx_label/_nvtx_color class attrs.
                    extra_update, cleanup_items = await self._sender.prepare(
                        vllm_preproc.mm_features, modality="image"
                    )
                    if extra_update is not None:
                        dynamo_preproc["extra_args"].update(extra_update)
                        nixl_transferred = True
                    else:
                        logger.debug(
                            "[mm-routing] sender returned None — no tensors to transfer"
                        )
                except Exception:
                    logger.warning(
                        "[mm-routing] sender failed, backend will run HF processor",
                        exc_info=True,
                    )
                    cleanup_items = []

        elif self.is_kv_router:
            logger.debug("[mm-routing] No mm_features — text-only request")
        _nvtx.end_range(rng_routing)

        return mm_routing_info, cleanup_items, nixl_transferred

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
        with _nvtx.annotate("mm_frontend:generator", color="blue"):
            async for item in self._generator_inner(request):
                yield item

    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        # vLLM's Pydantic model requires image_url.detail to be 'auto'/'low'/'high'.
        # The Rust HTTP layer accepts None/missing, so normalize before validation.
        messages = request.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    img_url = part.get("image_url")
                    if isinstance(img_url, dict) and img_url.get("detail") is None:
                        img_url["detail"] = "auto"

        # Images are fetched by vLLM's renderer via DynamoMediaConnector,
        # which wraps our ImageLoader (LRU cache + in-flight dedup).
        # No data URI encoding needed.
        with _nvtx.annotate("mm_frontend:preprocess_chat", color="yellow"):
            pre = await preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                renderer=self.input_processor.renderer,
                tool_parser_class=self.tool_parser_class,
                exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
                enable_auto_tool_choice=self.enable_auto_tool_choice,
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

        # The renderer's process_for_engine() always returns a fully processed
        # EngineInput (TokenInputs or MultiModalInputs) with a "type" key.
        # Pass it directly to process_inputs() — no need to rebuild a
        # TokensPrompt, and this avoids the deprecation warning.
        prompt_inputs = engine_prompt
        if request_for_sampling.cache_salt is not None:
            prompt_inputs["cache_salt"] = request_for_sampling.cache_salt
        if request_for_sampling.mm_processor_kwargs is not None:
            prompt_inputs[
                "mm_processor_kwargs"
            ] = request_for_sampling.mm_processor_kwargs

        with _nvtx.annotate("mm_frontend:process_inputs", color="orange"):
            vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
                request_id,
                prompt_inputs,
                sampling_params,
                GENERATION_TASKS,  # vLLM 0.17.0: required supported_tasks arg
            )

        InputProcessor.assign_request_id(vllm_preproc)

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
            "routing": request.get("routing"),
        }

        # Extract MM routing metadata and prepare transfer.
        cleanup_items: list = []
        try:
            (
                mm_routing_info,
                cleanup_items,
                nixl_transferred,
            ) = await self._prepare_mm_routing(vllm_preproc, dynamo_preproc)

            # Forward multimodal URLs so the backend handler can load the media.
            # Only skip when ALL features were transferred — a partial transfer
            # (some features had data=None due to processor cache) still needs
            # URLs for the backend to process the missing features.
            n_features = (
                len(vllm_preproc.mm_features) if vllm_preproc.mm_features else 0
            )
            n_with_data = sum(
                1 for f in (vllm_preproc.mm_features or []) if f.data is not None
            )
            all_transferred = nixl_transferred and n_with_data == n_features
            if not all_transferred:
                mm_data = extract_mm_urls(request.get("messages") or [])
                if mm_data:
                    dynamo_preproc["multi_modal_data"] = mm_data

            # Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
            if request_for_sampling.mm_processor_kwargs is not None:
                dynamo_preproc[
                    "mm_processor_kwargs"
                ] = request_for_sampling.mm_processor_kwargs

            post = StreamingPostProcessor(
                tokenizer=self.tokenizer,
                request_for_sampling=request_for_sampling,
                sampling_params=sampling_params,
                prompt_token_ids=tokens,
                tool_parser=tool_parser,
                reasoning_parser_class=self.reasoning_parser_class,
                chat_template_kwargs=chat_template_kwargs,
            )

            async for item in self._generate_and_stream(
                request_id,
                request,
                dynamo_preproc,
                tokens,
                vllm_preproc,
                post,
                mm_routing_info=mm_routing_info,
            ):
                yield item
        finally:
            if cleanup_items and self._sender is not None:
                await self._sender.cleanup(cleanup_items)

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        vllm_preproc: EngineCoreRequest,
        post: StreamingPostProcessor,
        mm_routing_info: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.output_processor.add_request(vllm_preproc, None)

        try:
            rng_route = _nvtx.start_range("mm_frontend:kv_router_generate", color="red")
            if self.is_kv_router:
                kv_kwargs: dict[str, Any] = {
                    "token_ids": tokens,
                    "model": dynamo_preproc["model"],
                    "stop_conditions": dynamo_preproc["stop_conditions"],
                    "sampling_options": dynamo_preproc["sampling_options"],
                    "output_options": dynamo_preproc["output_options"],
                    "multi_modal_data": dynamo_preproc.get("multi_modal_data"),
                }
                if dynamo_preproc.get("extra_args"):
                    kv_kwargs["extra_args"] = dynamo_preproc["extra_args"]
                    ea = dynamo_preproc["extra_args"]
                    logger.debug(
                        "[mm-routing] extra_args keys=%s, has_nixl=%s, "
                        "n_hashes=%d, n_placeholders=%d",
                        list(ea.keys()),
                        "mm_kwargs_nixl" in ea,
                        len(ea.get("mm_hashes", [])),
                        len(ea.get("mm_placeholders", [])),
                    )
                # Forward mm_processor_kwargs (e.g. use_audio_in_video) to backend.
                mm_proc_kwargs = dynamo_preproc.get("mm_processor_kwargs")
                if mm_proc_kwargs is not None:
                    if "extra_args" not in kv_kwargs or kv_kwargs["extra_args"] is None:
                        kv_kwargs["extra_args"] = {}
                    kv_kwargs["extra_args"]["mm_processor_kwargs"] = mm_proc_kwargs
                if mm_routing_info is not None:
                    kv_kwargs["mm_routing_info"] = mm_routing_info
                    logger.debug(
                        "[mm-routing] KvRouter.generate() called with "
                        "mm_routing_info (%d routing tokens, %d blocks)",
                        len(mm_routing_info.get("routing_token_ids", [])),
                        len(mm_routing_info.get("block_mm_infos", [])),
                    )
                else:
                    logger.debug(
                        "[mm-routing] KvRouter.generate() called without "
                        "mm_routing_info (text-only)"
                    )
                dynamo_stream = await self.router.generate(**kv_kwargs)
            else:
                dynamo_stream = await self.router.generate(
                    dynamo_preproc, annotated=False
                )
            _nvtx.end_range(rng_route)

            rng_stream = _nvtx.start_range(
                "mm_frontend:stream_response", color="purple"
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
            _nvtx.end_range(rng_stream)
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
        if self.config.preprocess_workers != 0:
            raise RuntimeError(
                "preprocess_workers is not supported for vllm processor. "
                "Use the sglang processor for worker-pool preprocessing."
            )
        loop = asyncio.get_running_loop()

        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_model(source_path, ignore_weights=True)

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"
        trust_remote_code = self.config.trust_remote_code
        enable_auto_tool_choice = getattr(self.flags, "enable_auto_tool_choice", False)

        model_config = ModelConfig(
            model=source_path,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            trust_remote_code=trust_remote_code,
        )
        # Use processor_only cache so tensor data persists across requests.
        # The default "lru" sender cache drops tensor data on cache hits
        # (designed for disagg where P1 holds tensors), but we need the
        # data to pickle and send via NIXL on repeated requests.
        if model_config.multimodal_config is not None:
            nixl_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
            if nixl_enabled:
                model_config.multimodal_config.mm_processor_cache_type = (
                    "processor_only"
                )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format=load_format),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        # Register dynamo's ImageLoader as vLLM's media connector so the
        # renderer uses our LRU cache + in-flight dedup for image fetching.
        # This eliminates data URI encoding overhead entirely.
        if os.environ.get("VLLM_MEDIA_CONNECTOR") != "dynamo":
            os.environ["VLLM_MEDIA_CONNECTOR"] = "dynamo"
        import dynamo.common.multimodal.media_connector  # noqa: F401

        input_processor = InputProcessor(vllm_config)
        tokenizer = input_processor.get_tokenizer()

        # Resolve stream_interval: env var override > backend config > default (20)
        stream_interval = self.stream_interval
        if not os.getenv("DYN_VLLM_STREAM_INTERVAL"):
            backend_interval = (
                mdc.runtime_config().get("runtime_data", {}).get("stream_interval")
            )
            if backend_interval is not None:
                try:
                    stream_interval = max(1, int(backend_interval))
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid stream_interval=%r from backend runtime_config, "
                        "using default=%d",
                        backend_interval,
                        stream_interval,
                    )

        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=stream_interval,
        )
        logger.info("vLLM OutputProcessor stream_interval=%d", stream_interval)

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

        block_size = self.config.kv_cache_block_size or 16

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            block_size=block_size,
            enable_auto_tool_choice=enable_auto_tool_choice,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
