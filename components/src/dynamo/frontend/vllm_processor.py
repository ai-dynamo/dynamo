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
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
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

        pre = await preprocess_chat_request(
            request,
            tokenizer=self.tokenizer,
            renderer=self.input_processor.renderer,
            tool_parser_class=self.tool_parser_class,
            exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
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

        # This calls update_from_generation_config and update_from_tokenizer on SamplingParams
        prompt_inputs = TokensPrompt(prompt_token_ids=tokens)
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
        }

        # Forward multimodal URLs so the backend handler can load the media.
        mm_data = extract_mm_urls(request.get("messages") or [])
        if mm_data:
            dynamo_preproc["multi_modal_data"] = mm_data

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
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.output_processor.add_request(vllm_preproc, None)

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

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
