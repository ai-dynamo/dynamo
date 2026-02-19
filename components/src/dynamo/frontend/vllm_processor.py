#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import logging
import multiprocessing
import os
import queue
import time
import uuid
from argparse import Namespace
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as _futures_wait
from typing import Any

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.inputs.data import TokensPrompt
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
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
from dynamo.runtime import DistributedRuntime

from .prepost import (
    StreamingPostProcessor,
    preprocess_chat_request,
    preprocess_chat_request_sync,
)

logger = logging.getLogger(__name__)


_MASK_64_BITS = (1 << 64) - 1
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "eos": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
    "cancelled": FinishReason.ABORT,
    "content_filter": FinishReason.STOP,
}


def random_uuid() -> str:
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"  # 16 hex chars


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


# --- Worker process globals (initialized once per process by _init_worker) ---
_w_input_processor: InputProcessor | None = None
_w_tokenizer: Any = None
_w_tool_parser_class: type[ToolParser] | None = None
_w_reasoning_parser_class: type[ReasoningParser] | None = None
_WORKER_STOP_SENTINEL = "__dynamo_worker_stop__"


def _init_worker(
    model_path: str,
    tokenizer_mode: str,
    config_format: str,
    load_format: str,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> None:
    """Initialize a worker process with its own VllmConfig and InputProcessor."""
    global _w_input_processor, _w_tokenizer, _w_tool_parser_class, _w_reasoning_parser_class

    model_config = ModelConfig(
        model=model_path,
        tokenizer_mode=tokenizer_mode,
        config_format=config_format,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format=load_format),
        cache_config=CacheConfig(),
    )

    _w_input_processor = InputProcessor(vllm_config)
    _w_tokenizer = _w_input_processor.get_tokenizer()

    if tool_parser_name:
        _w_tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
    else:
        _w_tool_parser_class = None

    if reasoning_parser_name:
        _w_reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(
            reasoning_parser_name
        )
    else:
        _w_reasoning_parser_class = None


def _worker_warmup() -> bool:
    """Dummy task to ensure worker process is fully initialized."""
    return True


def _request_handler(
    input_q: multiprocessing.Queue,
    output_q: multiprocessing.Queue,
    request: dict[str, Any],
    request_id: str,
    model_name: str,
) -> None:
    """Full request lifecycle in a worker process.

    Phase A: Preprocessing (pydantic validation, template rendering, tokenization)
    Phase B: Setup output processing
    Phase C: Token processing loop (detokenization, reasoning/tool parsing)
    Phase D: Cleanup
    """

    output_processor = None
    vllm_request_id = None

    try:
        # Phase A: Preprocessing
        pre = preprocess_chat_request_sync(
            request,
            tokenizer=_w_tokenizer,
            renderer=_w_input_processor.renderer,
            tool_parser_class=_w_tool_parser_class,
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
            max_tokens = None

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        for k, v in _w_input_processor.generation_config_fields.items():
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

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

        vllm_preproc: EngineCoreRequest = _w_input_processor.process_inputs(
            request_id,
            prompt_inputs,
            sampling_params,
        )
        InputProcessor.assign_request_id(vllm_preproc)

        sp = vllm_preproc.sampling_params
        if sp.n != 1:
            output_q.put(
                (
                    "error",
                    {
                        "error": {
                            "message": (
                                f"Unsupported value: 'n={sp.n}'. "
                                "This endpoint currently supports only n=1."
                            ),
                            "type": "invalid_request_error",
                            "param": "n",
                            "code": "unsupported_value",
                        }
                    },
                )
            )
            return

        dynamo_preproc = {
            "model": model_name,
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
            "eos_token_ids": [vllm_preproc.eos_token_id]
            if vllm_preproc.eos_token_id is not None
            else [],
            "annotations": [],
        }

        output_q.put(("preprocess", dynamo_preproc, tokens))

        # Phase B: Setup output processing
        output_processor = OutputProcessor(
            _w_tokenizer,
            log_stats=False,
            stream_interval=1,
        )
        vllm_request_id = vllm_preproc.request_id
        output_processor.add_request(vllm_preproc, None)

        post = StreamingPostProcessor(
            tokenizer=_w_tokenizer,
            request_for_sampling=request_for_sampling,
            sampling_params=sampling_params,
            prompt_token_ids=tokens,
            tool_parser=tool_parser,
            reasoning_parser_class=_w_reasoning_parser_class,
            chat_template_kwargs=chat_template_kwargs,
        )

        # Phase C: Token processing loop
        while True:
            engine_response = input_q.get()

            if engine_response == _WORKER_STOP_SENTINEL:
                break

            if (
                engine_response is None
                or not isinstance(engine_response, dict)
                or "token_ids" not in engine_response
            ):
                output_q.put(
                    (
                        "error",
                        {
                            "id": request_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "error",
                                }
                            ],
                            "created": int(time.time()),
                            "model": model_name,
                            "object": "chat.completion.chunk",
                        },
                    )
                )
                break

            raw_finish_reason = engine_response.get("finish_reason")
            finish_reason = map_finish_reason(raw_finish_reason)
            stop_reason = engine_response.get("stop_reason")

            vllm_response = EngineCoreOutput(
                request_id=vllm_request_id,
                new_token_ids=engine_response["token_ids"],
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )

            vllm_out = output_processor.process_outputs([vllm_response])

            if vllm_out.reqs_to_abort:
                pass

            choices = []
            if not vllm_out.request_outputs:
                output_q.put(("skip",))
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
                    "model": model_name,
                    "object": "chat.completion.chunk",
                }
                if usage := engine_response.get("completion_usage"):
                    dynamo_out["usage"] = usage

                output_q.put(("chunk", dynamo_out))
            else:
                output_q.put(("skip",))

    except Exception as e:
        try:
            output_q.put(
                (
                    "error",
                    {
                        "error": {
                            "message": f"Worker error: {e}",
                            "type": "internal_error",
                        }
                    },
                )
            )
        except Exception:
            pass

    finally:
        # Phase D: Cleanup
        if output_processor is not None and vllm_request_id is not None:
            if vllm_request_id in output_processor.request_states:
                output_processor.abort_requests([vllm_request_id], internal=True)


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        router,  # Client or KvRouter
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        debug_perf: bool = False,
        preprocess_pool: ProcessPoolExecutor | None = None,
        preprocess_workers: int = 0,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self.debug_perf = debug_perf
        self.preprocess_pool = preprocess_pool
        # Manager provides proxy Queue objects that are picklable and work
        # across processes (unlike raw multiprocessing.Queue which relies on
        # shared pipe FDs that don't transfer to already-forked pool workers).
        self._mp_manager = multiprocessing.Manager() if preprocess_pool else None
        # Two dedicated thread pools for blocking queue reads, sized from
        # preprocess_workers.  Preprocessing waits and streaming waits MUST
        # use separate pools: under high concurrency every waiting request
        # parks a thread on output_q.get for its preprocessing result.  If
        # those threads come from the same pool as streaming reads, active
        # requests that finished preprocessing can never acquire a thread to
        # read token results → circular deadlock.
        if preprocess_pool is not None:
            from concurrent.futures import ThreadPoolExecutor

            # Preprocessing pool: only needs as many threads as workers, since
            # only that many requests can be actively preprocessing.  Excess
            # requests queue cheaply inside the ThreadPoolExecutor (no thread
            # consumed until scheduled).
            self._preproc_wait_executor = ThreadPoolExecutor(
                max_workers=max(4, preprocess_workers)
            )
            # Streaming pool: also bounded by workers — each worker handles
            # one request's token loop at a time.
            self._stream_wait_executor = ThreadPoolExecutor(
                max_workers=max(4, preprocess_workers)
            )
        else:
            self._preproc_wait_executor = None
            self._stream_wait_executor = None

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a worker using the router.
        """

        # ** VllmProcessor.generator called: {'messages': [{'role': 'user', 'content': 'What is the capital of Tuvalu?'}], 'model': '/home/grahamk/llms/Qwen3-0.6B', 'max_completion_tokens': 1000, 'stream': False}

        if self.debug_perf:
            from .perf_instrumentation import enter_generator, exit_generator

            active = enter_generator()
            t_start = time.monotonic()
            logger.info("[perf] generator enter: active_requests=%d", active)

        try:
            if self.preprocess_pool is None:
                # Single process
                async for item in self._generator_inner(request):
                    yield item
            else:
                # Multi process
                async for item in self._generator_inner_pool(request):
                    yield item
        finally:
            if self.debug_perf:
                active = exit_generator()
                elapsed_ms = (time.monotonic() - t_start) * 1000.0
                logger.info(
                    "[perf] generator exit: total=%.2fms active_requests=%d",
                    elapsed_ms,
                    active,
                )

    async def _generator_inner(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        if self.debug_perf:
            t0 = time.monotonic()

        pre = await preprocess_chat_request(
            request,
            tokenizer=self.tokenizer,
            renderer=self.input_processor.renderer,
            tool_parser_class=self.tool_parser_class,
        )

        if self.debug_perf:
            t1 = time.monotonic()
            logger.info(
                "[perf] preprocess_chat_request: %.2fms (request=%s)",
                (t1 - t0) * 1000.0,
                request_id,
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
        for k, v in self.input_processor.generation_config_fields.items():
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

        if self.debug_perf:
            t2 = time.monotonic()

        vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
            request_id,
            prompt_inputs,
            sampling_params,
            # arrival_time: float | None = None,
            # lora_request: LoRARequest | None = None,
            # tokenization_kwargs: dict[str, Any] | None = None,
            # trace_headers: Mapping[str, str] | None = None,
            # priority: int = 0,
            # data_parallel_rank: int | None = None,
        )

        if self.debug_perf:
            t3 = time.monotonic()
            logger.info(
                "[perf] input_processor.process_inputs: %.2fms (request=%s tokens=%d)",
                (t3 - t2) * 1000.0,
                request_id,
                len(tokens),
            )

        InputProcessor.assign_request_id(vllm_preproc)

        # Processed: EngineCoreRequest(request_id='a2b76a85cd65e151', prompt_token_ids=[3838, 374, 279, 6722, 315, 28649, 25510, 30], mm_features=None, sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None), pooling_params=None, eos_token_id=151645, arrival_time=1769036937.9417946, lora_request=None, cache_salt=None, data_parallel_rank=None, prompt_embeds=None, client_index=0, current_wave=0, priority=0, trace_headers=None)

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

        prompt = None
        self.output_processor.add_request(
            vllm_preproc,
            prompt,
            # parent_req: ParentRequest | None = None,
            # request_index: int = 0,
            # queue: RequestOutputCollector | None = None,
        )
        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            # protocols.common.StopConditions
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "stop_token_ids": sp.stop_token_ids,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
            },
            # protocols.common.SamplingOptions
            # Is there a better way than typing it out like this?
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
            # protocols.common.OutputOptions
            "output_options": {
                "logprobs": sp.logprobs,
                "prompt_logprobs": sp.prompt_logprobs,
                "skip_special_tokens": sp.skip_special_tokens,
            },
            "eos_token_ids": [vllm_preproc.eos_token_id]
            if vllm_preproc.eos_token_id is not None
            else [],
            "annotations": [],
            # "prompt_embeds": vllm_preproc.prompt_embeds,
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

        # dynamo_response: Annotated
        token_count = 0
        output_proc_total_ms = 0.0
        post_proc_total_ms = 0.0

        try:
            # Dynamo Router. This goes to the backend, waits, gets the streaming response, returns it.
            # Stream is AsyncResponseStream
            if self.is_kv_router:
                dynamo_stream = await self.router.generate(
                    token_ids=tokens,
                    model=dynamo_preproc["model"],
                    stop_conditions=dynamo_preproc["stop_conditions"],
                    sampling_options=dynamo_preproc["sampling_options"],
                    output_options=dynamo_preproc["output_options"],
                )
            else:
                # Round robin or random, depending on cmd line flag
                dynamo_stream = await self.router.generate(dynamo_preproc)

            async for dynamo_response in dynamo_stream:
                # dynamo_response looks like this for regular router:
                # Stream got: Annotated(data={'token_ids': [7281]}, event=None, comment=[], id=None)
                # For KV router is is only the inner map: {'token_ids': [7281]}

                if self.is_kv_router:
                    engine_response = dynamo_response
                else:
                    engine_response = dynamo_response.data()

                # engine_response:
                # Normal: {'token_ids': [151658]}
                # Last: {'token_ids': [151645], 'finish_reason': 'stop', 'completion_usage': {'prompt_tokens': 190, 'completion_tokens': 168, 'total_tokens': 358, 'prompt_tokens_details': {'cached_tokens': 176}}}

                if engine_response is None or "token_ids" not in engine_response:
                    logger.error("No outputs from engine for request %s", request_id)
                    yield {
                        "id": request_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "error",
                            }
                        ],
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
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
                    # new_logprobs=new_logprobs,
                    # new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    # pooling_output=pooler_output,
                    # events=request.take_events(),
                    # kv_transfer_params=kv_transfer_params,
                    # trace_headers=request.trace_headers,
                    # num_cached_tokens=request.num_cached_tokens,
                    # num_nans_in_logits=request.num_nans_in_logits,
                )

                if self.debug_perf:
                    t_op0 = time.monotonic()

                # Let vllm handle all post-processing
                vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                    [vllm_response]
                )

                if self.debug_perf:
                    t_op1 = time.monotonic()
                    output_proc_total_ms += (t_op1 - t_op0) * 1000.0

                if vllm_out.reqs_to_abort:
                    # Router has no abort API; we cannot propagate aborts.
                    pass

                # Vec<ChatChoiceStream>
                choices = []
                if not vllm_out.request_outputs:
                    continue
                for output in vllm_out.request_outputs[0].outputs:
                    choice = post.process_output(output)
                    if choice:
                        choices.append(choice)

                if self.debug_perf:
                    t_op2 = time.monotonic()
                    post_proc_total_ms += (t_op2 - t_op1) * 1000.0
                    token_count += len(engine_response["token_ids"])

                if choices:
                    # dynamo_out: NvCreateChatCompletionStreamResponse
                    dynamo_out = {
                        "id": request_id,
                        "choices": choices,
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
                    }
                    if usage := engine_response.get("completion_usage"):
                        # The engine only includes this on the last response
                        dynamo_out["usage"] = usage

                    # Rust handles HTTP / Server Sent Events back to user
                    yield dynamo_out
        finally:
            if vllm_preproc.request_id in self.output_processor.request_states:
                self.output_processor.abort_requests(
                    [vllm_preproc.request_id], internal=True
                )
            if self.debug_perf and token_count > 0:
                logger.info(
                    "[perf] stream done: request=%s tokens=%d "
                    "output_processor_total=%.2fms (%.3fms/tok) "
                    "post_processor_total=%.2fms (%.3fms/tok)",
                    request_id,
                    token_count,
                    output_proc_total_ms,
                    output_proc_total_ms / token_count,
                    post_proc_total_ms,
                    post_proc_total_ms / token_count,
                )

    async def _generator_inner_pool(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process a request using the worker pool.

        Preprocessing and per-token output processing run in a worker process
        (own GIL). The main process only does router I/O and queue transfer.
        """
        request_id = random_uuid()
        loop = asyncio.get_running_loop()
        input_q = self._mp_manager.Queue()
        output_q = self._mp_manager.Queue()

        future = self.preprocess_pool.submit(
            _request_handler,
            input_q,
            output_q,
            request,
            request_id,
            request["model"],
        )

        try:
            # Wait for preprocessing result from worker in background thread (output_q is sync)
            try:
                msg = await loop.run_in_executor(
                    self._preproc_wait_executor, output_q.get, True, 120
                )
            except queue.Empty:
                logger.error(
                    "Worker preprocessing timed out for request %s", request_id
                )
                yield {
                    "error": {
                        "message": "Worker preprocessing timed out",
                        "type": "internal_error",
                    }
                }
                return

            if msg[0] == "error":
                yield msg[1]
                return

            _, dynamo_preproc, tokens = msg

            # Route via Rust router (stays in main process)
            if self.is_kv_router:
                dynamo_stream = await self.router.generate(
                    token_ids=tokens,
                    model=dynamo_preproc["model"],
                    stop_conditions=dynamo_preproc["stop_conditions"],
                    sampling_options=dynamo_preproc["sampling_options"],
                    output_options=dynamo_preproc["output_options"],
                )
            else:
                dynamo_stream = await self.router.generate(dynamo_preproc)

            # Stream tokens to worker, yield processed results back
            async for dynamo_response in dynamo_stream:
                if self.is_kv_router:
                    engine_response = dynamo_response
                else:
                    engine_response = dynamo_response.data()

                input_q.put(engine_response)

                try:
                    msg = await loop.run_in_executor(
                        self._stream_wait_executor, output_q.get, True, 60
                    )
                except queue.Empty:
                    logger.error(
                        "Worker token processing timed out for request %s",
                        request_id,
                    )
                    yield {
                        "error": {
                            "message": "Worker token processing timed out",
                            "type": "internal_error",
                        }
                    }
                    break

                if msg[0] == "chunk":
                    yield msg[1]
                elif msg[0] == "error":
                    yield msg[1]
                    break
                # "skip" → continue

        finally:
            input_q.put(_WORKER_STOP_SENTINEL)  # Signal worker to clean up
            try:
                future.result(timeout=5)
            except Exception:
                pass


class EngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        config: FrontendConfig,
        flags: Namespace,
        debug_perf: bool = False,
    ):
        self.runtime = runtime
        self.router_config = router_config
        self.config = config
        self.flags = flags
        self.debug_perf = debug_perf

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
            stream_interval=1,
        )

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

        (namespace_name, component_name, endpoint_name) = instance_id.triple()
        generate_endpoint = (
            self.runtime.namespace(namespace_name)
            .component(component_name)
            .endpoint(endpoint_name)
        )

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
        preprocess_workers = getattr(self.config, "preprocess_workers", 0)
        if preprocess_workers > 0:
            logger.info(
                "Creating preprocess worker pool with %d workers for model %s",
                preprocess_workers,
                source_path,
            )
            preprocess_pool = ProcessPoolExecutor(
                max_workers=preprocess_workers,
                initializer=_init_worker,
                initargs=(
                    source_path,
                    tokenizer_mode,
                    config_format,
                    load_format,
                    tool_parser_name,
                    reasoning_parser_name,
                ),
            )
            # Warm up all workers to ensure initialization completes
            futures = [
                preprocess_pool.submit(_worker_warmup)
                for _ in range(preprocess_workers)
            ]
            done, not_done = _futures_wait(futures, timeout=120)
            if not_done:
                for f in not_done:
                    f.cancel()
                preprocess_pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    "Timed out waiting for preprocess worker pool warmup"
                )
            try:
                for f in done:
                    f.result()  # Raises if initializer failed
            except Exception:
                preprocess_pool.shutdown(wait=False, cancel_futures=True)
                raise
            logger.info("Preprocess worker pool ready (%d workers)", preprocess_workers)

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            router,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            debug_perf=self.debug_perf,
            preprocess_pool=preprocess_pool,
            preprocess_workers=preprocess_workers,
        )

        return PythonAsyncEngine(gen.generator, loop)
