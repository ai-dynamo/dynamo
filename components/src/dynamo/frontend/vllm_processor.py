#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import logging
import os
import time
import uuid
from argparse import Namespace

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from dynamo.llm import (
    KvPushRouter,
    ModelCardInstanceId,
    ModelDeploymentCard,
    PythonAsyncEngine,
    RouterConfig,
    RouterMode,
    fetch_llm,
)
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


def random_uuid() -> str:
    MASK_64_BITS = (1 << 64) - 1
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        router,  # Client or KvPushRouter
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.router = router
        self.is_kv_router = isinstance(router, KvPushRouter)
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(self, request):
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a worker using the router.
        """

        # ** VllmProcessor.generator called: {'messages': [{'role': 'user', 'content': 'What is the capital of Tuvalu?'}], 'model': '/home/grahamk/llms/Qwen3-0.6B', 'max_completion_tokens': 1000, 'stream': False}

        # There seem to be two incompatible versions of apply_chat_template, depending on the model
        try:
            # tokenizer is a subclass of PreTrainedTokenizerBase (part of `transformers` library, not vllm)
            # This is not the type the source code declares.
            tokens = self.tokenizer.apply_chat_template(
                conversation=request["messages"],
                tools=request.get("tools", None),
                tokenize=True,
                add_generation_prompt=request.get("add_generation_prompt", True),
            )
        except TypeError:
            # tokenizer is an impl of TokenizerLike. This is the declared type.

            # For "role":"user" messages, mistral-common only allows 'role' and 'content', nothing else or pydantic validation breaks.
            # This deletes the optional 'name' field.
            filtered_messages = [
                {k: v for k, v in d.items() if k != "name"} for d in request["messages"]
            ]
            tokens = self.tokenizer.apply_chat_template(
                messages=filtered_messages,
                tools=request.get("tools", None),
                tokenize=True,
                add_generation_prompt=request.get("add_generation_prompt", True),
            )

        if "max_completion_tokens" in request:
            max_tokens = request["max_completion_tokens"]
        elif "max_tokens" in request:
            max_tokens = request["max_tokens"]
        else:
            # Match what Rust does
            max_tokens = 8192

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        # Only affects eos_token_id
        sampling_params.update_from_generation_config(
            self.input_processor.generation_config_fields, self.tokenizer.eos_token_id
        )
        # generation_config.json
        for k, v in self.input_processor.generation_config_fields.items():
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)
        # User request
        for k, v in request.items():
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

        request_id = random_uuid()
        # This calls update_from_generation_config and update_from_tokenizer on SamplingParams
        vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
            request_id,
            TokensPrompt(prompt_token_ids=tokens),
            sampling_params,
            # arrival_time: float | None = None,
            # lora_request: LoRARequest | None = None,
            # tokenization_kwargs: dict[str, Any] | None = None,
            # trace_headers: Mapping[str, str] | None = None,
            # priority: int = 0,
            # data_parallel_rank: int | None = None,
        )
        # TODO: Copy this from our request if present
        # vllm does not set this in process_inputs, but requires it in add_request
        vllm_preproc.external_req_id = request_id

        # Processed: EngineCoreRequest(request_id='a2b76a85cd65e151', prompt_token_ids=[3838, 374, 279, 6722, 315, 28649, 25510, 30], mm_features=None, sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None), pooling_params=None, eos_token_id=151645, arrival_time=1769036937.9417946, lora_request=None, cache_salt=None, data_parallel_rank=None, prompt_embeds=None, client_index=0, current_wave=0, priority=0, trace_headers=None)

        self.output_processor.add_request(
            vllm_preproc,
            request["messages"][0]["content"],  # prompt
            # parent_req: ParentRequest | None = None,
            # request_index: int = 0,
            # queue: RequestOutputCollector | None = None,
        )

        # Convert to a Python object that has fields that match our PreprocessedRequest
        sp = vllm_preproc.sampling_params
        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            # protocols.common.StopConditions
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
            },
            # protocols.common.SamplingOptions
            "sampling_options": {
                # Is there a better way than typing it out like this?
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
            "eos_token_ids": [vllm_preproc.eos_token_id],
            "annotations": [],
            # "prompt_embeds": vllm_preproc.prompt_embeds,
        }

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

        # tool parser is stateful, each request needs a fresh one
        tool_parser = (
            self.tool_parser_class(self.tokenizer) if self.tool_parser_class else None
        )
        previous_text = ""
        previous_token_ids = []
        in_progress_tool_call: DeltaToolCall | None = None

        # dynamo_response: Annotated
        async for dynamo_response in dynamo_stream:
            # dynamo_response looks like this for regular router:
            # Stream got: Annotated(data={'token_ids': [7281]}, event=None, comment=[], id=None)
            # For KV router is is only the inner map: {'token_ids': [7281]}

            if self.is_kv_router:
                output = dynamo_response
            else:
                output = dynamo_response.data()

            if output is None or "token_ids" not in output:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "token_ids": [],
                }
                break

            finish_reason = output.get("finish_reason")
            vllm_response = EngineCoreOutput(
                request_id=request_id,
                new_token_ids=output["token_ids"],
                finish_reason=finish_reason,
                # new_logprobs=new_logprobs,
                # new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                # pooling_output=pooler_output,
                # stop_reason=request.stop_reason,
                # events=request.take_events(),
                # kv_transfer_params=kv_transfer_params,
                # trace_headers=request.trace_headers,
                # num_cached_tokens=request.num_cached_tokens,
                # num_nans_in_logits=request.num_nans_in_logits,
            )

            # Let vllm handle all post-processing
            vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                [vllm_response]
            )

            # vllm
            # RequestOutput: OutputProcessorOutput(request_outputs=[RequestOutput(request_id=9dbe240d8de78db3, prompt='What is the capital of Tuvalu?', prompt_token_ids=[3838, 374, 279, 6722, 315, 28649, 25510, 30], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' The', token_ids=[576], cumulative_logprob=None, logprobs=None, finish_reason=None, stop_reason=None)], finished=False, metrics=RequestStateStats(num_generation_tokens=0, arrival_time=1769118902.2172132, queued_ts=0.0, scheduled_ts=0.0, first_token_ts=0.0, last_token_ts=0.0, first_token_latency=0.0, is_corrupted=False), lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})], reqs_to_abort=[])

            # Vec<ChatChoiceStream>
            choices = []
            for output in vllm_out.request_outputs[0].outputs:
                delta_text = output.text
                current_text = previous_text + delta_text
                current_token_ids = previous_token_ids + output.token_ids

                if tool_parser:
                    delta_message: DeltaMessage | None = (
                        tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=output.token_ids,
                            request=ChatCompletionRequest(**request),
                        )
                    )
                else:
                    delta_message = DeltaMessage(content=delta_text)

                if delta_message is None:
                    # tokens being held back, might be tool call marker
                    pass

                elif delta_message.tool_calls:
                    if delta_message.tool_calls[0].function:
                        # delta_message.tool_calls = DeltaToolCall objects to stream
                        if in_progress_tool_call is None:
                            in_progress_tool_call = delta_message.tool_calls[0]
                            if not in_progress_tool_call.function.arguments:
                                # Ensure + works later
                                in_progress_tool_call.function.arguments = ""
                        else:
                            delta_arg = delta_message.tool_calls[0]
                            in_progress_tool_call.function.arguments += (
                                delta_arg.function.arguments
                            )
                    else:
                        logger.warn(
                            f"Unknown tool call type in: {delta_message.tool_calls[0]}. Only type 'function' supported."
                        )

                elif delta_message.content:
                    # Stream content to user
                    # ChatCompletionStreamResponseDelta
                    delta = {"content": delta_message.content, "role": "assistant"}
                    if in_progress_tool_call:
                        delta["tool_calls"] = [
                            in_progress_tool_call.model_dump(exclude_none=True)
                        ]
                        in_progress_tool_call = None
                    choices.append(
                        {
                            "index": output.index,
                            "delta": delta,
                            # TODO: These three likely need converting
                            "finish_reason": output.finish_reason,
                            "stop_reason": output.stop_reason,
                            "logprobs": output.logprobs,
                        }
                    )

                elif in_progress_tool_call:
                    # Empty content of any kind. Send any outstanding tool calls.
                    choices.append(
                        {
                            "index": output.index,
                            # ChatCompletionStreamResponseDelta
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    in_progress_tool_call.model_dump(exclude_none=True)
                                ],
                            },
                            # TODO: These three likely need converting
                            "finish_reason": output.finish_reason,
                            "stop_reason": output.stop_reason,
                            "logprobs": output.logprobs,
                        }
                    )
                    in_progress_tool_call = None

                previous_text = current_text
                previous_token_ids = current_token_ids

            if choices:
                # dynamo_out: NvCreateChatCompletionStreamResponse
                dynamo_out = {
                    "id": request_id,
                    "choices": choices,
                    "created": int(time.time()),
                    "model": request["model"],
                    "object": "chat.completion.chunk",
                    # usage (from output.metrics maybe)
                }
                # Rust handles Server Sent Events back to user
                yield dynamo_out


class EngineFactory:
    def __init__(
        self,
        runtime: DistributedRuntime,
        router_config: RouterConfig,
        flags: Namespace,
    ):
        self.runtime = runtime
        self.router_config = router_config
        self.flags = flags

    async def engine_factory(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
    ) -> PythonAsyncEngine:
        """
        Called by Rust when a model is discovered.
        """
        logger.debug(f"Engine_factory called with MDC: {mdc.to_json_str()}")
        loop = asyncio.get_running_loop()

        source_path = mdc.source_path()
        if not os.path.exists(source_path):
            await fetch_llm(source_path)

        model_config = ModelConfig(
            model=source_path,
        )
        tokenizer = cached_tokenizer_from_config(model_config)
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format="dummy"),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        input_processor = InputProcessor(vllm_config, tokenizer)
        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=1,
        )

        tool_parser_name = (
            self.flags.tool_call_parser or mdc.runtime_config()["tool_call_parser"]
        )
        if tool_parser_name:
            tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
        else:
            tool_parser_class = None

        (namespace_name, component_name, endpoint_name) = instance_id.triple()
        generate_endpoint = (
            self.runtime.namespace(namespace_name)
            .component(component_name)
            .endpoint(endpoint_name)
        )

        if self.router_config.router_mode == RouterMode.KV:
            router = KvPushRouter(
                endpoint=generate_endpoint,
                block_size=self.flags.kv_cache_block_size or 16,
                kv_router_config=self.router_config.kv_router_config,
            )
        else:
            router = await generate_endpoint.client2(self.router_config.router_mode)

        gen = VllmProcessor(
            tokenizer, input_processor, router, output_processor, tool_parser_class
        )

        return PythonAsyncEngine(gen.generator, loop)
