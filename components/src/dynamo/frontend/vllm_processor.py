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

from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from dynamo.llm import (
    ModelCardInstanceId,
    ModelDeploymentCard,
    PythonAsyncEngine,
    fetch_llm,
)
from dynamo.runtime import Client, DistributedRuntime

logger = logging.getLogger(__name__)


def random_uuid() -> str:
    MASK_64_BITS = (1 << 64) - 1
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        output_processor: OutputProcessor,
        router: Client,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.router = router

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(self, request):
        """TODO: document"""

        # ** VllmProcessor.generator called: {'messages': [{'role': 'user', 'content': 'What is the capital of Tuvalu?'}], 'model': '/home/grahamk/llms/Qwen3-0.6B', 'max_completion_tokens': 1000, 'stream': False}
        print(f"** VllmProcessor.generator request: {request}")

        # tokenizer is CachedQwen2TokenizerFast, subclass of PreTrainedTokenizerBase (part of `transformers` library, not vllm)
        templated = self.tokenizer.apply_chat_template(
            conversation=request["messages"],
            tokenize=False,
        )

        print(f"*** Templated: {templated}")

        if "max_completion_tokens" in request:
            max_tokens = request["max_completion_tokens"]
        elif "max_tokens" in request:
            max_tokens = request["max_tokens"]
        else:
            max_tokens = 8192

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        for k, v in request.items():
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

        request_id = random_uuid()
        # This calls update_from_generation_config and update_from_tokenizer on SamplingParams
        vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
            request_id,
            templated,
            sampling_params,
            # arrival_time: float | None = None,
            # lora_request: LoRARequest | None = None,
            # tokenization_kwargs: dict[str, Any] | None = None,
            # trace_headers: Mapping[str, str] | None = None,
            # priority: int = 0,
            # data_parallel_rank: int | None = None,
        )

        # Processed: EngineCoreRequest(request_id='a2b76a85cd65e151', prompt_token_ids=[3838, 374, 279, 6722, 315, 28649, 25510, 30], mm_features=None, sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None), pooling_params=None, eos_token_id=151645, arrival_time=1769036937.9417946, lora_request=None, cache_salt=None, data_parallel_rank=None, prompt_embeds=None, client_index=0, current_wave=0, priority=0, trace_headers=None)
        print(f"Processed: {vllm_preproc}")

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
            "token_ids": vllm_preproc.prompt_token_ids,
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
        dynamo_stream = await self.router.random(dynamo_preproc)

        # dynamo_response: Annotated
        async for dynamo_response in dynamo_stream:
            # Mock
            # Stream got: Annotated(data={'token_ids': [1714], 'tokens': [' method'], 'text': ' method', 'cum_log_probs': None, 'log_probs': None, 'top_logprobs': None, 'finish_reason': None, 'index': None}, event=None, comment=[], id=None)
            #
            # vllm
            # Stream got: Annotated(data={'token_ids': [7281]}, event=None, comment=[], id=None)
            print(f"Stream got: {dynamo_response}")

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

            print(f"RequestOutput: {vllm_out}")

            # Vec<ChatChoiceStream>
            choices = []
            for output in vllm_out.request_outputs[0].outputs:
                choices.append(
                    {
                        "index": output.index,
                        # ChatCompletionStreamResponseDelta
                        "delta": {"content": output.text, "role": "assistant"},
                        # TODO: These three likely need converting, it won't just work
                        "finish_reason": output.finish_reason,
                        "stop_reason": output.stop_reason,
                        "logprobs": output.logprobs,
                    }
                )
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
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime

    async def engine_factory(
        self, instance_id: ModelCardInstanceId, mdc: ModelDeploymentCard
    ) -> PythonAsyncEngine:
        """
        Called by Rust when a model is discovered.
        """
        logger.info(f"Engine_factory called with MDC: {mdc.to_json_str()}")
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

        (namespace_name, component_name, endpoint_name) = instance_id.triple()
        generate_endpoint = (
            self.runtime.namespace(namespace_name)
            .component(component_name)
            .endpoint(endpoint_name)
        )
        router = await generate_endpoint.client()
        gen = VllmProcessor(tokenizer, input_processor, output_processor, router)

        return PythonAsyncEngine(gen.generator, loop)
