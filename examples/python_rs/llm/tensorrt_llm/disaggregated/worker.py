# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import json
import os
import signal

import uvloop
from common.base_engine import BaseTensorrtLLMEngine
from common.disagg_processor import ChatProcessor, parse_chat_message_content
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.processor import merge_promises
from common.protocol import (
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
    DisaggregatedTypeConverter,
)
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import MpiCommSession
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    parse_disagg_config_file,
    split_world_comm,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionRequest

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("debug")


def update_args_from_disagg_config(
    engine_config: LLMAPIConfig, server_config: CtxGenServerConfig
):
    # Overwrite the LLM API config with the disaggregated config
    # Allows for different configs for context and generation servers
    engine_config.extra_args.update(**server_config.other_args)
    engine_config.update_sub_configs(server_config.other_args)
    return engine_config


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        engine_config: LLMAPIConfig,
        disagg_config: DisaggServerConfig,
        instance_idx: int,
        sub_comm,
    ):
        self.disagg_config = disagg_config
        self.instance_idx = instance_idx
        self.server_config: CtxGenServerConfig = disagg_config.server_configs[
            instance_idx
        ]
        engine_config = update_args_from_disagg_config(
            engine_config, self.server_config
        )

        # needed for disagg
        self._mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
        engine_config.extra_args["_mpi_session"] = self._mpi_session
        super().__init__(engine_config)

    @triton_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    async def generate_chat(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        logger.debug(f"Received request: {request}")
        chat_processor = ChatProcessor(self._model, self._tokenizer, request)

        self._ongoing_request_count += 1

        try:
            conversation = []
            for message in request.messages:
                conversation.extend(parse_chat_message_content(message))
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            prompt: str = self._tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()
            disaggregated_params = (
                DisaggregatedTypeConverter.to_llm_disaggregated_params(
                    request.disaggregated_params
                )
            )

            final_result = None
            async for result in self._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
                disaggregated_params=disaggregated_params,
            ):
                final_result = result
                logger.debug(f"Generated result: {result}")
                if self.server_config.type == "ctx":
                    disaggregated_response = chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=True,
                    )
                    disaggregated_response.disaggregated_params = (
                        DisaggregatedTypeConverter.to_oai_disaggregated_params(
                            result.outputs[0].disaggregated_params
                        )
                    )
                    yield disaggregated_response.model_dump_json()
                else:
                    yield chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=False,
                    ).model_dump_json(
                        exclude_unset=True, exclude={"disaggregated_params"}
                    )

            if request.stream_options and request.stream_options.include_usage:
                yield chat_processor.create_final_stream_response(
                    request.id,
                    final_result,
                ).model_dump_json(exclude_unset=True, exclude={"disaggregated_params"})

        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

        self._ongoing_request_count -= 1

    @triton_endpoint(CompletionRequest, DisaggCompletionStreamResponse)
    async def generate_completions(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        self._ongoing_request_count += 1
        logger.debug(f"[worker] Received completions request: {request}")

        if not isinstance(request.prompt, str):
            # Check if it's a list and contains integers
            if isinstance(request.prompt, list) and len(request.prompt) == 1:
                request.prompt = request.prompt[0]
            elif not isinstance(request.prompt, list) or not all(
                isinstance(x, int) for x in request.prompt
            ):
                raise ValueError(
                    "Disaggregated server currently only supports single string prompt or list of integers in request"
                )

        sampling_params = request.to_sampling_params()
        llm_disaggregated_params = (
            DisaggregatedTypeConverter.to_llm_disaggregated_params(
                request.disaggregated_params
            )
        )

        # only 1 prompt is supported for now
        promise = self._llm_engine.generate_async(
            request.prompt,
            sampling_params,
            streaming=request.stream,
            disaggregated_params=llm_disaggregated_params,
        )
        generator = merge_promises([promise])
        num_choices = 1 if request.n is None else request.n
        if request.stream:
            response_generator = self.completions_processor.create_completion_generator(
                request, generator, num_choices
            )
            async for response in response_generator:
                yield json.loads(response)
        else:
            raise RuntimeError("Non-streaming is not supported")

        self._ongoing_request_count -= 1


@triton_worker()
async def worker(
    runtime: DistributedRuntime,
    engine_config: LLMAPIConfig,
    disagg_config: DisaggServerConfig,
    instance_idx: int,
    sub_comm,
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    server_type = disagg_config.server_configs[instance_idx].type
    logger.info(f"Starting {server_type} server")

    component = runtime.namespace("triton-init").component(
        f"tensorrt-llm-{server_type}"
    )
    await component.create_service()

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")
    engine = TensorrtLLMEngine(engine_config, disagg_config, instance_idx, sub_comm)
    await asyncio.gather(
        completions_endpoint.serve_endpoint(engine.generate_completions),
        chat_endpoint.serve_endpoint(engine.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    if args.llmapi_disaggregated_config is None or not os.path.exists(
        args.llmapi_disaggregated_config
    ):
        raise ValueError(
            "llmapi_disaggregated_config file does not exist or not provided"
        )

    disagg_config: DisaggServerConfig = parse_disagg_config_file(
        args.llmapi_disaggregated_config
    )

    logger.info(f"Parsed disaggregated config: {disagg_config}")

    is_leader, instance_idx, sub_comm = split_world_comm(disagg_config.server_configs)
    os.environ["TRTLLM_USE_MPI_KVCACHE"] = "1"
    set_mpi_comm(sub_comm)

    logger.info(f"is_leader: {is_leader}, instance_idx: {instance_idx}")

    if is_leader:
        asyncio.run(worker(engine_config, disagg_config, instance_idx, sub_comm))
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(f"rank{COMM_WORLD} should not have executor")
