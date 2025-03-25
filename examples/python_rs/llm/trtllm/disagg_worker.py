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
import copy
import os
import signal
from dataclasses import asdict
from typing import AsyncGenerator, Optional

import uvloop
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    DisaggregatedTypeConverter,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
    TRTLLMWorkerResponseOutput,
)
from common.utils import wait_for_workers
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
from tensorrt_llm.serve.openai_protocol import DisaggregatedParams

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import Client, DistributedRuntime, dynamo_endpoint, dynamo_worker

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
        trt_llm_engine_config: TensorrtLLMEngineConfig,
        disagg_config: DisaggServerConfig,
        instance_idx: int,
        sub_comm,
        prefill_client: Optional[Client] = None,
    ):
        self.disagg_config = disagg_config
        self.instance_idx = instance_idx
        self.server_config: CtxGenServerConfig = disagg_config.server_configs[
            instance_idx
        ]
        self.prefill_client = prefill_client
        trt_llm_engine_config.engine_config = update_args_from_disagg_config(
            trt_llm_engine_config.engine_config, self.server_config
        )

        # needed for disagg
        self._mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
        trt_llm_engine_config.engine_config.extra_args[
            "_mpi_session"
        ] = self._mpi_session

        super().__init__(trt_llm_engine_config)

    async def _get_remote_prefill_response(self, request):
        prefill_request = copy.deepcopy(request)
        prefill_request.sampling_params["max_tokens"] = 1
        prefill_request.disaggregated_params = DisaggregatedParams(
            request_type="context_only"
        )

        ctx_responses = [
            ctx_response
            async for ctx_response in await self.prefill_client.round_robin(
                prefill_request.model_dump_json()
            )
        ]
        if len(ctx_responses) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )
        logger.debug(
            f"[worker - {self.server_config.type}] received response from context server: {ctx_responses[0].data()}"
        )
        ctx_response_obj = TRTLLMWorkerResponse.model_validate_json(
            ctx_responses[0].data()
        )
        ctx_response_obj.outputs = [
            TRTLLMWorkerResponseOutput(**ctx_response_obj.outputs[0])
        ]
        assert ctx_response_obj.outputs[0].disaggregated_params is not None

        return ctx_response_obj

    @dynamo_endpoint(TRTLLMWorkerRequest, AsyncGenerator)
    async def generate(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if self._error_queue.qsize() > 0:
            error = self._error_queue.get()
            raise error

        logger.debug(
            f"[worker - {self.server_config.type}] Received request: {request}"
        )
        self._ongoing_request_count += 1

        try:
            worker_inputs = request.prompt
            disaggregated_params = None

            # TODO: add disagg router check here.
            if self.prefill_client is not None:
                ctx_response_obj = await self._get_remote_prefill_response(request)
                # Append token generated by prefill to worker inputs
                worker_inputs = (
                    ctx_response_obj.prompt_token_ids
                    + ctx_response_obj.outputs[0].token_ids
                )
                disaggregated_params = (
                    DisaggregatedTypeConverter.to_llm_disaggregated_params(
                        DisaggregatedParams(
                            **ctx_response_obj.outputs[0].disaggregated_params
                        )
                    )
                )
                disaggregated_params.request_type = "generation_only"

            logger.debug(
                f"[worker - {self.server_config.type}] Worker inputs: {worker_inputs}, disaggregated params: {disaggregated_params}"
            )
            async for response in self._llm_engine.generate_async(
                inputs=worker_inputs,
                sampling_params=request.to_sampling_params(),
                disaggregated_params=disaggregated_params,
                streaming=True,
            ):
                yield TRTLLMWorkerResponse(
                    request_id=request.id,
                    prompt=response.prompt,
                    prompt_token_ids=response.prompt_token_ids,
                    outputs=[asdict(response.outputs[0])],
                    finished=response.finished,
                ).model_dump_json(exclude_unset=True)

        except CppExecutorError:
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

        self._start_threads()
        self._ongoing_request_count -= 1

    @dynamo_endpoint(TRTLLMWorkerRequest, AsyncGenerator)
    async def generate_prefill(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if self._error_queue.qsize() > 0:
            error = self._error_queue.get()
            raise error

        logger.debug(
            f"[worker - {self.server_config.type}] Received remote prefill request: {request}"
        )
        self._ongoing_request_count += 1

        try:
            disaggregated_params = (
                DisaggregatedTypeConverter.to_llm_disaggregated_params(
                    request.disaggregated_params
                )
            )
            # Input tokens directly
            assert request.tokens is not None, "Remote prefill request must have tokens"
            async for response in self._llm_engine.generate_async(
                inputs=request.tokens.tokens,
                sampling_params=request.to_sampling_params(),
                disaggregated_params=disaggregated_params,
                streaming=True,
            ):
                response.outputs[
                    0
                ].disaggregated_params = DisaggregatedTypeConverter.to_oai_disaggregated_params(
                    response.outputs[0].disaggregated_params
                )
                yield TRTLLMWorkerResponse(
                    request_id=request.id,
                    prompt_token_ids=response.prompt_token_ids,
                    outputs=[asdict(response.outputs[0])],
                    finished=response.finished,
                ).model_dump_json(exclude_unset=True)

        except CppExecutorError:
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

        self._start_threads()
        self._ongoing_request_count -= 1


@dynamo_worker()
async def worker(
    runtime: DistributedRuntime,
    engine_config: LLMAPIConfig,
    disagg_config: DisaggServerConfig,
    instance_idx: int,
    sub_comm,
    args,
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    server_type = disagg_config.server_configs[instance_idx].type
    logger.info(f"Starting {server_type} server")

    namespace_str = "dynamo"
    component_str = "tensorrt-llm-prefill" if server_type == "ctx" else "tensorrt-llm"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    generate_endpoint = component.endpoint("generate")

    if server_type == "ctx" and args.publish_kv_cache_events:
        logger.warning(
            "KV routing is not supported for ctx server. "
            "Setting publish_kv_cache_events to False"
        )
        args.publish_kv_cache_events = False
        args.publish_stats = False

    # TODO: check if disagg router is enabled.
    if args.remote_prefill and server_type == "gen":
        prefill_client = (
            await runtime.namespace("dynamo")
            .component("tensorrt-llm-prefill")
            .endpoint("generate")
            .client()
        )
        # TODO read from disaggregated config
        await wait_for_workers(prefill_client, 1)
        logger.info(f"[worker - {server_type}] Prefill workers {prefill_client} ready")
    else:
        prefill_client = None

    trt_llm_engine_config = TensorrtLLMEngineConfig(
        namespace_str=namespace_str,
        component_str=component_str,
        engine_config=engine_config,
        publish_stats=args.publish_stats,
        publish_kv_cache_events=args.publish_kv_cache_events,
        kv_block_size=args.kv_block_size,
    )

    trt_llm_engine_config.worker_id = generate_endpoint.lease_id()

    if args.publish_stats:
        trt_llm_engine_config.kv_metrics_publisher = KvMetricsPublisher()

    engine = TensorrtLLMEngine(
        trt_llm_engine_config,
        disagg_config,
        instance_idx,
        sub_comm,
        prefill_client,
    )

    if server_type == "ctx":
        coros = [
            generate_endpoint.serve_endpoint(engine.generate_prefill),
        ]
    else:
        coros = [
            generate_endpoint.serve_endpoint(engine.generate),
        ]

    if args.publish_stats:
        coros.append(
            trt_llm_engine_config.kv_metrics_publisher.create_endpoint(component)
        )

    await asyncio.gather(*coros)


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
        asyncio.run(
            worker(
                engine_config,
                disagg_config,
                instance_idx,
                sub_comm,
                args,
            )
        )
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(f"rank{COMM_WORLD} should not have executor")
