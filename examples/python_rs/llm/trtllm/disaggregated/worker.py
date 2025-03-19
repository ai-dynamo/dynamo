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
import os
import signal
from typing import AsyncGenerator

import uvloop
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest
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

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

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
    ):
        self.disagg_config = disagg_config
        self.instance_idx = instance_idx
        self.server_config: CtxGenServerConfig = disagg_config.server_configs[
            instance_idx
        ]
        trt_llm_engine_config.engine_config = update_args_from_disagg_config(
            trt_llm_engine_config.engine_config, self.server_config
        )

        # needed for disagg
        self._mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
        trt_llm_engine_config.engine_config.extra_args[
            "_mpi_session"
        ] = self._mpi_session

        super().__init__(trt_llm_engine_config)

    @dynamo_endpoint(TRTLLMWorkerRequest, AsyncGenerator)
    async def generate(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if self._error_queue.qsize() > 0:
            error = self._error_queue.get()
            raise error

        logger.debug(f"[worker] Received request: {request}")
        self._ongoing_request_count += 1

        try:
            yield self._llm_engine.generate_async()

        except CppExecutorError:
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

        self._start_threads()
        self._ongoing_request_count -= 1

    # @dynamo_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    # async def generate_chat(self, request):
    #     if self._llm_engine is None:
    #         raise RuntimeError("Engine not initialized")

    #     if self._error_queue.qsize() > 0:
    #         error = self._error_queue.get()
    #         raise error

    #     logger.debug(f"Received request: {request}")

    #     self._ongoing_request_count += 1

    #     try:
    #         async for response in chat_generator(self, request, is_disaggregated=True):
    #             yield response

    #     except CppExecutorError:
    #         signal.raise_signal(signal.SIGINT)
    #     except Exception as e:
    #         raise RuntimeError("Failed to generate: " + str(e))

    #     # Start the publishing threads with first request submission
    #     self._start_threads()
    #     self._ongoing_request_count -= 1

    # @dynamo_endpoint(AdaptedCompletionRequest, DisaggCompletionStreamResponse)
    # async def generate_completions(self, request):
    #     logger.debug(f"[worker] worker_id: {self._worker_id} received request")
    #     if self._llm_engine is None:
    #         raise RuntimeError("Engine not initialized")

    #     if self._error_queue.qsize() > 0:
    #         error = self._error_queue.get()
    #         raise error

    #     self._ongoing_request_count += 1
    #     logger.debug(f"[worker] Received completions request: {request}")

    #     try:
    #         async for response in completion_generator(
    #             self, request, is_disaggregated=True
    #         ):
    #             yield response
    #     except CppExecutorError:
    #         signal.raise_signal(signal.SIGINT)
    #     except Exception as e:
    #         raise RuntimeError("Failed to generate: " + str(e))

    #     # Start the publishing threads with first request submission
    #     self._start_threads()
    #     self._ongoing_request_count -= 1


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
    component_str = f"tensorrt-llm-{server_type}"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    generate_endpoint = component.endpoint("generate")

    if server_type == "gen" and args.publish_kv_cache_events:
        logger.warning("KV routing is not supported for gen server")

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
    )

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
