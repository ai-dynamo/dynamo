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
from typing import Optional
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import parse_tensorrt_llm_args, LLMAPIConfig
from common.protocol import (
    TRTLLMWorkerRequest,
)
from components.prefill_worker import TensorRTLLMPrefillWorker

from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.llmapi import MpiCommSession
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    parse_disagg_config_file,
    split_world_comm,
)
from tensorrt_llm.logger import logger

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service, depends
from dynamo.sdk.lib.config import ServiceConfig

logger.set_level("debug")

def update_args_from_disagg_config(
    engine_config: LLMAPIConfig, server_config: CtxGenServerConfig
):
    # Overwrite the LLM API config with the disaggregated config
    # Allows for different configs for context and generation servers
    engine_config.extra_args.update(**server_config.other_args)
    engine_config.update_sub_configs(server_config.other_args)
    return engine_config

@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker(BaseTensorrtLLMEngine):
    prefill_worker = depends(TensorRTLLMPrefillWorker)

    def __init__(self):
        print("Initializing TensorRT-LLM Worker", flush=True)
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        self.args, self.engine_config = parse_tensorrt_llm_args(config_args)
        self.do_remote_prefill = self.args.remote_prefill
        self.disagg_config: Optional[DisaggServerConfig] = None

        if self.do_remote_prefill:
            if self.args.llmapi_disaggregated_config is None or not os.path.exists(
                self.args.llmapi_disaggregated_config
            ):  
                raise ValueError(
                    "llmapi_disaggregated_config file does not exist or not provided"
                )
            self.disagg_config: DisaggServerConfig = parse_disagg_config_file(
                self.args.llmapi_disaggregated_config
            )
            print(f"Parsed disaggregated config: {self.disagg_config}", flush=True)

        if self.disagg_config is not None:
            is_leader, instance_idx, sub_comm = split_world_comm(self.disagg_config.server_configs)
            os.environ["TRTLLM_USE_MPI_KVCACHE"] = "1"
            set_mpi_comm(sub_comm)

            self.instance_idx = instance_idx
            self.server_config: CtxGenServerConfig = self.disagg_config.server_configs[
                self.instance_idx
            ]
            self.engine_config = update_args_from_disagg_config(
                self.engine_config, self.server_config
            )

            if not is_leader:
                with MPICommExecutor(sub_comm) as executor:
                    if executor is not None:
                        raise RuntimeError(f"rank{COMM_WORLD} should not have executor")

            # needed for disagg
            self._mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
            self.engine_config.extra_args[
                "_mpi_session"
            ] = self._mpi_session

        if self.args.router == "kv":
            if self.do_remote_prefill:
                raise RuntimeError("KV router not supported for CTX worker in disaggregated mode")
            else:
                publish_stats = True
                publish_events = True
        else:
            publish_stats = False
            publish_events = False

        trt_llm_engine_config = TensorrtLLMEngineConfig(
            namespace_str="dynamo",
            component_str=class_name,
            engine_config=self.engine_config,
            publish_stats=publish_stats,
            publish_kv_cache_events=publish_events,
            kv_block_size=self.args.block_size,
        )

        if publish_stats:
            trt_llm_engine_config.kv_metrics_publisher = KvMetricsPublisher()

        trt_llm_engine_config.worker_id = dynamo_context["endpoints"][0].lease_id()
        logger.info(f"Generate endpoint ID: {trt_llm_engine_config.worker_id}")

        self.trtllm_engine_args = trt_llm_engine_config

    @async_on_start
    async def async_init(self):
        super().__init__(self.trtllm_engine_args)
        if self.do_remote_prefill:
            runtime = dynamo_context["runtime"]
            comp_ns, comp_name = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
            self.prefill_client = (
                await runtime.namespace(comp_ns)
                    .component(comp_name)
                    .endpoint("generate")
                    .client()
            )
            while len(self.prefill_client.endpoint_ids()) < self.min_workers:
                print(
                    f"Waiting for prefill workers to be ready.\n"
                    f" Current: {len(self.prefill_client.endpoint_ids())},"
                    f" Required: {self.min_workers}"
                )
            await asyncio.sleep(2)
        print("TensorRT-LLM Worker initialized")

    @dynamo_endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
