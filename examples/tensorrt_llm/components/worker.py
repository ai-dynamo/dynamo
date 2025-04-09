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
import logging
import os
from typing import Optional

from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest
from common.utils import ServerType
from components.prefill_worker import TensorRTLLMPrefillWorker
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    parse_disagg_config_file,
)

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        logger.info("Initializing TensorRT-LLM Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        self.args, self.engine_config = parse_tensorrt_llm_args(config_args)
        self.do_remote_prefill = self.args.remote_prefill
        self.disagg_config: Optional[DisaggServerConfig] = None
        self.min_workers = 0

        if self.do_remote_prefill:
            self.min_workers = self.args.min_workers
            if self.args.llmapi_disaggregated_config is None or not os.path.exists(
                self.args.llmapi_disaggregated_config
            ):
                raise ValueError(
                    "llmapi_disaggregated_config file does not exist or not provided"
                )
            self.disagg_config: DisaggServerConfig = parse_disagg_config_file(
                self.args.llmapi_disaggregated_config
            )
            self.server_config: CtxGenServerConfig = None

            for config in self.disagg_config.server_configs:
                # Select the first context server config
                if config.type == "gen":
                    self.server_config = config
                    break

            if self.server_config is None:
                raise ValueError(
                    "No generation server config found. Please check the disaggregated config file."
                )

            self.engine_config = update_args_from_disagg_config(
                self.engine_config, self.server_config
            )

        logger.info(f"Engine config: {self.engine_config}")

        if self.args.router == "kv":
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
        super().__init__(self.trtllm_engine_args, ServerType.GEN)
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
                logger.info(
                    f"Waiting for prefill workers to be ready.\n"
                    f" Current: {len(self.prefill_client.endpoint_ids())},"
                    f" Required: {self.min_workers}"
                )
                await asyncio.sleep(2)
        logger.info("TensorRT-LLM Worker initialized")

    @dynamo_endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
