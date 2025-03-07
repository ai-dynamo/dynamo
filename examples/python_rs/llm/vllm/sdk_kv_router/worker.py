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

import bentoml

with bentoml.importing():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.logger import logger as vllm_logger
    from vllm.sampling_params import RequestOutputKind
    from common.base_engine import BaseVllmEngine
    from common.protocol import MyRequestOutput, vLLMGenerateRequest
    from vllm.engine.multiprocessing.client import MQLLMEngineClient

from dynemo.llm import KvMetricsPublisher
from dynemo.sdk import (
    async_onstart,
    dynemo_context,
    dynemo_endpoint,
    server_context,
    service,
)

lease_id = None

## TODO: metrics_publisher.create_endpoint(worker_component),


@service(
    dynemo={
        "enabled": True,
        "namespace": "dynemo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmEngine(BaseVllmEngine):
    """
    vLLM Inference Engine
    """

    def __init__(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.engine_args = AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            block_size=64,
            max_model_len=16384,
        )
        VLLM_WORKER_ID = dynemo_context["endpoints"][0].lease_id()
        os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
        os.environ["VLLM_KV_NAMESPACE"] = "dynemo"
        os.environ["VLLM_KV_COMPONENT"] = "vllm"
        vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{server_context.worker_index - 1}"
        self.metrics_publisher = KvMetricsPublisher()
        self.engine_client: Optional[MQLLMEngineClient] = None
        super().__init__(self.engine_args)

    async def create_metrics_publisher_endpoint(self):
        component = dynemo_context["component"]
        await self.metrics_publisher.create_endpoint(component)

    @async_onstart
    async def init_engine(self):
        if self.engine_client is None:
            await super().initialize()
            print("vLLM worker initialized")
        assert self.engine_client is not None, "engine_client was not initialized"
        self.engine_client.set_metrics_publisher(self.metrics_publisher)
        self.metrics_publisher.publish(0, 1024, 0, 1024)
        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(lambda _: print("metrics publisher endpoint created"))

    @dynemo_endpoint()
    async def generate(self, request: vLLMGenerateRequest):
        sampling_params = request.sampling_params
        # rust HTTP requires Delta streaming
        sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.engine_client.generate(  # type: ignore
            request.engine_prompt, sampling_params, request.request_id
        ):
            # MyRequestOutput takes care of serializing the response as
            # vLLM's RequestOutput is not serializable by default
            resp = MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
            yield resp
