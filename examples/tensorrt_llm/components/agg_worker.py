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
import signal
from dataclasses import asdict

from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest, TRTLLMWorkerResponse
from tensorrt_llm.executor import CppExecutorError

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self):
        logger.info("Initializing TensorRT-LLM Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        self.args, self.engine_config = parse_tensorrt_llm_args(config_args)

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

        self.trtllm_engine_args = trt_llm_engine_config

    @async_on_start
    async def async_init(self):
        super().__init__(self.trtllm_engine_args)
        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(lambda _: print("metrics publisher endpoint created"))
        print("TensorRT-LLM Worker initialized")

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self.trtllm_engine_args.kv_metrics_publisher.create_endpoint(component)

    @dynamo_endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if self._error_queue.qsize() > 0:
            error = self._error_queue.get()
            raise error

        self._ongoing_request_count += 1

        try:
            # TODO: combine with disagg worker
            # TODO: only send tokens. Should be pretty simple.
            async for response in self._llm_engine.generate_async(
                inputs=request.prompt,
                sampling_params=request.to_sampling_params(),
                disaggregated_params=None,
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
