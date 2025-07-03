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
import logging
import signal

from common.base_engine import BaseEngineConfig, BaseTensorrtLLMEngine
from common.parser import parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest

from dynamo.sdk import async_on_start, dynamo_context, endpoint, on_shutdown, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMPrefillWorker(BaseTensorrtLLMEngine):
    def __init__(self):
        logger.info("Initializing TensorRT-LLM Prefill Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args = parse_tensorrt_llm_args(config_args)
        lease_id = dynamo_context["endpoints"][0].lease_id()
        namespace, _ = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore

        engine_config = BaseEngineConfig(
            namespace=namespace,
            component=class_name,
            endpoint="generate",
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            kv_block_size=args.kv_block_size,
            extra_engine_args=args.extra_engine_args,
            publish_events_and_metrics=False,
            disaggregation_mode="prefill",
            remote_prefill_endpoint=None,
            lease_id=lease_id,
        )

        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)

        super().__init__(config=engine_config)

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        await self.initialize(runtime)
        logger.info("TensorRT-LLM Prefill Worker initialized")

    def graceful_shutdown(self, signum, frame):
        """
        Gracefully shutdown the worker by shutting down the dynamo runtime.
        This will
            1. disable the generate endpoint so no new requests are accepted.
            2. wait until all in-flight requests are completed.
            3. finish the awaiting for the endpoint service.
            4. rely on python's garbage collection to clean up the GPU.
        """
        logger.info("Shutting down dynamo runtime...")
        dynamo_context["runtime"].shutdown()
        logger.info("Dynamo runtime shutdown complete.")

    @on_shutdown
    async def async_cleanup(self):
        logger.info("Cleaning up TensorRT-LLM Prefill Worker")
        await self.cleanup()
        logger.info("TensorRT-LLM Prefill Worker cleanup completed")

    @endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
