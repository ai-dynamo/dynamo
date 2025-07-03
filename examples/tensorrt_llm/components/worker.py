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
from components.prefill_worker import TensorRTLLMPrefillWorker

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import (
    async_on_start,
    depends,
    dynamo_context,
    endpoint,
    on_shutdown,
    service,
)
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
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
        args = parse_tensorrt_llm_args(config_args)
        lease_id = dynamo_context["endpoints"][0].lease_id()
        namespace, _ = TensorRTLLMWorker.dynamo_address()  # type: ignore
        endpoint_name = "generate"
        publish_events_and_metrics = args.router == "kv"
        prefill_class_name = "TensorRTLLMPrefillWorker"

        if args.enable_disagg:
            disaggregation_mode = "decode"
        else:
            disaggregation_mode = "prefill_and_decode"

        engine_config = BaseEngineConfig(
            namespace=namespace,
            component=class_name,
            endpoint=endpoint_name,
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            kv_block_size=args.kv_block_size,
            extra_engine_args=args.extra_engine_args,
            publish_events_and_metrics=publish_events_and_metrics,
            disaggregation_mode=disaggregation_mode,
            remote_prefill_endpoint=f"dyn://{namespace}.{prefill_class_name}.generate",
            lease_id=lease_id,
        )

        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)

        super().__init__(config=engine_config)

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        await self.initialize(runtime)

        logger.info("Registering LLM for discovery")
        endpoint = (
            runtime.namespace(self._config.namespace)
            .component(self._config.component)
            .endpoint(self._config.endpoint)
        )

        try:
            await register_llm(
                ModelType.Backend,
                endpoint,
                self._config.model_path,
                self._config.served_model_name,
                kv_cache_block_size=self._config.kv_block_size,
            )
            logger.info("Successfully registered LLM for discovery")
        except Exception as e:
            logger.error(f"Failed to register LLM for discovery: {e}")
            raise

        logger.info("TensorRT-LLM Worker initialized")

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
        logger.info("Cleaning up TensorRT-LLM Worker")
        await self.cleanup()
        logger.info("TensorRT-LLM Worker cleanup completed")

    @endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
