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

from common.aggregated_strategy import AggregatedStrategy
from common.base_engine import BaseEngineConfig
from common.decode_first_strategy import DecodeFirstStrategy
from common.parser import parse_tensorrt_llm_args
from common.prefill_first_strategy import PrefillFirstStrategy
from common.protocol import TRTLLMWorkerRequest
from components.next_worker import TensorRTLLMNextWorker

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
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker:
    next_worker = depends(TensorRTLLMNextWorker)

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
        next_class_name = "TensorRTLLMNextWorker"

        if args.enable_disagg:
            if not args.disaggregation_mode:
                disaggregation_mode = "decode"
            else:
                disaggregation_mode = args.disaggregation_mode
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
            next_endpoint=f"dyn://{namespace}.{next_class_name}.generate",
            lease_id=lease_id,
        )

        if disaggregation_mode == "prefill_and_decode":
            self._engine = AggregatedStrategy(engine_config)
        elif disaggregation_mode == "decode":
            self._engine = DecodeFirstStrategy(engine_config)
        elif disaggregation_mode == "prefill":
            self._engine = PrefillFirstStrategy(engine_config)
        else:
            raise ValueError(f"Invalid disaggregation mode: {disaggregation_mode}")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        await self._engine.initialize(runtime)

        endpoint = (
            runtime.namespace(self._engine._config.namespace)
            .component(self._engine._config.component)
            .endpoint(self._engine._config.endpoint)
        )

        try:
            await register_llm(
                ModelType.Backend,
                endpoint,
                self._engine._config.model_path,
                self._engine._config.served_model_name,
                kv_cache_block_size=self._engine._config.kv_block_size,
            )
            logger.info("Successfully registered LLM for discovery")
        except Exception as e:
            logger.error(f"Failed to register LLM for discovery: {e}")
            raise

        logger.info(
            f"TensorRT-LLM Worker initialized with {self._engine.__class__.__name__}"
        )

    @on_shutdown
    async def async_cleanup(self):
        await self._engine.cleanup()

    @endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in self._engine.generate(request):
            yield response
