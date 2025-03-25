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
from typing import AsyncGenerator

import uvloop
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    DisaggregatedTypeConverter,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
)
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
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
            disaggregated_params = None
            if request.disaggregated_params is not None:
                disaggregated_params = (
                    DisaggregatedTypeConverter.to_llm_disaggregated_params(
                        request.disaggregated_params
                    )
                )

            async for response in self._llm_engine.generate_async(
                inputs=request.prompt,
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


@dynamo_worker()
async def trtllm_worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig, args):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    namespace_str = "dynamo"
    component_str = "tensorrt-llm"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    generate_endpoint = component.endpoint("generate")

    trt_llm_engine_config = TensorrtLLMEngineConfig(
        namespace_str=namespace_str,
        component_str=component_str,
        engine_config=engine_config,
        publish_stats=args.publish_stats,
        publish_kv_cache_events=args.publish_kv_cache_events,
        kv_block_size=args.kv_block_size,
    )

    if args.publish_stats:
        trt_llm_engine_config.kv_metrics_publisher = KvMetricsPublisher()

    trt_llm_engine_config.worker_id = generate_endpoint.lease_id()

    engine = TensorrtLLMEngine(trt_llm_engine_config)

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

    asyncio.run(trtllm_worker(engine_config, args))
