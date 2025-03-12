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

import uvloop
from disagg_router import PyDisaggregatedRouter
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import MyRequestOutput, vLLMGenerateRequest
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import EngineClient
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.logger import logger as vllm_logger
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker


class RequestHandler:
    def __init__(
        self,
        model_name: str,
        engine_client: EngineClient,
        prefill_client,
        do_remote_prefill: bool,
        disaggregated_router: PyDisaggregatedRouter = None,
    ):
        self.model_name = model_name
        self.client = engine_client
        self.prefill_client = prefill_client
        self.openai_serving_chat = None
        self.initialized = False
        self.do_remote_prefill = (
            do_remote_prefill  # remote prefill is still controlled by the router
        )
        self.disaggregated_router = disaggregated_router

        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = model_name
        vllm_logger.info(
            "Prefill queue: %s:%s",
            self._prefill_queue_nats_server,
            self._prefill_queue_stream_name,
        )

        print("RequestHandler initialized")

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    @dynamo_endpoint(vLLMGenerateRequest, MyRequestOutput)
    async def generate(self, request):
        # TODO: consider prefix hit when deciding prefill locally or remotely
        if self.disaggregated_router is not None:
            disagg_router_decision = self.disaggregated_router.prefill_remote(
                len(request.engine_prompt["prompt_token_ids"]), request.prefix_hit_rate
            )
        else:
            # always prefill remotely if no disaggregated router is provided
            disagg_router_decision = True

        if self.do_remote_prefill and disagg_router_decision:
            remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
            )
            vllm_logger.debug(
                "Prefilling remotely for request %s with length %s",
                request.request_id,
                len(request.engine_prompt["prompt_token_ids"]),
            )
        else:
            remote_prefill_params = None
            vllm_logger.debug(
                "Prefilling locally for request %s with length %s",
                request.request_id,
                len(request.engine_prompt["prompt_token_ids"]),
            )

        # rust HTTP requires Delta streaming
        request.sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.client.generate(
            prompt=request.engine_prompt,
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    component = runtime.namespace("dynamo-init").component("vllm")
    await component.create_service()

    endpoint = component.endpoint("generate")

    if engine_args.remote_prefill:
        prefill_client = (
            await runtime.namespace("dynamo-init")
            .component("prefill")
            .endpoint("generate")
            .client()
        )
    else:
        prefill_client = None

    if engine_args.router == "kv":
        # TODO: do we need these env vars?
        VLLM_WORKER_ID = endpoint.lease_id()
        os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
        vllm_logger.info("Generate endpoint ID: %s", VLLM_WORKER_ID)

        VLLM_KV_NAMESPACE = "dynamo-init"
        os.environ["VLLM_KV_NAMESPACE"] = str(VLLM_KV_NAMESPACE)

        VLLM_KV_COMPONENT = "vllm"
        os.environ["VLLM_KV_COMPONENT"] = str(VLLM_KV_COMPONENT)

        metrics_publisher = KvMetricsPublisher()

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        served_model_name = (
            engine_args.served_model_name
            if engine_args.served_model_name is not None
            else "vllm"
        )

        if engine_args.router == "kv":
            engine_client.set_metrics_publisher(metrics_publisher)

            # Initially send dummy metrics to kick start,
            # vLLM will not update stat until forward pass is triggered
            metrics_publisher.publish(
                0,  # request_active_slots
                1024,  # request_total_slots
                0,  # kv_active_blocks
                1024,  # kv_total_blocks
                0,  # num_requests_waiting
                0.0,  # gpu_cache_usage_perc
                0.0,  # gpu_prefix_cache_hit_rate
            )

        if engine_args.remote_prefill:
            metadata = engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo-init", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

        if engine_args.conditional_disagg:
            disaggregated_router = PyDisaggregatedRouter(
                runtime,
                served_model_name,
                max_local_prefill_length=engine_args.max_local_prefill_length,
            )
        else:
            disaggregated_router = None

        endpoints = [
            endpoint.serve_endpoint(
                RequestHandler(
                    model_name=served_model_name,
                    engine_client=engine_client,
                    prefill_client=prefill_client,
                    do_remote_prefill=engine_args.remote_prefill,
                    disaggregated_router=disaggregated_router,
                ).generate
            )
        ]
        if engine_args.router == "kv":
            endpoints.append(metrics_publisher.create_endpoint(component))
        await asyncio.gather(*endpoints)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()

    if engine_args.remote_prefill:
        if engine_args.enable_chunked_prefill is not False:
            print("Chunked prefill is not supported yet, setting to False")
            engine_args.enable_chunked_prefill = False

        if engine_args.preemption_mode != "swap":
            print("Preemption mode is not supported yet, setting to swap")
            engine_args.preemption_mode = "swap"

        if engine_args.pipeline_parallel_size != 1:
            print("Pipeline parallel size is not supported yet, setting to 1")
            engine_args.pipeline_parallel_size = 1

    asyncio.run(worker(engine_args))
