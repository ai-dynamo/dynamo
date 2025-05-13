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
import signal

from components.disagg_router import PyDisaggregatedRouter
from components.prefill_worker import PrefillWorker
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import MyRequestOutput, vLLMGenerateRequest
from utils.vllm import RouterType, parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker:
    prefill_worker = depends(PrefillWorker)

    def __init__(self):
        self.client = None
        self.disaggregated_router: PyDisaggregatedRouter = None  # type: ignore
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self.namespace, _ = VllmWorker.dynamo_address()  # type: ignore
        self._prefill_queue_stream_name = f"{self.namespace}_prefill_queue"
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == RouterType.KV:
            if not self.engine_args.enable_prefix_caching:
                logger.info(
                    "When using KV router, prefix caching must be enabled, setting to True"
                )
                self.engine_args.enable_prefix_caching = True

            VLLM_WORKER_ID = dynamo_context["endpoints"][0].lease_id()
            os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
            os.environ["VLLM_KV_NAMESPACE"] = "dynamo"
            os.environ["VLLM_KV_COMPONENT"] = class_name

        self.metrics_publisher = KvMetricsPublisher()

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")
        self.engine_client.set_metrics_publisher(self.metrics_publisher)
        # Initially send dummy metrics to kick start,
        # vLLM will not update stat until forward pass is triggered
        self.metrics_publisher.publish(
            0,  # request_active_slots
            1024,  # request_total_slots
            0,  # kv_active_blocks
            1024,  # kv_total_blocks
            0,  # num_requests_waiting
            0.0,  # gpu_cache_usage_perc
            0.0,  # gpu_prefix_cache_hit_rate
        )
        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logger.info("metrics publisher endpoint created")
        )

        runtime = dynamo_context["runtime"]

        if self.engine_args.remote_prefill:
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

        if self.engine_args.conditional_disagg:
            self.disaggregated_router = PyDisaggregatedRouter(
                runtime,
                self.namespace,
                max_local_prefill_length=self.engine_args.max_local_prefill_length,
                max_prefill_queue_size=self.engine_args.max_prefill_queue_size,
            )
            await self.disaggregated_router.async_init()
        else:
            self.disaggregated_router = None

        # Set up signal handler for graceful shutdown
        # TODO: move to dynamo sdk
        loop = asyncio.get_running_loop()

        def signal_handler():
            # Schedule the shutdown coroutine instead of calling it directly
            asyncio.create_task(self.graceful_shutdown(runtime))

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        logger.info("VllmWorker has been initialized")

    async def graceful_shutdown(self, runtime):
        logger.info("Received shutdown signal, shutting down DistributedRuntime")
        runtime.shutdown()
        logger.info("DistributedRuntime shutdown complete")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        logger.info("Creating metrics publisher endpoint with primary lease")
        await self.metrics_publisher.create_endpoint(component)

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    # TODO: use the same child lease for metrics publisher endpoint and generate endpoint
    @dynamo_endpoint()
    async def generate(self, request: vLLMGenerateRequest):
        # TODO: consider prefix hit when deciding prefill locally or remotely

        if self.disaggregated_router is not None:
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                prefill_queue_size = await prefill_queue.get_queue_size()
            disagg_router_decision = await self.disaggregated_router.prefill_remote(
                len(request.engine_prompt["prompt_token_ids"]),
                request.prefix_hit_rate,
                prefill_queue_size,
            )
        else:
            # always prefill remotely if no disaggregated router is provided
            disagg_router_decision = True

        if self.do_remote_prefill and disagg_router_decision:
            remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
            )
            logger.info(
                f"Prefilling remotely for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
            )
        else:
            remote_prefill_params = None
            logger.info(
                f"Prefilling locally for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
            )

        # rust HTTP requires Delta streaming
        request.sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.engine_client.generate(
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
