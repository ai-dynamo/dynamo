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

import abc
import asyncio
import copy
import logging
import os
import signal
import socket
import uuid
from typing import Optional

from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, PreprocessedRequest, vLLMGenerateRequest
from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.llm import (
    ModelType,
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import Component
from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, component: Component, dp_rank: int) -> None:
        self.inner = WorkerMetricsPublisher()
        self.inner.create_endpoint(component, dp_rank=dp_rank)
        self.dp_rank = dp_rank

    def record(
        self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]
    ):
        # request_total_slots and kv_total_blocks are properties of model + gpu
        # we should only publish them once, not every metric update
        # they should be part of some runtime metadata tied to MDC or put in etcd ?
        hit_rate = 0
        if scheduler_stats.prefix_cache_stats.queries > 0:
            hit_rate = (
                scheduler_stats.prefix_cache_stats.hits
                / scheduler_stats.prefix_cache_stats.queries
            )

        # TODO Manage DP Ranks in metrics aggregation.
        self.inner.publish(
            request_active_slots=scheduler_stats.num_running_reqs,
            request_total_slots=0,  # TODO - remove from metrics
            kv_active_blocks=0,  # TODO - need to calculate this
            kv_total_blocks=0,  # TODO - remove from metrics
            num_requests_waiting=scheduler_stats.num_waiting_reqs,  # used in current cost function
            gpu_cache_usage_perc=scheduler_stats.gpu_cache_usage,  # used in current cost function
            gpu_prefix_cache_hit_rate=hit_rate,
            data_parallel_rank=self.dp_rank,
        )

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, component: Component) -> None:
        self.component = component

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        return DynamoStatLoggerPublisher(self.component, dp_rank)

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)


BLOCK_SIZE = 16


class VllmBaseWorker(abc.ABC):
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        if not self.engine_args.block_size:
            logger.info(f"block_size not set, default to {BLOCK_SIZE}")
            self.engine_args.block_size = BLOCK_SIZE

        os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests

        model_config = self.engine_args.create_model_config()
        self.default_sampling_params = model_config.get_diff_sampling_param()

        self.kv_publishers = []

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_host_and_port()

    async def async_init(self):
        # Taken from build_async_engine_client_from_engine_args()
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = self.engine_args.create_engine_config(usage_context=usage_context)

        # Explicitly pass our custom stat logger for metrics
        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=[StatLoggerFactory(dynamo_context["component"])],
            disable_log_requests=self.engine_args.disable_log_requests,
            disable_log_stats=self.engine_args.disable_log_stats,
        )

        logger.info("VllmWorker has been initialized")

        base_zmq_endpoint = "tcp://127.0.0.1:5557"
        dp_rank_size = vllm_config.parallel_config.data_parallel_size

        # Store references to prevent garbage collection

        for dp_rank in range(dp_rank_size):
            zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
                base_zmq_endpoint, data_parallel_rank=dp_rank
            )
            zmq_config = ZmqKvEventPublisherConfig(
                worker_id=dynamo_context["endpoints"][0].lease_id(),
                kv_block_size=self.engine_args.block_size,
                zmq_endpoint=zmq_endpoint,
            )

            try:
                publisher = ZmqKvEventPublisher(
                    component=dynamo_context["component"], config=zmq_config
                )
                self.kv_publishers.append(publisher)
            except Exception as e:
                logger.error(
                    f"Failed to create ZmqKvEventPublisher for dp_rank {dp_rank}: {e}"
                )

        logger.debug(
            f"Successfully created {len(self.kv_publishers)} ZmqKvEventPublishers out of {dp_rank_size} expected"
        )

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.shutdown()
            for publisher in self.kv_publishers:
                publisher.shutdown()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def set_side_channel_host_and_port(
        self, hostname: Optional[str] = None, port: Optional[int] = None
    ):
        """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
        This sets the port number for the side channel.
        """
        if hostname is None:
            hostname = socket.gethostname()
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_HOST to %s", hostname)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = hostname
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_PORT to %s", port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPrefillWorker(VllmBaseWorker):
    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmPrefillWorker has been initialized")

    @endpoint()
    async def prefill(self, request: vLLMGenerateRequest):
        gen = self.engine_client.generate(
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )
        async for response in gen:
            logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            ).model_dump_json()


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker(VllmBaseWorker):
    prefill_worker = depends(VllmPrefillWorker)

    @async_on_start
    async def async_init(self):
        await super().async_init()
        comp_ns, comp_name = VllmDecodeWorker.dynamo_address()  # type: ignore
        runtime = dynamo_context["runtime"]
        for served_model_name in self.engine_args.served_model_name:
            logger.info(
                f"Registering endpoint generate with model {self.engine_args.model} and served_model_name {served_model_name}"
            )
            endpoint = (
                runtime.namespace(comp_ns).component(comp_name).endpoint("generate")
            )
            await register_llm(
                ModelType.Backend,
                endpoint,
                self.engine_args.model,
                served_model_name,
                context_length=self.engine_args.max_model_len,
                kv_cache_block_size=self.engine_args.block_size,
            )

        if self.engine_args.enable_disagg:
            comp_ns, comp_name = VllmPrefillWorker.dynamo_address()  # type: ignore
            self.prefill_worker_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("prefill")
                .client()
            )
        else:
            self.prefill_worker_client = None

        logger.info("VllmDecodeWorker has been initialized")

    @endpoint()
    async def generate(self, request: PreprocessedRequest):
        data_parallel_rank = request.dp_rank
        vllm_request = self._prepare_request(request)

        logger.debug("Prepared requests with id %s", vllm_request.request_id)

        if self.engine_args.enable_disagg:
            logger.debug("Sending request to prefill")
            prefill_response = await self.send_request_to_prefill(vllm_request)
            extra_args = vllm_request.sampling_params.extra_args or {}
            extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
            vllm_request.sampling_params.extra_args = extra_args

            logger.debug(
                "Prefill kv transfer params: %s",
                vllm_request.sampling_params.extra_args["kv_transfer_params"],
            )

        logger.debug("Sending request to decode")

        gen = self.engine_client.generate(
            prompt=vllm_request.prompt,
            sampling_params=vllm_request.sampling_params,
            request_id=vllm_request.request_id,
            data_parallel_rank=data_parallel_rank,
        )
        logger.debug("Streaming response")
        async for res in self._stream_response(gen):
            yield res

    async def send_request_to_prefill(
        self, request: vLLMGenerateRequest
    ) -> MyRequestOutput:
        logger.debug("Sending request to prefill")

        prefill_request = copy.deepcopy(request)
        extra_args = prefill_request.sampling_params.extra_args or {}
        extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        prefill_request.sampling_params.extra_args = extra_args
        prefill_request.sampling_params.max_tokens = 1
        prefill_request.sampling_params.min_tokens = 1

        logger.debug("Prefill request: %s", prefill_request.model_dump_json())

        async for prefill_response in await self.prefill_worker_client.round_robin(
            prefill_request.model_dump_json()
        ):
            return MyRequestOutput.model_validate_json(prefill_response.data())

    def _prepare_request(self, request: PreprocessedRequest) -> vLLMGenerateRequest:
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request.token_ids)

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request.sampling_options.model_dump().items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request.stop_conditions.max_tokens
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        return vLLMGenerateRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

    async def _stream_response(self, gen):
        num_output_tokens_so_far = 0
        async for res in gen:
            # res is vllm's RequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks
