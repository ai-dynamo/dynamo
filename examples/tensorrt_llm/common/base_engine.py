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
import copy
import logging
import os
import signal
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from queue import Queue
from typing import Any, Optional

from common.parser import LLMAPIConfig
from common.protocol import (
    DisaggregatedTypeConverter,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
    TRTLLMWorkerResponseOutput,
)
from common.utils import ManagedThread, ServerType
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    parse_disagg_config_file,
)
from tensorrt_llm.serve.openai_protocol import DisaggregatedParams

from dynamo.llm import KvEventPublisher, KvMetricsPublisher
from dynamo.sdk import dynamo_context

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


class DisaggRequestType(Enum):
    CONTEXT_ONLY = "context_only"
    GENERATION_ONLY = "generation_only"


def update_args_from_disagg_config(
    engine_config: LLMAPIConfig, server_config: CtxGenServerConfig
):
    # Update the LLM API config with the disaggregated config
    # Allows for different configs for context and generation servers
    engine_config.extra_args.update(**server_config.other_args)
    engine_config.update_sub_configs(server_config.other_args)
    return engine_config


def get_sampling_params(sampling_params):
    # Removes keys starting with '_' from the sampling params which gets
    # added by the LLM API. TRTLLM does not support creating SamplingParams
    # from a dictionary with keys starting with '_'.
    cleaned_dict = {
        key: value for key, value in sampling_params.items() if not key.startswith("_")
    }
    return SamplingParams(**cleaned_dict)


class BaseTensorrtLLMEngine:
    def __init__(
        self,
        namespace_str: str = "dynamo",
        component_str: str = "tensorrt-llm",
        worker_id: Optional[str] = None,
        engine_config: LLMAPIConfig = None,
        remote_prefill: bool = False,
        min_workers: int = 0,
        disagg_config_file: Optional[str] = None,
        block_size: int = 32,
        router: str = "round_robin",
        server_type: ServerType = ServerType.GEN,
    ):
        self._namespace_str = namespace_str
        self._component_str = component_str
        self._worker_id = worker_id
        self._remote_prefill = remote_prefill
        self._min_workers = 0
        self._kv_block_size = block_size
        self._router = router
        self._server_type = server_type
        self._prefill_client = None
        self._error_queue: Queue = Queue()
        self._kv_metrics_publisher = None

        if self._remote_prefill or self._server_type == ServerType.CTX:
            self._min_workers = min_workers
            if disagg_config_file is None or not os.path.exists(disagg_config_file):
                raise ValueError(
                    "llmapi_disaggregated_config file does not exist or not provided"
                )
            disagg_config = parse_disagg_config_file(disagg_config_file)
            server_config: CtxGenServerConfig = None

            for config in disagg_config.server_configs:
                # Select the first context server config
                if config.type == server_type.value:
                    server_config = config
                    break

            if server_config is None:
                server_type_str = (
                    "generation" if server_type == ServerType.GEN else "context"
                )
                raise ValueError(
                    f"No {server_type_str} server config found. Please check the disaggregated config file."
                )

            engine_config = update_args_from_disagg_config(engine_config, server_config)

        if router == "kv":
            self._publish_stats = True
            self._publish_events = True
        else:
            self._publish_stats = False
            self._publish_events = False

        if self._publish_stats:
            self._kv_metrics_publisher = KvMetricsPublisher()

        if self._publish_events:
            if self._worker_id is None:
                raise ValueError("Worker ID is None!")

            runtime = dynamo_context["runtime"]
            kv_listener = runtime.namespace(self._namespace_str).component(
                self._component_str
            )
            self._kv_event_publisher = KvEventPublisher(
                kv_listener, int(self._worker_id), self._kv_block_size
            )
            logger.info("KvEventPublisher is initialized")

        self._engine_config = engine_config

    def _init_engine(self):
        logger.info("Initializing engine")
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine: Optional[Any] = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )

        self.publish_kv_cache_events_thread = None
        self.publish_stats_thread = None

        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            logger.error(f"Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

        try:
            if self._publish_stats:
                self._init_publish_metrics_thread()
        except Exception as e:
            logger.error(f"Failed to initialize publish metrics threads: {e}")
            raise e

        try:
            if self._publish_events:
                self._init_publish_kv_cache_events_thread()
        except Exception as e:
            logger.error(f"Failed to initialize publish events threads: {e}")
            raise e

    def _init_publish_metrics_thread(self):
        # Need to publish stats once so that worker can be selected.
        # Publishing some dummy values...
        request_active_slots = 0
        request_total_slots = 4
        kv_active_block = 0
        kv_total_blocks = 4
        num_requests_waiting = 0
        gpu_cache_usage_perc = 0.0
        gpu_prefix_cache_hit_rate = 0.0

        num_requests_waiting = 0
        gpu_cache_usage_perc = 0.0
        gpu_prefix_cache_hit_rate = 0.0

        if self._kv_metrics_publisher is None:
            logger.error("KV metrics publisher not initialized!")
            return

        self._kv_metrics_publisher.publish(
            request_active_slots,
            request_total_slots,
            kv_active_block,
            kv_total_blocks,
            num_requests_waiting,
            gpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate,
        )

        # Prepare threads for publishing stats but don't start them yet.
        # TRTLLM needs to start generating tokens first before stats
        # can be retrieved.
        self.publish_stats_thread = ManagedThread(
            self.publish_stats_task,
            error_queue=self._error_queue,
            name="publish_stats_thread",
        )

    def _init_publish_kv_cache_events_thread(self):
        if self._kv_event_publisher is None:
            logger.error("KV event publisher not initialized!")
            return

        # A set to store the block hash of partial block (i.e. block containing less than kv_block_size tokens) hashes.
        # It is used to prevent sending remove event to kv router since partial blocks are not stored.
        self._partial_block_hashes = set()

        # Prepare threads for publishing kv cache events but don't start them yet.
        # TRTLLM needs to start generating tokens first before kv cache events
        # can be retrieved.
        self.publish_kv_cache_events_thread = ManagedThread(
            self.publish_kv_cache_events_task,
            error_queue=self._error_queue,
            name="publish_kv_cache_events_thread",
        )

    async def publish_stats_task(self):
        """
        Publish stats to the metrics publisher.
        """
        if self._llm_engine is None:
            logger.error("LLM engine not initialized!")
            return

        if self._kv_metrics_publisher is None:
            logger.error("KV metrics publisher not initialized!")
            return False

        stats = self._llm_engine.get_stats_async(timeout=5)
        async for stat in stats:
            request_active_slots = stat["numActiveRequests"]
            request_total_slots = stat["maxNumActiveRequests"]
            kv_active_block = stat["kvCacheStats"]["usedNumBlocks"]
            kv_total_blocks = stat["kvCacheStats"]["maxNumBlocks"]
            reused_blocks = stat["kvCacheStats"]["reusedBlocks"]
            freeNumBlocks = stat["kvCacheStats"]["freeNumBlocks"]
            allocTotalBlocks = stat["kvCacheStats"]["allocTotalBlocks"]
            allocNewBlocks = stat["kvCacheStats"]["allocNewBlocks"]
            # NOTE: num paused requests is always 0 when using guarantee no evict scheduler (default).
            num_requests_waiting = (
                stat["numQueuedRequests"]
                + stat["inflightBatchingStats"]["numPausedRequests"]
            )
            gpu_cache_usage_perc = allocTotalBlocks / kv_total_blocks
            gpu_prefix_cache_hit_rate = stat["kvCacheStats"]["cacheHitRate"]

            logger.debug(
                f"Publishing stats: request_active_slots: {request_active_slots}, request_total_slots: {request_total_slots}, kv_active_block: {kv_active_block}, kv_total_blocks: {kv_total_blocks}, num_requests_waiting: {num_requests_waiting}, reused_blocks: {reused_blocks}, freeNumBlocks: {freeNumBlocks}, allocTotalBlocks: {allocTotalBlocks}, allocNewBlocks: {allocNewBlocks}, gpu_cache_usage_perc: {gpu_cache_usage_perc}, gpu_prefix_cache_hit_rate: {gpu_prefix_cache_hit_rate}"
            )

            self._kv_metrics_publisher.publish(
                request_active_slots,
                request_total_slots,
                kv_active_block,
                kv_total_blocks,
                num_requests_waiting,
                gpu_cache_usage_perc,
                gpu_prefix_cache_hit_rate,
            )

        return True

    async def publish_kv_cache_events_task(self):
        """
        Publish kv cache events to the events publisher.
        """
        if self._llm_engine is None:
            logger.error("LLM engine not initialized!")
            return

        events = self._llm_engine.get_kv_cache_events_async(timeout=5)
        async for event in events:
            event_id = event["event_id"]
            data = event["data"]
            if data["type"] == "stored":
                parent_hash = data["parent_hash"]
                token_ids = []
                num_block_tokens = []
                block_hashes = []
                for block in data["blocks"]:
                    token_num_in_block = len(block["tokens"])
                    block_hash = block["block_hash"]
                    if token_num_in_block > self._kv_block_size:
                        logger.error(
                            f"Block {block_hash} contains {token_num_in_block} tokens, which is greater than kv_block_size {self._kv_block_size}"
                        )
                        return
                    if token_num_in_block < self._kv_block_size:
                        logger.debug(
                            f"Early stop when block {block_hash} containing {token_num_in_block} tokens not equal to kv_block_size {self._kv_block_size}"
                        )
                        self._partial_block_hashes.add(block_hash)
                        break
                    num_block_tokens.append(token_num_in_block)
                    block_hashes.append(block_hash)
                    for token in block["tokens"]:
                        token_ids.append(int(token["token_id"]))

                # Note: Currently data does not have lora_id.
                # Using 0 as default value. If later data has
                # lora_id, we need to verify if this is correct.
                lora_id = data.get("lora_id", 0)
                self._kv_event_publisher.publish_stored(
                    event_id,
                    token_ids,
                    num_block_tokens,
                    block_hashes,
                    lora_id,
                    parent_hash,
                )
            elif data["type"] == "removed":
                block_hashes = []
                for block_hash in data["block_hashes"]:
                    if block_hash in self._partial_block_hashes:
                        logger.debug(
                            f"Skipping removing block hash {block_hash} since it is a partial block"
                        )
                        self._partial_block_hashes.remove(block_hash)
                        continue
                    block_hashes.append(block_hash)
                self._kv_event_publisher.publish_removed(event_id, block_hashes)
        return True

    def _start_threads(self):
        if (
            self.publish_kv_cache_events_thread
            and not self.publish_kv_cache_events_thread.is_alive()
        ):
            # [NOTE:] TRTLLM needs the stats to be collected on the same loop as the request handler.
            self._stats_loop = asyncio.get_running_loop()
            self.publish_kv_cache_events_thread.set_loop(self._stats_loop)
            self.publish_kv_cache_events_thread.start()
            logger.debug("Started kv cache events thread")

        if self.publish_stats_thread and not self.publish_stats_thread.is_alive():
            self._stats_loop = asyncio.get_running_loop()
            self.publish_stats_thread.set_loop(self._stats_loop)
            self.publish_stats_thread.start()
            logger.debug("Started stats thread")

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        model=self._engine_config.model_name,
                        **self._engine_config.to_dict(),
                    ),
                )
                yield llm
            finally:
                if "llm" in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                logger.info("Engine loaded and ready to serve...")

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Stop the publishing threads
                if self.publish_stats_thread and self.publish_stats_thread.is_alive():
                    self.publish_stats_thread.stop()
                    self.publish_stats_thread.join()
                if (
                    self.publish_kv_cache_events_thread
                    and self.publish_kv_cache_events_thread.is_alive()
                ):
                    self.publish_kv_cache_events_thread.stop()
                    self.publish_kv_cache_events_thread.join()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    logger.info(
                        "Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        logger.info("Shutdown complete")

    async def _get_remote_prefill_response(self, request):
        prefill_request = copy.deepcopy(request)
        prefill_request.sampling_params["max_tokens"] = 1
        prefill_request.disaggregated_params = DisaggregatedParams(
            request_type=DisaggRequestType.CONTEXT_ONLY.value
        )

        if self._prefill_client is None:
            raise ValueError("Prefill client not initialized")

        # TODO: Use smart KV router to determine which prefill worker to use.
        ctx_responses = [
            ctx_response
            async for ctx_response in await self._prefill_client.round_robin(
                prefill_request.model_dump_json()
            )
        ]
        if len(ctx_responses) > 1:
            raise ValueError(
                "Prefill worker returned more than one response. This is currently not supported in remote prefill mode."
            )
        logger.debug(
            f"Received response from prefill worker: {ctx_responses[0].data()}"
        )
        ctx_response_obj = TRTLLMWorkerResponse.model_validate_json(
            ctx_responses[0].data()
        )
        ctx_response_obj.outputs = [
            TRTLLMWorkerResponseOutput(**ctx_response_obj.outputs[0])
        ]
        assert ctx_response_obj.outputs[0].disaggregated_params is not None

        return ctx_response_obj

    async def generate(self, request: TRTLLMWorkerRequest):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if not self._error_queue.empty():
            raise self._error_queue.get()

        self._ongoing_request_count += 1

        try:
            worker_inputs = request.tokens.tokens

            disaggregated_params = (
                DisaggregatedTypeConverter.to_llm_disaggregated_params(
                    request.disaggregated_params
                )
            )

            if self._remote_prefill and self._server_type == ServerType.GEN:
                ctx_response_obj = await self._get_remote_prefill_response(request)

                worker_inputs = ctx_response_obj.prompt_token_ids
                disaggregated_params = (
                    DisaggregatedTypeConverter.to_llm_disaggregated_params(
                        DisaggregatedParams(
                            **ctx_response_obj.outputs[0].disaggregated_params
                        )
                    )
                )
                disaggregated_params.request_type = (
                    DisaggRequestType.GENERATION_ONLY.value
                )

            logger.debug(
                f"Worker inputs: {worker_inputs}, disaggregated params: {disaggregated_params}"
            )

            sampling_params = get_sampling_params(request.sampling_params)
            async for response in self._llm_engine.generate_async(
                inputs=worker_inputs,
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
                streaming=False
                if self._server_type == ServerType.CTX
                else request.streaming,
            ):
                # Convert the disaggregated params to OAI format so
                # it can be sent over the network.
                response.outputs[
                    0
                ].disaggregated_params = DisaggregatedTypeConverter.to_oai_disaggregated_params(
                    response.outputs[0].disaggregated_params
                )

                yield TRTLLMWorkerResponse(
                    request_id=request.id,
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
