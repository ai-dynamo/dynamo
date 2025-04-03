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
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Any, Optional
import copy
import signal
from dataclasses import asdict

from common.chat_processor import ChatProcessor, CompletionsProcessor
from common.parser import LLMAPIConfig
from common.utils import ManagedThread
from common.protocol import (
    DisaggregatedTypeConverter,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
    TRTLLMWorkerResponseOutput,
)

from tensorrt_llm.serve.openai_protocol import DisaggregatedParams
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.logger import logger
from transformers import AutoTokenizer

from dynamo.llm import KvMetricsPublisher

from .kv_cache_event_publisher import KVCacheEventPublisher

logger.set_level("info")


class ChatProcessorMixin:
    def __init__(self, engine_config: LLMAPIConfig):
        self._engine_config = engine_config
        logger.info(f"Using LLM API config: {self._engine_config.to_dict()}")
        # model name for chat processor
        self._model_name = self._engine_config.model_name
        logger.info(f"Set model name: {self._model_name}")

        # model for LLMAPI input
        self._model = self._model_name

        if self._engine_config.model_path:
            self._model = self._engine_config.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_path
            )
            logger.info(f"Using model from path: {self._engine_config.model_path}")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_name
            )

        if self._engine_config.extra_args.get("tokenizer", None):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.extra_args.get("tokenizer", None)
            )

        self.chat_processor = ChatProcessor(self._model_name, self._tokenizer)
        self.completions_processor = CompletionsProcessor(
            self._model_name, self._tokenizer
        )


@dataclass
class TensorrtLLMEngineConfig:
    namespace_str: str = "dynamo"
    component_str: str = "tensorrt-llm"
    engine_config: LLMAPIConfig = None
    worker_id: Optional[str] = None
    kv_metrics_publisher: Optional[KvMetricsPublisher] = None
    publish_stats: bool = False
    publish_kv_cache_events: bool = False
    # default block size is 32 for pytorch backend
    kv_block_size: int = 32


class BaseTensorrtLLMEngine(ChatProcessorMixin):
    def __init__(
        self,
        trt_llm_engine_config: TensorrtLLMEngineConfig,
    ):
        super().__init__(trt_llm_engine_config.engine_config)
        self._namespace_str = trt_llm_engine_config.namespace_str
        self._component_str = trt_llm_engine_config.component_str
        self._worker_id = trt_llm_engine_config.worker_id
        self._kv_metrics_publisher = trt_llm_engine_config.kv_metrics_publisher
        self._publish_stats = trt_llm_engine_config.publish_stats
        self._publish_kv_cache_events = trt_llm_engine_config.publish_kv_cache_events
        self._kv_block_size = trt_llm_engine_config.kv_block_size
        self._error_queue: Optional[Queue] = None

        self._init_engine()

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

        self._error_queue = Queue()
        try:
            if self._publish_stats:
                self._init_publish_metrics_thread()

            if self._publish_kv_cache_events:
                self._init_publish_kv_cache_events_thread()
        except Exception as e:
            logger.error(f"Failed to initialize publish metrics threads: {e}")
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
        if self._worker_id is None:
            logger.error("Worker ID not initialized!")
            return

        # TODO: Use python bindings to publish kv cache events once they
        # are available.
        lib_path = "/opt/dynamo/bindings/lib/libdynamo_llm_capi.so"
        self._kv_cache_events_publisher = KVCacheEventPublisher(
            self._namespace_str,
            self._component_str,
            int(self._worker_id),
            lib_path,
            self._kv_block_size,
        )

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
        async for event_list in events:
            for event in event_list:
                data = event["data"]
                if data["type"] == "stored":
                    parent_hash = data["parent_hash"]
                    for block in data["blocks"]:
                        tokens = []
                        for token in block["tokens"]:
                            tokens.append(int(token["token_id"]))

                        # Note: Currently data does not have lora_id.
                        # Using 0 as default value. If later data has
                        # lora_id, we need to verify if this is correct.
                        lora_id = data.get("lora_id", 0)
                        self._kv_cache_events_publisher.stored_event(
                            parent_hash,
                            block["block_hash"],
                            tokens,
                            lora_id,
                        )
                elif data["type"] == "removed":
                    for block_hash in data["block_hashes"]:
                        self._kv_cache_events_publisher.removed_event(block_hash)
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
                    lambda: LLM(model=self._model, **self._engine_config.to_dict()),
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
            request_type="context_only"
        )

        # TODO: Use smart KV router to determine which prefill worker to use.
        ctx_responses = [
            ctx_response
            async for ctx_response in await self.prefill_client.round_robin(
                prefill_request.model_dump_json()
            )
        ]
        if len(ctx_responses) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )
        logger.debug(
            f"[worker] received response from context server: {ctx_responses[0].data()}"
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

        if self._error_queue.qsize() > 0:
            error = self._error_queue.get()
            raise error

        self._ongoing_request_count += 1

        try:
            worker_inputs = request.prompt
            disaggregated_params = None

            if self.do_remote_prefill:
                ctx_response_obj = await self._get_remote_prefill_response(request)
                worker_inputs = (ctx_response_obj.prompt_token_ids + ctx_response_obj.outputs[0].token_ids)
                disaggregated_params = (
                    DisaggregatedTypeConverter.to_llm_disaggregated_params(
                        DisaggregatedParams(
                            **ctx_response_obj.outputs[0].disaggregated_params
                        )
                    )
                )
                disaggregated_params.request_type = "generation_only"

            logger.debug(
                f"[worker context] Worker inputs: {worker_inputs}, disaggregated params: {disaggregated_params}"
            )

            async for response in self._llm_engine.generate_async(
                inputs=worker_inputs,
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
