# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import logging
import threading
import traceback
import weakref
from queue import Queue
from typing import Callable, Optional, Union

from dynamo.llm import KvEventPublisher, KvMetricsPublisher

logging.basicConfig(level=logging.DEBUG)


class ManagedThread(threading.Thread):
    """
    A thread that runs a task and handles errors.
    """

    def __init__(
        self,
        task: Optional[Union[Callable[..., bool], weakref.WeakMethod]],
        error_queue: Optional[Queue] = None,
        name: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.task = task
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.loop = loop
        self.daemon = True
        self._current_future: Optional[concurrent.futures.Future] = None

        self._stop_event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def run(self):
        while not self._stop_event.is_set():
            task: Optional[Union[Callable[..., bool], weakref.WeakMethod]] = self.task
            if isinstance(task, weakref.WeakMethod):
                task = task()
                if task is None:
                    # Normally, this should not happen.
                    logging.warning("WeakMethod is expired.")
                    break

            if task is None:
                break

            try:
                if self.loop is None:
                    logging.error("[ManagedThread] Loop not initialized!")
                    break
                self._current_future = asyncio.run_coroutine_threadsafe(
                    task(**self.kwargs), self.loop
                )
                _ = self._current_future.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                logging.debug(f"Thread {self.name} was cancelled")
                break
            except Exception as e:
                logging.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                if self.error_queue is not None:
                    self.error_queue.put(e)

        logging.info(f"Thread {self.name} stopped.")

    def stop(self):
        self._stop_event.set()
        if self._current_future and not self._current_future.done():
            self._current_future.cancel()


class Publishers:
    """
    A class to retrieve stats and kv cache events from TRTLLM engine and publish them to the metrics and events publishers.
    """

    def __init__(self, component, engine, kv_listener, worker_id, kv_block_size):
        self.component = component
        self.engine = engine
        self.kv_listener = kv_listener
        self.worker_id = worker_id
        self.kv_block_size = kv_block_size

        # Needed by the events and metrics publishers
        self.metrics_publisher = None
        self.kv_event_publisher = None
        self.publish_kv_cache_events_thread = None
        self.publish_stats_thread = None
        # A set to store the block hash of partial block (i.e. block containing less than kv_block_size tokens) hashes.
        # It is used to prevent sending remove event to kv router since partial blocks are not stored.
        self.partial_block_hashes = set()
        self.error_queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._setup()

    async def _create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return
        await self.metrics_publisher.create_endpoint(self.component)

    def _setup(self):
        # Setup the metrics publisher
        self.metrics_publisher = KvMetricsPublisher()
        self._init_publish_metrics_thread()
        task = asyncio.create_task(self._create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

        # Setup the kv cache events publisher
        self.kv_event_publisher = KvEventPublisher(
            self.kv_listener, self.worker_id, self.kv_block_size
        )
        self._init_publish_kv_cache_events_thread()

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

        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return

        self.metrics_publisher.publish(
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
            self._publish_stats_task,
            error_queue=self.error_queue,
            name="publish_stats_thread",
        )

    def _init_publish_kv_cache_events_thread(self):
        if self.kv_event_publisher is None:
            logging.error("KV event publisher not initialized!")
            return

        # Prepare threads for publishing kv cache events but don't start them yet.
        # TRTLLM needs to start generating tokens first before kv cache events
        # can be retrieved.
        self.publish_kv_cache_events_thread = ManagedThread(
            self._publish_kv_cache_events_task,
            error_queue=self.error_queue,
            name="publish_kv_cache_events_thread",
        )

    async def _publish_stats_task(self):
        """
        Publish stats to the metrics publisher.
        """
        if self.engine is None:
            logging.error("LLM engine not initialized!")
            return

        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return False

        stats = self.engine.llm.get_stats_async(timeout=5)
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

            logging.debug(
                f"Publishing stats: request_active_slots: {request_active_slots}, request_total_slots: {request_total_slots}, kv_active_block: {kv_active_block}, kv_total_blocks: {kv_total_blocks}, num_requests_waiting: {num_requests_waiting}, reused_blocks: {reused_blocks}, freeNumBlocks: {freeNumBlocks}, allocTotalBlocks: {allocTotalBlocks}, allocNewBlocks: {allocNewBlocks}, gpu_cache_usage_perc: {gpu_cache_usage_perc}, gpu_prefix_cache_hit_rate: {gpu_prefix_cache_hit_rate}"
            )

            self.metrics_publisher.publish(
                request_active_slots,
                request_total_slots,
                kv_active_block,
                kv_total_blocks,
                num_requests_waiting,
                gpu_cache_usage_perc,
                gpu_prefix_cache_hit_rate,
            )

        return True

    async def _publish_kv_cache_events_task(self):
        """
        Publish kv cache events to the events publisher.
        """
        if self.engine is None:
            logging.error("LLM engine not initialized!")
            return

        if self.kv_event_publisher is None:
            logging.error("KV event publisher not initialized!")
            return

        events = self.engine.llm.get_kv_cache_events_async(timeout=5)
        async for event in events:
            logging.debug(f"KV cache event received: {event}")
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
                    if token_num_in_block > self.kv_block_size:
                        logging.error(
                            f"Block {block_hash} contains {token_num_in_block} tokens, which is greater than kv_block_size {self.kv_block_size}"
                        )
                        return
                    if token_num_in_block < self.kv_block_size:
                        logging.debug(
                            f"Early stop when block {block_hash} containing {token_num_in_block} tokens not equal to kv_block_size {self.kv_block_size}"
                        )
                        self.partial_block_hashes.add(block_hash)
                        break
                    num_block_tokens.append(token_num_in_block)
                    block_hashes.append(block_hash)
                    for token in block["tokens"]:
                        token_ids.append(int(token["token_id"]))

                # Note: Currently data does not have lora_id.
                # Using 0 as default value. If later data has
                # lora_id, we need to verify if this is correct.
                lora_id = data.get("lora_id", 0)

                logging.debug(
                    f"publish stored event: event_id: {event_id}, token_ids: {token_ids}, num_block_tokens: {num_block_tokens}, block_hashes: {block_hashes}, lora_id: {lora_id}, parent_hash: {parent_hash}"
                )
                self.kv_event_publisher.publish_stored(
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
                    if block_hash in self.partial_block_hashes:
                        logging.debug(
                            f"Skipping removing block hash {block_hash} since it is a partial block"
                        )
                        self.partial_block_hashes.remove(block_hash)
                        continue
                    block_hashes.append(block_hash)

                logging.debug(
                    f"publish removed event: event_id: {event_id}, block_hashes: {block_hashes}"
                )
                self.kv_event_publisher.publish_removed(event_id, block_hashes)
        return True

    def start_publish_threads(self):
        if (
            self.publish_kv_cache_events_thread
            and not self.publish_kv_cache_events_thread.is_alive()
        ):
            # REVISIT
            # [NOTE:] TRTLLM needs the stats to be collected on the same loop as the request handler.
            self._stats_loop = asyncio.get_running_loop()
            self.publish_kv_cache_events_thread.set_loop(self._stats_loop)
            self.publish_kv_cache_events_thread.start()
            logging.debug("Started kv cache events thread")

        if self.publish_stats_thread and not self.publish_stats_thread.is_alive():
            self._stats_loop = asyncio.get_running_loop()
            self.publish_stats_thread.set_loop(self._stats_loop)
            self.publish_stats_thread.start()
            logging.debug("Started stats thread")

    def check_error_queue(self):
        if not self.error_queue.empty():
            logging.error("Error in publishers error queue")
            return self.error_queue.get()
        return None

    async def cleanup(self):
        """Cleanup threads and resources"""
        self._stop_event.set()
        # Add timeout to prevent hanging
        cleanup_timeout = 5.0  # seconds

        if self.publish_stats_thread and self.publish_stats_thread.is_alive():
            self.publish_stats_thread.stop()
            self.publish_stats_thread.join(timeout=cleanup_timeout)
            if self.publish_stats_thread.is_alive():
                logging.warning("Stats thread did not stop within timeout")

        if (
            self.publish_kv_cache_events_thread
            and self.publish_kv_cache_events_thread.is_alive()
        ):
            self.publish_kv_cache_events_thread.stop()
            self.publish_kv_cache_events_thread.join(timeout=cleanup_timeout)
            if self.publish_kv_cache_events_thread.is_alive():
                logging.warning("KV cache events thread did not stop within timeout")
