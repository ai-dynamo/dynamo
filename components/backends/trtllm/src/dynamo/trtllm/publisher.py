# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import logging
import threading
import traceback
import weakref
from contextlib import asynccontextmanager
from queue import Queue
from typing import Awaitable, Callable, Optional, Union

from dynamo.llm import (
    ForwardPassMetrics,
    KvEventPublisher,
    KvStats,
    WorkerMetricsPublisher,
    WorkerStats,
)

logging.basicConfig(level=logging.DEBUG)


def _to_signed_i64(value: int | None) -> int | None:
    """Convert a Python int to signed 64-bit range by two's complement."""
    if value is None:
        return None

    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class ManagedThread(threading.Thread):
    """
    A thread that runs a task and handles errors.
    """

    def __init__(
        self,
        task: Optional[Union[Callable[..., Awaitable[bool]], weakref.WeakMethod]],
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
            task: Optional[
                Union[Callable[..., Awaitable[bool]], weakref.WeakMethod]
            ] = self.task
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

                # Call the task function to get the coroutine
                coro = task(**self.kwargs)
                if not asyncio.iscoroutine(coro):
                    logging.error(f"Task {task} did not return a coroutine")
                    break

                self._current_future = asyncio.run_coroutine_threadsafe(coro, self.loop)
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


class Publisher:
    """
    A class to retrieve stats and kv cache events from TRTLLM engine and publish them to the metrics and events publishers.
    """

    def __init__(
        self, component, engine, kv_listener, worker_id, kv_block_size, metrics_labels
    ):
        self.component = component
        self.engine = engine
        self.kv_listener = kv_listener
        self.worker_id = worker_id
        self.kv_block_size = kv_block_size
        self.max_window_size = None
        self.metrics_labels = metrics_labels

        # The first few kv events from the model engine are always "created" type events.
        # Use these events to capture the max_window_size of the model.
        # When the first event that is not a "created" type is received, the publisher will set this to False to stop processing "created" type events.
        self.processing_initial_created_events = True

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

    async def _create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return
        await self.metrics_publisher.create_endpoint(
            self.component, self.metrics_labels
        )

    def initialize(self):
        # Setup the metrics publisher
        self.metrics_publisher = WorkerMetricsPublisher()
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

        worker_stats = WorkerStats(
            request_active_slots=request_active_slots,
            request_total_slots=request_total_slots,
            num_requests_waiting=num_requests_waiting,
            data_parallel_rank=None,
        )

        kv_stats = KvStats(
            kv_active_blocks=kv_active_block,
            kv_total_blocks=kv_total_blocks,
            gpu_cache_usage_perc=gpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,
        )

        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=None,
        )
        self.metrics_publisher.publish(metrics)

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

            worker_stats = WorkerStats(
                request_active_slots=request_active_slots,
                request_total_slots=request_total_slots,
                num_requests_waiting=num_requests_waiting,
                data_parallel_rank=None,
            )

            kv_stats = KvStats(
                kv_active_blocks=kv_active_block,
                kv_total_blocks=kv_total_blocks,
                gpu_cache_usage_perc=gpu_cache_usage_perc,
                gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,
            )

            # TODO: get spec_decode_stats from engine
            spec_decode_stats = None

            metrics = ForwardPassMetrics(
                worker_stats=worker_stats,
                kv_stats=kv_stats,
                spec_decode_stats=spec_decode_stats,
            )
            self.metrics_publisher.publish(metrics)

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
            # drop the events that is not emitted from the global attention layer.
            if self.should_drop_event(event):
                continue

            event_id = event["event_id"]
            data = event["data"]
            if data["type"] == "stored":
                self.processing_initial_created_events = False
                parent_hash = _to_signed_i64(data["parent_hash"])
                token_ids = []
                num_block_tokens = []
                block_hashes = []
                for block in data["blocks"]:
                    token_num_in_block = len(block["tokens"])
                    block_hash = _to_signed_i64(block["block_hash"])
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
                self.processing_initial_created_events = False
                block_hashes = []
                for block_hash in data["block_hashes"]:
                    block_hash = _to_signed_i64(block_hash)
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
            elif data["type"] == "created" and self.processing_initial_created_events:
                self.update_max_window_size(event)

        return True

    def start(self):
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

    def update_max_window_size(self, event):
        if "window_size" in event:
            window_size = event["window_size"]
            if self.max_window_size is None or window_size > self.max_window_size:
                self.max_window_size = window_size
                logging.debug(
                    f"kv events max_window_size has been updated to {self.max_window_size}"
                )

    # The global attention layer will emit the KV event with the max_window_size.
    # We only want to keep the KV event that has the max_window_size to ensure
    # the accuracy of KV routing.
    # TRTLLM emits a "created" event at the very beginning when it creates the KV cache,
    # so we can use the "created" event to identify the max_window_size of the global
    # attention layer in the model engine.
    def should_drop_event(self, event):
        # There are two cases for KV event filtering:
        #
        # 1. If "window_size" is NOT in the KV event:
        #    "window_size" was added to KV events only recently, so some older versions of TRTLLM
        #    might not include it. In this case, the publisher will assume that all events are
        #    from the global attention layer.
        #
        # 2. If "window_size" is present in the KV event:
        #    The publisher will not drop any KV events until all initial "created" KV events
        #    have been processed in order to capture the max_window_size.
        #    After processing all "created" events, the publisher will only accept KV events
        #    whose window_size is equal to the max_window_size to ensure accurate routing.
        if "window_size" not in event or self.processing_initial_created_events:
            return False

        if event["window_size"] != self.max_window_size:
            return True

        return False


@asynccontextmanager
async def get_publisher(
    component, engine, kv_listener, worker_id, kv_block_size, metrics_labels
):
    publisher = Publisher(
        component, engine, kv_listener, worker_id, kv_block_size, metrics_labels
    )
    try:
        publisher.initialize()
        yield publisher
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await publisher.cleanup()
