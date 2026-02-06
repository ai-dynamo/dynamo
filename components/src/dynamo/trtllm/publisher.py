# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM KV Event Publisher Module

This module contains the Publisher class that retrieves KV cache events from TensorRT-LLM
and publishes them either to ZMQ (for consolidator) or NATS (direct to router).

Key Components:
- ZmqKvEventPublisher: Pure Python ZMQ PUBLISHER that publishes TensorRT-LLM KV events
  to ZMQ (so the consolidator can subscribe). This is different from KvEventPublisher
  in dynamo.llm, which is a Rust-based class that can optionally subscribe from a ZMQ
  source and publishes to NATS.
- Publisher: Main class that coordinates event publishing (ZMQ or NATS) and metrics publishing.

Event Flow:
- With Consolidator: Engine → ZmqKvEventPublisher (ZMQ PUB) → Consolidator → KvEventPublisher (dynamo.llm, ZMQ SUB) → NATS → Router
- Without Consolidator: Engine → KvEventPublisher (NATS PUB) → Router
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
import traceback
import weakref
from contextlib import asynccontextmanager
from queue import Queue
from typing import Awaitable, Callable, Dict, Optional, Union

import msgpack
import zmq

from dynamo.llm import KvEventPublisher, WorkerMetricsPublisher

logging.basicConfig(level=logging.DEBUG)

# Use non-blocking RPC calls; control overhead with backoff sleeps.
_STATS_TIMEOUT_SEC = 0.01
_KV_EVENTS_TIMEOUT_SEC = 0.0
_PUBLISH_MIN_SLEEP_SEC = 0.01
_PUBLISH_MAX_SLEEP_SEC = 0.1
_PUBLISH_BACKOFF_FACTOR = 2.0
_KV_EVENTS_MIN_SLEEP_SEC = 0.005
_KV_EVENTS_MAX_SLEEP_SEC = 0.02
_KV_EVENTS_BACKOFF_FACTOR = 1.5


def _to_signed_i64(value: int | None) -> int | None:
    """Convert a Python int to signed 64-bit range by two's complement."""
    if value is None:
        return None

    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class ZmqKvEventPublisher:
    """
    Pure Python ZMQ PUBLISHER for TensorRT-LLM KV events.

    This class publishes TensorRT-LLM's KV cache events to ZMQ so that the consolidator
    can subscribe to them. This is different from KvEventPublisher in dynamo.llm,
    which is a Rust-based class that can optionally subscribe from a ZMQ source
    and publishes to NATS.

    Event Format: [timestamp, [events], data_parallel_rank]
    Message Format: multipart ZMQ message [topic, sequence, payload] where payload is
    msgpack-serialized batch.
    When attention DP is enabled for DeepSeek-style models, `data_parallel_rank` is set to the attention DP rank.
    Otherwise, it defaults to 0.

    Usage:
        Used by Publisher class when consolidator is enabled (zmq_endpoint provided).
        Publishes events from TensorRT-LLM engine to ZMQ for consolidator to consume.
    """

    def __init__(self, zmq_endpoint: str, kv_block_size: int, topic: str = ""):
        """
        Initialize ZMQ publisher.

        Args:
            zmq_endpoint: ZMQ endpoint to bind to (e.g., "tcp://*:20081")
            kv_block_size: Size of KV cache blocks in tokens
            topic: ZMQ topic to publish on (empty string for all topics)
        """
        self.zmq_endpoint = zmq_endpoint
        self.kv_block_size = kv_block_size
        self.topic = topic
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(zmq_endpoint)
        self.sequence = 0
        self.data_parallel_rank = (
            0  # TensorRT-LLM doesn't use DP for now (but does support attention DP)
        )
        logging.info(
            f"TensorRT-LLM: ZMQ KV event publisher initialized - bound to {zmq_endpoint} "
            f"with topic '{topic}', kv_block_size={kv_block_size}"
        )

    def publish_stored(
        self,
        token_ids: list[int],
        num_block_tokens: list[int],
        block_hashes: list[int],
        lora_id: int = 0,
        parent_hash: Optional[int] = None,
        attention_dp_rank: int = 0,
    ):
        """Publish a BlockStored event.

        Note: event_id is managed internally via self.sequence counter.
        """
        # Convert block hashes to signed i64 format
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]
        parent_hash_signed = (
            _to_signed_i64(parent_hash) if parent_hash is not None else None
        )

        # Create event in the same format as vLLM's ZmqEventPublisher:
        # All blocks should have the same size (kv_block_size)
        event = {
            "type": "BlockStored",
            "block_hashes": block_hashes_signed,
            "parent_block_hash": parent_hash_signed,
            "token_ids": token_ids,
            "block_size": self.kv_block_size,
            "lora_id": lora_id if lora_id != 0 else None,
        }

        self._publish_event(event, attention_dp_rank)

    def publish_removed(self, block_hashes: list[int], attention_dp_rank: int = 0):
        """Publish a BlockRemoved event.

        Note: event_id is managed internally via self.sequence counter.
        """
        # Convert block hashes to signed i64 format (vLLM compatibility)
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]

        event = {
            "type": "BlockRemoved",
            "block_hashes": block_hashes_signed,
        }

        self._publish_event(event, attention_dp_rank)

    def publish_all_cleared(self):
        """Publish an AllBlocksCleared event."""
        event = {"type": "AllBlocksCleared"}
        self._publish_event(event)

    def _publish_event(self, event: dict, attention_dp_rank: int = 0):
        """Publish a single event to ZMQ in vLLM batch format."""
        try:
            # Create batch in vLLM format: [timestamp, [events], data_parallel_rank]
            # The third element (data_parallel_rank) is used by the router for dp_rank routing
            timestamp = time.time()
            batch = [timestamp, [event], attention_dp_rank]
            event_type = event.get("type", "Unknown")
            logging.debug(
                f"TensorRT-LLM: ZMQ publisher sending {event_type} event (dp_rank={attention_dp_rank}) to {self.zmq_endpoint}"
            )

            # Serialize with msgpack (vLLM uses msgpack/rmp_serde compatible format)
            payload = msgpack.packb(batch, use_bin_type=True)

            # Create multipart message: [topic, sequence, payload]
            # Format matches what consolidator expects: 3 frames [topic, sequence, payload]
            sequence_bytes = self.sequence.to_bytes(8, byteorder="big")
            self.sequence += 1

            # Send multipart message (blocking send to ensure delivery)
            # Topic is empty string for "all topics" (vLLM compatibility)
            self.socket.send_multipart(
                [self.topic.encode(), sequence_bytes, payload], flags=0
            )
        except Exception as e:
            logging.error(f"Failed to publish ZMQ event: {e}", exc_info=True)

    def shutdown(self):
        """Shutdown the ZMQ publisher."""
        if self.socket:
            self.socket.close()
        if self.ctx:
            self.ctx.term()
        logging.info("ZMQ KV event publisher shut down")


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
    Main publisher class for TensorRT-LLM KV events and metrics.

    Retrieves KV cache events and stats from TensorRT-LLM engine and publishes them:
    - KV Events: Routes to either ZMQ (if consolidator enabled) or NATS (if no consolidator)
    - Metrics: Always publishes to NATS via WorkerMetricsPublisher

    Publisher Selection Logic:
    - If zmq_endpoint provided: Uses ZmqKvEventPublisher (ZMQ PUB) → Consolidator → NATS
    - If zmq_endpoint None: Uses KvEventPublisher (NATS PUB) → Router directly

    Note: The ZmqKvEventPublisher used here is the pure Python ZMQ publisher defined
    in this module, not the Rust-based KvEventPublisher from dynamo.llm (which is
    used in main.py as the worker-side subscriber from consolidator to NATS).
    """

    def __init__(
        self,
        component,
        engine,
        kv_listener,
        worker_id,
        kv_block_size,
        metrics_labels,
        zmq_endpoint: Optional[str] = None,
        enable_local_indexer: bool = False,
    ):
        self.component = component
        self.engine = engine
        self.kv_listener = kv_listener
        self.worker_id = worker_id
        self.kv_block_size = kv_block_size
        self.max_window_size = None
        self.metrics_labels = metrics_labels
        self.enable_local_indexer = enable_local_indexer
        self.attention_dp_size = engine.get_attention_dp_size()

        # The first few kv events from the model engine are always "created" type events.
        # Use these events to capture the max_window_size of the model.
        # When the first event that is not a "created" type is received, the publisher will set this to False to stop processing "created" type events.
        self.processing_initial_created_events = True

        # Needed by the events and metrics publishers
        self.metrics_publisher = None
        self.kv_event_publishers: Optional[
            Dict[int, KvEventPublisher]
        ] = None  # One per attention_dp_rank
        self.zmq_kv_event_publisher = None  # ZMQ publisher for consolidator
        self.publish_kv_cache_events_thread: Optional[ManagedThread] = None
        self.publish_stats_thread: Optional[ManagedThread] = None
        # A set to store the block hash of partial block (i.e. block containing less than kv_block_size tokens) hashes.
        # It is used to prevent sending remove event to kv router since partial blocks are not stored.
        self.partial_block_hashes: set[int] = set()
        self.error_queue: Queue = Queue()
        self._stop_event = threading.Event()
        # Track the last engine event_id to assert consecutive event IDs from the engine
        self._last_engine_event_id: Optional[int] = None

        # Initialize ZMQ publisher if endpoint is provided (consolidator enabled)
        if zmq_endpoint:
            logging.info(
                f"TensorRT-LLM: Initializing ZMQ KV event publisher with endpoint={zmq_endpoint}"
            )
            self.zmq_kv_event_publisher = ZmqKvEventPublisher(
                zmq_endpoint, self.kv_block_size
            )
        else:
            logging.info(
                "TensorRT-LLM: ZMQ endpoint not provided, ZMQ publisher will not be initialized"
            )

    async def _create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return
        await self.metrics_publisher.create_endpoint(self.component)

    def initialize(self):
        # Setup the metrics publisher
        self.metrics_publisher = WorkerMetricsPublisher()
        self._init_publish_metrics_thread()
        task = asyncio.create_task(self._create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

        # Setup the kv cache events publisher
        # Publisher selection based on consolidator configuration:
        # - With consolidator: Use ZmqKvEventPublisher (this module) → ZMQ → Consolidator → NATS → Router
        # - Without consolidator: Use KvEventPublisher → NATS → Router (direct)
        # Note: The worker-side KvEventPublisher (from dynamo.llm) that subscribes from
        # consolidator and publishes to NATS is created separately in main.py, not here.
        if self.zmq_kv_event_publisher:
            logging.info(
                "KV Event Consolidator enabled - using ZMQ publisher only. "
                "Consolidator will publish consolidated events to NATS."
            )
            self.kv_event_publishers = None
        else:
            # No consolidator: use NATS publisher (router subscribes directly)
            # Create one KvEventPublisher per attention_dp_rank (similar to vLLM's DP pattern)
            self.kv_event_publishers = {}
            for rank in range(self.attention_dp_size):
                self.kv_event_publishers[rank] = KvEventPublisher(
                    self.kv_listener,
                    self.worker_id,
                    self.kv_block_size,
                    dp_rank=rank,
                    enable_local_indexer=self.enable_local_indexer,
                )
            logging.info(
                f"Created {self.attention_dp_size} KV event publisher(s) for attention DP ranks"
            )

        # Always initialize the thread - it routes to either ZMQ or NATS publisher
        self._init_publish_kv_cache_events_thread()

    def _init_publish_metrics_thread(self):
        # Need to publish stats once so that worker can be selected.
        if self.metrics_publisher is None:
            logging.error("KV metrics publisher not initialized!")
            return

        # Publish initial metrics with 0 active blocks
        self.metrics_publisher.publish(None, 0)

        # Prepare threads for publishing stats but don't start them yet.
        # TRTLLM needs to start generating tokens first before stats
        # can be retrieved.
        self.publish_stats_thread = ManagedThread(
            self._publish_stats_task,
            error_queue=self.error_queue,
            name="publish_stats_thread",
        )

    def _init_publish_kv_cache_events_thread(self):
        # The _publish_kv_cache_events_task will route to the appropriate publisher
        # Prepare threads for publishing kv cache events but don't start them yet.
        # TRTLLM needs to start generating tokens first before kv cache events
        # can be retrieved.
        self.publish_kv_cache_events_thread = ManagedThread(
            self._publish_kv_cache_events_task,
            error_queue=self.error_queue,
            name="publish_kv_cache_events_thread",
        )

    async def _polling_loop(
        self,
        fetch_fn,
        handler_fn,
        min_sleep: float,
        max_sleep: float,
        backoff_factor: float,
    ):
        sleep_s = min_sleep
        while not self._stop_event.is_set():
            had_data = False
            try:
                async for item in fetch_fn():
                    had_data = True
                    handler_fn(item)
            except (asyncio.TimeoutError, TimeoutError, asyncio.QueueEmpty):
                pass
            except Exception as e:
                logging.warning(f"Publisher polling loop error: {e}", exc_info=True)

            if not had_data:
                await asyncio.sleep(sleep_s)
                sleep_s = min(max_sleep, sleep_s * backoff_factor)
            else:
                sleep_s = min_sleep

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

        def handle_stat(stat):
            kv_active_blocks = stat["kvCacheStats"]["usedNumBlocks"]
            logging.debug(f"Publishing stats: kv_active_blocks: {kv_active_blocks}")
            # TRT-LLM doesn't use data parallelism currently (dp_rank=None)
            self.metrics_publisher.publish(None, kv_active_blocks)

        await self._polling_loop(
            lambda: self.engine.llm.get_stats_async(timeout=_STATS_TIMEOUT_SEC),
            handle_stat,
            _PUBLISH_MIN_SLEEP_SEC,
            _PUBLISH_MAX_SLEEP_SEC,
            _PUBLISH_BACKOFF_FACTOR,
        )
        return True

    async def _publish_kv_cache_events_task(self):
        """
        Publish kv cache events to the events publisher.
        Routes to ZMQ (if kv event consolidation is enabled) or NATS (if no kv event consolidation).
        """
        if self.engine is None:
            logging.error("LLM engine not initialized!")
            return

        # Check that at least one publisher is available
        if not self.kv_event_publishers and self.zmq_kv_event_publisher is None:
            logging.error("No KV event publisher initialized (neither NATS nor ZMQ)!")
            return

        await self._polling_loop(
            lambda: self.engine.llm.get_kv_cache_events_async(
                timeout=_KV_EVENTS_TIMEOUT_SEC
            ),
            self._handle_kv_event,
            _KV_EVENTS_MIN_SLEEP_SEC,
            _KV_EVENTS_MAX_SLEEP_SEC,
            _KV_EVENTS_BACKOFF_FACTOR,
        )
        return True

    def _handle_kv_event(self, event):
        logging.debug(f"KV cache event received: {event}")
        # drop the events that is not emitted from the global attention layer.
        if self.should_drop_event(event):
            return

        event_id = event["event_id"]

        # Check for consecutive event IDs from the engine
        if self._last_engine_event_id is not None:
            expected_id = self._last_engine_event_id + 1
            if event_id != expected_id:
                logging.warning(
                    f"Non-consecutive engine event_id: expected {expected_id}, got {event_id}"
                )
        self._last_engine_event_id = event_id

        data = event["data"]
        if data["type"] == "stored":
            self.processing_initial_created_events = False
            parent_hash = _to_signed_i64(data["parent_hash"])
            token_ids: list[int] = []
            num_block_tokens: list[int] = []
            block_hashes: list[int] = []
            for block in data["blocks"]:
                token_num_in_block = len(block["tokens"])
                block_hash = _to_signed_i64(block["block_hash"])
                if token_num_in_block > self.kv_block_size:
                    logging.error(
                        f"Block {block_hash} contains {token_num_in_block} tokens, which is greater than kv_block_size {self.kv_block_size}"
                    )
                    return
                if block_hash is None:
                    logging.warning(
                        f"Skipping block with None hash containing {token_num_in_block} tokens"
                    )
                    continue
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

            # Get attention_dp_rank from event (TRT-LLM includes this in KVCacheEvent)
            # Default to 0 for backwards compatibility with older TRT-LLM versions
            attention_dp_rank = event.get("attention_dp_rank", 0)

            logging.debug(
                f"publish stored event: engine_event_id: {event_id}, attention_dp_rank: {attention_dp_rank}, token_ids: {token_ids}, num_block_tokens: {num_block_tokens}, block_hashes: {block_hashes}, lora_id: {lora_id}, parent_hash: {parent_hash}"
            )
            # Publish to ZMQ if consolidator is enabled, otherwise publish to NATS
            # Note: event_id is managed internally by the publisher (monotonic counter per dp_rank)
            if self.zmq_kv_event_publisher:
                # Consolidator enabled: publish to ZMQ only
                self.zmq_kv_event_publisher.publish_stored(
                    token_ids,
                    num_block_tokens,
                    block_hashes,
                    lora_id,
                    parent_hash,
                    attention_dp_rank,
                )
            elif self.kv_event_publishers:
                # No consolidator: publish to NATS (router subscribes directly)
                # Route to correct publisher based on attention_dp_rank
                publisher = self.kv_event_publishers.get(attention_dp_rank)
                if publisher:
                    publisher.publish_stored(
                        token_ids,
                        num_block_tokens,
                        block_hashes,
                        lora_id,
                        parent_hash,
                    )
                else:
                    logging.warning(
                        f"No publisher for attention_dp_rank={attention_dp_rank}, "
                        f"available ranks: {list(self.kv_event_publishers.keys())}"
                    )
        elif data["type"] == "removed":
            self.processing_initial_created_events = False
            removed_block_hashes: list[int] = []
            for block_hash in data["block_hashes"]:
                block_hash = _to_signed_i64(block_hash)
                if block_hash is None:
                    continue
                if block_hash in self.partial_block_hashes:
                    logging.debug(
                        f"Skipping removing block hash {block_hash} since it is a partial block"
                    )
                    self.partial_block_hashes.remove(block_hash)
                    continue
                removed_block_hashes.append(block_hash)

            # Get attention_dp_rank from event (TRT-LLM includes this in KVCacheEvent)
            attention_dp_rank = event.get("attention_dp_rank", 0)

            logging.debug(
                f"publish removed event: engine_event_id: {event_id}, attention_dp_rank: {attention_dp_rank}, block_hashes: {removed_block_hashes}"
            )
            # Publish to ZMQ if consolidator is enabled, otherwise publish to NATS
            # Note: event_id is managed internally by the publisher (monotonic counter per dp_rank)
            if self.zmq_kv_event_publisher:
                # Consolidator enabled: publish to ZMQ only
                self.zmq_kv_event_publisher.publish_removed(
                    removed_block_hashes, attention_dp_rank
                )
            elif self.kv_event_publishers:
                # No consolidator: publish to NATS (router subscribes directly)
                # Route to correct publisher based on attention_dp_rank
                publisher = self.kv_event_publishers.get(attention_dp_rank)
                if publisher:
                    publisher.publish_removed(removed_block_hashes)
                else:
                    logging.warning(
                        f"No publisher for attention_dp_rank={attention_dp_rank}, "
                        f"available ranks: {list(self.kv_event_publishers.keys())}"
                    )
        elif data["type"] == "created" and self.processing_initial_created_events:
            self.update_max_window_size(event)

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

        # Shutdown ZMQ publisher if it exists
        if self.zmq_kv_event_publisher:
            self.zmq_kv_event_publisher.shutdown()

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
    component,
    engine,
    kv_listener,
    worker_id,
    kv_block_size,
    metrics_labels,
    zmq_endpoint: Optional[str] = None,
    enable_local_indexer: bool = False,
):
    publisher = Publisher(
        component,
        engine,
        kv_listener,
        worker_id,
        kv_block_size,
        metrics_labels,
        zmq_endpoint=zmq_endpoint,
        enable_local_indexer=enable_local_indexer,
    )
    try:
        publisher.initialize()
        yield publisher
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await publisher.cleanup()
