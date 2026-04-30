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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from queue import Queue
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import msgpack
import zmq
from prometheus_client import CollectorRegistry

from dynamo.common.utils.prometheus import LLMBackendMetrics
from dynamo.llm import FpmDirectPublisher, KvEventPublisher, WorkerMetricsPublisher

logging.basicConfig(level=logging.DEBUG)

# Create a dedicated registry for dynamo_component metrics
# This ensures these metrics are isolated and can be exposed via their own callback
DYNAMO_COMPONENT_REGISTRY = CollectorRegistry()


# Use non-blocking RPC calls; control overhead with backoff sleeps.
_STATS_TIMEOUT_SEC = 0.01
_KV_EVENTS_TIMEOUT_SEC = 0.0
_PUBLISH_MIN_SLEEP_SEC = 0.01
_PUBLISH_MAX_SLEEP_SEC = 0.1
_PUBLISH_BACKOFF_FACTOR = 2.0
_KV_EVENTS_MIN_SLEEP_SEC = 0.005
_KV_EVENTS_MAX_SLEEP_SEC = 0.02
_KV_EVENTS_BACKOFF_FACTOR = 1.5

# InflightBatchingStats fields the FPM publisher consumes. As of
# NVIDIA/TensorRT-LLM#13199 (merged 2026-04-27) all 11 fields live nested
# inside iterationStats["inflightBatchingStats"]. The first-stat schema
# probe requires the nested dict to be present and to carry every key
# below; any missing field disables the publisher so we do not emit
# all-zero snapshots that the Planner would misread as "worker idle".
#
# Mapping to the Dynamo planner-level fields:
#   numContextRequests        -> scheduled_num_prefill_requests
#   numCtxTokens              -> scheduled_sum_prefill_tokens (compute this
#                                iter; excludes KV-read tokens, matches
#                                vLLM TTFT-prediction semantics)
#   numCtxKvTokens            -> scheduled_sum_prefill_kv_tokens
#   numGenRequests            -> scheduled_num_decode_requests
#   numGenKvTokens            -> scheduled_sum_decode_kv_tokens
#   numQueuedContextRequests  -> queued_num_prefill_requests
#   numQueuedCtxTokens        -> queued_sum_prefill_tokens
#   numPausedRequests
#     + numQueuedGenRequests  -> queued_num_decode_requests   (composite)
#   numPausedKvTokens
#     + numQueuedGenKvTokens  -> queued_sum_decode_kv_tokens  (composite)
# Paused-request double-counting (also present in numScheduledRequests
# upstream) is intentional: the Planner reads queued_decode as a
# KV-pressure preemption signal where paused-decodes carry the same
# semantic weight as queued-gen-only requests.
_FPM_REQUIRED_IBS_FIELDS = (
    "numContextRequests",
    "numCtxTokens",
    "numCtxKvTokens",
    "numGenRequests",
    "numGenKvTokens",
    "numQueuedContextRequests",
    "numQueuedCtxTokens",
    "numQueuedGenRequests",
    "numQueuedGenKvTokens",
    "numPausedRequests",
    "numPausedKvTokens",
)


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

    def __init__(self, zmq_endpoint: str, kv_block_size: int, topic: str = "") -> None:
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
        parent_hash: Optional[int] = None,
        block_mm_infos: Optional[list[dict | None]] = None,
        attention_dp_rank: int = 0,
        lora_name: Optional[str] = None,
    ) -> None:
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
        event: dict[str, Any] = {
            "type": "BlockStored",
            "block_hashes": block_hashes_signed,
            "parent_block_hash": parent_hash_signed,
            "token_ids": token_ids,
            "block_size": self.kv_block_size,
        }
        if lora_name is not None:
            event["lora_name"] = lora_name

        # Add multimodal info if present
        if block_mm_infos is not None:
            event["block_mm_infos"] = block_mm_infos

        self._publish_event(event, attention_dp_rank)

    def publish_removed(
        self, block_hashes: list[int], attention_dp_rank: int = 0
    ) -> None:
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

    def publish_all_cleared(self) -> None:
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

    def shutdown(self) -> None:
        """Shutdown the ZMQ publisher."""
        if self.socket:
            self.socket.close()
        if self.ctx:
            self.ctx.term()
        logging.info("ZMQ KV event publisher shut down")


class ManagedThread(threading.Thread):
    """
    A thread that runs a task and handles errors.

    Each ManagedThread owns a private asyncio event loop. Previously the thread
    submitted its coroutine to a captured request-handler loop via
    run_coroutine_threadsafe(), making publisher work compete with HTTP request
    handling on the same event loop. Now the publisher's polling work runs on
    a dedicated loop in a real OS thread, decoupled from the request loop.
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
        # `loop` is accepted for ABI compatibility but is no longer used: the
        # thread constructs and owns its own loop in run().
        self.loop = loop
        self.daemon = True
        self._owned_loop: Optional[asyncio.AbstractEventLoop] = None

        self._stop_event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        # ABI-preserving no-op; see class docstring.
        self.loop = loop

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._owned_loop = loop

        try:
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
                    coro = task(**self.kwargs)
                    if not asyncio.iscoroutine(coro):
                        logging.error(f"Task {task} did not return a coroutine")
                        break

                    loop.run_until_complete(coro)
                except (asyncio.CancelledError, concurrent.futures.CancelledError):
                    logging.debug(f"Thread {self.name} was cancelled")
                    break
                except Exception as e:
                    logging.error(
                        f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                    )
                    if self.error_queue is not None:
                        self.error_queue.put(e)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._owned_loop = None

        logging.info(f"Thread {self.name} stopped.")

    def stop(self) -> None:
        self._stop_event.set()
        # If the owned loop is still alive, schedule a task cancellation onto it
        # so any in-flight polling coroutine breaks out of `await sleep()`. This
        # is only needed when the upstream Publisher._stop_event hasn't been set
        # before stop() — normally cleanup() sets that first and the coroutine
        # exits naturally on its next iteration check.
        owned_loop = self._owned_loop
        if owned_loop is not None and not owned_loop.is_closed():
            try:
                owned_loop.call_soon_threadsafe(self._cancel_running_tasks)
            except RuntimeError:
                # Loop already stopped/closed; nothing more to do.
                pass

    def _cancel_running_tasks(self) -> None:
        """Cancel any running task on the owned loop. Runs in the loop's thread."""
        loop = self._owned_loop
        if loop is None:
            return
        for task in asyncio.all_tasks(loop):
            task.cancel()


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
        endpoint: Any,
        engine: Any,
        worker_id: Any,
        kv_block_size: int,
        metrics_labels: Any,
        component_gauges: LLMBackendMetrics,
        zmq_endpoint: Optional[str] = None,
        enable_local_indexer: bool = False,
        metrics_collector: Any = None,
    ) -> None:
        self.endpoint = endpoint
        self.engine = engine
        self.worker_id = worker_id
        self.kv_block_size = kv_block_size
        self.max_window_size = None
        self.metrics_labels = metrics_labels
        self.component_gauges = component_gauges
        self.enable_local_indexer = enable_local_indexer
        self.metrics_collector = metrics_collector
        self.attention_dp_size = engine.get_attention_dp_size()
        # FPM publisher is gated off under attention-DP (attention_dp_size > 1)
        # until per-rank FPM emission lands in a follow-up PR. Today only rank
        # 0 receives stats (rank-0-only RPC gate in rpc_worker_mixin.py), so
        # leaving the publisher on under attention-DP would make ranks 1..N-1
        # emit permanent zero-heartbeats -- the Planner would interpret those
        # as "idle workers" and take bad scaling decisions ("planner poison").
        # get_attention_dp_size() returns tensor_parallel_size iff
        # enable_attention_dp=True AND tp_size>1 (engine.py:118-120), so
        # attention_dp_size == 1 is False only when effective attention-DP
        # size > 1 -- precisely the planner-poison case we suppress.
        self.fpm_enabled = self.attention_dp_size == 1

        # The first few kv events from the model engine are always "created" type events.
        # Use these events to capture the max_window_size of the model.
        # When the first event that is not a "created" type is received, the publisher will set this to False to stop processing "created" type events.
        self.processing_initial_created_events = True

        # Needed by the events and metrics publishers
        self.metrics_publisher: Optional[WorkerMetricsPublisher] = None
        # FpmDirectPublisher: publishes ForwardPassMetrics for the planner.
        # Allocated with one per-DP-rank channel; the Rust side handles the
        # 1 s idle heartbeat internally.
        self.fpm_publisher: Optional[FpmDirectPublisher] = None
        # One-shot schema probe gate. The first IterationStats delivered to
        # handle_stat is checked against _FPM_REQUIRED_STAT_FIELDS; on mismatch
        # the publisher is shut down and None'd. Prevents silent planner poison
        # when running against a TRT-LLM version that predates #13199.
        self._fpm_schema_checked: bool = False
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
        await self.metrics_publisher.create_endpoint(self.endpoint)

    def initialize(self) -> None:
        # Setup the metrics publisher
        self.metrics_publisher = WorkerMetricsPublisher()
        self._init_publish_metrics_thread()
        task = asyncio.create_task(self._create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

        # Setup the ForwardPassMetrics publisher. Only instantiated when FPM
        # is enabled (non-attention-DP: single-rank or plain-TP). Under
        # attention-DP, self.fpm_publisher stays None and handle_stat's FPM
        # publish branch is short-circuited via the `if self.fpm_publisher is
        # not None:` guard. See the Publisher.__init__ comment on self.fpm_enabled
        # for the rationale.
        if self.fpm_enabled:
            try:
                self.fpm_publisher = FpmDirectPublisher(
                    endpoint=self.endpoint,
                    worker_id=str(self.worker_id),
                    dp_size=self.attention_dp_size,
                )
                logging.info(
                    f"FpmDirectPublisher initialized with dp_size={self.attention_dp_size}"
                )
            except RuntimeError as e:
                # PyO3 surfaces all FpmDirectPublisher::new failures as
                # PyRuntimeError (Endpoint missing, tokio runtime missing,
                # etc.). Catch only that — any other exception here would
                # signal a programming error worth surfacing.
                logging.warning(
                    f"Failed to initialize FpmDirectPublisher; FPM emission disabled: {e}"
                )
                self.fpm_publisher = None
        else:
            logging.info(
                "FPM publisher disabled under attention-DP "
                f"(effective dp_size={self.attention_dp_size}); "
                "per-rank FPM emission is a follow-up."
            )
            self.fpm_publisher = None

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
                    endpoint=self.endpoint,
                    worker_id=self.worker_id,
                    kv_block_size=self.kv_block_size,
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
        # TRT-LLM doesn't use data parallelism currently (dp_rank="0")
        self.metrics_publisher.publish(None, kv_used_blocks=0)
        self.component_gauges.set_total_blocks("0", 0)
        self.component_gauges.set_gpu_cache_usage("0", 0.0)

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

    def _check_fpm_schema(self, stat: dict) -> None:
        """One-shot probe: disable FPM publisher if the TRT-LLM IterationStats
        nested ``inflightBatchingStats`` dict is missing or incomplete.

        Runs exactly once (gated by ``self._fpm_schema_checked``). Strict: if
        the nested dict is absent or any field in ``_FPM_REQUIRED_IBS_FIELDS``
        is missing, the publisher is shut down and set to ``None`` so the
        subsequent ``if self.fpm_publisher is not None:`` short-circuit
        suppresses all FPM emission for the lifetime of this worker. This
        prevents silent planner poison when running against a TRT-LLM that
        predates NVIDIA/TensorRT-LLM#13199 — otherwise every field would
        default to 0 and the emitted snapshot would be byte-identical to the
        idle heartbeat, making the Planner treat a loaded worker as idle.
        """
        self._fpm_schema_checked = True
        if self.fpm_publisher is None:
            return
        ibs = stat.get("inflightBatchingStats")
        if not isinstance(ibs, dict):
            logging.warning(
                "TRT-LLM IterationStats has no 'inflightBatchingStats' dict; "
                "disabling FpmDirectPublisher. Upgrade TRT-LLM past "
                "NVIDIA/TensorRT-LLM#13199 to enable FPM."
            )
            self._disable_fpm_publisher()
            return
        missing = [f for f in _FPM_REQUIRED_IBS_FIELDS if f not in ibs]
        if not missing:
            return
        logging.warning(
            "TRT-LLM inflightBatchingStats is missing required FPM fields %s; "
            "disabling FpmDirectPublisher to prevent planner poison. "
            "Upgrade TRT-LLM past NVIDIA/TensorRT-LLM#13199 to enable FPM.",
            missing,
        )
        self._disable_fpm_publisher()

    def _disable_fpm_publisher(self) -> None:
        """Shut down ``self.fpm_publisher`` (best effort) and None it out."""
        publisher = self.fpm_publisher
        if publisher is None:
            return
        try:
            publisher.shutdown()
        except RuntimeError as e:
            logging.warning(
                f"FpmDirectPublisher shutdown after schema mismatch failed: {e}"
            )
        self.fpm_publisher = None

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
            kv_total_blocks = stat["kvCacheStats"]["maxNumBlocks"]
            logging.debug(f"Publishing stats: kv_active_blocks: {kv_active_blocks}")
            # TRT-LLM doesn't use data parallelism currently (dp_rank=None for NATS, "0" for Prometheus)
            assert self.metrics_publisher is not None
            self.metrics_publisher.publish(None, kv_used_blocks=kv_active_blocks)

            # Publish Prometheus metrics
            self.component_gauges.set_total_blocks("0", kv_total_blocks)

            # Calculate and publish GPU cache usage percentage
            gpu_cache_usage = (
                kv_active_blocks / kv_total_blocks if kv_total_blocks > 0 else 0.0
            )
            self.component_gauges.set_gpu_cache_usage("0", gpu_cache_usage)

            # Log iteration stats to TRT-LLM MetricsCollector (PR #11243)
            # This populates trtllm_kv_cache_hit_rate and trtllm_kv_cache_utilization gauges
            if self.metrics_collector and hasattr(
                self.metrics_collector, "log_iteration_stats"
            ):
                try:
                    self.metrics_collector.log_iteration_stats(stat)
                except Exception as e:
                    logging.warning(f"Failed to log iteration stats: {e}")

            # Publish ForwardPassMetrics. TRT-LLM tags each stat dict with
            # top-level attentionDpRank inside BaseWorker._stats_serializer;
            # under non-attention-DP it defaults to 0. The FPM source fields
            # live nested under stat["inflightBatchingStats"] (camelCase from
            # NLOHMANN serialization). Variance fields are not yet computed
            # in TRT-LLM's PyExecutor and default to 0.0 on the Rust side.
            #
            # The first stat delivered here triggers a one-shot schema probe:
            # if the nested IBS dict is missing or any required field is
            # absent (e.g. running against a TRT-LLM that predates #13199)
            # the probe shuts down the publisher, which flips the guard
            # below to short-circuit FPM emission for the rest of this
            # worker's lifetime.
            if self.fpm_publisher is not None and not self._fpm_schema_checked:
                self._check_fpm_schema(stat)
            if self.fpm_publisher is not None:
                try:
                    ibs = stat.get("inflightBatchingStats") or {}
                    # numCtxTokens is the prefill compute volume *this iter*
                    # (excludes prefix-cache/chunked carryover counted in
                    # numCtxKvTokens). Mapped to scheduled_sum_prefill_tokens
                    # to match vLLM's TTFT-prediction semantics on the
                    # planner side.
                    sched_num_prefill = int(ibs.get("numContextRequests", 0))
                    sched_sum_prefill_tokens = int(ibs.get("numCtxTokens", 0))
                    sched_sum_prefill_kv_tokens = int(ibs.get("numCtxKvTokens", 0))
                    sched_num_decode = int(ibs.get("numGenRequests", 0))
                    sched_sum_decode_kv_tokens = int(ibs.get("numGenKvTokens", 0))
                    queued_num_prefill = int(ibs.get("numQueuedContextRequests", 0))
                    queued_sum_prefill_tokens = int(ibs.get("numQueuedCtxTokens", 0))
                    # Composite: paused-decodes + queued-gen-only requests.
                    # Both represent decode work blocked from progressing
                    # this iter due to KV pressure or pending KV transfer.
                    # numPausedRequests is also counted in numScheduledRequests
                    # upstream — the double-count is intentional because the
                    # Planner reads queued_decode as a preemption-pressure
                    # signal where paused-decodes carry the same weight as
                    # queued-gen-only requests.
                    queued_num_decode = int(ibs.get("numPausedRequests", 0)) + int(
                        ibs.get("numQueuedGenRequests", 0)
                    )
                    queued_sum_decode_kv_tokens = int(
                        ibs.get("numPausedKvTokens", 0)
                    ) + int(ibs.get("numQueuedGenKvTokens", 0))
                    # iterLatencyMS is ms; the Rust snapshot expects seconds.
                    wall_time_secs = float(stat.get("iterLatencyMS", 0.0)) / 1000.0
                    self.fpm_publisher.publish(
                        dp_rank=int(stat.get("attentionDpRank", 0)),
                        scheduled_num_prefill_requests=sched_num_prefill,
                        scheduled_sum_prefill_tokens=sched_sum_prefill_tokens,
                        scheduled_sum_prefill_kv_tokens=sched_sum_prefill_kv_tokens,
                        scheduled_num_decode_requests=sched_num_decode,
                        scheduled_sum_decode_kv_tokens=sched_sum_decode_kv_tokens,
                        queued_num_prefill_requests=queued_num_prefill,
                        queued_sum_prefill_tokens=queued_sum_prefill_tokens,
                        queued_num_decode_requests=queued_num_decode,
                        queued_sum_decode_kv_tokens=queued_sum_decode_kv_tokens,
                        wall_time_secs=wall_time_secs,
                    )
                except Exception as e:
                    # Defensive (broad on purpose): the FPM publish path is
                    # cold compared to ActiveLoad/Prometheus and we'd rather
                    # drop a single FPM snapshot than poison the existing
                    # metrics pipeline if TRT-LLM ever emits an unexpected
                    # stat shape.
                    logging.warning(f"FPM publish failed: {e}")

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
            # Tighter per-block walk: inner per-token .append() loop replaced
            # with extend(comprehension); attribute lookups
            # (`self.kv_block_size`, `self.partial_block_hashes`) hoisted to
            # locals so the tight loop avoids LOAD_ATTR per iteration. mm_keys
            # handling skipped entirely when absent. For ISL=2048 / 32-token
            # blocks this trims ~64x32=2048 Python ops per prefill to ~64 ops
            # plus a single fused comprehension, reducing GIL-held time on
            # the publisher thread (which would otherwise contend with HTTP
            # request handling under load).
            self.processing_initial_created_events = False
            parent_hash = _to_signed_i64(data["parent_hash"])
            token_ids: list[int] = []
            num_block_tokens: list[int] = []
            block_hashes: list[int] = []
            block_mm_infos: list[dict | None] = []
            kv_block_size = self.kv_block_size
            partial_block_hashes = self.partial_block_hashes
            for block in data["blocks"]:
                block_tokens = block["tokens"]
                token_num_in_block = len(block_tokens)
                if token_num_in_block > kv_block_size:
                    logging.error(
                        f"Block contains {token_num_in_block} tokens, which is greater than kv_block_size {kv_block_size}"
                    )
                    return
                block_hash = _to_signed_i64(block["block_hash"])
                if block_hash is None:
                    logging.warning(
                        f"Skipping block with None hash containing {token_num_in_block} tokens"
                    )
                    continue
                if token_num_in_block < kv_block_size:
                    partial_block_hashes.add(block_hash)
                    break
                num_block_tokens.append(token_num_in_block)
                block_hashes.append(block_hash)
                token_ids.extend(int(t["token_id"]) for t in block_tokens)

                mm_keys = block.get("mm_keys")
                if mm_keys:
                    mm_hashes = [
                        int(mk["hash"][:16], 16)
                        for mk in mm_keys
                        if mk.get("type") == "mm_key" and mk.get("hash")
                    ]
                    if mm_hashes:
                        block_mm_infos.append(
                            {
                                "mm_objects": [
                                    {"mm_hash": h, "offsets": []} for h in mm_hashes
                                ]
                            }
                        )
                    else:
                        block_mm_infos.append(None)
                else:
                    block_mm_infos.append(None)

            lora_name = data.get("lora_name")

            # Get attention_dp_rank from event (TRT-LLM includes this in KVCacheEvent)
            # Default to 0 for backwards compatibility with older TRT-LLM versions
            attention_dp_rank = event.get("attention_dp_rank", 0)

            logging.debug(
                f"publish stored event: engine_event_id: {event_id}, attention_dp_rank: {attention_dp_rank}, token_ids: {token_ids}, num_block_tokens: {num_block_tokens}, block_hashes: {block_hashes}, lora_name: {lora_name}, parent_hash: {parent_hash}"
            )
            # Publish to ZMQ if consolidator is enabled, otherwise publish to NATS
            # Note: event_id is managed internally by the publisher (monotonic counter per dp_rank)
            if self.zmq_kv_event_publisher:
                # Consolidator enabled: publish to ZMQ only
                self.zmq_kv_event_publisher.publish_stored(
                    token_ids,
                    num_block_tokens,
                    block_hashes,
                    parent_hash,
                    block_mm_infos,
                    attention_dp_rank,
                    lora_name,
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
                        parent_hash,
                        block_mm_infos,
                        lora_name=lora_name,
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

    def start(self) -> None:
        # Each ManagedThread owns its own asyncio loop now, so we no longer
        # capture the request-handler loop and pass it via set_loop(). The
        # threads run their polling coroutines on private loops, off the
        # request loop.
        if (
            self.publish_kv_cache_events_thread
            and not self.publish_kv_cache_events_thread.is_alive()
        ):
            self.publish_kv_cache_events_thread.start()
            logging.debug("Started kv cache events thread")

        if self.publish_stats_thread and not self.publish_stats_thread.is_alive():
            self.publish_stats_thread.start()
            logging.debug("Started stats thread")

    def check_error_queue(self) -> Optional[Exception]:
        if not self.error_queue.empty():
            logging.error("Error in publishers error queue")
            return self.error_queue.get()
        return None

    async def cleanup(self) -> None:
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

        # Shutdown FpmDirectPublisher (stops the per-rank serialization tasks
        # and the event-plane publisher task on the Rust side). PyO3 surfaces
        # shutdown failures as PyRuntimeError; narrower catch keeps real
        # programming errors visible.
        if self.fpm_publisher is not None:
            try:
                self.fpm_publisher.shutdown()
            except RuntimeError as e:
                logging.warning(f"FpmDirectPublisher shutdown failed: {e}")

    def update_max_window_size(self, event: dict) -> None:
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
    def should_drop_event(self, event: dict) -> bool:
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
    endpoint: Any,
    engine: Any,
    worker_id: Any,
    kv_block_size: int,
    metrics_labels: Any,
    component_gauges: LLMBackendMetrics,
    zmq_endpoint: Optional[str] = None,
    enable_local_indexer: bool = False,
    metrics_collector: Any = None,
) -> AsyncGenerator[Publisher, None]:
    publisher = Publisher(
        endpoint,
        engine,
        worker_id,
        kv_block_size,
        metrics_labels,
        component_gauges=component_gauges,
        zmq_endpoint=zmq_endpoint,
        enable_local_indexer=enable_local_indexer,
        metrics_collector=metrics_collector,
    )
    try:
        publisher.initialize()
        yield publisher
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await publisher.cleanup()
