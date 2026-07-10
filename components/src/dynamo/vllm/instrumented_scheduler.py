# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InstrumentedScheduler -- vLLM AsyncScheduler subclass that emits
ForwardPassMetrics over ZMQ PUB on every forward pass completion.

Scheduling modes
----------------
vLLM's EngineCore has two execution modes selected at startup:

* **Sync** (``batch_queue`` is None, uses ``EngineCore.step``):
  ``schedule() -> execute_model() [blocking] -> update_from_output()``
  One schedule per forward pass, CPU blocks while GPU runs.

* **Async** (``batch_queue_size=2``, uses ``step_with_batch_queue``):
  The engine overlaps scheduling with GPU execution to hide CPU overhead.
  ``schedule(N)`` is called and the batch is submitted, then the engine
  returns early.  On the next loop iteration ``schedule(N+1)`` runs
  (while the GPU is still processing batch N), then the engine blocks
  until batch N completes and calls ``update_from_output(N)``.
  This means ``schedule()`` is called **twice** before the first
  ``update_from_output()``.

  ``AsyncScheduler`` handles this by adding *output placeholders* in
  ``_update_after_schedule()``: ``num_output_placeholders += 1`` keeps
  ``num_new_tokens == 1`` for every running request, so the next
  ``schedule()`` can schedule all requests again without waiting for
  the sampled token from ``update_from_output()``.

Why we extend AsyncScheduler (not Scheduler)
---------------------------------------------
vLLM's ``--scheduler-cls`` only accepts a single class; it does not
auto-select between ``Scheduler`` and ``AsyncScheduler`` based on the
engine mode.  We extend ``AsyncScheduler`` because:

1. If we extended ``Scheduler`` (without placeholders), the second
   ``schedule()`` call in async mode would see ``num_new_tokens == 0``
   for all requests already advanced by ``_update_after_schedule``,
   producing partial batches (e.g. 22/28 split of 50 requests) with
   incorrect per-batch ``sum_decode_kv_tokens`` and other metrics.

2. ``AsyncScheduler`` is a thin wrapper (adds placeholders in
   ``_update_after_schedule`` and decrements them in
   ``_update_request_with_output``).  The placeholder logic is
   harmless in sync mode: placeholders are added and immediately
   consumed within the same step (``0 -> 1 -> 0`` per iteration).

3. A single subclass that works correctly in both sync and async
   engine modes avoids the need for mode detection or two classes.

How metrics are measured
------------------------
* **Emission point**: ``update_from_output()``, called once per
  completed GPU forward pass (after the engine pops the batch result).
  Empty batches (``total_num_scheduled_tokens == 0``) are skipped.
* **scheduled_requests**: extracted from the ``SchedulerOutput``
  parameter passed to ``update_from_output`` (the EngineCore always
  passes the correct output for the batch being processed, even in
  async mode where multiple batches are in flight).
* **queued_requests**: computed from ``self.waiting`` at emit time.
* **wall_time**: approximates the GPU forward pass time for each batch.
  In steady state, measured as the interval between consecutive
  ``update_from_output()`` calls (accurate because CPU scheduling
  overlaps with GPU execution).  For the first batch after engine idle
  (no previous ``update_from_output``), falls back to a per-batch
  ``schedule()``-to-``update_from_output()`` timestamp recorded via a
  FIFO queue.  ``wall_time`` is ``0.0`` only for heartbeats.

Serialization and ZMQ send are handled by a background thread
(same approach as vLLM's ZmqEventPublisher) so the scheduler
hot path only pays for accumulation + queue.put().

Inject via:
    --scheduler-cls "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Literal

import msgspec.structs
import zmq
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager
from vllm.v1.request import Request, RequestStatus

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    WelfordAccumulator,
    encode,
)
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.constants import MAX_BENCHMARK_GRID_POINTS

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

configure_dynamo_logging()
logger = logging.getLogger(__name__)

DEFAULT_FPM_PORT = 20380
ENV_FPM_PORT = "DYN_FORWARDPASS_METRIC_PORT"
ENV_FPM_WORKER_ID = "DYN_FPM_WORKER_ID"
ENV_FPM_BENCHMARK_OUTPUT_PATH = "DYN_FPM_BENCHMARK_OUTPUT_PATH"


# ---------------------------------------------------------------------------
# Benchmark mode dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    mode: Literal["prefill", "decode", "agg"] = "agg"
    warmup_iterations: int = 5
    output_path: str = "/tmp/benchmark_results.json"
    timeout: int = 300


class _BenchPhase(enum.Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    PREFILL_SWEEP = "prefill_sweep"
    DECODE_SWEEP = "decode_sweep"
    DONE = "done"


@dataclass
class BenchmarkPoint:
    point_type: str  # "prefill" or "decode"
    benchmark_id: int = 0
    total_prefill_tokens: int = 0
    total_kv_read_tokens: int = 0
    batch_size: int = 1
    expected_cudagraph_mode: str = "NONE"
    expected_capture_size: int | None = None
    padding_tokens: int | None = None
    sample_reasons: list[str] = field(default_factory=list)


@dataclass
class BenchmarkPointResult:
    point: BenchmarkPoint
    fpms: list = field(default_factory=list)


@dataclass
class SkippedBenchmarkPoint:
    point: BenchmarkPoint
    reason: str


def _benchmark_point_digest(point: BenchmarkPoint) -> str:
    payload = json.dumps(
        asdict(point),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _balanced_partition(
    total: int,
    count: int,
    *,
    unit: int = 1,
    minimum_units: int = 0,
) -> list[int]:
    """Split an exact total as evenly as possible in ``unit`` increments."""
    if count < 1:
        raise ValueError("count must be positive")
    if unit < 1:
        raise ValueError("unit must be positive")
    if total < 0 or total % unit != 0:
        raise ValueError("total must be a non-negative multiple of unit")

    total_units = total // unit
    required_units = count * minimum_units
    if total_units < required_units:
        raise ValueError("total is too small for the requested minimum")

    quotient, remainder = divmod(total_units - required_units, count)
    return [
        (minimum_units + quotient + int(index < remainder)) * unit
        for index in range(count)
    ]


def _powers_of_two_up_to(limit: int) -> list[int]:
    if limit < 1:
        return []
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _cudagraph_axis_points(
    capture_sizes: Sequence[int],
    limit: int,
) -> list[int]:
    """Return all ``{C, C+1}`` boundaries plus a geometric eager tail."""
    if limit < 1:
        return []

    configured_captures = sorted(
        {int(size) for size in capture_sizes if int(size) >= 1}
    )
    if not configured_captures:
        return sorted(set(_powers_of_two_up_to(limit) + [limit]))
    captures = [size for size in configured_captures if size <= limit]

    points: set[int] = set()
    for capture_size in captures:
        points.add(capture_size)
        if capture_size < limit:
            points.add(capture_size + 1)

    if configured_captures[-1] <= limit:
        tail: list[int] = []
        value = configured_captures[-1] * 2
        while value < limit:
            tail.append(value)
            value *= 2
        points.update(tail)
    points.add(limit)
    return sorted(points)


# ---------------------------------------------------------------------------
# Attention-DP benchmark synchronization
# ---------------------------------------------------------------------------


class _BenchmarkSynchronizer:
    """Align one measured benchmark iteration across attention-DP ranks.

    Rank 0 owns a ROUTER socket and every other rank owns a DEALER socket. A
    scheduler first constructs its complete measured ``SchedulerOutput``, then
    blocks here until every rank reports the same benchmark point. Rank 0 uses
    a prepare/arm handshake before sending ``GO`` so a rank that disappeared
    after READY cannot release the remaining ranks into an ADP collective. The
    scheduler records its schedule timestamp after ``GO``, so barrier wait is
    excluded from the measured iteration wall time. Small post-GO delivery and
    model-runner launch skew can remain because this operates at scheduler level.
    """

    MAX_SYNC_TIMEOUT_SECONDS = 10
    FINAL_GO_GRACE_SECONDS = 1

    def __init__(
        self,
        *,
        dp_rank: int,
        dp_size: int,
        master_ip: str,
        port: int,
        timeout: float,
        endpoint: str | None = None,
    ) -> None:
        if dp_size < 2:
            raise ValueError("benchmark synchronization requires dp_size >= 2")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.port = port
        self._timeout_ms = max(
            1,
            min(int(timeout * 1000), self.MAX_SYNC_TIMEOUT_SECONDS * 1000),
        )
        self._run_id = uuid.uuid4().hex if dp_rank == 0 else None
        self._ctx = zmq.Context.instance()

        if dp_rank == 0:
            self._socket = self._ctx.socket(zmq.ROUTER)
            self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self._endpoint = endpoint or f"tcp://*:{port}"
            self._socket.bind(self._endpoint)
        else:
            self._socket = self._ctx.socket(zmq.DEALER)
            self._socket.setsockopt(zmq.IDENTITY, str(dp_rank).encode())
            self._endpoint = endpoint or f"tcp://{master_ip}:{port}"
            self._socket.connect(self._endpoint)
        self._socket.setsockopt(zmq.LINGER, 0)

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_ms / 1000

    def close(self) -> None:
        self._socket.close(linger=0)

    def synchronize(
        self,
        point: BenchmarkPoint,
        output_summary: dict | None = None,
        validation_error: str | None = None,
    ) -> str:
        ready = {
            "type": "ready",
            "dp_rank": self.dp_rank,
            "benchmark_id": point.benchmark_id,
            "point_digest": _benchmark_point_digest(point),
            "output_summary": output_summary or {},
            "validation_error": validation_error,
        }
        if self.dp_rank == 0:
            return self._coordinate(ready)

        deadline = time.monotonic() + self.timeout_seconds
        self._socket.send_json(ready)
        self._follower_phase(deadline, point.benchmark_id, "prepare", "prepared")
        deadline = time.monotonic() + self.timeout_seconds
        self._follower_phase(deadline, point.benchmark_id, "arm", "armed")
        go_deadline = (
            time.monotonic() + self.timeout_seconds + self.FINAL_GO_GRACE_SECONDS
        )
        reply = self._recv_follower(go_deadline, point.benchmark_id, "go")
        run_id = reply.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("attention-DP benchmark GO did not include run_id")
        if self._run_id is not None and self._run_id != run_id:
            raise RuntimeError(
                "attention-DP benchmark run_id changed during the sweep: "
                f"expected={self._run_id} actual={run_id}"
            )
        self._run_id = run_id
        return run_id

    def collect_result(self, point: BenchmarkPoint, fpms: list[dict]) -> list[dict]:
        """Gather one completed iteration and return every rank's FPMs."""
        result = {
            "type": "result",
            "dp_rank": self.dp_rank,
            "benchmark_id": point.benchmark_id,
            "point_digest": _benchmark_point_digest(point),
            "fpms": fpms,
        }
        if self.dp_rank == 0:
            return self._coordinate_results(result)

        deadline = time.monotonic() + self.timeout_seconds
        self._socket.send_json(result)
        reply = self._recv_follower(deadline, point.benchmark_id, "group")
        rank_results = reply.get("rank_results")
        if not isinstance(rank_results, list):
            raise RuntimeError("attention-DP benchmark group has no rank results")
        return rank_results

    def _follower_phase(
        self,
        deadline: float,
        benchmark_id: int,
        receive_type: str,
        send_type: str,
    ) -> None:
        self._recv_follower(deadline, benchmark_id, receive_type)
        self._socket.send_json(
            {
                "type": send_type,
                "dp_rank": self.dp_rank,
                "benchmark_id": benchmark_id,
            }
        )

    def _recv_follower(
        self, deadline: float, benchmark_id: int, expected_type: str
    ) -> dict:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        if time.monotonic() >= deadline or not self._socket.poll(
            remaining_ms, zmq.POLLIN
        ):
            raise TimeoutError(
                "timed out waiting for attention-DP benchmark "
                f"{expected_type} for benchmark_id={benchmark_id}"
            )
        reply = self._socket.recv_json()
        if not isinstance(reply, dict):
            raise RuntimeError("invalid attention-DP benchmark reply")
        if reply.get("type") == "error":
            raise RuntimeError(
                "attention-DP benchmark synchronization failed: "
                f"{reply.get('error', reply)}"
            )
        if reply.get("type") != expected_type:
            raise RuntimeError(
                "attention-DP benchmark protocol mismatch: "
                f"expected={expected_type} actual={reply.get('type')}"
            )
        if reply.get("benchmark_id") != benchmark_id:
            raise RuntimeError(
                "attention-DP benchmark message id mismatch: "
                f"expected={benchmark_id} actual={reply.get('benchmark_id')}"
            )
        return reply

    def _coordinate(self, local_ready: dict) -> str:
        expected_id = local_ready["benchmark_id"]
        expected_digest = local_ready["point_digest"]
        expected_summary = local_ready["output_summary"]
        identities: dict[int, bytes] = {}
        last_identity: bytes | None = None
        validation_errors = []
        if local_ready.get("validation_error"):
            validation_errors.append(f"rank 0: {local_ready['validation_error']}")
        deadline = time.monotonic() + self.timeout_seconds

        try:
            while len(identities) < self.dp_size - 1:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                if time.monotonic() >= deadline or not self._socket.poll(
                    remaining_ms, zmq.POLLIN
                ):
                    raise TimeoutError(
                        "timed out waiting for attention-DP benchmark ranks "
                        f"for benchmark_id={expected_id}; "
                        f"ready_ranks={[0, *sorted(identities)]}"
                    )
                frames = self._socket.recv_multipart()
                if len(frames) != 2:
                    raise RuntimeError(
                        "invalid attention-DP benchmark READY multipart message"
                    )
                identity, payload = frames
                last_identity = identity
                message = json.loads(payload)
                rank = message.get("dp_rank")
                if (
                    message.get("type") != "ready"
                    or not isinstance(rank, int)
                    or not 1 <= rank < self.dp_size
                ):
                    raise RuntimeError(
                        f"invalid attention-DP benchmark READY message: {message}"
                    )
                if rank in identities:
                    raise RuntimeError(
                        f"duplicate attention-DP benchmark READY from rank {rank}"
                    )
                if identity != str(rank).encode():
                    raise RuntimeError(
                        f"attention-DP benchmark identity mismatch for rank {rank}"
                    )
                if message.get("benchmark_id") != expected_id:
                    validation_errors.append(
                        "attention-DP benchmark id mismatch: "
                        f"rank0={expected_id} rank{rank}="
                        f"{message.get('benchmark_id')}"
                    )
                if message.get("point_digest") != expected_digest:
                    validation_errors.append(
                        "attention-DP benchmark point mismatch for "
                        f"benchmark_id={expected_id} on rank {rank}"
                    )
                if message.get("output_summary") != expected_summary:
                    validation_errors.append(
                        "attention-DP benchmark SchedulerOutput mismatch for "
                        f"benchmark_id={expected_id} on rank {rank}"
                    )
                if message.get("validation_error"):
                    validation_errors.append(
                        f"rank {rank}: {message['validation_error']}"
                    )
                identities[rank] = identity
        except Exception as error:
            error_identities = set(identities.values())
            if last_identity is not None:
                error_identities.add(last_identity)
            self._notify_error(error_identities, str(error))
            raise

        if validation_errors:
            validation_message = "; ".join(validation_errors)
            self._notify_error(identities.values(), validation_message)
            raise RuntimeError(validation_message)

        assert self._run_id is not None
        try:
            self._send_to_all(
                identities,
                {"type": "prepare", "benchmark_id": expected_id},
            )
            self._coordinate_phase(
                identities,
                deadline,
                benchmark_id=expected_id,
                expected_type="prepared",
            )
            self._send_to_all(
                identities,
                {"type": "arm", "benchmark_id": expected_id},
            )
            self._coordinate_phase(
                identities,
                deadline,
                benchmark_id=expected_id,
                expected_type="armed",
            )
            self._send_to_all(
                identities,
                {
                    "type": "go",
                    "benchmark_id": expected_id,
                    "run_id": self._run_id,
                },
            )
        except Exception as error:
            self._notify_error(identities.values(), str(error))
            raise
        return self._run_id

    def _coordinate_phase(
        self,
        identities: dict[int, bytes],
        deadline: float,
        *,
        benchmark_id: int,
        expected_type: str,
    ) -> None:
        pending = set(identities)
        identity_to_rank = {
            identity: dp_rank for dp_rank, identity in identities.items()
        }
        while pending:
            identity, message = self._recv_router(deadline, benchmark_id)
            rank = identity_to_rank.get(identity)
            if rank is None or message.get("dp_rank") != rank:
                raise RuntimeError(
                    "attention-DP benchmark phase came from an unknown rank"
                )
            if rank not in pending or message.get("type") != expected_type:
                raise RuntimeError(
                    "attention-DP benchmark phase mismatch: "
                    f"rank={rank} expected={expected_type} "
                    f"actual={message.get('type')}"
                )
            pending.remove(rank)

    def _coordinate_results(self, local_result: dict) -> list[dict]:
        benchmark_id = local_result["benchmark_id"]
        point_digest = local_result["point_digest"]
        deadline = time.monotonic() + self.timeout_seconds
        identities: dict[int, bytes] = {}
        rank_results = [
            {"dp_rank": 0, "fpms": local_result["fpms"]},
        ]
        try:
            while len(identities) < self.dp_size - 1:
                identity, message = self._recv_router(deadline, benchmark_id)
                rank = message.get("dp_rank")
                if (
                    message.get("type") != "result"
                    or not isinstance(rank, int)
                    or not 1 <= rank < self.dp_size
                ):
                    raise RuntimeError(
                        f"invalid attention-DP benchmark result: {message}"
                    )
                if rank in identities or identity != str(rank).encode():
                    raise RuntimeError(
                        f"duplicate or mismatched result from ADP rank {rank}"
                    )
                if message.get("point_digest") != point_digest:
                    raise RuntimeError(
                        "attention-DP benchmark result point mismatch for "
                        f"benchmark_id={benchmark_id} on rank {rank}"
                    )
                fpms = message.get("fpms")
                if not isinstance(fpms, list):
                    raise RuntimeError(
                        f"attention-DP benchmark rank {rank} sent invalid FPMs"
                    )
                identities[rank] = identity
                rank_results.append({"dp_rank": rank, "fpms": fpms})

            rank_results.sort(key=lambda result: result["dp_rank"])
            self._send_to_all(
                identities,
                {
                    "type": "group",
                    "benchmark_id": benchmark_id,
                    "rank_results": rank_results,
                },
            )
            return rank_results
        except Exception as error:
            self._notify_error(identities.values(), str(error))
            raise

    def _recv_router(self, deadline: float, benchmark_id: int) -> tuple[bytes, dict]:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        if time.monotonic() >= deadline or not self._socket.poll(
            remaining_ms, zmq.POLLIN
        ):
            raise TimeoutError(
                "timed out waiting for attention-DP ranks for "
                f"benchmark_id={benchmark_id}"
            )
        frames = self._socket.recv_multipart()
        if len(frames) != 2:
            raise RuntimeError("invalid attention-DP benchmark multipart message")
        identity, payload = frames
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise RuntimeError("invalid attention-DP benchmark message")
        if message.get("benchmark_id") != benchmark_id:
            raise RuntimeError(
                "attention-DP benchmark id mismatch: "
                f"expected={benchmark_id} actual={message.get('benchmark_id')}"
            )
        return identity, message

    def _send_to_all(self, identities: dict[int, bytes], message: dict) -> None:
        payload = json.dumps(message).encode()
        for rank in sorted(identities):
            self._socket.send_multipart((identities[rank], payload))

    def _notify_error(self, identities, error: str) -> None:
        payload = json.dumps({"type": "error", "error": error}).encode()
        for identity in identities:
            try:
                self._socket.send_multipart((identity, payload))
            except zmq.ZMQError:
                logger.debug(
                    "Could not notify disconnected ADP benchmark rank",
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# Background publisher thread
# ---------------------------------------------------------------------------


class _FpmPublisherThread:
    """Background thread that serializes and sends ForwardPassMetrics over ZMQ.

    Also emits periodic heartbeats when idle.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    HEARTBEAT_INTERVAL: float = 1.0

    def __init__(
        self,
        endpoint: str,
        worker_id: str,
        dp_rank: int,
        max_queue_size: int = 10_000,
        start_paused: bool = False,
    ) -> None:
        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()
        self._worker_id = worker_id
        self._dp_rank = dp_rank
        self._publishing = threading.Event()
        if not start_paused:
            self._publishing.set()

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fpm-zmq-publisher"
        )
        self._thread.start()

    def publish(self, metrics: ForwardPassMetrics) -> None:
        if not self._running or not self._publishing.is_set():
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass

    def resume(self) -> None:
        """Enable live publishing after startup self-benchmarking finishes."""
        self._publishing.set()

    def shutdown(self) -> None:
        self._running = False
        self._publishing.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            self._pub.close(linger=0)
        except Exception:
            pass

    def _run(self) -> None:
        topic = b""
        last_publish = time.monotonic()

        while self._running or not self._queue.empty():
            if not self._publishing.wait(timeout=self.HEARTBEAT_INTERVAL):
                continue
            try:
                metrics = self._queue.get(timeout=self.HEARTBEAT_INTERVAL)
                if metrics is None:
                    break
            except queue.Empty:
                if time.monotonic() - last_publish >= self.HEARTBEAT_INTERVAL:
                    metrics = ForwardPassMetrics(
                        worker_id=self._worker_id,
                        dp_rank=self._dp_rank,
                    )
                else:
                    continue

            try:
                seq = next(self._seq)
                metrics = msgspec.structs.replace(metrics, counter_id=seq)
                payload = encode(metrics)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)
                last_publish = time.monotonic()
            except zmq.Again:
                pass
            except Exception:
                logger.warning("FPM publisher send failed", exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler subclass
# ---------------------------------------------------------------------------


class InstrumentedScheduler(AsyncScheduler):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            **kwargs,
        )
        self._bench_hash_block_size = (
            block_size if hash_block_size is None else hash_block_size
        )

        dp_rank = self._resolve_dp_rank(vllm_config.parallel_config)
        self._fpm_worker_id = os.environ.get(ENV_FPM_WORKER_ID, "")
        self._fpm_dp_rank = dp_rank

        self._schedule_times: deque[float] = deque()
        self._last_update_time: float = 0.0
        self._prompt_len_per_req: dict[str, int] = {}
        self._bench_active: bool = False
        self._bench_phase: _BenchPhase = _BenchPhase.IDLE
        self._bench_synchronizer: _BenchmarkSynchronizer | None = None

        base_port = int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT)))
        self._bench_init(vllm_config)

        port = base_port + dp_rank
        try:
            self._publisher = _FpmPublisherThread(
                f"tcp://*:{port}",
                worker_id=self._fpm_worker_id,
                dp_rank=dp_rank,
                start_paused=self._bench_active,
            )
        except Exception:
            if self._bench_synchronizer is not None:
                self._bench_synchronizer.close()
            raise

        logger.info(
            "InstrumentedScheduler: ZMQ PUB bound on tcp://*:%d "
            "(worker_id=%s, dp_rank=%d)",
            port,
            self._fpm_worker_id,
            dp_rank,
        )

    @staticmethod
    def _resolve_dp_rank(parallel_config) -> int:
        # ``data_parallel_index`` always holds the true global DP rank of the
        # engine process. For dense (non-MoE) models in external DP mode,
        # vLLM resets ``data_parallel_rank`` to 0 in every child process but
        # preserves ``data_parallel_index`` (see ``vllm/v1/engine/core.py``:
        # ``parallel_config.data_parallel_index = dp_rank`` then
        # ``parallel_config.data_parallel_rank = 0``). Reading the rank field
        # would make every DP child compute ``base_port + 0`` and the second
        # ``bind()`` would fail with "Address already in use".
        dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = getattr(parallel_config, "data_parallel_rank", 0) or 0
        return dp_rank

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def has_requests(self) -> bool:
        if self._bench_active:
            return True
        return super().has_requests()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if self._bench_active and self._bench_phase != _BenchPhase.IDLE:
            try:
                output = self._bench_step()
            except Exception as error:
                logger.exception("Benchmark step failed, cleaning up")
                self._bench_abort(error)
                raise
            if output is not None:
                self.kv_cache_manager.new_step_starts()
                self._update_after_schedule(output)
                try:
                    self._bench_synchronize_output(output)
                except Exception as error:
                    logger.exception("Benchmark synchronization failed")
                    self._bench_abort(error)
                    raise
                self._schedule_times.append(time.monotonic())
                return output

            if (
                self._bench_phase == _BenchPhase.DECODE_SWEEP
                and self._bench_active_req_ids
            ):
                empty = SchedulerOutput(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens={},
                    total_num_scheduled_tokens=0,
                    scheduled_spec_decode_tokens={},
                    scheduled_encoder_inputs={},
                    num_common_prefix_blocks=(
                        [0] * self.kv_cache_manager.num_kv_cache_groups
                    ),
                    finished_req_ids=self.finished_req_ids,
                    free_encoder_mm_hashes=[],
                )
                # See _bench_inject_fake_decode for the rationale; the
                # parent scheduler attaches connector metadata to every
                # SchedulerOutput when a connector is configured, so
                # benchmark-built outputs must do the same or the worker
                # asserts on bind_connector_metadata. Use direct
                # attribute access (not getattr-with-default) so a
                # future vLLM bump that drops these attributes from the
                # parent fails loudly instead of being silently masked.
                if self.connector is not None:
                    empty.kv_connector_metadata = self.connector.build_connector_meta(
                        empty
                    )
                if self.ec_connector is not None:
                    empty.ec_connector_metadata = (
                        self.ec_connector.build_connector_meta(empty)
                    )
                self._update_after_schedule(empty)
                return empty

        return self._schedule_and_record_time(throttle_prefills)

    def _schedule_and_record_time(
        self, throttle_prefills: bool = False
    ) -> SchedulerOutput:
        output = super().schedule(throttle_prefills)
        if output.total_num_scheduled_tokens > 0:
            if self._bench_active:
                try:
                    self._bench_synchronize_output(output)
                except Exception as error:
                    logger.exception("Benchmark synchronization failed")
                    self._bench_abort(error)
                    raise
            self._schedule_times.append(time.monotonic())
        return output

    def shutdown(self) -> None:
        if self._bench_active and self._bench_active_req_ids:
            logger.warning(
                "Benchmark interrupted, cleaning up %d requests",
                len(self._bench_active_req_ids),
            )
            self._bench_cleanup_requests()
        if self._bench_synchronizer is not None:
            self._bench_synchronizer.close()
            self._bench_synchronizer = None
        self._publisher.shutdown()
        super().shutdown()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: "ModelRunnerOutput",
    ):
        model_output_arrival = (
            time.monotonic() if scheduler_output.total_num_scheduled_tokens > 0 else 0.0
        )
        result = super().update_from_output(scheduler_output, model_runner_output)

        if scheduler_output.total_num_scheduled_tokens > 0:
            t_sched = self._schedule_times.popleft() if self._schedule_times else 0.0
            scheduled = self._extract_scheduled(scheduler_output)
            is_benchmark_point = self._bench_active and (
                self._bench_should_record_scheduled(scheduled)
            )
            wall_time = self._iteration_wall_time(
                model_output_arrival,
                t_sched,
                is_benchmark_point=is_benchmark_point,
            )
            self._last_update_time = model_output_arrival

            metrics = self._extract_metrics(
                scheduler_output,
                self._compute_queued(),
                wall_time,
                scheduled=scheduled,
            )
            self._publish_or_record_metrics(metrics)
        else:
            self._last_update_time = 0.0

        self._cleanup_finished(scheduler_output)
        return result

    def _publish_or_record_metrics(self, metrics: ForwardPassMetrics) -> None:
        """Keep benchmark FPMs local; publish only post-benchmark traffic."""
        if not self._bench_active:
            self._publisher.publish(metrics)
            return
        if not self._bench_should_record_fpm(metrics):
            return

        point = self._bench_current_point
        assert point is not None
        benchmark_metrics = msgspec.structs.replace(
            metrics,
            counter_id=point.benchmark_id,
        )
        self._bench_current_fpms.append(
            json.loads(msgspec.json.encode(benchmark_metrics))
        )

    # ------------------------------------------------------------------
    # Metric extraction (single-pass with WelfordAccumulator, no lists)
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        output: SchedulerOutput,
        queued: QueuedRequestMetrics | None,
        wall_time: float,
        scheduled: ScheduledRequestMetrics | None = None,
    ) -> ForwardPassMetrics:
        return ForwardPassMetrics(
            worker_id=self._fpm_worker_id,
            dp_rank=self._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=(
                scheduled if scheduled is not None else self._extract_scheduled(output)
            ),
            queued_requests=queued or QueuedRequestMetrics(),
        )

    def _iteration_wall_time(
        self,
        now: float,
        t_sched: float,
        *,
        is_benchmark_point: bool,
    ) -> float:
        if is_benchmark_point:
            return now - t_sched if t_sched > 0 else 0.0
        if self._last_update_time > 0:
            return now - self._last_update_time
        return now - t_sched if t_sched > 0 else 0.0

    def _extract_scheduled(self, output: SchedulerOutput) -> ScheduledRequestMetrics:
        new_reqs: list[NewRequestData] = output.scheduled_new_reqs
        cached: CachedRequestData = output.scheduled_cached_reqs
        num_scheduled = output.num_scheduled_tokens

        num_prefill = 0
        sum_prefill_tokens = 0
        prefill_lengths = WelfordAccumulator()
        sum_prefill_kv_tokens = 0
        decode_kv = WelfordAccumulator()

        for req in new_reqs:
            if self._bench_new_request_counts_as_decode(req.req_id):
                decode_kv.add(req.num_computed_tokens)
                continue
            num_prefill += 1
            sum_prefill_tokens += num_scheduled.get(req.req_id, 0)
            prompt_len = len(req.prompt_token_ids) if req.prompt_token_ids else 0
            prefill_lengths.add(prompt_len)
            sum_prefill_kv_tokens += req.num_computed_tokens
            self._prompt_len_per_req[req.req_id] = prompt_len

        for i, req_id in enumerate(cached.req_ids):
            if cached.is_context_phase(req_id):
                num_prefill += 1
                sum_prefill_tokens += num_scheduled.get(req_id, 0)
                prefill_lengths.add(self._prompt_len_per_req.get(req_id, 0))
                sum_prefill_kv_tokens += cached.num_computed_tokens[i]
            else:
                decode_kv.add(cached.num_computed_tokens[i])

        return ScheduledRequestMetrics(
            num_prefill_requests=num_prefill,
            sum_prefill_tokens=sum_prefill_tokens,
            var_prefill_length=prefill_lengths.variance(),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=decode_kv.n,
            sum_decode_kv_tokens=decode_kv.s,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _bench_new_request_counts_as_decode(self, req_id: str) -> bool:
        """Synthetic decode benchmark requests register as new requests."""
        return (
            getattr(self, "_bench_active", False)
            and getattr(self, "_bench_phase", _BenchPhase.IDLE)
            == _BenchPhase.DECODE_SWEEP
            and req_id in getattr(self, "_bench_active_req_ids", set())
        )

    def _bench_should_record_fpm(self, metrics: ForwardPassMetrics) -> bool:
        """Keep only the forward-pass type represented by the current point."""
        return self._bench_should_record_scheduled(metrics.scheduled_requests)

    def _bench_should_record_scheduled(
        self, scheduled: ScheduledRequestMetrics
    ) -> bool:
        point = getattr(self, "_bench_current_point", None)
        if point is None:
            return False
        if point.point_type == "prefill":
            return scheduled.num_prefill_requests > 0
        return scheduled.num_decode_requests > 0

    def _compute_queued(self) -> QueuedRequestMetrics:
        """Single-pass aggregation over ``self.waiting`` and ``self.skipped_waiting``.

        vLLM's scheduler parks requests in two queues:

        * ``self.waiting`` holds requests in ``WAITING`` (new, never scheduled)
          and ``PREEMPTED`` (were decoding, evicted back for memory) states.
        * ``self.skipped_waiting`` holds "blocked-waiting" requests awaiting an
          async precondition — see ``Scheduler._is_blocked_waiting_status`` /
          ``Scheduler._enqueue_waiting_request``:

              WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
                                          -- grammar/structured-output compile
                                          -- WAITING_FOR_FSM on older vLLM
              WAITING_FOR_REMOTE_KVS      -- disagg decode-engine KV transfer
              WAITING_FOR_STREAMING_REQ   -- streaming request handshake

        A ``WAITING_FOR_REMOTE_KVS`` request is a **decode** request: the
        prefill engine has already computed its KV and is transferring it; once
        finished the request goes straight to decode without a local prefill
        step. ``num_computed_tokens`` is pre-set to the transferred KV length
        (see ``Scheduler.schedule`` at the ``load_kv_async`` branch), so it is
        the correct decode-KV-context value for FPM purposes.

        ``WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`` /
        ``WAITING_FOR_STREAMING_REQ`` have no KV computed yet — they are queued
        prefill requests blocked on a precondition.

        Only iterating ``self.waiting`` (the previous behaviour) silently
        misses every ``WAITING_FOR_REMOTE_KVS`` request on the decode engine
        in disaggregated serving, and misclassifies it as queued prefill if it
        ever transiently appears in ``self.waiting``.
        """
        prefill = WelfordAccumulator()
        decode_kv = WelfordAccumulator()

        for request in self.waiting:
            if request.status == RequestStatus.PREEMPTED:
                decode_kv.add(request.num_computed_tokens)
            else:
                prefill.add(request.num_tokens)

        for request in self.skipped_waiting:
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Disagg decode side: KV already computed on the prefill
                # engine and being transferred. Next schedule() step will
                # start generating -- count as queued decode.
                decode_kv.add(request.num_computed_tokens)
            else:
                # Structured-output waits / WAITING_FOR_STREAMING_REQ:
                # no KV yet, essentially a queued prefill awaiting a
                # precondition.
                prefill.add(request.num_tokens)

        return QueuedRequestMetrics(
            num_prefill_requests=prefill.n,
            sum_prefill_tokens=prefill.s,
            var_prefill_length=prefill.variance(),
            num_decode_requests=decode_kv.n,
            sum_decode_kv_tokens=decode_kv.s,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    # ------------------------------------------------------------------
    # State cleanup
    # ------------------------------------------------------------------

    def _cleanup_finished(self, output: SchedulerOutput) -> None:
        for req_id in output.finished_req_ids:
            self._prompt_len_per_req.pop(req_id, None)

    # ------------------------------------------------------------------
    # Benchmark mode
    # ------------------------------------------------------------------

    def _bench_init(self, vllm_config: "VllmConfig") -> None:
        """Parse benchmark config and initialise state machine."""
        bench_cfg = vllm_config.additional_config.get("benchmark")
        if not bench_cfg:
            self._bench_active = False
            return

        cfg = bench_cfg if isinstance(bench_cfg, dict) else {}
        # additional_config values arrive as strings from JSON; coerce to
        # the types that BenchmarkConfig expects.
        _INT_FIELDS = {"warmup_iterations", "timeout"}
        for k in _INT_FIELDS:
            if k in cfg and not isinstance(cfg[k], int):
                cfg[k] = int(cfg[k])
        known = {f.name for f in BenchmarkConfig.__dataclass_fields__.values()}
        self._bench_config = BenchmarkConfig(
            **{k: v for k, v in cfg.items() if k in known}
        )
        if self._bench_config.timeout <= 0:
            raise ValueError("benchmark timeout must be positive")
        self._bench_config.output_path = os.environ.get(
            ENV_FPM_BENCHMARK_OUTPUT_PATH,
            self._bench_config.output_path,
        )

        if (
            self._bench_config.mode in {"decode", "agg"}
            and getattr(vllm_config, "speculative_config", None) is not None
        ):
            raise ValueError(
                "decode self-benchmarking does not yet support speculative "
                "decoding because its CUDA graph key also depends on the "
                "decode query length"
            )

        compilation_config = getattr(vllm_config, "compilation_config", None)
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        self._bench_cudagraph_mode = getattr(cudagraph_mode, "name", None) or "NONE"
        self._bench_cudagraph_capture_sizes = sorted(
            {
                int(size)
                for size in (
                    getattr(compilation_config, "cudagraph_capture_sizes", None) or []
                )
                if int(size) > 0
            }
        )
        self._bench_max_cudagraph_capture_size = int(
            getattr(compilation_config, "max_cudagraph_capture_size", None)
            or (
                self._bench_cudagraph_capture_sizes[-1]
                if self._bench_cudagraph_capture_sizes
                else 0
            )
        )

        mixed_mode = (
            cudagraph_mode.mixed_mode()
            if cudagraph_mode is not None
            and callable(getattr(cudagraph_mode, "mixed_mode", None))
            else cudagraph_mode
        )
        decode_mode = (
            cudagraph_mode.decode_mode()
            if cudagraph_mode is not None
            and callable(getattr(cudagraph_mode, "decode_mode", None))
            else cudagraph_mode
        )
        self._bench_prefill_cudagraph_mode = getattr(mixed_mode, "name", None) or "NONE"
        self._bench_decode_cudagraph_mode = getattr(decode_mode, "name", None) or "NONE"
        self._bench_prefill_capture_sizes = (
            list(self._bench_cudagraph_capture_sizes)
            if self._bench_prefill_cudagraph_mode != "NONE"
            else []
        )
        self._bench_decode_capture_sizes = (
            [
                size
                for size in self._bench_cudagraph_capture_sizes
                if size <= self.max_num_running_reqs
            ]
            if self._bench_decode_cudagraph_mode != "NONE"
            else []
        )

        dp_rank = self._fpm_dp_rank
        if dp_rank > 0:
            base, ext = os.path.splitext(self._bench_config.output_path)
            self._bench_config.output_path = f"{base}_dp{dp_rank}{ext}"

        try:
            os.unlink(self._bench_config.output_path)
        except FileNotFoundError:
            pass

        self._bench_active = True
        self._bench_phase = _BenchPhase.WARMUP
        self._bench_grid: deque[BenchmarkPoint] = deque()
        self._bench_current_point: BenchmarkPoint | None = None
        self._bench_results: list[BenchmarkPointResult] = []
        self._bench_iteration_groups: list[dict] = []
        self._bench_skipped_points: list[SkippedBenchmarkPoint] = []
        self._bench_missing_phases: list[str] = []
        self._bench_current_fpms: list[dict] = []
        self._bench_active_req_ids: set[str] = set()
        self._bench_seq = 0
        self._bench_grid_built = False
        self._bench_expected_points = 0
        self._bench_drain_pending = False
        self._bench_pending_seed_point: BenchmarkPoint | None = None
        self._bench_pending_seed_salts: list[str] | None = None
        self._bench_grid_error: str | None = None
        self._bench_grid_digest: str | None = None
        self._bench_feasible_max_decode_batch_size = 0
        self._bench_sync_pending = False
        self._bench_point_deadline = 0.0
        self._bench_point_result_timeout_seconds = (
            min(
                float(self._bench_config.timeout),
                float(_BenchmarkSynchronizer.MAX_SYNC_TIMEOUT_SECONDS),
            )
            * 0.8
        )

        parallel_config = vllm_config.parallel_config
        self._bench_dp_size = parallel_config.data_parallel_size
        self._bench_run_id = uuid.uuid4().hex
        if self._bench_dp_size > 1:
            sync_port = (
                int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT)))
                + self._bench_dp_size
            )
            self._bench_synchronizer = _BenchmarkSynchronizer(
                dp_rank=dp_rank,
                dp_size=self._bench_dp_size,
                master_ip=parallel_config.data_parallel_master_ip,
                port=sync_port,
                timeout=self._bench_config.timeout,
            )
            if self._bench_synchronizer.run_id is not None:
                self._bench_run_id = self._bench_synchronizer.run_id
            logger.info(
                "Attention-DP benchmark synchronization enabled: "
                "rank=%d size=%d port=%d",
                dp_rank,
                self._bench_dp_size,
                sync_port,
            )

        # Build block_hasher so benchmark requests work with prefix caching.
        if self.cache_config.enable_prefix_caching:
            caching_hash_fn = get_hash_fn_by_name(
                self.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)
            self._bench_block_hasher = get_request_block_hasher(
                self._bench_hash_block_size, caching_hash_fn
            )
        else:
            self._bench_block_hasher = None

        logger.info(
            "Benchmark mode enabled: %s (cudagraph_mode=%s, capture_sizes=%s)",
            self._bench_config,
            self._bench_cudagraph_mode,
            self._bench_cudagraph_capture_sizes,
        )

    # -- Grid generation ------------------------------------------------

    def _bench_build_grid(self) -> None:
        """Generate the sweep grid once scheduler limits are known."""
        if self._bench_grid_built:
            return
        self._bench_grid_built = True
        mode = self._bench_config.mode
        if mode in ("prefill", "agg"):
            points_before = len(self._bench_grid)
            self._bench_generate_prefill_grid()
            if len(self._bench_grid) == points_before:
                self._bench_missing_phases.append("prefill")
                logger.warning("Benchmark prefill phase generated no points")
        if mode in ("decode", "agg"):
            points_before = len(self._bench_grid)
            self._bench_generate_decode_grid()
            if len(self._bench_grid) == points_before:
                self._bench_missing_phases.append("decode")
                logger.warning("Benchmark decode phase generated no points")
        self._bench_expected_points = len(self._bench_grid)
        if self._bench_expected_points > MAX_BENCHMARK_GRID_POINTS:
            prefill_points = [
                point for point in self._bench_grid if point.point_type == "prefill"
            ]
            decode_points = [
                point for point in self._bench_grid if point.point_type == "decode"
            ]
            dimensions = (
                "prefill("
                f"P={len({p.total_prefill_tokens for p in prefill_points})}, "
                f"Rp={len({p.batch_size for p in prefill_points})}, "
                f"Kp={len({p.total_kv_read_tokens for p in prefill_points})}, "
                f"points={len(prefill_points)}); "
                "decode("
                f"D={len({p.batch_size for p in decode_points})}, "
                f"Kd={len({p.total_kv_read_tokens for p in decode_points})}, "
                f"points={len(decode_points)})"
            )
            self._bench_grid_error = (
                f"benchmark grid contains {self._bench_expected_points} points; "
                f"maximum is {MAX_BENCHMARK_GRID_POINTS}; {dimensions}. "
                "Reduce the configured engine limits, model length, KV-cache "
                "capacity, or CUDA graph capture set. Sampling points are fixed "
                "and are never dropped automatically."
            )
            logger.error(self._bench_grid_error)
            self._bench_grid.clear()
            self._bench_phase = _BenchPhase.DONE
            return

        for benchmark_id, point in enumerate(self._bench_grid, start=1):
            point.benchmark_id = benchmark_id
        grid_payload = json.dumps(
            [asdict(point) for point in self._bench_grid],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._bench_grid_digest = hashlib.sha256(grid_payload).hexdigest()
        logger.info("Benchmark grid: %d points (%s mode)", len(self._bench_grid), mode)

    def _bench_generate_prefill_grid(self) -> None:
        max_tokens = self.max_num_scheduled_tokens
        if max_tokens < 1:
            logger.warning(
                "max_num_scheduled_tokens=%d too small, skipping prefill grid",
                max_tokens,
            )
            return

        total_prefill_tokens = _cudagraph_axis_points(
            self._bench_prefill_capture_sizes,
            max_tokens,
        )
        for total_tokens in total_prefill_tokens:
            for batch_size in self._bench_prefill_batch_sizes(total_tokens):
                for total_kv_read_tokens in self._bench_prefill_kv_read_points(
                    total_tokens, batch_size
                ):
                    if not self._bench_prefill_point_feasible(
                        total_tokens, batch_size, total_kv_read_tokens
                    ):
                        continue
                    (
                        capture_size,
                        padding_tokens,
                        sample_reasons,
                    ) = self._bench_cudagraph_metadata(
                        total_tokens,
                        self._bench_prefill_capture_sizes,
                        max_tokens,
                    )
                    self._bench_grid.append(
                        BenchmarkPoint(
                            point_type="prefill",
                            total_prefill_tokens=total_tokens,
                            total_kv_read_tokens=total_kv_read_tokens,
                            batch_size=batch_size,
                            expected_cudagraph_mode=(
                                self._bench_prefill_cudagraph_mode
                                if capture_size is not None
                                else "NONE"
                            ),
                            expected_capture_size=capture_size,
                            padding_tokens=padding_tokens,
                            sample_reasons=sample_reasons,
                        )
                    )

    def _bench_prefill_batch_sizes(self, total_tokens: int) -> list[int]:
        """Return power-of-two request counts plus the legal maximum."""
        upper_bound = min(
            total_tokens,
            self.max_num_running_reqs,
            self.max_num_scheduled_tokens,
        )
        legal_batches = [
            batch_size
            for batch_size in range(1, upper_bound + 1)
            if self._bench_prefill_point_feasible(total_tokens, batch_size, 0)
        ]
        if not legal_batches:
            return []

        legal_set = set(legal_batches)
        max_batch = legal_batches[-1]
        presets = [
            value for value in _powers_of_two_up_to(max_batch) if value in legal_set
        ]
        presets.append(max_batch)
        return sorted(set(presets))

    @staticmethod
    def _bench_cudagraph_metadata(
        num_tokens: int,
        capture_sizes: Sequence[int],
        axis_limit: int,
    ) -> tuple[int | None, int | None, list[str]]:
        captures = sorted({int(size) for size in capture_sizes if int(size) > 0})
        capture_size = next((size for size in captures if size >= num_tokens), None)
        reasons: list[str] = []
        if num_tokens in captures:
            reasons.append("capture")
        if num_tokens > 1 and num_tokens - 1 in captures:
            reasons.append("post_capture")
        if not captures:
            reasons.append("cudagraph_disabled")
            if num_tokens != axis_limit:
                reasons.append("geometric_axis")
        elif num_tokens > captures[-1]:
            reasons.append("eager_tail")
            if num_tokens != axis_limit:
                reasons.append("geometric_tail")
        if num_tokens == axis_limit:
            reasons.append("engine_limit")
        padding_tokens = capture_size - num_tokens if capture_size is not None else None
        return capture_size, padding_tokens, reasons

    def _bench_prefill_scheduled_tokens_per_req(
        self, isl: int, kv_read_tokens: int
    ) -> int:
        uncached_tokens = max(1, isl - kv_read_tokens)
        threshold = getattr(
            getattr(self, "scheduler_config", None),
            "long_prefill_token_threshold",
            0,
        )
        if 0 < threshold < uncached_tokens:
            scheduled_tokens = threshold
        else:
            scheduled_tokens = uncached_tokens

        if getattr(self, "need_mamba_block_aligned_split", False):
            # Mirror vLLM's initial waiting-request branch in
            # _mamba_block_aligned_split. Hybrid align-mode prefills may round
            # an otherwise feasible chunk down to a cache-block boundary.
            block_size = (
                getattr(self.cache_config, "block_size", None) or self.block_size
            )
            last_cache_position = isl - isl % block_size
            if getattr(self.kv_cache_manager, "use_eagle", False):
                last_cache_position = max(last_cache_position - block_size, 0)
            computed_after_schedule = kv_read_tokens + scheduled_tokens
            if computed_after_schedule < last_cache_position:
                scheduled_tokens = scheduled_tokens // block_size * block_size
            elif kv_read_tokens < last_cache_position < computed_after_schedule:
                scheduled_tokens = last_cache_position - kv_read_tokens

        return scheduled_tokens

    @staticmethod
    def _bench_prefill_new_token_lengths(
        total_prefill_tokens: int, batch_size: int
    ) -> list[int]:
        return _balanced_partition(
            total_prefill_tokens,
            batch_size,
            minimum_units=1,
        )

    def _bench_prefill_kv_read_lengths(
        self, total_kv_read_tokens: int, batch_size: int
    ) -> list[int]:
        if total_kv_read_tokens == 0:
            return [0] * batch_size
        return _balanced_partition(
            total_kv_read_tokens,
            batch_size,
            unit=max(1, self._bench_hash_block_size),
            minimum_units=1,
        )

    def _bench_prefill_point_feasible(
        self,
        total_prefill_tokens: int,
        batch_size: int,
        total_kv_read_tokens: int,
    ) -> bool:
        if (
            total_prefill_tokens < 1
            or total_prefill_tokens > self.max_num_scheduled_tokens
            or batch_size < 1
            or batch_size > self.max_num_running_reqs
        ):
            return False
        try:
            new_token_lengths = self._bench_prefill_new_token_lengths(
                total_prefill_tokens, batch_size
            )
            kv_read_lengths = self._bench_prefill_kv_read_lengths(
                total_kv_read_tokens, batch_size
            )
        except ValueError:
            return False

        prompt_lengths = [
            new_tokens + kv_read_tokens
            for new_tokens, kv_read_tokens in zip(new_token_lengths, kv_read_lengths)
        ]
        if any(prompt_len + 1 > self.max_model_len for prompt_len in prompt_lengths):
            return False
        if any(
            self._bench_prefill_scheduled_tokens_per_req(prompt_len, kv_read_tokens)
            != new_tokens
            for prompt_len, kv_read_tokens, new_tokens in zip(
                prompt_lengths, kv_read_lengths, new_token_lengths
            )
        ):
            return False

        required_blocks = sum(
            self._bench_prefill_blocks_per_req(prompt_len, kv_read_tokens)
            for prompt_len, kv_read_tokens in zip(prompt_lengths, kv_read_lengths)
        )
        if total_kv_read_tokens > 0:
            seed_prompt_lengths = [
                self._bench_seed_prompt_len(kv_read_tokens)
                for kv_read_tokens in kv_read_lengths
            ]
            if any(
                prompt_len + 1 > self.max_model_len
                for prompt_len in seed_prompt_lengths
            ):
                return False
            seed_required_blocks = sum(
                self._bench_prefill_blocks_per_req(prompt_len, 0)
                for prompt_len in seed_prompt_lengths
            )
            required_blocks = max(required_blocks, seed_required_blocks)
        return required_blocks <= self._bench_usable_blocks(batch_size)

    def _bench_prefill_blocks_per_req(self, isl: int, kv_read_tokens: int) -> int:
        tokens_with_lookahead = isl + getattr(self, "num_lookahead_tokens", 0)
        return self._bench_blocks_per_req(
            tokens_with_lookahead,
            has_cache_hit=kv_read_tokens > 0,
            apply_admission_cap=True,
        )

    def _bench_blocks_per_req(
        self,
        num_tokens: int,
        *,
        has_cache_hit: bool = False,
        apply_admission_cap: bool = False,
    ) -> int:
        """Predict the shared-pool block footprint of one request."""
        coordinator = getattr(
            getattr(self, "kv_cache_manager", None), "coordinator", None
        )
        managers = getattr(coordinator, "single_type_managers", ())
        manager_blocks: list[int] = []
        for manager in managers:
            if isinstance(manager, CrossAttentionManager):
                # Synthetic decoder-only requests have no encoder tokens, and
                # the coordinator therefore asks this manager for zero blocks.
                manager_blocks.append(0)
                continue
            block_size = getattr(manager, "block_size", None)
            if not isinstance(block_size, int) or block_size < 1:
                continue

            blocks = math.ceil(num_tokens / block_size)
            admission_cap = getattr(manager, "_max_admission_blocks_per_request", None)
            if (
                apply_admission_cap
                and isinstance(admission_cap, int)
                and admission_cap > 0
            ):
                # Sliding-window and chunked-local managers recycle old blocks
                # and expose their peak per-request reservation through this
                # same cap used by vLLM's full-sequence admission check.
                blocks = min(blocks, admission_cap)

            mamba_cache_mode = getattr(manager, "mamba_cache_mode", None)
            speculative_blocks = getattr(manager, "num_speculative_blocks", 0)
            if not isinstance(speculative_blocks, int):
                speculative_blocks = 0
            if mamba_cache_mode == "align":
                # Align-mode Mamba keeps one running-state block rather than a
                # dense sequence. A cache hit also pins one cached state block.
                blocks = 1 + speculative_blocks + int(has_cache_hit)
            elif mamba_cache_mode is not None:
                blocks += speculative_blocks

            manager_blocks.append(blocks)

        if manager_blocks:
            # Hybrid layouts allocate independently from one shared physical
            # block pool for every KV-cache group.
            return sum(manager_blocks)
        return math.ceil(num_tokens / self.block_size)

    def _bench_available_blocks(self) -> int:
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        get_num_free_blocks = getattr(block_pool, "get_num_free_blocks", None)
        if callable(get_num_free_blocks):
            # The live count already excludes the null block and permanent
            # manager reservations such as sink-attention blocks.
            return max(0, int(get_num_free_blocks()))
        return max(0, int(self.cache_config.num_gpu_blocks) - 1)

    def _bench_usable_blocks(
        self, batch_size: int, *, reserve_watermark: bool = False
    ) -> int:
        available_blocks = self._bench_available_blocks()
        if reserve_watermark or batch_size > 1:
            watermark_blocks = getattr(
                getattr(self, "kv_cache_manager", None), "watermark_blocks", 0
            )
            available_blocks -= max(0, int(watermark_blocks))
        return max(0, available_blocks)

    def _bench_prefill_kv_read_points(
        self, total_prefill_tokens: int, batch_size: int
    ) -> list[int]:
        if not getattr(self.cache_config, "enable_prefix_caching", True):
            return [0]

        max_kv_read_tokens = self._bench_max_prefill_kv_read_tokens(
            total_prefill_tokens, batch_size
        )
        hash_block_size = max(1, self._bench_hash_block_size)
        max_blocks = max_kv_read_tokens // hash_block_size
        if max_blocks < batch_size:
            return [0]

        block_presets = [0, batch_size]
        block_presets.extend(
            value for value in _powers_of_two_up_to(max_blocks) if value >= batch_size
        )
        block_presets.append(max_blocks)
        return [blocks * hash_block_size for blocks in sorted(set(block_presets))]

    def _bench_max_prefill_kv_read_tokens(
        self, total_prefill_tokens: int, batch_size: int
    ) -> int:
        try:
            new_token_lengths = self._bench_prefill_new_token_lengths(
                total_prefill_tokens, batch_size
            )
        except ValueError:
            return 0

        hash_block_size = max(1, self._bench_hash_block_size)
        max_blocks = sum(
            max(0, self.max_model_len - new_tokens - 1) // hash_block_size
            for new_tokens in new_token_lengths
        )
        if max_blocks < batch_size:
            return 0

        low = batch_size
        high = max_blocks
        best = 0
        while low <= high:
            mid = (low + high) // 2
            total_kv_read_tokens = mid * hash_block_size
            if self._bench_prefill_point_feasible(
                total_prefill_tokens, batch_size, total_kv_read_tokens
            ):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best * hash_block_size

    def _bench_eagle_cache_drop_tokens(self) -> int:
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        if self._bench_uses_per_group_cache_lookup():
            return 0
        return self.block_size if getattr(kv_cache_manager, "use_eagle", False) else 0

    def _bench_seed_prompt_len(self, kv_read_tokens: int) -> int:
        # EAGLE/MTP deliberately drops the last matched cache block. Seed one
        # additional block so the measured request still reads the grid target.
        return kv_read_tokens + self._bench_eagle_cache_drop_tokens()

    def _bench_uses_per_group_cache_lookup(self) -> bool:
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        return (
            getattr(self, "connector", None) is not None
            and getattr(self, "has_mamba_layers", False)
            and hasattr(coordinator, "find_longest_cache_hit_per_group")
        )

    def _bench_cached_kv_read_tokens(self, req: Request) -> int:
        coordinator = self.kv_cache_manager.coordinator
        if self._bench_uses_per_group_cache_lookup():
            _, per_group_hits = coordinator.find_longest_cache_hit_per_group(
                req.block_hashes,
                req.num_tokens - 1,
            )
            return max(per_group_hits, default=0)
        _, cached_tokens = coordinator.find_longest_cache_hit(
            req.block_hashes,
            req.num_tokens - 1,
        )
        return cached_tokens

    def _bench_generate_decode_grid(self) -> None:
        if self.max_model_len < 2:
            logger.warning("max_model_len too small for decode grid, skipping")
            return

        min_blocks_per_request = self._bench_blocks_per_req(2)
        if min_blocks_per_request < 1:
            feasible_max_batch = self.max_num_running_reqs
        else:
            feasible_max_batch = (
                self._bench_usable_blocks(
                    self.max_num_running_reqs, reserve_watermark=True
                )
                // min_blocks_per_request
            )
        feasible_max_batch = min(
            self.max_num_running_reqs,
            self.max_num_scheduled_tokens,
            feasible_max_batch,
        )
        self._bench_feasible_max_decode_batch_size = max(0, feasible_max_batch)
        if feasible_max_batch < 1:
            logger.warning("KV cache too small for decode grid, skipping")
            return

        batch_sizes = _cudagraph_axis_points(
            self._bench_decode_capture_sizes,
            feasible_max_batch,
        )
        for batch_size in batch_sizes:
            (
                capture_size,
                padding_tokens,
                sample_reasons,
            ) = self._bench_cudagraph_metadata(
                batch_size,
                self._bench_decode_capture_sizes,
                feasible_max_batch,
            )
            for total_kv_read_tokens in self._bench_decode_kv_read_points(batch_size):
                if not self._bench_decode_point_feasible(
                    batch_size, total_kv_read_tokens
                ):
                    continue
                self._bench_grid.append(
                    BenchmarkPoint(
                        point_type="decode",
                        total_kv_read_tokens=total_kv_read_tokens,
                        batch_size=batch_size,
                        expected_cudagraph_mode=(
                            self._bench_decode_cudagraph_mode
                            if capture_size is not None
                            else "NONE"
                        ),
                        expected_capture_size=capture_size,
                        padding_tokens=padding_tokens,
                        sample_reasons=sample_reasons,
                    )
                )

    @staticmethod
    def _bench_decode_context_lengths(
        total_kv_read_tokens: int, batch_size: int
    ) -> list[int]:
        return _balanced_partition(
            total_kv_read_tokens,
            batch_size,
            minimum_units=1,
        )

    def _bench_decode_point_feasible(
        self, batch_size: int, total_kv_read_tokens: int
    ) -> bool:
        if batch_size < 1 or batch_size > self.max_num_running_reqs:
            return False
        try:
            context_lengths = self._bench_decode_context_lengths(
                total_kv_read_tokens, batch_size
            )
        except ValueError:
            return False
        if any(context_len + 1 > self.max_model_len for context_len in context_lengths):
            return False
        required_blocks = sum(
            self._bench_blocks_per_req(context_len + 1)
            for context_len in context_lengths
        )
        return required_blocks <= self._bench_usable_blocks(
            batch_size, reserve_watermark=True
        )

    def _bench_max_decode_kv_read_tokens(self, batch_size: int) -> int:
        low = batch_size
        high = batch_size * (self.max_model_len - 1)
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if self._bench_decode_point_feasible(batch_size, mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _bench_decode_kv_read_points(self, batch_size: int) -> list[int]:
        max_kv_read_tokens = self._bench_max_decode_kv_read_tokens(batch_size)
        if max_kv_read_tokens < batch_size:
            return []
        presets = [batch_size]
        presets.extend(
            value
            for value in _powers_of_two_up_to(max_kv_read_tokens)
            if value >= batch_size
        )
        presets.append(max_kv_read_tokens)
        return sorted(set(presets))

    # -- Request injection / cleanup ------------------------------------

    def _bench_inject_prefill(
        self,
        prompt_lens: Sequence[int],
        max_tokens: int,
        cache_salts: Sequence[str] | None = None,
        expected_kv_read_tokens: Sequence[int] | None = None,
    ) -> int:
        """Build and atomically enqueue a possibly heterogeneous prefill batch."""
        batch_size = len(prompt_lens)
        if cache_salts is not None and len(cache_salts) != batch_size:
            raise ValueError("cache_salts must match prompt_lens")
        if (
            expected_kv_read_tokens is not None
            and len(expected_kv_read_tokens) != batch_size
        ):
            raise ValueError("expected_kv_read_tokens must match prompt_lens")

        requests: list[Request] = []
        for index, prompt_len in enumerate(prompt_lens):
            req_id = f"__bench_{self._bench_seq + index}"
            req = Request(
                request_id=req_id,
                prompt_token_ids=[0] * prompt_len,
                sampling_params=SamplingParams(max_tokens=max_tokens),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=cache_salts[index] if cache_salts is not None else req_id,
            )

            if expected_kv_read_tokens is not None:
                expected_tokens = expected_kv_read_tokens[index]
                actual_kv_read_tokens = self._bench_cached_kv_read_tokens(req)
                if actual_kv_read_tokens != expected_tokens:
                    logger.warning(
                        "Skipping benchmark point after seed cache validation "
                        "failed: expected_kv_read_tokens=%d "
                        "actual_kv_read_tokens=%d",
                        expected_tokens,
                        actual_kv_read_tokens,
                    )
                    return 0

            requests.append(req)

        self._bench_seq += len(requests)
        for req in requests:
            self.add_request(req)
            self._bench_active_req_ids.add(req.request_id)
        return len(requests)

    def _bench_inject_fake_decode(
        self, context_lengths: Sequence[int]
    ) -> SchedulerOutput:
        """Create fake decode requests with pre-allocated KV and return
        a custom SchedulerOutput that registers them with the model runner.

        We pad each synthetic prompt to ``ctx_len + 1`` tokens (rather than
        ``ctx_len``) so the input slot at position ``ctx_len`` -- the one
        the decode iteration reads from -- is part of the request's prompt
        and therefore guaranteed to be a valid token id (0). Without this
        padding the worker's async-scheduler bookkeeping writes a ``-1``
        placeholder into ``token_ids_cpu[req_idx, ctx_len]`` after
        sampling (gpu_model_runner._update_states_after_model_execute, see
        ``sampled_ids = [-1]`` for async scheduling). When the same input
        batch slot gets reused by a later benchmark batch, that ``-1``
        is read as the input token and the embedding lookup OOBs. Padding
        by one keeps the placeholder write at position ``ctx_len + 1``
        (out of the read range) and leaves position ``ctx_len`` untouched.
        Also allocate ``ctx_len + 1`` KV slots so block-table indexing for
        position ``ctx_len`` (block ``ctx_len // block_size`` -- which is
        a NEW block when ``ctx_len % block_size == 0``) stays in range.
        """
        new_reqs_data: list[NewRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}

        for ctx_len in context_lengths:
            req_id = f"__bench_{self._bench_seq}"
            padded_len = ctx_len + 1
            prompt = [0] * padded_len
            req = Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=100_000),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=req_id,
            )

            new_blocks = self.kv_cache_manager.allocate_slots(
                req, padded_len, delay_cache_blocks=True
            )
            if new_blocks is None:
                logger.warning(
                    "KV exhausted at ctx_len=%d after %d requests, truncating batch",
                    ctx_len,
                    len(new_reqs_data),
                )
                break

            req.num_computed_tokens = ctx_len
            req.status = RequestStatus.RUNNING

            self.requests[req_id] = req
            self.running.append(req)  # type: ignore[has-type]
            self._bench_active_req_ids.add(req_id)
            self._bench_seq += 1

            block_ids = new_blocks.get_block_ids()
            new_reqs_data.append(
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    mm_features=[],
                    sampling_params=req.sampling_params,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=ctx_len,
                    lora_request=None,
                    # vLLM >=0.22's v2 GPU model runner requires `prefill_token_ids`
                    # (asserted non-None in gpu/model_runner.add_requests, used as the
                    # request's `all_token_ids`). vLLM's own scheduler passes
                    # `req._all_token_ids` for new requests; mirror that here for the
                    # synthetic decode requests we build directly. Older runners ignore it.
                    prefill_token_ids=req._all_token_ids,
                )
            )
            num_scheduled_tokens[req_id] = 1

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if getattr(self, "needs_kv_cache_zeroing", False)
            else None
        )

        output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=len(new_reqs_data),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=([0] * self.kv_cache_manager.num_kv_cache_groups),
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # Mirror the parent scheduler's connector-metadata population (see
        # vllm/v1/core/sched/scheduler.py:912-923). Without this, the
        # gpu_model_runner asserts ``scheduler_output.kv_connector_metadata
        # is not None`` whenever a KV connector is configured (e.g. the
        # NixlConnector used by disagg workers), and EngineCore dies the
        # instant the decode sweep tries to run a synthetic batch.
        # Our fake decode reqs have their KV pre-allocated via
        # ``allocate_slots`` above, so ``build_connector_meta`` produces a
        # no-op metadata -- no transfers planned, just a non-None object
        # the worker-side ``bind_connector_metadata`` can consume.
        if self.connector is not None:
            output.kv_connector_metadata = self.connector.build_connector_meta(output)
        if self.ec_connector is not None:
            output.ec_connector_metadata = self.ec_connector.build_connector_meta(
                output
            )

        return output

    def _bench_cleanup_requests(self) -> None:
        """Free all resources held by active benchmark requests."""
        for req_id in list(self._bench_active_req_ids):
            req = self.requests.get(req_id)
            if req:
                self.kv_cache_manager.free(req)
                self.finished_req_ids.add(req_id)
                del self.requests[req_id]
        running = self.running  # type: ignore[has-type]
        self.running = [
            r for r in running if r.request_id not in self._bench_active_req_ids
        ]
        self._bench_active_req_ids.clear()
        self._schedule_times.clear()

    def _bench_synchronize_output(self, output: SchedulerOutput) -> None:
        """Release one measured point only after every ADP rank is ready."""
        if not self._bench_sync_pending or output.total_num_scheduled_tokens <= 0:
            return
        point = self._bench_current_point
        if point is None:
            raise RuntimeError("benchmark synchronization has no current point")

        scheduled = self._extract_scheduled(output)
        output_summary = {
            "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
            "num_prefill_requests": scheduled.num_prefill_requests,
            "sum_prefill_tokens": scheduled.sum_prefill_tokens,
            "sum_prefill_kv_tokens": scheduled.sum_prefill_kv_tokens,
            "num_decode_requests": scheduled.num_decode_requests,
            "sum_decode_kv_tokens": scheduled.sum_decode_kv_tokens,
        }
        validation_error = self._bench_output_validation_error(point, output_summary)
        if self._bench_synchronizer is not None:
            self._bench_run_id = self._bench_synchronizer.synchronize(
                point,
                output_summary,
                validation_error,
            )
        elif validation_error is not None:
            raise RuntimeError(validation_error)
        self._bench_sync_pending = False
        self._bench_point_deadline = (
            time.monotonic() + self._bench_point_result_timeout_seconds
        )

    @staticmethod
    def _bench_output_validation_error(
        point: BenchmarkPoint, summary: dict
    ) -> str | None:
        if point.point_type == "prefill":
            expected = {
                "total_num_scheduled_tokens": point.total_prefill_tokens,
                "num_prefill_requests": point.batch_size,
                "sum_prefill_tokens": point.total_prefill_tokens,
                "sum_prefill_kv_tokens": point.total_kv_read_tokens,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
            }
        else:
            expected = {
                "total_num_scheduled_tokens": point.batch_size,
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": point.batch_size,
                "sum_decode_kv_tokens": point.total_kv_read_tokens,
            }
        if summary == expected:
            return None
        return (
            f"benchmark_id={point.benchmark_id} SchedulerOutput does not match "
            f"the point: expected={expected} actual={summary}"
        )

    def _bench_deactivate(self) -> None:
        if self._bench_synchronizer is not None:
            self._bench_synchronizer.close()
            self._bench_synchronizer = None
        self._bench_active = False
        self._bench_phase = _BenchPhase.IDLE
        self._bench_sync_pending = False
        self._schedule_times.clear()
        self._last_update_time = 0.0
        self._publisher.resume()

    def _bench_abort(self, error: Exception) -> None:
        self._bench_cleanup_requests()
        self._bench_grid_error = str(error)
        try:
            self._bench_write_results()
        except Exception:
            logger.exception("Failed to write benchmark failure results")
        self._bench_deactivate()

    # -- State machine --------------------------------------------------

    def _bench_step(self) -> SchedulerOutput | None:
        """Advance the benchmark state machine.

        Returns a custom ``SchedulerOutput`` for fake-decode points, or
        ``None`` when normal scheduling should handle the current step
        (prefill / warmup / cleanup cycles).
        """
        self._bench_build_grid()

        if self._bench_phase == _BenchPhase.WARMUP:
            return self._bench_step_warmup()
        if self._bench_phase == _BenchPhase.PREFILL_SWEEP:
            return self._bench_step_prefill()
        if self._bench_phase == _BenchPhase.DECODE_SWEEP:
            return self._bench_step_decode()
        if self._bench_phase == _BenchPhase.DONE:
            self._bench_write_results()
            self._bench_deactivate()
            logger.info("Benchmark complete")
        return None

    def _bench_step_warmup(self) -> SchedulerOutput | None:
        if not self._bench_active_req_ids:
            iters = self._bench_config.warmup_iterations
            if iters > 0:
                self._bench_inject_prefill(prompt_lens=[256], max_tokens=iters)
                logger.info("Benchmark warmup: 1 prefill + %d decode steps", iters)
            else:
                self._bench_transition_after_warmup()
            return None

        still_alive = any(rid in self.requests for rid in self._bench_active_req_ids)
        if not still_alive:
            self._bench_transition_after_warmup()
        return None

    def _bench_transition_after_warmup(self) -> None:
        self._bench_cleanup_requests()
        self._bench_current_fpms.clear()
        mode = self._bench_config.mode
        if mode in ("prefill", "agg"):
            self._bench_phase = _BenchPhase.PREFILL_SWEEP
            logger.info("Benchmark: entering PREFILL_SWEEP")
        else:
            self._bench_phase = _BenchPhase.DECODE_SWEEP
            logger.info("Benchmark: entering DECODE_SWEEP")

    def _bench_drain_if_pending(self) -> bool:
        """If a drain cycle is pending, discard stale FPMs and return True."""
        if not self._bench_drain_pending:
            return False
        self._bench_drain_pending = False
        self._bench_current_fpms.clear()
        self._schedule_times.clear()
        return True

    def _bench_step_prefill(self) -> SchedulerOutput | None:
        if self._bench_drain_if_pending():
            pass  # fall through to inject next point

        elif self._bench_active_req_ids:
            still_alive = any(
                rid in self.requests for rid in self._bench_active_req_ids
            )
            if (
                self._bench_current_point is not None
                and not self._bench_current_fpms
                and self._bench_point_result_timed_out()
            ):
                self._bench_save_current_point()
            if still_alive:
                return None
            if self._bench_pending_seed_point is not None:
                # The seed pass exists only to populate prefix cache for the
                # measured request. Drain the async pipeline before reusing its
                # cache salt, and never record the seed's forward-pass metrics.
                self._bench_cleanup_requests()
                self._bench_drain_pending = True
                return None
            self._bench_save_current_point()
            self._bench_cleanup_requests()
            self._bench_drain_pending = True
            return None

        if self._bench_pending_seed_point is not None:
            point = self._bench_pending_seed_point
            cache_salts = self._bench_pending_seed_salts
            self._bench_pending_seed_point = None
            self._bench_pending_seed_salts = None
            assert cache_salts is not None

            new_token_lengths = self._bench_prefill_new_token_lengths(
                point.total_prefill_tokens, point.batch_size
            )
            kv_read_lengths = self._bench_prefill_kv_read_lengths(
                point.total_kv_read_tokens, point.batch_size
            )
            prompt_lengths = [
                new_tokens + kv_read_tokens
                for new_tokens, kv_read_tokens in zip(
                    new_token_lengths, kv_read_lengths
                )
            ]

            self._bench_current_point = point
            self._bench_current_fpms = []
            injected = self._bench_inject_prefill(
                prompt_lens=prompt_lengths,
                max_tokens=1,
                cache_salts=cache_salts,
                expected_kv_read_tokens=kv_read_lengths,
            )
            if injected != point.batch_size:
                self._bench_current_point = None
                self._bench_skip_point(point, "seed_cache_validation_failed")
                logger.warning(
                    "Skipping benchmark prefill point after KV-read seed miss: %s",
                    point,
                )
                return None

            self._bench_sync_pending = True
            logger.info(
                "Benchmark prefill: total_tokens=%d total_kv_reads=%d batch_size=%d",
                point.total_prefill_tokens,
                point.total_kv_read_tokens,
                point.batch_size,
            )
            return None

        next_point = self._bench_pop_next("prefill")
        if next_point is None:
            if self._bench_config.mode == "agg":
                self._bench_phase = _BenchPhase.DECODE_SWEEP
                logger.info("Benchmark: entering DECODE_SWEEP")
            else:
                self._bench_phase = _BenchPhase.DONE
            return None
        point = next_point

        self._bench_current_fpms = []
        new_token_lengths = self._bench_prefill_new_token_lengths(
            point.total_prefill_tokens, point.batch_size
        )
        kv_read_lengths = self._bench_prefill_kv_read_lengths(
            point.total_kv_read_tokens, point.batch_size
        )
        if point.total_kv_read_tokens > 0:
            cache_salts = [
                f"__bench_kv_seed_{self._bench_seq}_{index}"
                for index in range(point.batch_size)
            ]
            injected = self._bench_inject_prefill(
                prompt_lens=[
                    self._bench_seed_prompt_len(kv_read_tokens)
                    for kv_read_tokens in kv_read_lengths
                ],
                max_tokens=1,
                cache_salts=cache_salts,
            )
            if injected != point.batch_size:
                self._bench_skip_point(point, "seed_injection_failed")
                logger.warning(
                    "Skipping benchmark prefill point after KV-read seed "
                    "injection failed: %s",
                    point,
                )
                return None
            self._bench_pending_seed_point = point
            self._bench_pending_seed_salts = cache_salts
            logger.info(
                "Benchmark prefill seed: total_kv_tokens=%d batch_size=%d",
                point.total_kv_read_tokens,
                point.batch_size,
            )
            return None

        self._bench_current_point = point
        injected = self._bench_inject_prefill(
            prompt_lens=new_token_lengths,
            max_tokens=1,
        )
        if injected != point.batch_size:
            self._bench_current_point = None
            self._bench_skip_point(point, "prefill_injection_failed")
            return None
        self._bench_sync_pending = True
        logger.info(
            "Benchmark prefill: total_tokens=%d total_kv_reads=0 batch_size=%d",
            point.total_prefill_tokens,
            point.batch_size,
        )
        return None

    def _bench_step_decode(self) -> SchedulerOutput | None:
        if self._bench_drain_if_pending():
            pass  # fall through to inject next point

        elif self._bench_active_req_ids:
            if not self._bench_current_fpms:
                if not self._bench_point_result_timed_out():
                    return None
            self._bench_save_current_point()
            self._bench_cleanup_requests()
            self._bench_drain_pending = True
            return None

        point = self._bench_pop_next("decode")
        if point is None:
            self._bench_phase = _BenchPhase.DONE
            return None

        self._bench_current_point = point
        self._bench_current_fpms = []
        context_lengths = self._bench_decode_context_lengths(
            point.total_kv_read_tokens, point.batch_size
        )
        logger.info(
            "Benchmark decode: total_kv_reads=%d batch_size=%d",
            point.total_kv_read_tokens,
            point.batch_size,
        )
        output = self._bench_inject_fake_decode(context_lengths)
        if output.total_num_scheduled_tokens != point.batch_size:
            logger.warning(
                "Skipping benchmark decode point after request injection produced "
                "%d of %d requests: %s",
                output.total_num_scheduled_tokens,
                point.batch_size,
                point,
            )
            self._bench_cleanup_requests()
            self._bench_skip_point(point, "decode_injection_failed")
            self._bench_current_point = None
            return None
        self._bench_sync_pending = True
        return output

    def _bench_pop_next(self, point_type: str) -> BenchmarkPoint | None:
        while self._bench_grid:
            pt = self._bench_grid[0]
            if pt.point_type == point_type:
                return self._bench_grid.popleft()
            break
        return None

    def _bench_point_result_timed_out(self) -> bool:
        return (
            self._bench_point_deadline > 0
            and time.monotonic() >= self._bench_point_deadline
        )

    def _bench_save_current_point(self) -> None:
        if self._bench_current_point is not None:
            point = self._bench_current_point
            local_fpms = list(self._bench_current_fpms)
            if self._bench_synchronizer is not None:
                rank_results = self._bench_synchronizer.collect_result(
                    point, local_fpms
                )
            else:
                rank_results = [{"dp_rank": self._fpm_dp_rank, "fpms": local_fpms}]

            expected_ranks = list(range(self._bench_dp_size))
            actual_ranks = [result.get("dp_rank") for result in rank_results]
            if actual_ranks != expected_ranks:
                raise RuntimeError(
                    "attention-DP benchmark result ranks do not match: "
                    f"expected={expected_ranks} actual={actual_ranks}"
                )

            wall_times: list[float] = []
            validation_failure: tuple[int, str] | None = None
            for result in rank_results:
                dp_rank = result["dp_rank"]
                fpms = result.get("fpms")
                if not isinstance(fpms, list) or len(fpms) != 1:
                    raise RuntimeError(
                        "each self-benchmark point must produce exactly one FPM: "
                        f"benchmark_id={point.benchmark_id} rank={dp_rank} "
                        f"count={len(fpms) if isinstance(fpms, list) else 'invalid'}"
                    )
                fpm = fpms[0]
                if fpm.get("counter_id") != point.benchmark_id:
                    raise RuntimeError(
                        "self-benchmark FPM counter mismatch: "
                        f"rank={dp_rank} benchmark_id={point.benchmark_id} "
                        f"counter_id={fpm.get('counter_id')}"
                    )
                if fpm.get("dp_rank") != dp_rank:
                    raise RuntimeError(
                        "self-benchmark FPM rank mismatch: "
                        f"result_rank={dp_rank} fpm_rank={fpm.get('dp_rank')}"
                    )
                wall_times.append(float(fpm.get("wall_time", 0.0)))
                reason = self._bench_fpm_validation_failure(point, fpm)
                if reason is not None and validation_failure is None:
                    validation_failure = (dp_rank, reason)

            if validation_failure is not None:
                dp_rank, reason = validation_failure
                logger.warning(
                    "Skipping benchmark point after measured shape mismatch "
                    "on ADP rank %d: point=%s reason=%s",
                    dp_rank,
                    point,
                    reason,
                )
                self._bench_skip_point(point, reason)
                self._bench_current_point = None
                self._bench_current_fpms = []
                self._bench_point_deadline = 0.0
                return

            self._bench_results.append(
                BenchmarkPointResult(
                    point=point,
                    fpms=local_fpms,
                )
            )
            self._bench_iteration_groups.append(
                {
                    "benchmark_id": point.benchmark_id,
                    "point": asdict(point),
                    "expected_dp_ranks": expected_ranks,
                    "complete": True,
                    "wall_time": max(wall_times, default=0.0),
                    "rank_results": rank_results,
                }
            )
        self._bench_current_point = None
        self._bench_current_fpms = []
        self._bench_point_deadline = 0.0

    @staticmethod
    def _bench_fpm_validation_failure(point: BenchmarkPoint, fpm: dict) -> str | None:
        scheduled = fpm.get("scheduled_requests", {})
        batch_size_key = (
            "num_prefill_requests"
            if point.point_type == "prefill"
            else "num_decode_requests"
        )
        if scheduled.get(batch_size_key) != point.batch_size:
            return "measured_batch_size_mismatch"
        if point.point_type == "prefill":
            if scheduled.get("sum_prefill_tokens") != point.total_prefill_tokens:
                return "measured_prefill_tokens_mismatch"
            if scheduled.get("sum_prefill_kv_tokens") != point.total_kv_read_tokens:
                return "measured_kv_read_mismatch"
        elif scheduled.get("sum_decode_kv_tokens") != point.total_kv_read_tokens:
            return "measured_decode_context_mismatch"
        return None

    def _bench_skip_point(self, point: BenchmarkPoint, reason: str) -> None:
        self._bench_skipped_points.append(
            SkippedBenchmarkPoint(point=point, reason=reason)
        )

    # -- Results output -------------------------------------------------

    def _bench_write_results(self) -> None:
        completed_points = len(self._bench_results)
        skipped_points = len(self._bench_skipped_points)
        missing_phases = list(getattr(self, "_bench_missing_phases", []))
        error = getattr(self, "_bench_grid_error", None)
        dp_size = getattr(self, "_bench_dp_size", 1)
        iteration_groups = list(getattr(self, "_bench_iteration_groups", []))
        output = {
            "schema_version": 2,
            "artifact_type": "rank",
            "status": "failed" if error is not None else "complete",
            "valid": completed_points == self._bench_expected_points
            and len(iteration_groups) == completed_points
            and all(group.get("complete") for group in iteration_groups)
            and skipped_points == 0
            and not missing_phases
            and error is None,
            "run_id": getattr(self, "_bench_run_id", None),
            "grid_digest": getattr(self, "_bench_grid_digest", None),
            "dp": {
                "rank": getattr(self, "_fpm_dp_rank", 0),
                "size": dp_size,
            },
            "synchronization": {
                "enabled": dp_size > 1,
                "coordinator_rank": 0,
                "port": (
                    int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT))) + dp_size
                    if dp_size > 1
                    else None
                ),
            },
            "coverage": {
                "expected_points": self._bench_expected_points,
                "completed_points": completed_points,
                "skipped_points": skipped_points,
            },
            "config": asdict(self._bench_config),
            "limits": {
                "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
                "max_num_running_reqs": self.max_num_running_reqs,
                "max_model_len": self.max_model_len,
                "block_size": self.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
                "configured_max_batch_size": self.max_num_running_reqs,
                "feasible_max_batch_size": getattr(
                    self, "_bench_feasible_max_decode_batch_size", 0
                ),
            },
            "cudagraph": {
                "mode": getattr(self, "_bench_cudagraph_mode", "NONE"),
                "prefill_mode": getattr(self, "_bench_prefill_cudagraph_mode", "NONE"),
                "decode_mode": getattr(self, "_bench_decode_cudagraph_mode", "NONE"),
                "max_capture_size": getattr(
                    self, "_bench_max_cudagraph_capture_size", 0
                ),
                "capture_sizes": getattr(self, "_bench_cudagraph_capture_sizes", []),
                "prefill_capture_sizes": getattr(
                    self, "_bench_prefill_capture_sizes", []
                ),
                "decode_capture_sizes": getattr(
                    self, "_bench_decode_capture_sizes", []
                ),
            },
            "results": [
                {"point": asdict(r.point), "fpms": r.fpms} for r in self._bench_results
            ],
            "iteration_groups": iteration_groups,
            "skipped_points": [
                {"point": asdict(skipped.point), "reason": skipped.reason}
                for skipped in self._bench_skipped_points
            ],
            "missing_phases": missing_phases,
            "error": error,
        }
        dest = self._bench_config.output_path
        tmp = dest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2)
        os.replace(tmp, dest)
        logger.info(
            "Benchmark results written to %s (%d points)",
            dest,
            len(self._bench_results),
        )
