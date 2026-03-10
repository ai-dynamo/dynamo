# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InstrumentedScheduler -- vLLM Scheduler subclass that emits
ForwardPassMetrics over ZMQ PUB on every iteration.

Serialization and ZMQ send are handled by a background thread
(same approach as vLLM's ZmqEventPublisher) so the scheduler
hot path only pays for a queue.put().

Inject via:
    --scheduler-cls "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from itertools import count
from typing import TYPE_CHECKING

import zmq
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    encode,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

logger = logging.getLogger(__name__)

DEFAULT_FPM_PORT = 20380
ENV_FPM_PORT = "DYN_VLLM_FORWARDPASS_METRIC_PORT"


def _population_variance(values: list[int | float]) -> float:
    if len(values) == 0:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class _FpmPublisherThread:
    """Background thread that serializes and sends ForwardPassMetrics over ZMQ.

    The scheduler thread only calls ``publish(metrics)`` which is a
    non-blocking ``queue.put``.  Serialization (msgspec) and the ZMQ
    send happen entirely in the daemon thread.
    """

    SHUTDOWN_TIMEOUT: float = 1.0

    def __init__(self, endpoint: str, max_queue_size: int = 10_000) -> None:
        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fpm-zmq-publisher"
        )
        self._thread.start()

    def publish(self, metrics: ForwardPassMetrics) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass

    def shutdown(self) -> None:
        self._running = False
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
        while self._running or not self._queue.empty():
            try:
                metrics = self._queue.get(timeout=0.1)
                if metrics is None:
                    break
            except queue.Empty:
                continue

            try:
                payload = encode(metrics)
                seq_bytes = next(self._seq).to_bytes(8, "big")
                self._pub.send_multipart(
                    (topic, seq_bytes, payload), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                pass
            except Exception:
                logger.debug("FPM publisher send failed", exc_info=True)


class InstrumentedScheduler(Scheduler):

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            **kwargs,
        )

        dp_rank = getattr(vllm_config.parallel_config, "data_parallel_rank", 0) or 0
        self._fpm_worker_id = vllm_config.additional_config.get("fpm_worker_id", "")
        self._fpm_dp_rank = dp_rank

        self._schedule_time: float = 0.0
        self._pending_output: SchedulerOutput | None = None
        self._pending_waiting_snapshot: _WaitingSnapshot | None = None
        self._was_active: bool = False
        self._prompt_len_per_req: dict[str, int] = {}

        base_port = int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT)))
        port = base_port + dp_rank
        self._publisher = _FpmPublisherThread(f"tcp://*:{port}")

        logger.info(
            "InstrumentedScheduler: ZMQ PUB bound on tcp://*:%d "
            "(worker_id=%s, dp_rank=%d)",
            port,
            self._fpm_worker_id,
            dp_rank,
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        self._was_active = True
        self._schedule_time = time.monotonic()

        output = super().schedule()

        self._pending_output = output
        self._pending_waiting_snapshot = self._snapshot_waiting()

        return output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: "ModelRunnerOutput",
    ):
        result = super().update_from_output(scheduler_output, model_runner_output)

        wall_time = time.monotonic() - self._schedule_time

        if self._pending_output is not None:
            metrics = self._extract_metrics(
                self._pending_output,
                self._pending_waiting_snapshot,
                wall_time,
            )
            self._publisher.publish(metrics)

        self._pending_output = None
        self._pending_waiting_snapshot = None

        self._cleanup_finished(scheduler_output)

        return result

    def has_requests(self) -> bool:
        has = super().has_requests()
        if not has and self._was_active:
            self._publisher.publish(
                ForwardPassMetrics(
                    worker_id=self._fpm_worker_id,
                    dp_rank=self._fpm_dp_rank,
                    wall_time=0.0,
                )
            )
            self._was_active = False
        return has

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        output: SchedulerOutput,
        waiting: _WaitingSnapshot | None,
        wall_time: float,
    ) -> ForwardPassMetrics:
        scheduled = self._extract_scheduled(output)
        queued = self._extract_queued(waiting)
        return ForwardPassMetrics(
            worker_id=self._fpm_worker_id,
            dp_rank=self._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=scheduled,
            queued_requests=queued,
        )

    def _extract_scheduled(self, output: SchedulerOutput) -> ScheduledRequestMetrics:
        new_reqs: list[NewRequestData] = output.scheduled_new_reqs
        cached: CachedRequestData = output.scheduled_cached_reqs
        num_scheduled = output.num_scheduled_tokens

        prefill_token_counts: list[int] = []
        prefill_prompt_lengths: list[int] = []
        sum_prefill_kv_tokens = 0

        for req in new_reqs:
            tokens_this_step = num_scheduled.get(req.req_id, 0)
            prefill_token_counts.append(tokens_this_step)

            prompt_len = len(req.prompt_token_ids) if req.prompt_token_ids else 0
            prefill_prompt_lengths.append(prompt_len)
            sum_prefill_kv_tokens += req.num_computed_tokens

            self._prompt_len_per_req[req.req_id] = prompt_len

        for i, req_id in enumerate(cached.req_ids):
            if cached.is_context_phase(req_id):
                tokens_this_step = num_scheduled.get(req_id, 0)
                prefill_token_counts.append(tokens_this_step)

                prompt_len = self._prompt_len_per_req.get(req_id, 0)
                prefill_prompt_lengths.append(prompt_len)
                sum_prefill_kv_tokens += cached.num_computed_tokens[i]

        decode_kv_lengths: list[int] = []

        for i, req_id in enumerate(cached.req_ids):
            if not cached.is_context_phase(req_id):
                decode_kv_lengths.append(cached.num_computed_tokens[i])

        return ScheduledRequestMetrics(
            num_prefill_requests=len(prefill_token_counts),
            sum_prefill_tokens=sum(prefill_token_counts),
            var_prefill_length=_population_variance(prefill_prompt_lengths),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=len(decode_kv_lengths),
            sum_decode_kv_tokens=sum(decode_kv_lengths),
            var_decode_kv_tokens=_population_variance(decode_kv_lengths),
        )

    def _extract_queued(
        self, snapshot: _WaitingSnapshot | None
    ) -> QueuedRequestMetrics:
        if snapshot is None:
            return QueuedRequestMetrics()

        prefill_lengths: list[int] = []
        decode_kv_lengths: list[int] = []

        for status, num_tokens, num_computed in snapshot.entries:
            if status == RequestStatus.PREEMPTED:
                decode_kv_lengths.append(num_computed)
            else:
                prefill_lengths.append(num_tokens)

        return QueuedRequestMetrics(
            num_prefill_requests=len(prefill_lengths),
            sum_prefill_tokens=sum(prefill_lengths),
            var_prefill_length=_population_variance(prefill_lengths),
            num_decode_requests=len(decode_kv_lengths),
            sum_decode_kv_tokens=sum(decode_kv_lengths),
            var_decode_kv_tokens=_population_variance(decode_kv_lengths),
        )

    # ------------------------------------------------------------------
    # Waiting queue snapshot
    # ------------------------------------------------------------------

    def _snapshot_waiting(self) -> _WaitingSnapshot:
        entries: list[tuple[RequestStatus, int, int]] = []
        for request in self.waiting:
            entries.append(
                (request.status, request.num_tokens, request.num_computed_tokens)
            )
        return _WaitingSnapshot(entries)

    # ------------------------------------------------------------------
    # State cleanup
    # ------------------------------------------------------------------

    def _cleanup_finished(self, output: SchedulerOutput) -> None:
        for req_id in output.finished_req_ids:
            self._prompt_len_per_req.pop(req_id, None)


class _WaitingSnapshot:
    """Lightweight snapshot of the waiting queue state."""

    __slots__ = ("entries",)

    def __init__(self, entries: list[tuple[RequestStatus, int, int]]) -> None:
        self.entries = entries
