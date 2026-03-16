# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InstrumentedScheduler -- vLLM Scheduler subclass that emits
ForwardPassMetrics over ZMQ PUB on every iteration.

The scheduler thread does a single-pass accumulation (count, sum,
sum_of_squares) and produces a final ForwardPassMetrics struct.
Serialization and ZMQ send are handled by a background thread
(same approach as vLLM's ZmqEventPublisher) so the scheduler
hot path only pays for accumulation + queue.put().

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
    WelfordAccumulator,
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
ENV_FPM_PORT = "DYN_FORWARDPASS_METRIC_PORT"


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
    ) -> None:
        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()
        self._worker_id = worker_id
        self._dp_rank = dp_rank

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
        last_publish = time.monotonic()

        while self._running or not self._queue.empty():
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
                payload = encode(metrics)
                seq_bytes = next(self._seq).to_bytes(8, "big")
                self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)
                last_publish = time.monotonic()
            except zmq.Again:
                pass
            except Exception:
                logger.warning("FPM publisher send failed", exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler subclass
# ---------------------------------------------------------------------------


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
        self._pending_queued: QueuedRequestMetrics | None = None
        self._prompt_len_per_req: dict[str, int] = {}

        base_port = int(os.environ.get(ENV_FPM_PORT, str(DEFAULT_FPM_PORT)))
        port = base_port + dp_rank
        self._publisher = _FpmPublisherThread(
            f"tcp://*:{port}",
            worker_id=self._fpm_worker_id,
            dp_rank=dp_rank,
        )

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

    def shutdown(self) -> None:
        self._publisher.shutdown()
        super().shutdown()

    def schedule(self) -> SchedulerOutput:
        self._schedule_time = time.monotonic()

        output = super().schedule()

        self._pending_output = output
        self._pending_queued = self._compute_queued()

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
                self._pending_queued,
                wall_time,
            )
            self._publisher.publish(metrics)

        self._pending_output = None
        self._pending_queued = None

        self._cleanup_finished(scheduler_output)

        return result

    # ------------------------------------------------------------------
    # Metric extraction (single-pass with WelfordAccumulator, no lists)
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        output: SchedulerOutput,
        queued: QueuedRequestMetrics | None,
        wall_time: float,
    ) -> ForwardPassMetrics:
        return ForwardPassMetrics(
            worker_id=self._fpm_worker_id,
            dp_rank=self._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=self._extract_scheduled(output),
            queued_requests=queued or QueuedRequestMetrics(),
        )

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

    def _compute_queued(self) -> QueuedRequestMetrics:
        """Single-pass aggregation over self.waiting -- no intermediate list."""
        prefill = WelfordAccumulator()
        decode_kv = WelfordAccumulator()

        for request in self.waiting:
            if request.status == RequestStatus.PREEMPTED:
                decode_kv.add(request.num_computed_tokens)
            else:
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
