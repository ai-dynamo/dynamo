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
* **wall_time**: approximates the schedule-to-update_from_output
  latency described in ``ForwardPassMetrics``.  Measured as the time
  between consecutive ``update_from_output()`` calls.  This works
  because the EngineCore always blocks on ``future.result()`` (the
  GPU forward pass) right before calling ``update_from_output``, so
  the interval is dominated by GPU compute.  Assumption: CPU overhead
  (scheduling + output processing) between consecutive calls is small
  relative to GPU forward pass time.  ``wall_time`` is ``0.0`` for
  the first message after engine idle and for heartbeats.

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

import msgspec.structs
import zmq
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    WelfordAccumulator,
    encode,
)
from dynamo.runtime.logging import configure_dynamo_logging

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

configure_dynamo_logging()
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

        self._last_update_time: float = 0.0
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

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: "ModelRunnerOutput",
    ):
        result = super().update_from_output(scheduler_output, model_runner_output)
        now = time.monotonic()

        if scheduler_output.total_num_scheduled_tokens > 0:
            wall_time = (
                now - self._last_update_time if self._last_update_time > 0 else 0.0
            )
            self._last_update_time = now

            metrics = self._extract_metrics(
                scheduler_output, self._compute_queued(), wall_time
            )
            self._publisher.publish(metrics)
        else:
            self._last_update_time = 0.0

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
