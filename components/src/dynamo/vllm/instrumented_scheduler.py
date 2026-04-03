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
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Literal

import msgspec.structs
import numpy as np
import zmq
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

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
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

configure_dynamo_logging()
logger = logging.getLogger(__name__)

DEFAULT_FPM_PORT = 20380
ENV_FPM_PORT = "DYN_FORWARDPASS_METRIC_PORT"


# ---------------------------------------------------------------------------
# Benchmark mode dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    mode: Literal["prefill", "decode", "agg"] = "agg"
    prefill_isl_granularity: int = 16
    decode_length_granularity: int = 6
    decode_batch_size_granularity: int = 6
    warmup_iterations: int = 5
    output_path: str = "/tmp/benchmark_results.json"


class _BenchPhase(enum.Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    PREFILL_SWEEP = "prefill_sweep"
    DECODE_SWEEP = "decode_sweep"
    DONE = "done"


@dataclass
class BenchmarkPoint:
    point_type: str  # "prefill" or "decode"
    isl: int = 0
    context_length: int = 0
    batch_size: int = 0


@dataclass
class BenchmarkPointResult:
    point: BenchmarkPoint
    fpms: list = field(default_factory=list)


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

        self._schedule_times: deque[float] = deque()
        self._last_update_time: float = 0.0
        self._prompt_len_per_req: dict[str, int] = {}
        self._bench_active: bool = False
        self._bench_phase: _BenchPhase = _BenchPhase.IDLE

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

        self._bench_init(vllm_config)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def has_requests(self) -> bool:
        if self._bench_active:
            return True
        return super().has_requests()

    def schedule(self) -> SchedulerOutput:
        if self._bench_active and self._bench_phase != _BenchPhase.IDLE:
            try:
                output = self._bench_step()
            except Exception:
                logger.exception("Benchmark step failed, cleaning up")
                self._bench_cleanup_requests()
                self._bench_active = False
                self._bench_phase = _BenchPhase.IDLE
                return self._schedule_and_record_time()
            if output is not None:
                self.kv_cache_manager.new_step_starts()
                self._update_after_schedule(output)
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
                self._update_after_schedule(empty)
                return empty

        return self._schedule_and_record_time()

    def _schedule_and_record_time(self) -> SchedulerOutput:
        output = super().schedule()
        if output.total_num_scheduled_tokens > 0:
            self._schedule_times.append(time.monotonic())
        return output

    def shutdown(self) -> None:
        if self._bench_active and self._bench_active_req_ids:
            logger.warning(
                "Benchmark interrupted, cleaning up %d requests",
                len(self._bench_active_req_ids),
            )
            self._bench_cleanup_requests()
        self._publisher.shutdown()
        super().shutdown()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: "ModelRunnerOutput",
    ):
        result = super().update_from_output(scheduler_output, model_runner_output)

        if scheduler_output.total_num_scheduled_tokens > 0:
            now = time.monotonic()
            t_sched = self._schedule_times.popleft() if self._schedule_times else 0.0

            if self._last_update_time > 0:
                wall_time = now - self._last_update_time
            elif t_sched > 0:
                wall_time = now - t_sched
            else:
                wall_time = 0.0
            self._last_update_time = now

            metrics = self._extract_metrics(
                scheduler_output, self._compute_queued(), wall_time
            )
            self._publisher.publish(metrics)

            if self._bench_active:
                self._bench_current_fpms.append(
                    json.loads(msgspec.json.encode(metrics))
                )
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
        _INT_FIELDS = {
            "prefill_isl_granularity",
            "decode_length_granularity",
            "decode_batch_size_granularity",
            "warmup_iterations",
        }
        for k in _INT_FIELDS:
            if k in cfg and not isinstance(cfg[k], int):
                cfg[k] = int(cfg[k])
        known = {f.name for f in BenchmarkConfig.__dataclass_fields__.values()}
        self._bench_config = BenchmarkConfig(
            **{k: v for k, v in cfg.items() if k in known}
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
        self._bench_current_fpms: list[dict] = []
        self._bench_active_req_ids: set[str] = set()
        self._bench_seq = 0
        self._bench_grid_built = False
        self._bench_drain_pending = False

        # Build block_hasher so benchmark requests work with prefix caching.
        if self.cache_config.enable_prefix_caching:
            caching_hash_fn = get_hash_fn_by_name(
                self.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)
            self._bench_block_hasher = get_request_block_hasher(
                self.block_size, caching_hash_fn
            )
        else:
            self._bench_block_hasher = None

        logger.info("Benchmark mode enabled: %s", self._bench_config)

    # -- Grid generation ------------------------------------------------

    def _bench_build_grid(self) -> None:
        """Generate the sweep grid once scheduler limits are known."""
        if self._bench_grid_built:
            return
        self._bench_grid_built = True
        mode = self._bench_config.mode
        if mode in ("prefill", "agg"):
            self._bench_generate_prefill_grid()
        if mode in ("decode", "agg"):
            self._bench_generate_decode_grid()
        logger.info("Benchmark grid: %d points (%s mode)", len(self._bench_grid), mode)

    def _bench_generate_prefill_grid(self) -> None:
        n = max(1, self._bench_config.prefill_isl_granularity)
        max_tokens = self.max_num_scheduled_tokens
        if max_tokens < 10:
            logger.warning(
                "max_num_scheduled_tokens=%d too small, skipping prefill grid",
                max_tokens,
            )
            return
        isls = np.unique(np.linspace(10, max_tokens, n, dtype=int))
        for isl in isls:
            self._bench_grid.append(BenchmarkPoint(point_type="prefill", isl=int(isl)))

    def _bench_generate_decode_grid(self) -> None:
        n_len = max(1, self._bench_config.decode_length_granularity)
        n_bs = max(1, self._bench_config.decode_batch_size_granularity)
        total_kv_tokens = self.cache_config.num_gpu_blocks * self.block_size
        max_ctx = self.max_model_len - 10
        if max_ctx < self.block_size:
            logger.warning("max_model_len too small for decode grid, skipping")
            return
        ctx_lens = np.unique(np.linspace(self.block_size, max_ctx, n_len, dtype=int))
        for ctx_len in ctx_lens:
            ctx_len = int(ctx_len)
            max_batch = min(self.max_num_running_reqs, total_kv_tokens // ctx_len)
            if max_batch < 1:
                continue
            batch_sizes = np.unique(np.linspace(1, max_batch, n_bs, dtype=int))
            for bs in batch_sizes:
                self._bench_grid.append(
                    BenchmarkPoint(
                        point_type="decode",
                        context_length=ctx_len,
                        batch_size=int(bs),
                    )
                )

    # -- Request injection / cleanup ------------------------------------

    def _bench_inject_prefill(
        self, prompt_len: int, max_tokens: int, n: int = 1
    ) -> None:
        for _ in range(n):
            req_id = f"__bench_{self._bench_seq}"
            req = Request(
                request_id=req_id,
                prompt_token_ids=[0] * prompt_len,
                sampling_params=SamplingParams(max_tokens=max_tokens),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=req_id,
            )
            self.add_request(req)
            self._bench_active_req_ids.add(req_id)
            self._bench_seq += 1

    def _bench_inject_fake_decode(
        self, ctx_len: int, batch_size: int
    ) -> SchedulerOutput:
        """Create fake decode requests with pre-allocated KV and return
        a custom SchedulerOutput that registers them with the model runner."""
        new_reqs_data: list[NewRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}

        for _ in range(batch_size):
            req_id = f"__bench_{self._bench_seq}"
            prompt = [0] * ctx_len
            req = Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=100_000),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=req_id,
            )

            new_blocks = self.kv_cache_manager.allocate_slots(
                req, ctx_len, delay_cache_blocks=True
            )
            if new_blocks is None:
                logger.warning(
                    "KV exhausted at ctx_len=%d after %d requests, " "truncating batch",
                    ctx_len,
                    len(new_reqs_data),
                )
                break

            req.num_computed_tokens = ctx_len
            req.status = RequestStatus.RUNNING
            req.append_output_token_ids(0)

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
                )
            )
            num_scheduled_tokens[req_id] = 1

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if getattr(self, "needs_kv_cache_zeroing", False)
            else None
        )

        return SchedulerOutput(
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
            self._bench_active = False
            self._bench_phase = _BenchPhase.IDLE
            logger.info("Benchmark complete")
        return None

    def _bench_step_warmup(self) -> SchedulerOutput | None:
        if not self._bench_active_req_ids:
            iters = self._bench_config.warmup_iterations
            if iters > 0:
                self._bench_inject_prefill(prompt_len=256, max_tokens=iters)
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
            if still_alive:
                return None
            if not self._bench_current_fpms:
                return None
            self._bench_save_current_point()
            self._bench_cleanup_requests()
            self._bench_drain_pending = True
            return None

        point = self._bench_pop_next("prefill")
        if point is None:
            if self._bench_config.mode == "agg":
                self._bench_phase = _BenchPhase.DECODE_SWEEP
                logger.info("Benchmark: entering DECODE_SWEEP")
            else:
                self._bench_phase = _BenchPhase.DONE
            return None

        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_inject_prefill(prompt_len=point.isl, max_tokens=1)
        logger.info("Benchmark prefill: ISL=%d", point.isl)
        return None

    def _bench_step_decode(self) -> SchedulerOutput | None:
        if self._bench_drain_if_pending():
            pass  # fall through to inject next point

        elif self._bench_active_req_ids:
            if not self._bench_current_fpms:
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
        logger.info(
            "Benchmark decode: ctx_len=%d batch_size=%d",
            point.context_length,
            point.batch_size,
        )
        return self._bench_inject_fake_decode(point.context_length, point.batch_size)

    def _bench_pop_next(self, point_type: str) -> BenchmarkPoint | None:
        while self._bench_grid:
            pt = self._bench_grid[0]
            if pt.point_type == point_type:
                return self._bench_grid.popleft()
            break
        return None

    def _bench_save_current_point(self) -> None:
        if self._bench_current_point is not None and self._bench_current_fpms:
            self._bench_results.append(
                BenchmarkPointResult(
                    point=self._bench_current_point,
                    fpms=list(self._bench_current_fpms),
                )
            )
        self._bench_current_point = None
        self._bench_current_fpms = []

    # -- Results output -------------------------------------------------

    def _bench_write_results(self) -> None:
        output = {
            "config": asdict(self._bench_config),
            "limits": {
                "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
                "max_num_running_reqs": self.max_num_running_reqs,
                "max_model_len": self.max_model_len,
                "block_size": self.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
            },
            "results": [
                {"point": asdict(r.point), "fpms": r.fpms} for r in self._bench_results
            ],
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
