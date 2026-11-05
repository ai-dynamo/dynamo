# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reactive threshold autoscaler — the naive lower-bound baseline.

Scales each pool by a fixed step whenever an *instantaneous* utilization signal
crosses a threshold: prefill on queue depth (requests waiting), decode on
KV-cache utilization. There is no smoothing, no tolerance band, and no
stabilization window — those are exactly what a real production scaler such as
KEDA adds on top. So Reactive is deliberately the floor: it shows what
"just scale on the raw signal" costs in oscillation and SLO misses, the
lower bound the smarter autoscalers must beat.

``EngineProtocol`` implementation. Consumes ``worker_counts`` (current fleet)
and ``fpm_observations`` (per-worker queue/KV), reduced to fleet-level scalars
via the shared :mod:`autoscaling_arena.adapters.aggregation` helpers so every
adapter reads the substrate the same way.
"""

from __future__ import annotations

from typing import Optional

from autoscaling_arena.adapters._regression_bootstrap import _NoopRegressionBootstrap
from autoscaling_arena.adapters.aggregation import (
    aggregate_kv_util,
    aggregate_queue_depth,
    current_replica_target,
)

from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
)


class ReactiveAutoscaler(_NoopRegressionBootstrap):
    """Threshold scaler: ±``step`` per poll on queue depth / KV utilization.

    Args:
        mode: ``"disagg"`` scales prefill (on queue depth) and decode (on KV
            utilization) independently; ``"agg"`` scales the single pool on
            queue depth.
        min_prefill / max_prefill / min_decode / max_decode: replica clamps.
        prefill_queue_up / prefill_queue_down: queue-depth thresholds (waiting
            request count) to scale prefill up / down.
        decode_kv_up / decode_kv_down: KV-utilization fractions to scale decode
            up / down. Ignored when capacity is unknown.
        agg_queue_up / agg_queue_down: queue-depth thresholds for agg mode.
        poll_interval_s: virtual-clock cadence between decisions.
        step: replicas added/removed per crossing.
        capabilities: per-pool ``max_kv_tokens`` for the KV-utilization signal.
    """

    def __init__(
        self,
        *,
        mode: str = "disagg",
        min_prefill: int = 1,
        max_prefill: int = 16,
        min_decode: int = 1,
        max_decode: int = 8,
        prefill_queue_up: int = 4,
        prefill_queue_down: int = 1,
        decode_kv_up: float = 0.8,
        decode_kv_down: float = 0.3,
        agg_queue_up: int = 4,
        agg_queue_down: int = 1,
        poll_interval_s: float = 15.0,
        step: int = 1,
        capabilities: Optional[WorkerCapabilities] = None,
    ) -> None:
        self._is_disagg = mode == "disagg"
        self._min_prefill = min_prefill
        self._max_prefill = max_prefill
        self._min_decode = min_decode
        self._max_decode = max_decode
        self._prefill_queue_up = prefill_queue_up
        self._prefill_queue_down = prefill_queue_down
        self._decode_kv_up = decode_kv_up
        self._decode_kv_down = decode_kv_down
        self._agg_queue_up = agg_queue_up
        self._agg_queue_down = agg_queue_down
        self._poll_interval_s = poll_interval_s
        self._step = step
        self._capabilities = capabilities

    def _step_toward(
        self, current: int, *, up: bool, down: bool, lo: int, hi: int
    ) -> int:
        if up:
            return min(current + self._step, hi)
        if down:
            return max(current - self._step, lo)
        return current

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return ScheduledTick(
            at_s=start_s, need_worker_states=True, need_worker_fpm=True
        )

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        wc = tick_input.worker_counts
        fpm = tick_input.fpm_observations

        target_prefill: Optional[int] = None
        target_decode: Optional[int]

        if self._is_disagg:
            cur_p = current_replica_target(
                wc, role="prefill", minimum=self._min_prefill
            )
            qd = aggregate_queue_depth(fpm, pool="prefill")
            target_prefill = self._step_toward(
                cur_p,
                up=qd is not None and qd > self._prefill_queue_up,
                down=qd is not None and qd < self._prefill_queue_down,
                lo=self._min_prefill,
                hi=self._max_prefill,
            )

            cur_d = current_replica_target(wc, role="decode", minimum=self._min_decode)
            kv = aggregate_kv_util(fpm, self._capabilities, pool="decode")
            target_decode = self._step_toward(
                cur_d,
                up=kv is not None and kv > self._decode_kv_up,
                down=kv is not None and kv < self._decode_kv_down,
                lo=self._min_decode,
                hi=self._max_decode,
            )
        else:
            cur = current_replica_target(wc, role="decode", minimum=self._min_decode)
            qd = aggregate_queue_depth(fpm, pool="all")
            target_decode = self._step_toward(
                cur,
                up=qd is not None and qd > self._agg_queue_up,
                down=qd is not None and qd < self._agg_queue_down,
                lo=self._min_decode,
                hi=self._max_decode,
            )

        return PlannerEffects(
            scale_to=ScalingDecision(
                num_prefill=target_prefill, num_decode=target_decode
            ),
            next_tick=ScheduledTick(
                at_s=scheduled_tick.at_s + self._poll_interval_s,
                need_worker_states=True,
                need_worker_fpm=True,
            ),
        )

    async def shutdown(self) -> None:
        return None
