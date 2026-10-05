# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KEDA / Kubernetes-HPA autoscaler — the production rival (fidelity "A-port").

A faithful re-implementation of the KEDA Prometheus-scaler + Kubernetes HPA v2
control loop, driven once per ``pollingInterval`` against live fleet state. It is
NOT a strawman: it reproduces the behaviors that dominate a fair-vs-Planner
comparison — the per-replica ``AverageValue`` formula, the 10% tolerance
dead-band, MAX-across-triggers, the asymmetric stabilization windows (scale-up
immediate, scale-down held 300s), and the scale-up rate cap.

Algorithm + defaults are pinned to upstream docs:
- HPA: desiredReplicas = ceil(currentReplicas * currentMetricValue/targetValue);
  tolerance 0.1; sync 15s; max across metrics; stabilization picks the most
  conservative recommendation in the trailing window (max for down, min for up).
  https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- HPA default behavior: scaleUp stabilization 0s + (Pods 4 OR Percent 100)/15s
  selectPolicy Max; scaleDown stabilization 300s + Percent 100/15s.
- KEDA Prometheus scaler: metricType AverageValue (threshold is per-replica;
  query is the fleet-wide total). vLLM stack: vllm:num_requests_waiting
  threshold 5, vllm:gpu_cache_usage_perc target ~0.9, pollingInterval 15s.
  https://docs.vllm.ai/projects/production-stack/.../autoscaling-keda.html

Disagg: one independent HPA per pool — prefill on queue depth, decode on KV
utilization — each with its own recommendation history (a real ScaledObject
scales one Deployment).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

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

MetricMode = Literal["fleet_total", "value_avg"]


class _HpaController:
    """One HPA instance: stateful recommendation for a single scaled pool.

    ``metric_mode`` selects how the raw Prometheus value maps to a target:
    - ``fleet_total`` (queue depth, KEDA ``AverageValue`` default): the value is
      a fleet-wide total; target = ceil(total / threshold), usage ratio
      (per-replica avg / threshold) is replica-independent in effect.
    - ``value_avg`` (KV utilization, ``Value`` on an ``avg(...)`` query): the
      value is already a per-replica average fraction; target =
      ceil(current * value / threshold).
    """

    def __init__(
        self,
        *,
        threshold: float,
        metric_mode: MetricMode,
        min_replicas: int,
        max_replicas: int,
        tolerance: float = 0.10,
        scale_up_stabilization_s: float = 0.0,
        scale_down_stabilization_s: float = 300.0,
        scale_up_pods_policy: int = 4,
        scale_up_percent_policy: float = 100.0,
    ) -> None:
        self.threshold = threshold
        self.metric_mode = metric_mode
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.tolerance = tolerance
        self.scale_up_stabilization_s = scale_up_stabilization_s
        self.scale_down_stabilization_s = scale_down_stabilization_s
        self.scale_up_pods_policy = scale_up_pods_policy
        self.scale_up_percent_policy = scale_up_percent_policy
        # Trailing recommendation history: list of (t_seconds, raw_recommendation).
        self._history: list[tuple[float, int]] = []

    def recommend(self, t_s: float, current: int, metric_value: Optional[float]) -> int:
        cur = max(current, self.min_replicas)

        # 1) raw target + usage ratio (HPA formula, per metricType).
        if metric_value is None:
            # Missing metric this tick: hold (record current so the window stays dense).
            raw = cur
        else:
            if self.metric_mode == "fleet_total":
                per_replica = metric_value / cur if cur > 0 else metric_value
                usage_ratio = (
                    per_replica / self.threshold if self.threshold > 0 else 0.0
                )
                target = (
                    math.ceil(metric_value / self.threshold)
                    if self.threshold > 0
                    else cur
                )
            else:  # value_avg
                usage_ratio = (
                    metric_value / self.threshold if self.threshold > 0 else 0.0
                )
                target = (
                    math.ceil(cur * metric_value / self.threshold)
                    if self.threshold > 0
                    else cur
                )
            # 2) tolerance dead-band: skip scaling near the setpoint.
            raw = cur if abs(usage_ratio - 1.0) <= self.tolerance else target

        # 3) record + evict beyond the longest window we consult.
        self._history.append((t_s, raw))
        horizon = max(self.scale_up_stabilization_s, self.scale_down_stabilization_s)
        self._history = [(t, r) for (t, r) in self._history if t >= t_s - horizon]

        # 4) stabilization window: most-conservative recommendation in-window.
        # Upstream HPA clamps toward current so stabilization can NEVER reverse
        # the raw direction (a scale-down eval can't increase replicas, and
        # vice versa) — clamp to ``cur`` on each branch.
        if raw > cur:  # scaling up: MIN over the (short) up-window, never below current
            window = [
                r
                for (t, r) in self._history
                if t >= t_s - self.scale_up_stabilization_s
            ]
            windowed = max(cur, min(window)) if window else raw
        elif (
            raw < cur
        ):  # scaling down: MAX over the (long) down-window, never above current
            window = [
                r
                for (t, r) in self._history
                if t >= t_s - self.scale_down_stabilization_s
            ]
            windowed = min(cur, max(window)) if window else raw
        else:
            windowed = cur

        # 5) rate-limit policies (cap the per-tick delta from current).
        if windowed > cur:
            cap = max(
                self.scale_up_pods_policy,
                math.ceil(self.scale_up_percent_policy / 100.0 * cur),
            )
            desired = min(windowed, cur + cap)
        elif windowed < cur:
            desired = windowed  # scaleDown Percent 100: may drop straight toward min
        else:
            desired = cur

        # 6) clamp.
        return max(self.min_replicas, min(self.max_replicas, desired))


class KedaAutoscaler(_NoopRegressionBootstrap):
    """KEDA dual-trigger autoscaler over the common adapter interface.

    Args:
        mode: ``"disagg"`` runs one HPA per pool (prefill=queue, decode=KV);
            ``"agg"`` runs a single queue-depth HPA.
        capabilities: per-pool ``max_kv_tokens`` for the KV-utilization signal.
        poll_interval_s: KEDA pollingInterval (vLLM stack: 15s).
        queue_threshold: vllm:num_requests_waiting target (per-replica, AverageValue).
        kv_threshold: vllm:gpu_cache_usage_perc target utilization.
        min/max per pool: replica clamps (vLLM stack tutorial used 1..3).
        tolerance / stabilization / rate-policy: HPA defaults, overridable.
    """

    def __init__(
        self,
        *,
        mode: str = "disagg",
        capabilities: Optional[WorkerCapabilities] = None,
        poll_interval_s: float = 15.0,
        queue_threshold: float = 5.0,
        kv_threshold: float = 0.9,
        min_prefill: int = 1,
        max_prefill: int = 16,
        min_decode: int = 1,
        max_decode: int = 8,
        tolerance: float = 0.10,
        scale_up_stabilization_s: float = 0.0,
        scale_down_stabilization_s: float = 300.0,
    ) -> None:
        self._is_disagg = mode == "disagg"
        self._capabilities = capabilities
        self._poll_interval_s = poll_interval_s
        common = dict(
            tolerance=tolerance,
            scale_up_stabilization_s=scale_up_stabilization_s,
            scale_down_stabilization_s=scale_down_stabilization_s,
        )
        if self._is_disagg:
            self._prefill_hpa = _HpaController(
                threshold=queue_threshold,
                metric_mode="fleet_total",
                min_replicas=min_prefill,
                max_replicas=max_prefill,
                **common,
            )
            self._decode_hpa = _HpaController(
                threshold=kv_threshold,
                metric_mode="value_avg",
                min_replicas=min_decode,
                max_replicas=max_decode,
                **common,
            )
        else:
            self._agg_hpa = _HpaController(
                threshold=queue_threshold,
                metric_mode="fleet_total",
                min_replicas=min_decode,
                max_replicas=max_decode,
                **common,
            )

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return ScheduledTick(
            at_s=start_s, need_worker_states=True, need_worker_fpm=True
        )

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        t_s = scheduled_tick.at_s
        wc = tick_input.worker_counts
        fpm = tick_input.fpm_observations

        target_prefill: Optional[int] = None
        if self._is_disagg:
            cur_p = current_replica_target(
                wc,
                role="prefill",
                minimum=self._prefill_hpa.min_replicas,
            )
            queue = aggregate_queue_depth(fpm, pool="prefill")
            target_prefill = self._prefill_hpa.recommend(t_s, cur_p, queue)

            cur_d = current_replica_target(
                wc,
                role="decode",
                minimum=self._decode_hpa.min_replicas,
            )
            kv = aggregate_kv_util(fpm, self._capabilities, pool="decode")
            target_decode = self._decode_hpa.recommend(t_s, cur_d, kv)
        else:
            cur = current_replica_target(
                wc,
                role="decode",
                minimum=self._agg_hpa.min_replicas,
            )
            queue = aggregate_queue_depth(fpm, pool="all")
            target_decode = self._agg_hpa.recommend(t_s, cur, queue)

        return PlannerEffects(
            scale_to=ScalingDecision(
                num_prefill=target_prefill, num_decode=target_decode
            ),
            next_tick=ScheduledTick(
                at_s=t_s + self._poll_interval_s,
                need_worker_states=True,
                need_worker_fpm=True,
            ),
        )

    async def shutdown(self) -> None:
        return None
