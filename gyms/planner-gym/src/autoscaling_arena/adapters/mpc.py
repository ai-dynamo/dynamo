# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model predictive control for aggregate and disaggregated inference serving.

The capacity model combines prefill work, decode batching and KV-cache limits
into a sustainable request rate. A bounded online correction uses completions
observed while the fleet is saturated. Arrival forecasts use coarse bins,
EWMA levels, change-point detection and statistically significant trends.

Each candidate fleet is evaluated over the cold-start delay plus the planning
horizon. Pending replicas contribute GPU cost immediately and serving capacity
only after they become ready. The objective approximates good requests per
GPU-second using a Dinkelbach ratio objective with TTFT and ITL constraints.

Scale-up uses the mean arrival forecast. Scale-down requires no pending orders
and a reduced fleet that stays below 75% utilization and half the TTFT budget
under an 80th-percentile forecast for a full cold-start interval. A queue-spike
floor accounts for the capacity already on its way.

Disaggregated serving searches feasible prefill/decode pairs, accounts for
KV transfer, and uses the slower pool as the throughput limit. See docs/mpc.md
for the model equations and configuration.
"""

from __future__ import annotations

import abc
import logging
import math
from collections import deque
from collections.abc import Mapping
from typing import Any

from autoscaling_arena.adapters._regression_bootstrap import _NoopRegressionBootstrap
from autoscaling_arena.adapters.aggregation import (
    aggregate_kv_util,
    aggregate_queue_depth,
    current_replica_target,
)
from autoscaling_arena.adapters.outcome_feedback import (
    LatencyCalibration,
    OutcomeFeedback,
)
from autoscaling_arena.policy_contracts import POLICY_CONTRACTS

from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
)

_MAX_CONCURRENCY = 256.0


def _roofline_ttft_ms(isl: float, batch_size: float = 1.0) -> float:
    """Analytical TTFT fallback in ms: base + linear(ISL), sublinear in batch.

    Used when the AIS capacity model has no prefill estimate for an ISL, and by
    ``RooflineCapacityModel``.
    """
    base_ttft = 10.0 + 0.2 * max(1.0, isl)
    batch_factor = 1.0 + 0.3 * math.log1p(max(0.0, batch_size - 1))
    return base_ttft * batch_factor


class _ArrivalForecaster:
    """Arrival-rate forecaster over coarse bins.

    ``update`` takes the arrivals of one poll interval. Completed bins of
    ``bin_s`` seconds drive an EWMA level (time constant ``tau_s``) with a
    Poisson change-point reset, and a least-squares trend over
    ``trend_window_s`` that is applied only when significant.
    """

    #: Poisson standard deviations a bin must deviate by to reset the level.
    snap_sigma: float = 2.5

    def __init__(
        self,
        *,
        bin_s: float = 15.0,
        tau_s: float = 30.0,
        trend_window_s: float = 120.0,
        plan_s: float = 55.0,
    ) -> None:
        self._bin_s = max(1e-6, bin_s)
        self._alpha = min(1.0, self._bin_s / max(self._bin_s, tau_s))
        self._plan_s = max(1e-6, plan_s)
        self._history: deque[float] = deque(
            maxlen=max(2, round(trend_window_s / self._bin_s))
        )
        self._level: float | None = None
        self._acc_count = 0.0
        self._acc_time = 0.0

    @property
    def level(self) -> float:
        if self._level is None:
            return self._acc_count / self._acc_time if self._acc_time > 0 else 0.0
        return self._level

    def update(self, count: float, dt: float) -> None:
        self._acc_count += max(0.0, count)
        self._acc_time += max(0.0, dt)
        if self._acc_time + 1e-9 >= self._bin_s:
            self._close_bin()

    def _close_bin(self) -> None:
        rate = self._acc_count / self._acc_time
        if self._level is None:
            self._level = rate
        else:
            expected = self._level * self._acc_time
            if abs(self._acc_count - expected) > max(
                self.snap_sigma * math.sqrt(max(expected, 1.0)), 5.0
            ):
                self._level = rate  # change point: a real step, not noise
                self._history.clear()
            else:
                self._level = (1.0 - self._alpha) * self._level + self._alpha * rate
        self._history.append(rate)
        self._acc_count = 0.0
        self._acc_time = 0.0

    def _trend_per_s(self) -> float:
        n = len(self._history)
        if n < 4:
            return 0.0
        xs = [i * self._bin_s for i in range(n)]
        ys = list(self._history)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            return 0.0
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        sse = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
        se = math.sqrt(max(sse, 1e-12) / (n - 2) / sxx)
        return slope if abs(slope) > 2.0 * se else 0.0

    def forecast(self, steps: int, dt: float) -> tuple[list[float], list[float]]:
        """Mean and 80th-percentile arrival rate for each of ``steps`` plan steps."""
        level = self.level
        slope = self._trend_per_s()
        var = level * (
            self._alpha / ((2.0 - self._alpha) * self._bin_s) + 1.0 / self._plan_s
        )
        spread = 0.84 * math.sqrt(max(0.0, var))
        mean: list[float] = []
        hi: list[float] = []
        for k in range(steps):
            m = level + slope * (k + 1) * dt
            m = min(max(m, 0.5 * level), 2.0 * level) if level > 0 else max(0.0, m)
            mean.append(m)
            hi.append(m + spread)
        return mean, hi


logger = logging.getLogger(__name__)


class CapacityModel(abc.ABC):
    """Per-worker service-rate model built from prefill and decode step times."""

    #: Fraction of KV memory a worker may fill before the model treats it as
    #: saturated. The simulator preempts and re-prefills above this point.
    kv_fraction: float = 0.9

    def __init__(
        self,
        *,
        max_kv_tokens: int | None = None,
        max_num_seqs: int | None = None,
        prefill_parallelism: int = 1,
        decode_parallelism: int = 1,
    ) -> None:
        if prefill_parallelism < 1 or decode_parallelism < 1:
            raise ValueError("attention-DP parallelism must be positive")
        self.prefill_parallelism = prefill_parallelism
        self.decode_parallelism = decode_parallelism
        # KV capacity is reported per worker; sequence limits and AIS queries
        # describe one attention-DP rank.
        self._max_kv_tokens = (
            max_kv_tokens / decode_parallelism if max_kv_tokens is not None else None
        )
        self._max_concurrency = (
            float(max_num_seqs) if max_num_seqs else _MAX_CONCURRENCY
        )
        self._bias = 1.0

    # ---- engine primitives ------------------------------------------------ #

    @abc.abstractmethod
    def prefill_ms(self, isl: float) -> float:
        """Prefill time (ms) of one request with ``isl`` uncached tokens."""

    @abc.abstractmethod
    def decode_step_ms(self, concurrency: float, isl: float, osl: float) -> float:
        """Time (ms) of one decode step serving ``concurrency`` requests."""

    # ---- derived quantities ----------------------------------------------- #

    def kv_concurrency_cap(self, isl: float, osl: float) -> float:
        if not self._max_kv_tokens:
            return _MAX_CONCURRENCY
        tokens_per_request = max(1.0, isl + osl)
        return max(
            1.0,
            min(
                _MAX_CONCURRENCY,
                self.kv_fraction * self._max_kv_tokens / tokens_per_request,
            ),
        )

    def residence_s(
        self, concurrency: float, isl: float, osl: float, kv_hit: float
    ) -> float:
        t_p = self.prefill_ms(isl * (1.0 - kv_hit)) / 1000.0
        t_d = self.decode_step_ms(concurrency, isl, osl) / 1000.0
        return t_p + max(0.0, osl) * t_d

    def concurrency(
        self, load_per_worker: float, isl: float, osl: float, kv_hit: float
    ) -> float:
        """Little's-law concurrency per rank at the given worker arrival rate."""
        load_per_rank = load_per_worker / self.decode_parallelism
        cap = self.kv_concurrency_cap(isl, osl)
        c = 1.0
        for _ in range(30):
            c_new = min(
                cap, max(1.0, load_per_rank * self.residence_s(c, isl, osl, kv_hit))
            )
            if abs(c_new - c) < 0.05:
                c = c_new
                break
            c = c_new
        return c

    def saturated_concurrency(self, isl: float, osl: float) -> float:
        """Requests a saturated attention-DP rank holds, bounded by KV and sequences."""
        return max(1.0, min(self.kv_concurrency_cap(isl, osl), self._max_concurrency))

    def service_rate(
        self, load_per_worker: float, isl: float, osl: float, kv_hit: float = 0.0
    ) -> float:
        """Sustainable requests/s of one worker (bias applied).

        Capacity is the throughput of a *saturated* worker: a backlog keeps its
        concurrency at the cap, where decode batching is best. Prefill GPU time
        is serial per request; each decode step is shared by ``c`` requests::

            mu = 1 / ( T_p + osl * T_d(c_sat) / c_sat )

        Evaluating at the current load's concurrency instead would charge a
        lightly loaded worker almost a full decode step per token and report a
        capacity far below what it sustains once queued work arrives.
        ``load_per_worker`` is accepted for interface symmetry with
        :meth:`itl_ms`; it does not enter the capacity.
        """
        del load_per_worker
        isl = max(1.0, isl)
        osl = max(1.0, osl)
        kv_hit = min(max(kv_hit or 0.0, 0.0), 0.99)
        c = self.saturated_concurrency(isl, osl)
        t_p = self.prefill_ms(isl * (1.0 - kv_hit)) / 1000.0
        t_d = self.decode_step_ms(c, isl, osl) / 1000.0
        return self.decode_parallelism * self._bias / max(1e-3, t_p + osl * t_d / c)

    def itl_ms(
        self,
        load_per_worker: float,
        isl: float,
        osl: float,
        kv_hit: float = 0.0,
        saturated: bool = False,
    ) -> float:
        """Mean decode step time (ms).

        Saturated worker (backlog present): every decode step waits behind a
        prefill, so ``ITL = T_p + T_d(c_sat)``. Otherwise the processor-sharing
        form ``T_d(c) / (1 - lambda * T_p)`` at the Little's-law concurrency.
        """
        isl = max(1.0, isl)
        osl = max(1.0, osl)
        kv_hit = min(max(kv_hit or 0.0, 0.0), 0.99)
        t_p = self.prefill_ms(isl * (1.0 - kv_hit))
        if saturated:
            return t_p + self.decode_step_ms(
                self.saturated_concurrency(isl, osl), isl, osl
            )
        c = self.concurrency(load_per_worker, isl, osl, kv_hit)
        t_d = self.decode_step_ms(c, isl, osl)
        prefill_share = min(
            0.95, load_per_worker / self.prefill_parallelism * t_p / 1000.0
        )
        return t_d / (1.0 - prefill_share)

    # ---- online correction ------------------------------------------------ #

    @property
    def bias(self) -> float:
        return self._bias

    def observe_saturated(
        self,
        measured_rate_per_worker: float,
        load_per_worker: float,
        isl: float,
        osl: float,
        kv_hit: float = 0.0,
        alpha: float = 0.1,
    ) -> None:
        """Fold a saturated-tick measurement into the multiplicative bias.

        Call only when the queue was non-empty across the whole interval, so
        completions per active worker-second measure capacity, not demand.
        """
        if measured_rate_per_worker <= 0:
            return
        model = self.service_rate(load_per_worker, isl, osl, kv_hit) / self._bias
        if model <= 0:
            return
        ratio = measured_rate_per_worker / model
        ratio = min(4.0, max(0.25, ratio))
        self._bias = (1.0 - alpha) * self._bias + alpha * ratio


class AISCapacityModel(CapacityModel):
    """Capacity model over Dynamo's AIS (AISimulate) prefill and decode estimators.

    ``config`` is the canonical AIS performance-model identity of the engine
    whose TTFT the plan predicts (the same mapping Match Config renders into
    the engine's ``ais_perf_config``: model, system, backend, backend_version,
    tp, moe sizes, attention_dp, worker_type, estimation_mode, fallback_policy).
    ``decode_config`` supplies the deployed decode identity in disaggregated
    mode. Aggregate workers reuse one session. Sessions load on first prediction.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        max_kv_tokens: int | None = None,
        max_num_seqs: int | None = None,
        decode_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not config:
            raise ValueError(
                "AISCapacityModel needs the canonical AIS config of the TTFT engine"
            )
        self._config: dict[str, Any] = dict(config)
        self._decode_config = dict(
            decode_config if decode_config is not None else config
        )
        super().__init__(
            max_kv_tokens=max_kv_tokens,
            max_num_seqs=max_num_seqs,
            prefill_parallelism=int(self._config.get("attention_dp", 1)),
            decode_parallelism=int(self._decode_config.get("attention_dp", 1)),
        )
        self.fallbacks = 0  # AIS points the roofline model had to cover
        self._session = None
        self._decode_session = None
        self._prefill_cache: dict[int, float] = {}
        self._decode_cache: dict[tuple[int, int, int], float] = {}

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        from dynamo._internal.ais import create_session

        prefill_session = create_session(dict(self._config))
        decode_session = (
            prefill_session
            if self._decode_config == self._config
            else create_session(dict(self._decode_config))
        )
        self._session, self._decode_session = prefill_session, decode_session

    @staticmethod
    def _bucket(value: float, width: int) -> int:
        return max(width, round(value / width) * width)

    def prefill_ms(self, isl: float) -> float:
        self._ensure_loaded()
        key = self._bucket(isl, 64)
        if key not in self._prefill_cache:
            try:
                self._prefill_cache[key] = float(
                    self._session.predict_prefill(
                        batch_size=1, effective_isl=key, prefix=0
                    )
                )
            except (RuntimeError, ValueError) as exc:
                # AIS (fallback_policy "deny") raises ValueError for points it
                # does not support and RuntimeError when it has no finite
                # latency; the roofline covers those. Anything else propagates.
                self.fallbacks += 1
                logger.warning(
                    "AIS predict_prefill failed for isl=%d (%s); using roofline",
                    key,
                    exc,
                )
                self._prefill_cache[key] = _roofline_ttft_ms(key, 1)
        return self._prefill_cache[key]

    def decode_step_ms(self, concurrency: float, isl: float, osl: float) -> float:
        self._ensure_loaded()
        c = max(1, min(int(_MAX_CONCURRENCY), round(concurrency)))
        key = (c, self._bucket(isl, 256), self._bucket(osl, 16))
        if key not in self._decode_cache:
            try:
                total = float(
                    self._decode_session.predict_decode(
                        batch_size=c, isl=key[1], osl=key[2]
                    )
                )
                self._decode_cache[key] = total / key[2]
            except (RuntimeError, ValueError) as exc:
                self.fallbacks += 1
                logger.warning(
                    "AIS predict_decode failed for batch=%d isl=%d osl=%d (%s); "
                    "using roofline",
                    key[0],
                    key[1],
                    key[2],
                    exc,
                )
                self._decode_cache[key] = RooflineCapacityModel.decode_estimate(c)
        return self._decode_cache[key]


class RooflineCapacityModel(CapacityModel):
    """Dependency-free analytical fallback with the same interface."""

    def prefill_ms(self, isl: float) -> float:
        return _roofline_ttft_ms(isl, 1)

    def decode_step_ms(self, concurrency: float, isl: float, osl: float) -> float:
        return self.decode_estimate(concurrency)

    @staticmethod
    def decode_estimate(concurrency: float) -> float:
        return 4.0 + 0.17 * max(0.0, concurrency)


class _PoolBook:
    """Order book of one worker pool: pending replicas and their ready times."""

    def __init__(
        self, *, cold_start_s: float, dt: float, plan_steps: int, lag_steps: int
    ) -> None:
        self._cold_start_s = cold_start_s
        self._dt = dt
        self._plan_steps = plan_steps
        self._lag_steps = lag_steps
        self.orders: list[list] = []
        self.t_s = 0.0
        self.ready = 1
        self.expected = 1
        self.downsize_ticks = 0

    @property
    def pending(self) -> int:
        return sum(int(c) for _, c in self.orders)

    def reconcile(self, t_s: float, ready: int, expected: int) -> None:
        self.t_s, self.ready, self.expected = t_s, ready, expected
        starting = max(0, expected - ready)
        self.orders.sort(key=lambda o: o[0])
        pending = self.pending
        if pending > starting:
            surplus = pending - starting
            for order in self.orders:
                if surplus <= 0:
                    break
                if order[0] <= t_s:
                    take = min(order[1], surplus)
                    order[1] -= take
                    surplus -= take
            for order in reversed(self.orders):
                if surplus <= 0:
                    break
                take = min(order[1], surplus)
                order[1] -= take
                surplus -= take
            self.orders = [o for o in self.orders if o[1] > 0]
        elif pending < starting:
            self.orders.append([t_s + self._cold_start_s, starting - pending])

    def book(self, t_s: float, target: int) -> None:
        if target > self.expected:
            self.orders.append([t_s + self._cold_start_s, target - self.expected])
        elif target < self.expected:
            to_cancel = self.expected - target
            self.orders.sort(key=lambda o: o[0])
            for order in reversed(self.orders):
                if to_cancel <= 0:
                    break
                take = min(order[1], to_cancel)
                order[1] -= take
                to_cancel -= take
            self.orders = [o for o in self.orders if o[1] > 0]

    def schedule(self, candidate: int) -> list[int]:
        orders = sorted(([o[0], o[1]] for o in self.orders), key=lambda o: o[0])
        base = self.ready
        new = 0
        if candidate >= self.expected:
            new = candidate - self.expected
        else:
            to_cancel = self.expected - candidate
            for order in reversed(orders):
                if to_cancel <= 0:
                    break
                take = min(order[1], to_cancel)
                order[1] -= take
                to_cancel -= take
            orders = [o for o in orders if o[1] > 0]
            base = max(0, base - to_cancel)
        out = []
        for k in range(self._plan_steps):
            t_k = self.t_s + k * self._dt
            n = base + sum(c for r, c in orders if r <= t_k)
            if new and k >= self._lag_steps:
                n += new
            out.append(max(1, n))
        return out


class MPCAutoscaler(_NoopRegressionBootstrap):
    """MPC over a capacity model with pending replicas and guarded scale-down."""

    consumes_replay_telemetry = True
    telemetry_interval_s = 5.0
    policy_contract = POLICY_CONTRACTS["mpc"]

    def __init__(
        self,
        *,
        capacity_model: CapacityModel,
        mode: str = "agg",
        horizon: int = 5,
        slo_ttft_ms: float = 200.0,
        slo_itl_ms: float = 50.0,
        gpus_per_worker: int = 1,
        poll_interval_s: float = 5.0,
        scale_up_penalty: float = 0.0,
        scale_down_penalty: float = 0.0,
        reactive_queue_threshold: float = 10.0,
        min_replicas: int = 1,
        max_replicas: int = 16,
        cold_start_s: float = 30.0,
        rho_min: float = 1e-6,
        rho_window_s: float = 60.0,
        prefill_gpus_per_worker: int | None = None,
        kv_transfer_gbps: float | None = None,
        kv_bytes_per_token: int | None = None,
        capabilities: WorkerCapabilities | None = None,
        feasible_targets: tuple[tuple[int, int], ...] | None = None,
        short_output_itl: str = "skip",
    ) -> None:
        self._capacity = capacity_model
        self._mode = mode
        self._slo_ttft = slo_ttft_ms
        self._slo_itl = slo_itl_ms
        self._feedback = OutcomeFeedback(
            slo={"ttft_ms": slo_ttft_ms, "itl_ms": slo_itl_ms},
            short_output_itl=short_output_itl,
            window_s=max(poll_interval_s, rho_window_s),
        )
        self._latency_calibration = LatencyCalibration(
            slo_ttft_ms, slo_itl_ms, short_output_itl=short_output_itl
        )
        self._latency_prediction = None
        self._observed_outcomes = None
        self._gpus_per_worker = gpus_per_worker
        self._poll_interval_s = poll_interval_s
        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
        self._capabilities = capabilities
        if feasible_targets is None:
            prefills = (
                range(min_replicas, max_replicas + 1) if mode == "disagg" else (0,)
            )
            feasible_targets = tuple(
                (p, d) for p in prefills for d in range(min_replicas, max_replicas + 1)
            )
        self._fleet_targets = tuple(
            (p, d)
            for p, d in feasible_targets
            if min_replicas <= d <= max_replicas
            and (min_replicas <= p <= max_replicas if mode == "disagg" else p == 0)
        )
        if not self._fleet_targets:
            raise ValueError("MPC has no feasible fleet within this run's limits")
        if mode == "agg":
            self._min_replicas = min(d for _, d in self._fleet_targets)
            self._max_replicas = max(d for _, d in self._fleet_targets)

        self._scale_up_penalty = scale_up_penalty
        self._scale_down_penalty = scale_down_penalty

        self._reactive_queue_threshold = reactive_queue_threshold

        self._deadband_fraction = 0.05
        self._shape_tau_s = 60.0

        self._current_replicas: int = self._min_replicas
        self._history: deque[dict] = deque(maxlen=100)

        # Scale-down rule state and constants.
        self._downsize_utilisation = 0.75
        self._downsize_ticks = 0

        # Dinkelbach weight = trailing goodput per GPU-second.
        self._rho_min = max(1e-6, rho_min)
        self._rho_window_s = max(poll_interval_s, rho_window_s)
        self._rho = self._rho_min
        self._good_ema: float | None = None
        self._gpu_ema: float | None = None

        # Shape estimates for the plant model (60 s EWMAs).
        self._isl_now: float = 512.0
        self._isl_ema: float | None = None
        self._osl_ema: float = 128.0
        self._osl_seen = False
        self._kv_hit_ema: float = 0.0
        # Saturation bookkeeping for the online bias: previous tick's queue,
        # in-flight count and ready workers.
        self._prev_inflight: float | None = None
        self._prev_queue: int = 0
        self._prev_ready: int = 0
        self._traffic_observed = False

        # Pending orders are [ready_at_s, count]; the horizon includes startup lag.
        self._cold_start_s = max(0.0, cold_start_s)
        self._lag_steps = math.ceil(self._cold_start_s / max(1e-6, poll_interval_s))
        self._plan_steps = self._lag_steps + horizon
        self._orders: list[list] = []
        self._t_s: float = 0.0
        self._ready: int = self._min_replicas
        self._expected: int = self._min_replicas
        # Match the forecast's smoothing time to startup lag, with a 30 s
        # smoothing window when startup is instantaneous.
        self._forecaster = _ArrivalForecaster(
            bin_s=3 * poll_interval_s,
            tau_s=self._cold_start_s or 30.0,
            trend_window_s=120.0,
            plan_s=self._plan_steps * poll_interval_s,
        )

        # one order book per pool, per-pool GPU cost and KV transfer.
        self._prefill_gpus = prefill_gpus_per_worker or gpus_per_worker
        self._kv_transfer_gbps = kv_transfer_gbps
        self._kv_bytes_per_token = kv_bytes_per_token
        book_kwargs = {
            "cold_start_s": self._cold_start_s,
            "dt": poll_interval_s,
            "plan_steps": self._plan_steps,
            "lag_steps": self._lag_steps,
        }
        self._book_p = _PoolBook(**book_kwargs)
        self._book_d = _PoolBook(**book_kwargs)
        self._book_p.ready = self._book_p.expected = min_replicas
        self._book_d.ready = self._book_d.expected = min_replicas

    # ---- scheduling ------------------------------------------------------- #

    def on_telemetry(self, sample) -> None:
        self._feedback.observe(sample)

    def _observe_outcomes(self, now_s, fleet) -> None:
        self._observed_outcomes = self._feedback.view(now_s)
        if self._feedback.latest is not None:
            self._latency_calibration.observe(
                self._feedback.latest, self._latency_prediction, fleet
            )

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return ScheduledTick(
            at_s=start_s,
            need_worker_states=True,
            need_worker_fpm=True,
            need_traffic_metrics=True,
            use_full_traffic_metrics=True,
            traffic_metrics_duration_s=self._poll_interval_s,
        )

    def _make_effects(self, t_s: float, action: int) -> PlannerEffects:
        target_decode = action
        target_prefill = action if self._mode == "disagg" else None
        return PlannerEffects(
            scale_to=ScalingDecision(
                num_prefill=target_prefill, num_decode=target_decode
            ),
            next_tick=ScheduledTick(
                at_s=t_s + self._poll_interval_s,
                need_worker_states=True,
                need_worker_fpm=True,
                need_traffic_metrics=True,
                use_full_traffic_metrics=True,
                traffic_metrics_duration_s=self._poll_interval_s,
            ),
        )

    # ---- order book --------------------------------------------- #

    @property
    def pending_orders(self) -> int:
        return sum(int(c) for _, c in self._orders)

    def _reconcile_orders(self, t_s: float, ready: int, expected: int) -> None:
        """Align the order book with the harness's starting-worker count.

        Orders that the harness reports as ready are retired oldest-first;
        orders it no longer reports (cancelled by a scale-down we did not
        book, or ready early) are dropped newest-first; starting workers we
        never ordered get a full cold start from now.
        """
        self._t_s = t_s
        self._ready = ready
        self._expected = expected
        starting = max(0, expected - ready)
        self._orders.sort(key=lambda o: o[0])
        pending = self.pending_orders
        if pending > starting:
            surplus = pending - starting
            # Retire expired orders first (they became ready) ...
            for order in self._orders:
                if surplus <= 0:
                    break
                if order[0] <= t_s:
                    take = min(order[1], surplus)
                    order[1] -= take
                    surplus -= take
            # ... then the newest ones (cancelled).
            for order in reversed(self._orders):
                if surplus <= 0:
                    break
                take = min(order[1], surplus)
                order[1] -= take
                surplus -= take
            self._orders = [o for o in self._orders if o[1] > 0]
        elif pending < starting:
            self._orders.append([t_s + self._cold_start_s, starting - pending])

    def _book_decision(self, t_s: float, target: int) -> None:
        """Record new orders for a scale-up; cancel pending orders for a scale-down."""
        if target > self._expected:
            self._orders.append([t_s + self._cold_start_s, target - self._expected])
        elif target < self._expected:
            to_cancel = self._expected - target
            self._orders.sort(key=lambda o: o[0])
            for order in reversed(self._orders):
                if to_cancel <= 0:
                    break
                take = min(order[1], to_cancel)
                order[1] -= take
                to_cancel -= take
            self._orders = [o for o in self._orders if o[1] > 0]

    def _active_schedule(self, candidate: int) -> list[int]:
        """Active workers in each plan step if the fleet target becomes ``candidate``.

        Pending orders arrive at their booked time; new orders (candidate
        above the provisioned fleet) arrive after the cold start; a smaller
        target cancels pending orders newest-first and removes active
        workers at once. Late orders (booked time already passed) count from
        the first step.
        """
        dt = self._poll_interval_s
        orders = sorted(([o[0], o[1]] for o in self._orders), key=lambda o: o[0])
        base = self._ready
        new = 0
        if candidate >= self._expected:
            new = candidate - self._expected
        else:
            to_cancel = self._expected - candidate
            for order in reversed(orders):
                if to_cancel <= 0:
                    break
                take = min(order[1], to_cancel)
                order[1] -= take
                to_cancel -= take
            orders = [o for o in orders if o[1] > 0]
            base = max(0, base - to_cancel)
        schedule = []
        for k in range(self._plan_steps):
            t_k = self._t_s + k * dt
            n = base + sum(c for r, c in orders if r <= t_k)
            if new and k >= self._lag_steps:
                n += new
            schedule.append(max(1, n))
        return schedule

    # ---- plant model helpers --------------------------------------------- #

    def _mu(self, rps: float, n: int) -> float:
        """Per-worker service rate (req/s) for the fleet ``n`` offered ``rps``."""
        return self._capacity.service_rate(
            rps / max(1, n), self._isl_now, self._osl_ema, self._kv_hit_ema
        )

    def _prefill_ms(self) -> float:
        return self._capacity.prefill_ms(
            max(1.0, self._isl_now * (1.0 - self._kv_hit_ema))
        )

    def _min_stable_fleet(self, rps: float) -> int:
        """Smallest fleet whose service rate covers ``rps``."""
        for n in range(self._min_replicas, self._max_replicas + 1):
            if rps <= n * self._mu(rps, n):
                return n
        return self._max_replicas

    def _queue_plan(
        self, forecast_rps: list[float], queue_depth: float, candidate: int
    ) -> list[tuple]:
        """Per-step plan for ``candidate``.

        Each entry is ``(queue_after, ttft_arrivals_ms, served, n_active,
        ttft_served_ms, saturated)``: the backlog after the step, the TTFT a
        request arriving at the end of the step sees, the requests completed
        during the step, the active fleet, the wait the served requests saw,
        and whether the fleet ran with a backlog.
        """
        dt = self._poll_interval_s
        t_p = self._prefill_ms()
        schedule = self._active_schedule(candidate)
        queue = float(queue_depth)
        plan = []
        for k in range(self._plan_steps):
            rps = forecast_rps[min(k, len(forecast_rps) - 1)]
            n_active = schedule[k]
            mu = n_active * self._mu(rps, n_active)
            served = min(queue + rps * dt, mu * dt)
            ttft_served = queue / max(1e-6, mu) * 1000.0 + t_p
            saturated = queue > 0.5 or rps >= mu
            queue = max(0.0, queue + (rps - mu) * dt)
            plan.append(
                (
                    queue,
                    queue / max(1e-6, mu) * 1000.0 + t_p,
                    served,
                    n_active,
                    ttft_served,
                    saturated,
                )
            )
        return plan

    def _predict_ttft_ms(self, rps: float, queue_depth: float, n: int) -> float:
        """Worst TTFT over the plan if the fleet target becomes ``n`` now."""
        plan = self._queue_plan([rps], queue_depth, n)
        mu0 = self._active_schedule(n)[0] * self._mu(rps, self._active_schedule(n)[0])
        now = float(queue_depth) / max(1e-6, mu0) * 1000.0 + self._prefill_ms()
        return max([now] + [step[1] for step in plan])

    def _backlog_clears(
        self, forecast_rps: list[float], queue_depth: float, candidate: int
    ) -> bool:
        plan = self._queue_plan(forecast_rps, queue_depth, candidate)
        return plan[-1][0] <= 0.5

    def _downsize_feasible(
        self, forecast_hi: list[float], queue_depth: float, candidate: int
    ) -> bool:
        """Reduced fleet keeps utilisation, TTFT and ITL in bounds at every step."""
        schedule = self._active_schedule(candidate)
        for k, step in enumerate(self._queue_plan(forecast_hi, queue_depth, candidate)):
            ttft = step[1]
            rps = forecast_hi[min(k, len(forecast_hi) - 1)]
            n_active = schedule[k]
            if rps > self._downsize_utilisation * n_active * self._mu(rps, n_active):
                return False
            if ttft >= self._slo_ttft * 0.5:
                return False
            itl = self._step_itl_ms(rps, n_active, step[5])
            if self._latency_calibration.mean_itl(itl) >= self._slo_itl:
                return False
        return True

    def _warm_start_candidates(self) -> list[int]:
        cur = self._current_replicas
        result: list[int] = []
        seen: set[int] = set()
        for delta in range(self._max_replicas + 1):
            for c in (cur + delta, cur - delta):
                if c in seen or c < self._min_replicas or c > self._max_replicas:
                    continue
                seen.add(c)
                result.append(c)
        return result

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _good_fraction(self, ttft_ms: float, itl_ms: float) -> float:
        """Paired empirical SLO success, with a smooth prior until samples arrive."""
        prior = self._sigmoid(
            (self._slo_ttft - ttft_ms) / (0.1 * self._slo_ttft)
        ) * self._sigmoid((self._slo_itl - itl_ms) / (0.1 * self._slo_itl))
        return self._latency_calibration.fraction(ttft_ms, itl_ms, prior)

    def _step_itl_ms(self, rps: float, n_active: int, saturated: bool) -> float:
        return self._capacity.itl_ms(
            rps / max(1, n_active),
            self._isl_now,
            self._osl_ema,
            self._kv_hit_ema,
            saturated=saturated,
        )

    def _evaluate_candidate(
        self, n_replicas: int, forecast_rps: list[float], queue_depth: float
    ) -> float:
        """Negative Dinkelbach surrogate: -(good completions - rho * GPU-seconds)."""
        dt = self._poll_interval_s
        total = 0.0
        for k, (_, _, served, n_active, ttft_served, saturated) in enumerate(
            self._queue_plan(forecast_rps, queue_depth, n_replicas)
        ):
            rps = forecast_rps[min(k, len(forecast_rps) - 1)]
            itl = self._step_itl_ms(rps, n_active, saturated)
            total += served * self._good_fraction(ttft_served, itl)
            # Provisioned count is charged from step 0, starting workers included.
            total -= self._rho * n_replicas * self._gpus_per_worker * dt
        cost = -total
        delta = n_replicas - self._current_replicas
        if delta > 0:
            cost += self._scale_up_penalty * delta
        elif delta < 0:
            cost += self._scale_down_penalty * abs(delta)
        return cost

    def _update_rho(self, observed_rps: float, queue_depth: int) -> None:
        """Measured goodput/GPU-second; use the model until completions arrive."""
        dt = self._poll_interval_s
        ready = max(1, self._ready)
        mu = ready * self._mu(observed_rps, ready)
        saturated = queue_depth > 0 or observed_rps >= mu
        ttft = queue_depth / max(1e-6, mu) * 1000.0 + self._prefill_ms()
        itl = self._step_itl_ms(observed_rps, ready, saturated)
        self._latency_prediction = ((0, self._expected), ttft, itl, self._t_s)
        if (
            self._observed_outcomes
            and self._observed_outcomes["terminal_requests"] > 0
            and self._observed_outcomes["gpu_seconds"] > 0
        ):
            self._rho = max(self._rho_min, self._observed_outcomes["goodput_per_gpu"])
            return
        served = min(queue_depth + observed_rps * dt, mu * dt)
        good = served * self._good_fraction(ttft, itl)
        gpu = max(1, self._expected) * self._gpus_per_worker * dt
        alpha = dt / self._rho_window_s
        if self._good_ema is None or self._gpu_ema is None:
            self._good_ema, self._gpu_ema = good, gpu
        else:
            self._good_ema = (1.0 - alpha) * self._good_ema + alpha * good
            self._gpu_ema = (1.0 - alpha) * self._gpu_ema + alpha * gpu
        self._rho = max(self._rho_min, self._good_ema / max(1e-6, self._gpu_ema))

    # ---- online bias ------------------------------------------------------ #

    def _update_bias(
        self, fpm, ready: int, observed_rps: float, queue_depth: int
    ) -> None:
        """Measure capacity from completions during saturated intervals.

        completions = arrivals - d(queued + running). When the queue stayed
        above the reactive threshold across the interval and the ready fleet
        did not change, completions per ready worker-second is a capacity
        sample for the capacity model's bias.
        """
        running = 0
        if fpm is not None:
            for pool in (fpm.prefill, fpm.decode):
                if pool:
                    for m in pool.values():
                        running += m.scheduled_requests.num_prefill_requests
                        running += m.scheduled_requests.num_decode_requests
        inflight = float(queue_depth + running)
        if (
            self._prev_inflight is not None
            and ready > 0
            and ready == self._prev_ready
            and self._prev_queue > self._reactive_queue_threshold
            and queue_depth > self._reactive_queue_threshold
        ):
            arrivals = observed_rps * self._poll_interval_s
            completions = arrivals - (inflight - self._prev_inflight)
            if completions > 0:
                measured = completions / ready / self._poll_interval_s
                self._capacity.observe_saturated(
                    measured,
                    observed_rps / ready,
                    self._isl_now,
                    self._osl_ema,
                    self._kv_hit_ema,
                )
        self._prev_inflight = inflight
        self._prev_queue = queue_depth
        self._prev_ready = ready

    # ---- control loop ------------------------------------------------------ #

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        if self._mode == "disagg":
            return await self._tick_disagg(scheduled_tick, tick_input)
        t_s = scheduled_tick.at_s
        wc = tick_input.worker_counts
        traffic = tick_input.traffic
        fpm = tick_input.fpm_observations

        if wc is not None:
            self._current_replicas = current_replica_target(
                wc, role="decode", minimum=self._min_replicas
            )
        expected = self._current_replicas
        ready = wc.ready_num_decode if wc is not None else None
        ready = min(expected, ready) if ready is not None else expected
        self._reconcile_orders(t_s, ready, expected)
        self._observe_outcomes(t_s, (0, expected))

        observed_rps, observed_isl = self._observe_traffic(traffic)
        if not self._traffic_observed:
            return self._make_effects(t_s, expected)

        # No rank reported (None) means no observed backlog, as before the
        # helper distinguished "unreported" from an idle zero.
        queue_depth = aggregate_queue_depth(fpm, pool="all") or 0
        kv_util = (
            aggregate_kv_util(fpm, self._capabilities, pool="decode") if fpm else None
        )

        self._update_bias(fpm, self._ready, observed_rps, queue_depth)
        self._update_rho(observed_rps, queue_depth)

        self._forecaster.update(
            observed_rps * self._poll_interval_s, self._poll_interval_s
        )

        forecast_rps, forecast_hi = self._forecaster.forecast(
            self._plan_steps, self._poll_interval_s
        )

        # Lag-aware queue-spike floor: the smallest target whose
        # plan clears the backlog; never below the provisioned fleet while
        # orders are pending, so a marginal plan cannot cancel them.
        reactive_floor = self._min_replicas
        if queue_depth > self._reactive_queue_threshold:
            start = self._expected if self.pending_orders > 0 else self._min_replicas
            floor = self._max_replicas
            for n in range(start, self._max_replicas + 1):
                if self._backlog_clears(forecast_rps, queue_depth, n):
                    floor = n
                    break
            reactive_floor = max(reactive_floor, floor)
        reactive_floor = min(reactive_floor, self._max_replicas)

        peak_rps = max(forecast_rps) if forecast_rps else observed_rps
        min_needed = self._min_stable_fleet(peak_rps)

        # Scale-down rule: eligibility accrues only while no orders
        # are pending and one replica fewer stays feasible under the upper
        # forecast; it takes one cold start of consecutive ticks to unlock.
        if (
            self.pending_orders == 0
            and expected > self._min_replicas
            and self._downsize_feasible(forecast_hi, queue_depth, expected - 1)
        ):
            self._downsize_ticks += 1
        else:
            self._downsize_ticks = 0
        can_downsize = self._downsize_ticks >= max(1, self._lag_steps)

        best_cost = float("inf")
        best_action = expected
        current_cost = None
        consecutive_worse = 0

        for n_replicas in self._warm_start_candidates():
            if n_replicas < expected and not (
                can_downsize
                and self._downsize_feasible(forecast_hi, queue_depth, n_replicas)
            ):
                continue
            cost = self._evaluate_candidate(n_replicas, forecast_rps, queue_depth)
            if n_replicas == expected:
                current_cost = cost
            if cost < best_cost:
                best_cost = cost
                best_action = n_replicas
                consecutive_worse = 0
            else:
                consecutive_worse += 1
                if consecutive_worse >= 4:
                    break

        if current_cost is not None and best_action != expected:
            improvement = (current_cost - best_cost) / max(1e-6, abs(current_cost))
            if improvement < self._deadband_fraction:
                best_action = expected

        best_action = max(best_action, reactive_floor, self._min_replicas)
        best_action = min(best_action, self._max_replicas)

        if best_action != expected:
            self._downsize_ticks = 0
        self._book_decision(t_s, best_action)
        self._current_replicas = best_action
        self._record(
            t_s,
            observed_rps,
            observed_isl,
            queue_depth,
            kv_util,
            reactive_floor,
            best_action,
            forecast_rps,
            forecast_hi,
            min_needed,
            can_downsize=can_downsize,
        )
        return self._make_effects(t_s, best_action)

    # ---- disaggregated pools ------------------------------------ #

    def _observe_traffic(self, traffic) -> tuple[float, float]:
        """Update the shape EWMAs; return (observed rps, observed isl)."""
        observed_rps = 0.0
        observed_isl = 512.0
        if traffic is not None:
            duration = traffic.duration_s if traffic.duration_s else 0.0
            num_req = traffic.num_req if traffic.num_req else 0
            if duration > 0:
                self._traffic_observed = True
                observed_rps = num_req / duration
            observed_isl = float(traffic.isl) if traffic.isl else 512.0
            if num_req > 0:
                a = min(1.0, self._poll_interval_s / self._shape_tau_s)
                self._isl_ema = (
                    observed_isl
                    if self._isl_ema is None
                    else (1 - a) * self._isl_ema + a * observed_isl
                )
                if traffic.osl:
                    osl = float(traffic.osl)
                    self._osl_ema = (
                        osl if not self._osl_seen else (1 - a) * self._osl_ema + a * osl
                    )
                    self._osl_seen = True
                if traffic.kv_hit_rate is not None:
                    self._kv_hit_ema = (1 - a) * self._kv_hit_ema + a * float(
                        traffic.kv_hit_rate
                    )
        self._isl_now = self._isl_ema if self._isl_ema is not None else observed_isl
        return observed_rps, observed_isl

    def _kv_transfer_ms(self) -> float:
        if not self._kv_transfer_gbps or not self._kv_bytes_per_token:
            return 0.0
        bits = self._isl_now * self._kv_bytes_per_token * 8.0
        return bits / (self._kv_transfer_gbps * 1e9) * 1000.0

    def _mu_prefill(self) -> float:
        """Requests/s one prefill worker sustains: serial prefill plus KV transfer."""
        return 1000.0 / max(
            1e-3,
            self._prefill_ms() / self._capacity.prefill_parallelism
            + self._kv_transfer_ms(),
        )

    def _mu_decode(self) -> float:
        """Requests/s one decode worker sustains at its KV concurrency cap."""
        isl, osl = max(1.0, self._isl_now), max(1.0, self._osl_ema)
        c = self._capacity.saturated_concurrency(isl, osl)
        t_d = self._capacity.decode_step_ms(c, isl, osl) / 1000.0
        return self._capacity.decode_parallelism * c / max(1e-3, osl * t_d)

    def _decode_itl_ms(self, load_per_worker: float) -> float:
        """Decode step time at the pool's Little's-law concurrency (no prefill on the worker)."""
        isl, osl = max(1.0, self._isl_now), max(1.0, self._osl_ema)
        cap = self._capacity.saturated_concurrency(isl, osl)
        load_per_rank = load_per_worker / self._capacity.decode_parallelism
        c = 1.0
        for _ in range(30):
            t_d = self._capacity.decode_step_ms(c, isl, osl) / 1000.0
            c_new = min(cap, max(1.0, load_per_rank * osl * t_d))
            if abs(c_new - c) < 0.05:
                c = c_new
                break
            c = c_new
        return self._capacity.decode_step_ms(c, isl, osl)

    def _plan_disagg(
        self, forecast_rps: list[float], queue_depth: float, n_p: int, n_d: int
    ) -> list[tuple]:
        """Per-step (queue_after, ttft_served_ms, served, itl_ms) for the pair."""
        dt = self._poll_interval_s
        sched_p = self._book_p.schedule(n_p)
        sched_d = self._book_d.schedule(n_d)
        mu_p, mu_d = self._mu_prefill(), self._mu_decode()
        first_token_ms = self._prefill_ms() + self._kv_transfer_ms()
        queue = float(queue_depth)
        plan = []
        for k in range(self._plan_steps):
            rps = forecast_rps[min(k, len(forecast_rps) - 1)]
            cap_p, cap_d = sched_p[k] * mu_p, sched_d[k] * mu_d
            cap = min(cap_p, cap_d)
            served = min(queue + rps * dt, cap * dt)
            ttft = queue / max(1e-6, cap) * 1000.0 + first_token_ms
            decode_saturated = rps >= cap_d or (queue > 0.5 and cap_d <= cap_p)
            itl = (
                2.0 * self._slo_itl
                if decode_saturated
                else self._decode_itl_ms(rps / sched_d[k])
            )
            queue = max(0.0, queue + (rps - cap) * dt)
            plan.append((queue, ttft, served, itl))
        return plan

    def _cost_disagg(
        self,
        forecast_rps: list[float],
        queue_depth: float,
        n_p: int,
        n_d: int,
        plan: list[tuple] | None = None,
    ) -> float:
        dt = self._poll_interval_s
        gpu = n_p * self._prefill_gpus + n_d * self._gpus_per_worker
        total = 0.0
        if plan is None:
            plan = self._plan_disagg(forecast_rps, queue_depth, n_p, n_d)
        for _, ttft, served, itl in plan:
            total += served * self._good_fraction(ttft, itl) - self._rho * gpu * dt
        cost = -total
        cost += self._scale_up_penalty * max(
            0, n_p - self._book_p.expected
        ) + self._scale_up_penalty * max(0, n_d - self._book_d.expected)
        cost += self._scale_down_penalty * max(
            0, self._book_p.expected - n_p
        ) + self._scale_down_penalty * max(0, self._book_d.expected - n_d)
        return cost

    def _feasible_disagg(
        self, forecast_hi: list[float], queue_depth: float, n_p: int, n_d: int
    ) -> bool:
        """Both pools stay under the utilisation bound and both SLOs at every step."""
        sched_p = self._book_p.schedule(n_p)
        sched_d = self._book_d.schedule(n_d)
        mu_p, mu_d = self._mu_prefill(), self._mu_decode()
        for k, (_, ttft, _, itl) in enumerate(
            self._plan_disagg(forecast_hi, queue_depth, n_p, n_d)
        ):
            rps = forecast_hi[min(k, len(forecast_hi) - 1)]
            if rps > self._downsize_utilisation * sched_p[k] * mu_p:
                return False
            if rps > self._downsize_utilisation * sched_d[k] * mu_d:
                return False
            if (
                ttft >= self._slo_ttft * 0.5
                or self._latency_calibration.mean_itl(itl) >= self._slo_itl
            ):
                return False
        return True

    def _update_rho_disagg(self, observed_rps: float, queue_depth: int) -> None:
        dt = self._poll_interval_s
        cap_p = max(1, self._book_p.ready) * self._mu_prefill()
        cap_d = max(1, self._book_d.ready) * self._mu_decode()
        cap = min(cap_p, cap_d)
        ttft = (
            queue_depth / max(1e-6, cap) * 1000.0
            + self._prefill_ms()
            + self._kv_transfer_ms()
        )
        saturated_d = observed_rps >= cap_d or (queue_depth > 0 and cap_d <= cap_p)
        itl = (
            2.0 * self._slo_itl
            if saturated_d
            else self._decode_itl_ms(observed_rps / max(1, self._book_d.ready))
        )
        self._latency_prediction = (
            (self._book_p.expected, self._book_d.expected),
            ttft,
            itl,
            self._book_d.t_s,
        )
        if (
            self._observed_outcomes
            and self._observed_outcomes["terminal_requests"] > 0
            and self._observed_outcomes["gpu_seconds"] > 0
        ):
            self._rho = max(self._rho_min, self._observed_outcomes["goodput_per_gpu"])
            return
        served = min(queue_depth + observed_rps * dt, cap * dt)
        good = served * self._good_fraction(ttft, itl)
        gpu = (
            max(1, self._book_p.expected) * self._prefill_gpus
            + max(1, self._book_d.expected) * self._gpus_per_worker
        ) * dt
        alpha = dt / self._rho_window_s
        if self._good_ema is None or self._gpu_ema is None:
            self._good_ema, self._gpu_ema = good, gpu
        else:
            self._good_ema = (1.0 - alpha) * self._good_ema + alpha * good
            self._gpu_ema = (1.0 - alpha) * self._gpu_ema + alpha * gpu
        self._rho = max(self._rho_min, self._good_ema / max(1e-6, self._gpu_ema))

    def _reallocation_improves(
        self, forecast: list[float], queue_depth: float, pair: tuple[int, int]
    ) -> bool:
        """Allow a pool exchange when the coupled plan improves without a new bottleneck."""
        p, d = pair
        current = (self._book_p.expected, self._book_d.expected)
        if (p - current[0]) * (d - current[1]) >= 0:
            return False
        if self._book_p.pending or self._book_d.pending:
            return False
        before = self._plan_disagg(forecast, queue_depth, *current)
        after = self._plan_disagg(forecast, queue_depth, p, d)
        for old, new in zip(before, after):
            if new[0] > old[0] + 1e-6:
                return False
            if new[1] > max(self._slo_ttft, old[1]) + 1e-6:
                return False
            if new[3] > max(self._slo_itl, old[3]) + 1e-6:
                return False
        old_good = sum(s * self._good_fraction(t, i) for _, t, s, i in before)
        new_good = sum(s * self._good_fraction(t, i) for _, t, s, i in after)
        return after[-1][0] < before[-1][0] - 0.5 or new_good > old_good * 1.05 + 1e-6

    def _disagg_effects(self, t_s: float, p: int, d: int) -> PlannerEffects:
        return PlannerEffects(
            scale_to=ScalingDecision(num_prefill=p, num_decode=d),
            next_tick=self.initial_tick(t_s + self._poll_interval_s),
        )

    async def _tick_disagg(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        t_s = scheduled_tick.at_s
        wc = tick_input.worker_counts
        fpm = tick_input.fpm_observations
        lo = self._min_replicas

        exp_p = current_replica_target(wc, role="prefill", minimum=lo)
        exp_d = current_replica_target(wc, role="decode", minimum=lo)
        ready_p = wc.ready_num_prefill if wc is not None else None
        ready_d = wc.ready_num_decode if wc is not None else None
        ready_p = min(exp_p, ready_p) if ready_p is not None else exp_p
        ready_d = min(exp_d, ready_d) if ready_d is not None else exp_d
        self._book_p.reconcile(t_s, ready_p, exp_p)
        self._book_d.reconcile(t_s, ready_d, exp_d)
        self._observe_outcomes(t_s, (exp_p, exp_d))

        observed_rps, observed_isl = self._observe_traffic(tick_input.traffic)
        if not self._traffic_observed:
            return self._disagg_effects(t_s, exp_p, exp_d)
        # None (no rank reported) counts as no observed backlog.
        q_p = aggregate_queue_depth(fpm, pool="prefill") or 0
        q_d = aggregate_queue_depth(fpm, pool="decode") or 0
        queue_depth = q_p + q_d
        kv_util = (
            aggregate_kv_util(fpm, self._capabilities, pool="decode") if fpm else None
        )

        self._update_rho_disagg(observed_rps, queue_depth)
        self._forecaster.update(
            observed_rps * self._poll_interval_s, self._poll_interval_s
        )
        forecast_rps, forecast_hi = self._forecaster.forecast(
            self._plan_steps, self._poll_interval_s
        )

        def gpu_cost(n_p: int, n_d: int) -> int:
            return n_p * self._prefill_gpus + n_d * self._gpus_per_worker

        # Scale-down eligibility per pool.
        for book, other_n, is_p in (
            (self._book_p, exp_d, True),
            (self._book_d, exp_p, False),
        ):
            if book.pending == 0 and book.expected > lo:
                pair = (
                    (book.expected - 1, other_n)
                    if is_p
                    else (other_n, book.expected - 1)
                )
                feasible = self._feasible_disagg(forecast_hi, queue_depth, *pair)
            else:
                feasible = False
            book.downsize_ticks = book.downsize_ticks + 1 if feasible else 0
        can_down_p = self._book_p.downsize_ticks >= max(1, self._lag_steps)
        can_down_d = self._book_d.downsize_ticks >= max(1, self._lag_steps)

        current = (exp_p, exp_d)
        current_is_feasible = current in self._fleet_targets
        costs = {}
        plans = {}
        for a, b in self._fleet_targets:
            if (a < exp_p or b < exp_d) and current_is_feasible:
                guarded_downsize = (
                    (a >= exp_p or can_down_p)
                    and (b >= exp_d or can_down_d)
                    and self._feasible_disagg(forecast_hi, queue_depth, a, b)
                )
                if not guarded_downsize and not self._reallocation_improves(
                    forecast_hi, queue_depth, (a, b)
                ):
                    continue
            plans[a, b] = self._plan_disagg(forecast_rps, queue_depth, a, b)
            costs[a, b] = self._cost_disagg(
                forecast_rps, queue_depth, a, b, plan=plans[a, b]
            )
        best = min(costs, key=costs.__getitem__)
        best_cost = costs[best]
        current_cost = costs.get(current)
        if (
            current_cost is not None
            and best != current
            and (current_cost - best_cost) / max(1e-6, abs(current_cost))
            < self._deadband_fraction
        ):
            best = current

        # Choose a whole feasible pair. Independent per-pool floors can create
        # an impossible allocation that the external limiter would distort.
        recovery = None
        if queue_depth > self._reactive_queue_threshold:
            clearing = [pair for pair, plan in plans.items() if plan[-1][0] <= 0.5]
            if clearing:
                if best not in clearing:
                    best = min(clearing, key=costs.__getitem__)
            else:
                # Under sustained overload, maximize achievable queue relief;
                # spare capacity in the other pool has no benefit.
                best = min(
                    plans,
                    key=lambda pair: (
                        plans[pair][-1][0],
                        gpu_cost(*pair),
                        costs[pair],
                    ),
                )
            recovery = best
        n_p, n_d = best
        if n_p != exp_p:
            self._book_p.downsize_ticks = 0
        if n_d != exp_d:
            self._book_d.downsize_ticks = 0
        self._book_p.book(t_s, n_p)
        self._book_d.book(t_s, n_d)
        self._current_replicas = n_d
        self._history.append(
            {
                "t_s": t_s,
                "rps": observed_rps,
                "isl": observed_isl,
                "isl_ema": self._isl_now,
                "osl": self._osl_ema,
                "queue_prefill": q_p,
                "queue_decode": q_d,
                "kv_util": kv_util,
                "ready_p": ready_p,
                "observed_outcomes": self._observed_outcomes,
                "latency_calibration_samples": len(self._latency_calibration.residuals),
                "expected_p": exp_p,
                "pending_p": self._book_p.pending,
                "ready_d": ready_d,
                "expected_d": exp_d,
                "pending_d": self._book_d.pending,
                "mu_prefill": self._mu_prefill(),
                "mu_decode": self._mu_decode(),
                "rho": self._rho,
                "recovery_target": recovery,
                "can_downsize": (can_down_p, can_down_d),
                "decision": (n_p, n_d),
                "forecast_rps": forecast_rps[0] if forecast_rps else 0,
            }
        )
        return self._disagg_effects(t_s, n_p, n_d)

    def _record(
        self,
        t_s,
        rps,
        isl,
        queue,
        kv_util,
        reactive_floor,
        decision,
        forecast_rps,
        forecast_hi,
        min_needed,
        *,
        can_downsize: bool,
    ) -> None:
        self._history.append(
            {
                "t_s": t_s,
                "rps": rps,
                "isl": isl,
                "osl": self._osl_ema,
                "queue": queue,
                "kv_util": kv_util,
                "ready": self._ready,
                "observed_outcomes": self._observed_outcomes,
                "latency_calibration_samples": len(self._latency_calibration.residuals),
                "expected": self._expected,
                "pending_orders": self.pending_orders,
                "reactive_floor": reactive_floor,
                "min_needed": min_needed,
                "mu_per_worker": self._mu(rps, max(1, decision)),
                "bias": self._capacity.bias,
                "rho": self._rho,
                "can_downsize": can_downsize,
                "downsize_ticks": self._downsize_ticks,
                "decision": decision,
                "forecast_rps": forecast_rps[0] if forecast_rps else 0,
                "forecast_hi": forecast_hi[0] if forecast_hi else 0,
                "isl_ema": self._isl_now,
            }
        )

    async def shutdown(self) -> None:
        return None

    @property
    def decision_history(self) -> list[dict]:
        return list(self._history)
