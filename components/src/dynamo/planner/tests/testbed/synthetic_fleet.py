# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SyntheticFleet — truth model for α-class scenarios.

Implements: truth(t) = aic_prediction × bias_signal(t) × (1 + noise(t))

Truth-side response to actuation (§4.4):
  - cap_p ↓ → TTFT inflates as tdp_w / cap_p (compute-bound)
  - cap_d ↓ below decode_power_floor → ITL inflates (memory-bound)
  - offered_load > capacity → TTFT/ITL inflate via M/M/1 proxy
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from dynamo.planner.core.types import TrafficObservation

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.scenarios import (
        ActuationFaultEvent,
        AicFailureEvent,
        BiasRampEvent,
        BiasSineEvent,
        BiasStepEvent,
        BudgetChangeEvent,
        Event,
        FleetSpec,
        MdcUnavailableEvent,
        NodeDownEvent,
        NodeUpEvent,
        PromOutageEvent,
        PromStaleEvent,
        PromWindowCrossEvent,
        FrontendPostFaultEvent,
        ScenarioSpec,
        SystemSpec,
    )


@dataclass
class FleetState:
    """Mutable fleet state — what's currently running."""

    n_p_truth: int = 1
    n_d_truth: int = 4
    applied_cap_p: int = 500
    applied_cap_d: int = 425

    # Bias signals (scenario-controlled)
    bias_power_p: float = 1.0
    bias_power_d: float = 1.0
    bias_ttft: float = 1.0
    bias_itl: float = 1.0
    bias_capacity: float = 1.0

    # AR1 noise state (per signal)
    ar1_state: dict[str, float] = field(default_factory=dict)


@dataclass
class Observation:
    """One tick's truth-side observation (post-noise)."""

    traffic: TrafficObservation
    ttft_avg_s: float
    itl_avg_s: float
    power_w_prefill: float
    power_w_decode: float
    total_tokens_per_sec: float


class SyntheticFleet:
    """Truth model for α-class scenarios.

    Computes per-tick observations from AIC predictions multiplied by the
    current bias and noise, applying actuation effects (cap clamping, replica
    changes) according to fleet state.
    """

    def __init__(
        self,
        fleet_spec: "FleetSpec",
        system_spec: "SystemSpec",
        scenario: "ScenarioSpec",
        rng: random.Random,
    ) -> None:
        self._fleet = fleet_spec
        self._system = system_spec
        self._scenario = scenario
        self._rng = rng

        # Initialise state from fleet spec
        self.state = FleetState(
            n_p_truth=1,
            n_d_truth=fleet_spec.gpus_per_decode_engine,
            applied_cap_p=scenario.planner.prefill_engine_gpu_power_limit,
            applied_cap_d=scenario.planner.decode_engine_gpu_power_limit,
            bias_power_p=fleet_spec.bias.power_bias_prefill,
            bias_power_d=fleet_spec.bias.power_bias_decode,
            bias_ttft=fleet_spec.bias.ttft_bias,
            bias_itl=fleet_spec.bias.itl_bias,
            bias_capacity=fleet_spec.bias.capacity_bias,
        )

        # Active event tracking
        self._active_prom_outage: dict[str, int] = {}   # signal → end_tick
        self._active_prom_stale: Optional[tuple[int, int]] = None  # (end_tick, lag_ticks)
        self._active_actuation_fault: Optional[tuple[int, str]] = None  # (end_tick, mode)
        self._active_frontend_fault: Optional[tuple[int, float]] = None  # (end_tick, fraction)
        self._active_mdc_unavailable: int = 0  # end_tick
        self._sine_events: list[BiasSineEvent] = []
        self._ramp_events: list[BiasRampEvent] = []
        self._window_cross_events: dict[str, tuple[int, float]] = {}  # signal → (tick, weight)
        self._observation_history: list[Optional[Observation]] = []  # indexed by tick

    # ------------------------------------------------------------------
    # Main per-tick step
    # ------------------------------------------------------------------

    def step(self, tick: int, offered_load: float) -> Observation:
        """Apply pending events, compute the truth-side observation."""
        # Compute AIC predictions at current applied config
        aic_ttft_ms = self._system.aic_ttft_ms
        aic_itl_ms = self._system.aic_itl_ms
        aic_power_w_p = self._system.aic_power_w_prefill
        aic_power_w_d = self._system.aic_power_w_decode

        # Compute capacity
        base_capacity_tps = (
            self.state.n_d_truth
            * self._system.aic_itl_ms
            * 10.0  # rough: 10 tok/s per decode replica per GPU at base
        )
        # More realistic: seq_per_s × (isl + osl) per replica
        # Using AIC ITL: max_concurrency × 1000 / (itl_ms × osl)
        max_kv = self._system.max_kv_tokens
        osl = 150  # default osl from planner config
        isl = 3000
        max_concurrency = max(1, max_kv // (isl + osl))
        seq_per_s_per_replica = max_concurrency * 1000.0 / max(0.001, aic_itl_ms * osl)
        base_capacity_tps = seq_per_s_per_replica * self.state.n_d_truth * (isl + osl)
        capacity_tps = base_capacity_tps * self.state.bias_capacity

        # Cap effect on compute-bound prefill (TTFT inflates as tdp_w / cap_p)
        tdp = self._system.tdp_w
        cap_p = max(1, self.state.applied_cap_p)
        ttft_cap_factor = tdp / cap_p
        true_ttft_ms = aic_ttft_ms * ttft_cap_factor * self.state.bias_ttft

        # Cap effect on memory-bound decode
        floor = self._fleet.decode_power_floor_w or self._system.decode_power_floor_w
        cap_d = max(1, self.state.applied_cap_d)
        if cap_d < floor:
            itl_cap_factor = floor / cap_d
        else:
            itl_cap_factor = 1.0
        true_itl_ms = aic_itl_ms * itl_cap_factor * self.state.bias_itl

        # Queue saturation proxy (M/M/1 inflation when offered > capacity)
        utilization = min(0.99, offered_load / max(1.0, capacity_tps))
        queue_factor = 1.0 / max(0.01, 1.0 - utilization)
        queue_factor = min(queue_factor, 10.0)  # cap to avoid explosion
        true_ttft_ms *= queue_factor
        true_itl_ms *= queue_factor

        # Power signals
        true_power_w_p = aic_power_w_p * self.state.bias_power_p
        true_power_w_d = aic_power_w_d * self.state.bias_power_d

        # Apply sine biases
        for ev in self._sine_events:
            factor = 1.0 + ev.amplitude * math.sin(
                2 * math.pi * tick / max(1, ev.period_ticks) + ev.offset
            )
            if ev.signal in ("power_bias_prefill", "power_p"):
                true_power_w_p *= factor
            elif ev.signal in ("power_bias_decode", "power_d"):
                true_power_w_d *= factor
            elif ev.signal in ("ttft_bias", "ttft"):
                true_ttft_ms *= factor
            elif ev.signal in ("itl_bias", "itl"):
                true_itl_ms *= factor

        # Apply ramp biases
        for ev in self._ramp_events:
            if ev.start_tick <= tick <= ev.end_tick:
                t = (tick - ev.start_tick) / max(1, ev.end_tick - ev.start_tick)
                factor = ev.from_ + t * (ev.to - ev.from_)
                _apply_bias_factor(self.state, ev.signal, factor)

        # Apply noise
        true_ttft_ms *= 1.0 + self._noise("ttft", tick)
        true_itl_ms *= 1.0 + self._noise("itl", tick)
        true_power_w_p *= 1.0 + self._noise("power_per_gpu", tick)
        true_power_w_d *= 1.0 + self._noise("power_per_gpu", tick)
        capacity_tps *= 1.0 + self._noise("capacity", tick)

        # Apply prom window-cross mixing (one-tick mixed window after events)
        for signal, (cross_tick, weight_old) in list(self._window_cross_events.items()):
            if tick == cross_tick and tick > 0:
                prev = self._observation_history[-1] if self._observation_history else None
                if prev is not None:
                    if signal in ("power_d", "power_bias_decode"):
                        pre_val = prev.power_w_decode
                        true_power_w_d = weight_old * pre_val + (1 - weight_old) * true_power_w_d
                    elif signal in ("power_p", "power_bias_prefill"):
                        pre_val = prev.power_w_prefill
                        true_power_w_p = weight_old * pre_val + (1 - weight_old) * true_power_w_p
                del self._window_cross_events[signal]

        throughput = min(offered_load, max(0.0, capacity_tps))
        traffic = TrafficObservation(
            duration_s=self._scenario.interval_s,
            num_req=max(0, int(throughput / max(1, isl + osl))),
            isl=float(isl),
            osl=float(osl),
            kv_hit_rate=0.5,
            ttft_avg=true_ttft_ms / 1000.0,
            itl_avg=true_itl_ms / 1000.0,
            total_tokens_per_s=throughput,
            scheduled_prefill_tokens=max(0.0, throughput * 0.4),
            scheduled_decode_kv_tokens=max(0.0, throughput * 0.6),
        )

        obs = Observation(
            traffic=traffic,
            ttft_avg_s=true_ttft_ms / 1000.0,
            itl_avg_s=true_itl_ms / 1000.0,
            power_w_prefill=max(0.0, true_power_w_p),
            power_w_decode=max(0.0, true_power_w_d),
            total_tokens_per_sec=throughput,
        )

        self._observation_history.append(obs)
        return obs

    def observation_at(self, tick: int) -> Optional[Observation]:
        """Return observation for given tick (for prom_stale lag)."""
        if 0 <= tick < len(self._observation_history):
            return self._observation_history[tick]
        return None

    # ------------------------------------------------------------------
    # Event application (called by runner per tick)
    # ------------------------------------------------------------------

    def apply_event(self, event: "Event", tick: int) -> None:
        """Mutate fleet state based on the event."""
        from dynamo.planner.tests.testbed.scenarios import (
            BiasStepEvent,
            BiasRampEvent,
            BiasSineEvent,
            ActuationFaultEvent,
            NodeDownEvent,
            NodeUpEvent,
            PromOutageEvent,
            PromStaleEvent,
            PromWindowCrossEvent,
            BudgetChangeEvent,
            FrontendPostFaultEvent,
            MdcUnavailableEvent,
            AicFailureEvent,
        )

        if isinstance(event, BiasStepEvent):
            _apply_bias_factor(self.state, event.signal, event.value)
            if event.auto_inject_window_cross:
                self._window_cross_events[event.signal] = (tick + 1, 0.66)

        elif isinstance(event, BiasRampEvent):
            self._ramp_events.append(event)

        elif isinstance(event, BiasSineEvent):
            self._sine_events.append(event)

        elif isinstance(event, ActuationFaultEvent):
            self._active_actuation_fault = (tick + event.duration_ticks, event.mode)

        elif isinstance(event, NodeDownEvent):
            self.state.n_p_truth = max(0, self.state.n_p_truth - event.n_prefill_lost)
            self.state.n_d_truth = max(0, self.state.n_d_truth - event.n_decode_lost)

        elif isinstance(event, NodeUpEvent):
            self.state.n_p_truth += event.n_prefill_restored
            self.state.n_d_truth += event.n_decode_restored

        elif isinstance(event, PromOutageEvent):
            for sig in event.signals:
                self._active_prom_outage[sig] = tick + event.duration_ticks

        elif isinstance(event, PromStaleEvent):
            self._active_prom_stale = (tick + event.duration_ticks, event.lag_ticks)

        elif isinstance(event, PromWindowCrossEvent):
            self._window_cross_events[event.signal] = (tick, event.weight_old)

        elif isinstance(event, BudgetChangeEvent):
            # Budget change is handled by the runner (mutates planner config)
            pass

        elif isinstance(event, FrontendPostFaultEvent):
            self._active_frontend_fault = (tick + event.duration_ticks, event.failing_fraction)

        elif isinstance(event, MdcUnavailableEvent):
            self._active_mdc_unavailable = tick + event.duration_ticks

    def clear_expired_events(self, tick: int) -> None:
        """Remove expired timed events."""
        expired = [sig for sig, end in self._active_prom_outage.items() if tick >= end]
        for sig in expired:
            del self._active_prom_outage[sig]

        if self._active_prom_stale and tick >= self._active_prom_stale[0]:
            self._active_prom_stale = None

        if self._active_actuation_fault and tick >= self._active_actuation_fault[0]:
            self._active_actuation_fault = None

        if self._active_frontend_fault and tick >= self._active_frontend_fault[0]:
            self._active_frontend_fault = None

    # ------------------------------------------------------------------
    # Observability helpers for FakePrometheusClient
    # ------------------------------------------------------------------

    def is_signal_in_outage(self, signal: str) -> bool:
        return signal in self._active_prom_outage

    def prom_stale_lag(self) -> Optional[int]:
        if self._active_prom_stale:
            return self._active_prom_stale[1]
        return None

    def actuation_fault(self) -> Optional[str]:
        if self._active_actuation_fault:
            return self._active_actuation_fault[1]
        return None

    def frontend_fault(self) -> Optional[float]:
        if self._active_frontend_fault:
            return self._active_frontend_fault[1]
        return None

    def mdc_unavailable(self, tick: int) -> bool:
        return tick < self._active_mdc_unavailable

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _noise(self, signal: str, tick: int) -> float:
        noise_spec = getattr(self._fleet.noise, signal, None)
        if noise_spec is None:
            noise_spec = self._fleet.noise.power_per_gpu
        return _sample_noise(noise_spec, self._rng, f"{signal}_ar1", self.state.ar1_state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_bias_factor(state: FleetState, signal: str, value: float) -> None:
    if signal in ("power_bias_prefill", "power_p"):
        state.bias_power_p = value
    elif signal in ("power_bias_decode", "power_d"):
        state.bias_power_d = value
    elif signal in ("ttft_bias", "ttft"):
        state.bias_ttft = value
    elif signal in ("itl_bias", "itl"):
        state.bias_itl = value
    elif signal in ("capacity_bias", "capacity"):
        state.bias_capacity = value


def _sample_noise(
    spec: "NoiseModel",
    rng: random.Random,
    ar1_key: str,
    ar1_state: dict[str, float],
) -> float:
    from dynamo.planner.tests.testbed.scenarios import NoiseModel
    if spec.model == "gaussian":
        sigma = spec.sigma
        if sigma == 0.0:
            return 0.0
        raw = rng.gauss(0, sigma)
        return max(-3 * sigma, min(3 * sigma, raw))
    elif spec.model == "uniform":
        h = spec.half_width
        return rng.uniform(-h, h)
    elif spec.model == "ar1":
        prev = ar1_state.get(ar1_key, 0.0)
        eps = rng.gauss(0, spec.sigma)
        n = spec.rho * prev + eps
        ar1_state[ar1_key] = n
        return n
    return 0.0
