# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SyntheticPowerOverlay — γ-class power signal synthesiser.

Reads mocker FPM snapshots, computes per-component per-GPU power signals
using the deterministic formulas from §5.2 of the design, applies bias × noise
from the scenario timeline, and exposes results via ``observation_at(tick)``
(same interface as SyntheticFleet so FakePrometheusClient is class-agnostic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.fake_actuator import AppliedCaps
    from dynamo.planner.tests.testbed.scenarios import (
        OverlaySpec,
        ScenarioSpec,
        SystemSpec,
    )


@dataclass
class OverlayObservation:
    """Power signals synthesised from one tick's FPM snapshots."""

    tick: int
    power_w_prefill: float  # per-GPU average
    power_w_decode: float  # per-GPU average

    # Compatibility with SyntheticFleet.observation_at() interface:
    # FakePrometheusClient checks .ttft_avg_s, .itl_avg_s as well.
    # We expose None for latency fields — FakePrometheus returns None for
    # those signals from γ-class (latency comes from mocker traffic drain).
    ttft_avg_s: Optional[float] = None
    itl_avg_s: Optional[float] = None
    total_tokens_per_sec: Optional[float] = None
    traffic: Any = None


class SyntheticPowerOverlay:
    """Derives per-component power signals from mocker FPM snapshots.

    Implements the same ``observation_at(tick)`` + ``is_signal_in_outage()``
    + ``prom_stale_lag()`` interface as SyntheticFleet so FakePrometheusClient
    is a drop-in for both α and γ-class.
    """

    def __init__(
        self,
        overlay_spec: "OverlaySpec",
        system_spec: "SystemSpec",
        scenario: "ScenarioSpec",
        rng: random.Random,
    ) -> None:
        self._spec = overlay_spec
        self._system = system_spec
        self._scenario = scenario
        self._rng = rng

        # Current applied caps (updated by ReplayFakeActuator via notify_caps_changed)
        self._applied_cap_p: int = scenario.planner.prefill_engine_gpu_power_limit
        self._applied_cap_d: int = scenario.planner.decode_engine_gpu_power_limit

        self._history: dict[int, OverlayObservation] = {}
        self._latest: Optional[OverlayObservation] = None

        # Active observability faults (copied from fleet state interface)
        self._active_prom_outage: dict[str, int] = {}
        self._active_prom_stale: Optional[tuple[int, int]] = None
        self._ar1_state: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Main per-tick observation
    # ------------------------------------------------------------------

    def observe(
        self,
        fpm_snapshots: list[dict[str, Any]],
        applied_caps: "AppliedCaps",
        tick: int,
    ) -> None:
        """Synthesise power signals from this tick's FPM snapshots.

        Called after bridge.advance_to() returns snapshots for the tick.
        """
        prefill_snaps = [s for s in fpm_snapshots if s.get("component") == "prefill"]
        decode_snaps = [s for s in fpm_snapshots if s.get("component") == "decode"]

        # Honour the overlay spec's own bias defaults — these are the
        # "steady-state" multipliers the scenario author set on
        # ``overlay.bias.power_bias_*``. Events (BiasStepEvent / BiasRampEvent)
        # below can override the per-component bias dynamically per tick.
        bias_p = self._spec.bias.power_bias_prefill
        bias_d = self._spec.bias.power_bias_decode

        # Read scenario bias timeline for this tick
        for event in self._scenario.parsed_events():
            from dynamo.planner.tests.testbed.scenarios import (
                BiasRampEvent,
                BiasStepEvent,
            )

            if isinstance(event, BiasStepEvent) and event.at_tick <= tick:
                if "prefill" in event.signal:
                    bias_p = event.value
                elif "decode" in event.signal:
                    bias_d = event.value
            elif isinstance(event, BiasRampEvent):
                if event.start_tick <= tick <= event.end_tick:
                    t = (tick - event.start_tick) / max(
                        1, event.end_tick - event.start_tick
                    )
                    val = event.from_ + t * (event.to - event.from_)
                    if "prefill" in event.signal:
                        bias_p = val
                    elif "decode" in event.signal:
                        bias_d = val

        p_w = self._aggregate_power(
            prefill_snaps, applied_caps.cap_p, self._predict_prefill_power
        )
        d_w = self._aggregate_power(
            decode_snaps, applied_caps.cap_d, self._predict_decode_power
        )

        p_w *= bias_p * (1.0 + self._noise("power_per_gpu"))
        d_w *= bias_d * (1.0 + self._noise("power_per_gpu"))

        obs = OverlayObservation(
            tick=tick,
            power_w_prefill=max(0.0, p_w),
            power_w_decode=max(0.0, d_w),
        )
        self._history[tick] = obs
        self._latest = obs

    def observation_at(self, tick: int) -> Optional[OverlayObservation]:
        return self._history.get(tick)

    def notify_caps_changed(self, new_cap_p: int, new_cap_d: int) -> None:
        """Called by ReplayFakeActuator when caps are applied (possibly clamped)."""
        self._applied_cap_p = new_cap_p
        self._applied_cap_d = new_cap_d

    # ------------------------------------------------------------------
    # FakePrometheusClient interface stubs
    # ------------------------------------------------------------------

    def is_signal_in_outage(self, signal: str) -> bool:
        return signal in self._active_prom_outage

    def prom_stale_lag(self) -> Optional[int]:
        if self._active_prom_stale:
            return self._active_prom_stale[1]
        return None

    def set_prom_outage(self, signals: list[str], end_tick: int) -> None:
        for sig in signals:
            self._active_prom_outage[sig] = end_tick

    def clear_expired_events(self, tick: int) -> None:
        expired = [s for s, end in self._active_prom_outage.items() if tick >= end]
        for s in expired:
            del self._active_prom_outage[s]
        if self._active_prom_stale and tick >= self._active_prom_stale[0]:
            self._active_prom_stale = None

    # ------------------------------------------------------------------
    # Power prediction formulas (§5.2)
    # ------------------------------------------------------------------

    def _predict_prefill_power(self, snap: dict[str, Any], applied_cap_w: int) -> float:
        """Compute-bound regime. Power scales with GEMM intensity, clamped by cap."""
        tdp = self._system.tdp_w
        sku_min = self._system.sku_min_w
        sat_tokens = self._system.overlay_prefill_saturation_tokens
        base = 0.6 * tdp
        gemm_load = min(1.0, snap.get("sum_prefill_tokens", 0) / max(1, sat_tokens))
        aic_predicted = base + (applied_cap_w - base) * gemm_load
        return max(sku_min, min(applied_cap_w, aic_predicted))

    def _predict_decode_power(self, snap: dict[str, Any], applied_cap_w: int) -> float:
        """Memory-bound regime. Power scales with KV traffic, less sensitive to cap."""
        tdp = self._system.tdp_w
        sku_min = self._system.sku_min_w
        hbm_tokens = self._system.overlay_decode_hbm_tokens
        base = 0.5 * tdp
        hbm_load = min(1.0, snap.get("sum_decode_kv_tokens", 0) / max(1, hbm_tokens))
        aic_predicted = base + (applied_cap_w - base) * (0.3 + 0.7 * hbm_load)
        return max(sku_min, min(applied_cap_w, aic_predicted))

    def _aggregate_power(
        self,
        snaps: list[dict[str, Any]],
        applied_cap_w: int,
        predict_fn,
    ) -> float:
        if not snaps:
            # Fall back to system TDP × idle fraction when no FPMs
            return self._system.tdp_w * 0.5
        total = sum(predict_fn(s, applied_cap_w) for s in snaps)
        return total / len(snaps)

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _noise(self, signal: str) -> float:
        spec = getattr(self._spec.noise, signal, None) or self._spec.noise.power_per_gpu
        from dynamo.planner.tests.testbed.synthetic_fleet import _sample_noise

        return _sample_noise(spec, self._rng, f"{signal}_ar1", self._ar1_state)
