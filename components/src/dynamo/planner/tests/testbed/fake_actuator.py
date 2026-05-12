# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FakeActuator — α-class PlannerConnector implementation.

Replaces KubernetesConnector + Power Agent NVML for the testbed.  All
actuation lands in in-memory state.  Fault hooks inject:
  - ``rbac_denied``     → set_component_replicas raises 403-style RuntimeError
  - ``nvml_low``        → cap clamped up to sku_min_w
  - ``nvml_high``       → cap clamped down to sku_max_w
  - ``daemonset_absent``→ patch_pod_annotation silently no-ops (annotation recorded,
                          no side-effect in truth model → truth draws at TDP)
  - ``frontend_post``   → post_busy_threshold raises for a fraction of calls

The γ-class subclass (ReplayFakeActuator) is in replay/replay_fake_actuator.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.base import PlannerConnector

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics
    from dynamo.planner.tests.testbed.scenarios import ScenarioSpec, SystemSpec
    from dynamo.planner.tests.testbed.synthetic_fleet import SyntheticFleet


@dataclass
class AppliedCaps:
    cap_p: int
    cap_d: int


class FakeActuator(PlannerConnector):
    """α-class actuator — pure in-memory state.

    Tracks applied replica counts and per-GPU caps.  Reflects applied caps
    back into the SyntheticFleet truth model so the closed loop is consistent.
    """

    def __init__(
        self,
        scenario: "ScenarioSpec",
        fleet: "SyntheticFleet",
        metrics: "FakePlannerMetrics",
        system_spec: "SystemSpec",
    ) -> None:
        self._scenario = scenario
        self._fleet = fleet
        self._metrics = metrics
        self._sku_min_w = int(system_spec.sku_min_w)
        self._sku_max_w = int(system_spec.sku_max_w)

        self._applied_n_p: int = 1
        self._applied_n_d: int = (
            getattr(scenario.fleet, "gpus_per_decode_engine", 4)
            if scenario.fleet
            else 4
        )
        self._applied_cap_p: int = scenario.planner.prefill_engine_gpu_power_limit
        self._applied_cap_d: int = scenario.planner.decode_engine_gpu_power_limit
        self._annotations: dict[str, dict[str, str]] = {}  # pod_name → {key: value}

        # Fault state (controlled by runner via events)
        self._actuation_fault_mode: Optional[str] = None

    # ------------------------------------------------------------------
    # PlannerConnector ABC
    # ------------------------------------------------------------------

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if sub_component_type.name.lower() == "prefill":
            self._applied_n_p += 1
            self._fleet.state.n_p_truth += 1
        else:
            self._applied_n_d += 1
            self._fleet.state.n_d_truth += 1

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if sub_component_type.name.lower() == "prefill":
            self._applied_n_p = max(0, self._applied_n_p - 1)
            self._fleet.state.n_p_truth = max(0, self._fleet.state.n_p_truth - 1)
        else:
            self._applied_n_d = max(0, self._applied_n_d - 1)
            self._fleet.state.n_d_truth = max(0, self._fleet.state.n_d_truth - 1)

    # ------------------------------------------------------------------
    # Testbed-specific methods
    # ------------------------------------------------------------------

    def apply_replicas(self, n_p: int, n_d: int) -> None:
        """Apply desired replica counts (post power-budget clamp)."""
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        self._applied_n_p = max(0, n_p)
        self._applied_n_d = max(0, n_d)
        self._fleet.state.n_p_truth = self._applied_n_p
        self._fleet.state.n_d_truth = self._applied_n_d

    def apply_caps(self, cap_p: int, cap_d: int) -> None:
        """Apply per-GPU power caps, injecting NVML clamp faults if active."""
        clamped_p = self._clamp(cap_p, "prefill")
        clamped_d = self._clamp(cap_d, "decode")
        self._applied_cap_p = clamped_p
        self._applied_cap_d = clamped_d
        # Reflect back to truth model
        self._fleet.state.applied_cap_p = clamped_p
        self._fleet.state.applied_cap_d = clamped_d

    def patch_pod_annotation(self, pod_name: str, key: str, value: str) -> None:
        """Record annotation; if daemonset_absent, don't reflect to truth model."""
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if pod_name not in self._annotations:
            self._annotations[pod_name] = {}
        self._annotations[pod_name][key] = value
        if self._actuation_fault_mode != "daemonset_absent":
            # Reflect the cap into truth model
            try:
                cap_w = int(value)
                clamped = self._clamp_raw(cap_w)
                # Determine which component from annotation key
                if "prefill" in key.lower():
                    self._fleet.state.applied_cap_p = clamped
                else:
                    self._fleet.state.applied_cap_d = clamped
            except (ValueError, TypeError):
                pass

    async def post_busy_threshold(
        self, pod: str, model: str, port: int, **thresholds: float
    ) -> None:
        """Simulate frontend POST; raise for failing_fraction of calls."""
        fault_frac = self._fleet.frontend_fault()
        if fault_frac is not None:
            if random.random() < fault_frac:
                self._metrics.admission_partial_success_total.inc()
                raise RuntimeError(
                    f"503 Service Unavailable (synthetic POST fault to {pod})"
                )

    def applied_caps_snapshot(self) -> AppliedCaps:
        return AppliedCaps(cap_p=self._applied_cap_p, cap_d=self._applied_cap_d)

    # ------------------------------------------------------------------
    # Fault state helpers
    # ------------------------------------------------------------------

    def set_actuation_fault(self, mode: Optional[str]) -> None:
        """Set the active actuation fault mode ('rbac_denied', 'nvml_low', etc.)."""
        self._actuation_fault_mode = mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clamp(self, cap_w: int, component: str) -> int:
        clamped = self._clamp_raw(cap_w)
        if clamped > cap_w:
            self._metrics.power_agent_cap_clamped_total.labels(direction="min").inc()
        elif clamped < cap_w:
            self._metrics.power_agent_cap_clamped_total.labels(direction="max").inc()
        return clamped

    def _clamp_raw(self, cap_w: int) -> int:
        mode = self._actuation_fault_mode
        if mode == "nvml_low":
            return max(cap_w, self._sku_min_w)
        elif mode == "nvml_high":
            return min(cap_w, self._sku_max_w)
        return max(self._sku_min_w, min(self._sku_max_w, cap_w))
