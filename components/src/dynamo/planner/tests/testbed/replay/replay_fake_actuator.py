# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReplayFakeActuator — γ-class actuator.

Subclass of FakeActuator that wraps ``PlannerReplayBridge.apply_scaling()``
for replica changes.  Cap annotations and frontend POST faults behave
identically to the α-class parent.

Reflects applied caps back into SyntheticPowerOverlay so the truth model
uses the actually-applied (possibly clamped) cap.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.tests.testbed.fake_actuator import AppliedCaps, FakeActuator

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics
    from dynamo.planner.tests.testbed.replay.synthetic_power_overlay import SyntheticPowerOverlay
    from dynamo.planner.tests.testbed.scenarios import ScenarioSpec, SystemSpec


class ReplayFakeActuator(FakeActuator):
    """γ-class actuator.

    Wraps PlannerReplayBridge.apply_scaling for replica changes; otherwise
    identical fault-injection behaviour to FakeActuator.
    """

    def __init__(
        self,
        scenario: "ScenarioSpec",
        overlay: "SyntheticPowerOverlay",
        bridge: Any,  # PlannerReplayBridge (Rust pyclass)
        metrics: "FakePlannerMetrics",
        system_spec: "SystemSpec",
    ) -> None:
        # No SyntheticFleet in γ-class — overlay is the truth source.
        # We pass fleet=None but override the methods that use it.
        super().__init__(
            scenario=scenario,
            fleet=None,   # type: ignore[arg-type]
            metrics=metrics,
            system_spec=system_spec,
        )
        self._bridge = bridge
        self._overlay = overlay

        self._current_n_p = scenario.mocker.num_prefill_workers if scenario.mocker else 1
        self._current_n_d = scenario.mocker.num_decode_workers if scenario.mocker else 4

        # γ-class has no SyntheticFleet, so the FakeActuator parent's
        # post_busy_threshold() (which reads self._fleet.frontend_fault()) is
        # not safe — we override below and store the active fault locally.
        self._frontend_fault_fraction: Optional[float] = None

    # ------------------------------------------------------------------
    # Override: replica scaling goes through bridge
    # ------------------------------------------------------------------

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if sub_component_type.name.lower() == "prefill":
            self._current_n_p += 1
        else:
            self._current_n_d += 1
        self._bridge.apply_scaling(self._current_n_p, self._current_n_d)

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if sub_component_type.name.lower() == "prefill":
            self._current_n_p = max(0, self._current_n_p - 1)
        else:
            self._current_n_d = max(0, self._current_n_d - 1)
        self._bridge.apply_scaling(self._current_n_p, self._current_n_d)

    def apply_replicas(self, n_p: int, n_d: int) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        self._current_n_p = max(0, n_p)
        self._current_n_d = max(0, n_d)
        self._bridge.apply_scaling(self._current_n_p, self._current_n_d)

    # ------------------------------------------------------------------
    # Override: cap reflection goes into overlay (not fleet)
    # ------------------------------------------------------------------

    def apply_caps(self, cap_p: int, cap_d: int) -> None:
        clamped_p = self._clamp_raw(cap_p)
        clamped_d = self._clamp_raw(cap_d)
        if clamped_p > cap_p or clamped_d > cap_d:
            self._metrics.power_agent_cap_clamped_total.labels(direction="min").inc()
        if clamped_p < cap_p or clamped_d < cap_d:
            self._metrics.power_agent_cap_clamped_total.labels(direction="max").inc()
        self._applied_cap_p = clamped_p
        self._applied_cap_d = clamped_d
        self._overlay.notify_caps_changed(clamped_p, clamped_d)

    def patch_pod_annotation(self, pod_name: str, key: str, value: str) -> None:
        if self._actuation_fault_mode == "rbac_denied":
            raise RuntimeError("403 Forbidden (synthetic actuation fault)")
        if pod_name not in self._annotations:
            self._annotations[pod_name] = {}
        self._annotations[pod_name][key] = value
        if self._actuation_fault_mode != "daemonset_absent":
            try:
                cap_w = int(value)
                clamped = self._clamp_raw(cap_w)
                if "prefill" in key.lower():
                    self._overlay.notify_caps_changed(clamped, self._applied_cap_d)
                else:
                    self._overlay.notify_caps_changed(self._applied_cap_p, clamped)
            except (ValueError, TypeError):
                pass

    def applied_caps_snapshot(self) -> AppliedCaps:
        return AppliedCaps(cap_p=self._applied_cap_p, cap_d=self._applied_cap_d)

    # ------------------------------------------------------------------
    # Override: frontend POST fault state lives on the actuator (no fleet)
    # ------------------------------------------------------------------

    def set_frontend_fault_fraction(self, fraction: Optional[float]) -> None:
        """Called by the γ-runner when a FrontendPostFaultEvent fires/expires."""
        self._frontend_fault_fraction = fraction

    async def post_busy_threshold(
        self, pod: str, model: str, port: int, **thresholds: float
    ) -> None:
        frac = self._frontend_fault_fraction
        if frac is not None and random.random() < frac:
            self._metrics.admission_partial_success_total.inc()
            raise RuntimeError(
                f"503 Service Unavailable (synthetic POST fault to {pod})"
            )
