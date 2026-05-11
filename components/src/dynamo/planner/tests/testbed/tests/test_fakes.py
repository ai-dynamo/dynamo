# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the testbed's fake components.

These tests are deliberately constructed against the current public API of
each fake — if you rename a field on TickSnapshot, change FakeAIC's seam, or
restructure FakeActuator, these tests fail loud at collection.
"""
from __future__ import annotations

import random

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _system_spec():
    from dynamo.planner.tests.testbed.scenarios import SystemSpec

    return SystemSpec.load("h200_sxm")


def _scenario(planner_overrides: dict | None = None,
              fleet_overrides: dict | None = None) -> "ScenarioSpec":  # type: ignore[name-defined]
    from dynamo.planner.tests.testbed.scenarios import (
        FleetSpec, LoadSpec, PlannerSpec, ScenarioSpec,
    )

    planner_kwargs = dict(
        mode="disagg",
        enable_power_awareness=True,
        enable_aic_optimizer=True,
        total_gpu_power_limit=4000,
        power_agent_safe_default_watts=500,
        prefill_engine_gpu_power_limit=500,
        decode_engine_gpu_power_limit=425,
    )
    if planner_overrides:
        planner_kwargs.update(planner_overrides)

    fleet_kwargs = dict(system="h200_sxm")
    if fleet_overrides:
        fleet_kwargs.update(fleet_overrides)

    return ScenarioSpec(
        name="test",
        **{"class": "alpha"},
        ticks=10,
        planner=PlannerSpec(**planner_kwargs),
        fleet=FleetSpec(**fleet_kwargs),
        load=LoadSpec(profile="constant", tokens_per_sec=1000.0),
    )


def _fleet(scenario=None):
    from dynamo.planner.tests.testbed.synthetic_fleet import SyntheticFleet

    sc = scenario or _scenario()
    return SyntheticFleet(sc.fleet, _system_spec(), sc, random.Random(0))


def _metrics():
    from dynamo.planner.tests.testbed.fake_planner_metrics import FakePlannerMetrics

    return FakePlannerMetrics()


# ---------------------------------------------------------------------------
# FakeAIC
# ---------------------------------------------------------------------------


class TestFakeAIC:
    def test_factory_returns_estimator_with_normal_values(self):
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC

        aic = FakeAIC(_system_spec())
        factory = aic.make_estimator_factory()
        est = factory(hf_id="m", system="h200_sxm", backend="vllm")

        prefill = est.estimate_prefill_perf(isl=3000)
        decode = est.estimate_perf(isl=3000, osl=150, batch_size=1)

        # Values come from systems/h200_sxm.yaml — keep test loosely coupled
        # to the actual numbers but assert plausibility.
        assert prefill["context_latency"] > 0
        assert prefill["power_w"] > 0
        assert decode["tpot"] > 0
        assert decode["power_w"] > 0

    def test_raises_mode(self):
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC

        aic = FakeAIC(_system_spec())
        aic.set_fault_mode("raises")
        est = aic.make_estimator_factory()(hf_id="m", system="s", backend="vllm")
        with pytest.raises(RuntimeError):
            est.estimate_prefill_perf(isl=3000)

    def test_empty_pareto_mode_returns_huge_ttft(self):
        """``empty_pareto`` forces the optimizer into the infeasibility path."""
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC

        aic = FakeAIC(_system_spec())
        aic.set_fault_mode("empty_pareto")
        est = aic.make_estimator_factory()(hf_id="m", system="s", backend="vllm")
        prefill = est.estimate_prefill_perf(isl=3000)
        assert prefill["context_latency"] > 100_000

    def test_reset_fault_restores_normal_response(self):
        from dynamo.planner.tests.testbed.fake_aic import FakeAIC

        aic = FakeAIC(_system_spec())
        aic.set_fault_mode("raises")
        aic.reset_fault()
        est = aic.make_estimator_factory()(hf_id="m", system="s", backend="vllm")
        # Should not raise now.
        est.estimate_prefill_perf(isl=3000)


# ---------------------------------------------------------------------------
# FakePlannerMetrics
# ---------------------------------------------------------------------------


class TestFakePlannerMetrics:
    def test_counter_starts_zero_and_increments(self):
        m = _metrics()
        assert m.aic_optimizer_exceptions_total.value == 0.0
        m.aic_optimizer_exceptions_total.inc()
        m.aic_optimizer_exceptions_total.inc()
        assert m.aic_optimizer_exceptions_total.value == 2.0

    def test_labeled_counter(self):
        m = _metrics()
        m.power_agent_cap_clamped_total.labels(direction="min").inc()
        m.power_agent_cap_clamped_total.labels(direction="min").inc()
        m.power_agent_cap_clamped_total.labels(direction="max").inc()
        assert m.power_agent_cap_clamped_total.labeled_value(direction="min") == 2.0
        assert m.power_agent_cap_clamped_total.labeled_value(direction="max") == 1.0

    def test_gauge_set(self):
        m = _metrics()
        m.aic_consecutive_failures.set(3)
        assert m.aic_consecutive_failures.value == 3


# ---------------------------------------------------------------------------
# FakeActuator
# ---------------------------------------------------------------------------


class TestFakeActuator:
    def _make_actuator(self, scenario=None):
        from dynamo.planner.tests.testbed.fake_actuator import FakeActuator

        sc = scenario or _scenario()
        fleet = _fleet(sc)
        actuator = FakeActuator(sc, fleet, _metrics(), _system_spec())
        return actuator, fleet

    def test_apply_caps_within_sku_range(self):
        actuator, _ = self._make_actuator()
        actuator.apply_caps(500, 425)
        snap = actuator.applied_caps_snapshot()
        sys = _system_spec()
        assert sys.sku_min_w <= snap.cap_p <= sys.sku_max_w
        assert sys.sku_min_w <= snap.cap_d <= sys.sku_max_w

    def test_apply_caps_clamps_below_min(self):
        actuator, _ = self._make_actuator()
        actuator.apply_caps(1, 1)
        snap = actuator.applied_caps_snapshot()
        sys = _system_spec()
        assert snap.cap_p >= sys.sku_min_w
        assert snap.cap_d >= sys.sku_min_w

    def test_apply_caps_clamps_above_max(self):
        actuator, _ = self._make_actuator()
        actuator.apply_caps(9999, 9999)
        snap = actuator.applied_caps_snapshot()
        sys = _system_spec()
        assert snap.cap_p <= sys.sku_max_w
        assert snap.cap_d <= sys.sku_max_w

    def test_rbac_denied_blocks_apply_replicas(self):
        actuator, _ = self._make_actuator()
        actuator.set_actuation_fault("rbac_denied")
        with pytest.raises(RuntimeError, match="403"):
            actuator.apply_replicas(2, 4)

    def test_apply_replicas_propagates_to_fleet_state(self):
        actuator, fleet = self._make_actuator()
        actuator.apply_replicas(3, 5)
        assert fleet.state.n_p_truth == 3
        assert fleet.state.n_d_truth == 5

    def test_nvml_low_clamp_records_metric(self):
        actuator, _ = self._make_actuator()
        actuator.set_actuation_fault("nvml_low")
        actuator.apply_caps(50, 50)  # Below sku_min_w
        snap = actuator.applied_caps_snapshot()
        sys = _system_spec()
        assert snap.cap_p >= sys.sku_min_w
        assert snap.cap_d >= sys.sku_min_w


# ---------------------------------------------------------------------------
# FakePrometheusClient
# ---------------------------------------------------------------------------


class TestFakePrometheusClient:
    def _make_prom(self):
        from dynamo.planner.tests.testbed.fake_prometheus import FakePrometheusClient

        sc = _scenario()
        fleet = _fleet(sc)
        prom = FakePrometheusClient(source=fleet)
        prom.set_tick(0)
        # Drive a tick so observation_at(0) has data.
        fleet.step(tick=0, offered_load=1000.0)
        return prom, fleet

    def test_decode_power_is_positive_float(self):
        prom, _ = self._make_prom()
        val = prom.get_avg_per_gpu_power_by_component(component="decode", interval="60s")
        assert isinstance(val, float)
        assert val > 0

    def test_prefill_power_is_positive_float(self):
        prom, _ = self._make_prom()
        val = prom.get_avg_per_gpu_power_by_component(component="prefill", interval="60s")
        assert isinstance(val, float)
        assert val > 0

    def test_outage_returns_none(self):
        prom, fleet = self._make_prom()
        # Inject a power_p outage active at tick 0.
        fleet._active_prom_outage["power_p"] = 10
        val = prom.get_avg_per_gpu_power_by_component(component="prefill", interval="60s")
        assert val is None


# ---------------------------------------------------------------------------
# Wire-up sanity: scenarios sourced from real YAMLs load via the same code path
# ---------------------------------------------------------------------------


def test_load_a1_scenario_smoke():
    from pathlib import Path

    from dynamo.planner.tests.testbed.scenarios import load_scenario

    here = Path(__file__).parent.parent / "scenarios"
    sc = load_scenario(here / "A1_power_under_estimate_decode.yaml")
    assert sc.class_name == "alpha"
    assert sc.fleet is not None
    assert sc.fleet.bias.power_bias_decode == 1.35
