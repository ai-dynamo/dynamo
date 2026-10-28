# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before replay imports.

"""Replay host actuation of planner retune decisions.

The retune controller is replaced by a stub that returns canned decisions, so
these tests cover only what ``ReplayPlannerAdapter`` does with them: planner
changes, the router channel key, the tick capture, and the final report.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.offline.replay_adapter as replay_adapter_module
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.load.predictors import ConstantPredictor
from dynamo.planner.core.types import RetuneDecision, RetuneParams, TrafficObservation
from dynamo.planner.offline.replay_adapter import create_replay_planner_adapter
from dynamo.planner.tests.retune_stubs import (
    ROUTER_MAP,
    StubTickResult,
    caps,
    install_stub_controller,
    retune_inputs,
    retune_only_config,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_PRESWEEP = {
    "winner": "constant_last",
    "fields": {"load_predictor": "constant", "load_predictor_log1p": False},
    "losses": {"constant_last": 0.5, "arima_raw": 0.75, "prophet_w20_raw": None},
    "eval_steps": 4,
    "wall_s": 0.25,
}

_QUIET = {"report_interval_hours": None, "live_dashboard_port": 0}


@pytest.fixture
def stub_controller(monkeypatch):
    return install_stub_controller(monkeypatch)


def _load_scaling_config() -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        optimization_target="throughput",
        retune={
            "interval_seconds": 60,
            "planner": {
                "load_sensitivity": {"preset": False},
                "load_scaling_down_sensitivity": {"choices": [70, 80, 90]},
            },
        },
        **_QUIET,
    )


def _predictor_config() -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        retune={
            "interval_seconds": 60,
            "planner": {"load_predictor": {"preset": ["arima_raw", "constant_last"]}},
        },
        **_QUIET,
    )


def _adapter(config: PlannerConfig, *, capture_details: bool = True):
    return create_replay_planner_adapter(
        planner_config=config,
        capabilities=caps(),
        capture_details=capture_details,
        retune_inputs=None if config.retune is None else retune_inputs(),
    )


def _snapshot(ordinal: int, router_policies=None) -> dict:
    snapshot = {
        "tick_ordinal": ordinal,
        "active_prefill_count": 0,
        "active_decode_count": 2,
        "active_prefill_ids": [],
        "active_decode_ids": [0, 1],
    }
    if router_policies is not None:
        snapshot["router_policies"] = router_policies
    return snapshot


def _decision(
    outcome: str,
    planner_changes: dict | None = None,
    *,
    predictor_warmup=None,
    predictor_presweep=None,
) -> RetuneDecision:
    previous = RetuneParams(planner={"load_scaling_down_sensitivity": 80})
    chosen = RetuneParams(planner=dict(planner_changes or {}))
    return RetuneDecision(
        decision_id="retune-0-w0",
        outcome=outcome,  # type: ignore[arg-type]
        window_index=0,
        window_start_s=0.0,
        window_end_s=60.0,
        as_of_s=120.0,
        launched_at_s=120.0,
        decided_at_s=125.0,
        num_requests=12,
        num_candidates=3,
        previous=previous,
        chosen=chosen if outcome in ("applied", "advisory") else previous,
        baseline_score=10.0,
        best_score=12.0,
        planner_changes=dict(planner_changes or {}),
        predictor_presweep=predictor_presweep,
        predictor_warmup=predictor_warmup,
        sweep_wall_s=1.5,
        top=[('{"planner":{},"router":{}}', 12.0)],
    )


def _warmup() -> tuple[TrafficObservation, ...]:
    return tuple(
        TrafficObservation(duration_s=180.0, num_req=num_req, isl=512.0, osl=64.0)
        for num_req in (10.0, 12.0, 14.0)
    )


def test_on_tick_always_returns_the_router_parameters_key():
    # Every kv_router planner replay reports its router policies; without
    # retune the tick capture keeps its previous shape.
    policies = [
        {
            "worker_type": "aggregated",
            "revision": 0,
            "parameters": ROUTER_MAP["aggregated"],
        }
    ]
    with _adapter(PlannerConfig(mode="agg", **_QUIET)) as adapter:
        adapter.start()
        decision = adapter.on_tick(_snapshot(0, router_policies=policies))
        details = adapter.finalize()

    assert decision == {
        "target_prefill": None,
        "target_decode": None,
        "next_tick_ms": decision["next_tick_ms"],
        "router_parameters": None,
    }
    (tick,) = details.ticks
    assert set(tick["effects"]) == {
        "scale_to",
        "next_tick",
        "diagnostics",
        "batch_drain_limits",
    }
    assert "router_policies" not in tick
    assert details.retune_events is None
    assert "retune_events" not in details.to_dict()


def test_on_tick_forwards_desired_router_map_and_records_router_policies(
    stub_controller,
):
    policies = [
        {
            "worker_type": "aggregated",
            "revision": 2,
            "parameters": ROUTER_MAP["aggregated"],
        }
    ]
    with _adapter(retune_only_config(**_QUIET)) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            StubTickResult(router_parameters=ROUTER_MAP)
        )
        decision = adapter.on_tick(_snapshot(0, router_policies=policies))
        details = adapter.finalize()

    assert decision["router_parameters"] == ROUTER_MAP
    assert decision["target_prefill"] is None and decision["target_decode"] is None
    (tick,) = details.ticks
    assert tick["router_policies"] == policies
    assert tick["runtime_decision"]["router_parameters"] == ROUTER_MAP
    # The effects keep the shape of a planner without retune.
    assert set(tick["effects"]) == {
        "scale_to",
        "next_tick",
        "diagnostics",
        "batch_drain_limits",
    }
    # With retune configured, the events are reported even when empty.
    assert details.retune_events == []
    assert details.to_dict()["retune_events"] == []


def test_advisory_never_hands_rust_the_router_map(stub_controller):
    policies = [
        {
            "worker_type": "aggregated",
            "revision": 0,
            "parameters": ROUTER_MAP["aggregated"],
        }
    ]
    with _adapter(retune_only_config(advisory=True, **_QUIET)) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            StubTickResult(router_parameters=ROUTER_MAP)
        )
        decision = adapter.on_tick(_snapshot(0, router_policies=policies))
        details = adapter.finalize()

    assert decision["router_parameters"] is None
    (tick,) = details.ticks
    assert tick["runtime_decision"]["router_parameters"] is None
    assert tick["router_policies"] == policies
    assert details.metadata["advisory"] is True
    assert details.retune_events == []
    assert details.recommended_scaling_events == []


def test_an_applied_decision_under_advisory_fails_the_replay(stub_controller):
    config = _load_scaling_config()
    config.advisory = True
    with _adapter(config, capture_details=False) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            StubTickResult(
                decisions=[_decision("applied", {"load_scaling_down_sensitivity": 70})]
            )
        )
        with pytest.raises(RuntimeError, match="advisory mode"):
            adapter.on_tick(_snapshot(0))

    assert config.load_scaling_down_sensitivity == 80


@pytest.mark.parametrize(
    ("outcome", "value"),
    [("applied", 70), ("kept", 80), ("advisory", 80)],
)
def test_actuated_planner_changes_apply_from_the_next_tick(
    stub_controller, monkeypatch, outcome, value
):
    def no_predictor(*args, **kwargs):
        raise AssertionError("no predictor field changed")

    monkeypatch.setattr(replay_adapter_module, "prepare_predictor", no_predictor)
    config = _load_scaling_config()
    with _adapter(config, capture_details=False) as adapter:
        adapter.start()
        controller = stub_controller.instances[0]
        controller.results.append(
            StubTickResult(
                decisions=[_decision(outcome, {"load_scaling_down_sensitivity": 70})]
            )
        )
        adapter.on_tick(_snapshot(0))
        assert config.load_scaling_down_sensitivity == value
        adapter.on_tick(_snapshot(1))
        details = adapter.finalize()

    # The engine observed the old value in the decision tick and the actuated
    # value from the next tick on; kept and advisory never actuate.
    assert [
        values["load_scaling_down_sensitivity"]
        for values in controller.seen_planner_values
    ] == [80, value]
    # Retune events are reported without detail capture.
    assert details.ticks == []
    assert [
        (event["decision_id"], event["outcome"]) for event in details.retune_events
    ] == [("retune-0-w0", outcome)]
    assert details.to_dict()["retune_events"] == details.retune_events


def test_predictor_change_installs_a_predictor_warmed_with_the_decision_history(
    stub_controller, monkeypatch
):
    prepared = []
    real_prepare = replay_adapter_module.prepare_predictor

    def recording_prepare(config, capabilities, observations):
        prepared.append((config.load_predictor, observations))
        return real_prepare(config, capabilities, observations)

    monkeypatch.setattr(replay_adapter_module, "prepare_predictor", recording_prepare)
    config = _predictor_config()
    warmup = _warmup()
    with _adapter(config) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            StubTickResult(
                decisions=[
                    _decision(
                        "applied",
                        {"load_predictor": "constant"},
                        predictor_warmup=warmup,
                        predictor_presweep=_PRESWEEP,
                    )
                ]
            )
        )
        adapter.on_tick(_snapshot(0))
        predict = adapter._engine._builtins["load_predict"]
        details = adapter.finalize()

    assert prepared == [("constant", warmup)]
    assert config.load_predictor == "constant"
    assert isinstance(predict._num_req_predictor, ConstantPredictor)
    assert predict._num_req_predictor.predict_next() == 14.0

    # The retune events keep the full pre-sweep record and the warmup length.
    json.dumps(details.ticks)
    (event,) = details.retune_events
    assert event["predictor_warmup"] == 3
    assert event["predictor_presweep"] == _PRESWEEP
    assert event["planner_changes"] == {"load_predictor": "constant"}


def test_report_digests_the_startup_config(stub_controller):
    config = _load_scaling_config()
    with _adapter(config) as adapter:
        adapter.start()
        startup_dump = config.model_dump(mode="json", by_alias=True)
        stub_controller.instances[0].results.append(
            StubTickResult(
                decisions=[_decision("applied", {"load_scaling_down_sensitivity": 90})]
            )
        )
        adapter.on_tick(_snapshot(0))
        details = adapter.finalize()

    assert config.model_dump(mode="json", by_alias=True) != startup_dump
    digest = details.metadata["planner_config_digest"]
    assert digest == adapter._planner_metadata(startup_dump)["planner_config_digest"]
    assert digest != adapter._planner_metadata()["planner_config_digest"]
