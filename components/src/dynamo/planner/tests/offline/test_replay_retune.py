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
import sys
import types
from dataclasses import dataclass, field

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.offline.replay_adapter as replay_adapter_module
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.load.predictors import ConstantPredictor
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneDecision,
    RetuneInputs,
    RetuneParams,
    TrafficObservation,
    WorkerCapabilities,
)
from dynamo.planner.offline.replay_adapter import create_replay_planner_adapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_CONTROLLER_MODULE = "dynamo.planner.retune.controller"
_ROUTER_MAP = {
    "aggregated": {
        "decode_active_request_weight": 0.0,
        "disk_cache_hit_weight": 0.25,
        "host_cache_hit_weight": 0.75,
        "overlap_score_credit": 1.0,
        "overlap_score_credit_decay": 0.0,
        "prefill_load_scale": 1.0,
        "router_temperature": 0.0,
        "shared_cache_multiplier": 0.0,
    }
}
_PRESWEEP = {
    "winner": "constant_last",
    "fields": {"load_predictor": "constant", "load_predictor_log1p": False},
    "losses": {"constant_last": 0.5, "arima_raw": 0.75, "prophet_w20_raw": None},
    "eval_steps": 4,
    "wall_s": 0.25,
}


@dataclass(frozen=True)
class _TickResult:
    decisions: list[RetuneDecision] = field(default_factory=list)
    router_parameters: dict[str, dict[str, float]] | None = None
    audit_events: list[str] = field(default_factory=list)


class _StubController:
    """Returns one queued tick result per ``on_tick``."""

    instances: list[_StubController] = []

    def __init__(self, config, inputs, capabilities, clock, scale_interval_s):
        self.config = config
        self.results: list[_TickResult] = []
        self.seen_planner_values: list[dict[str, object]] = []
        _StubController.instances.append(self)

    def start(self, start_s: float) -> None:
        del start_s

    async def on_tick(self, now_s, worker_counts, feedback):
        del now_s, worker_counts, feedback
        self.seen_planner_values.append(
            {
                "load_scaling_down_sensitivity": self.config.load_scaling_down_sensitivity,
                "load_predictor": self.config.load_predictor,
            }
        )
        if self.results:
            return self.results.pop(0)
        return _TickResult()

    def close(self) -> None:
        pass


@pytest.fixture
def stub_controller(monkeypatch):
    module = types.ModuleType(_CONTROLLER_MODULE)
    module.RetuneController = _StubController  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _CONTROLLER_MODULE, module)
    _StubController.instances = []
    return _StubController


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


_QUIET = {"report_interval_hours": None, "live_dashboard_port": 0}


def _retune_only_config() -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={"interval_seconds": 60, "router": {"router_temperature": [0.0, 0.5]}},
        **_QUIET,
    )


def _load_scaling_config() -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        optimization_target="throughput",
        retune={
            "interval_seconds": 60,
            "planner": {"load_scaling_down_sensitivity": [70, 80, 90]},
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
            "planner": {"load_predictor": ["arima", "constant"]},
        },
        **_QUIET,
    )


def _inputs() -> RetuneInputs:
    return RetuneInputs(engine=RetuneEngineSpec(), trace_paths=("trace.jsonl",))


def _adapter(config: PlannerConfig, *, capture_details: bool = True):
    return create_replay_planner_adapter(
        planner_config=config,
        capabilities=_caps(),
        capture_details=capture_details,
        retune_inputs=None if config.retune is None else _inputs(),
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
    window_index: int = 0,
) -> RetuneDecision:
    previous = RetuneParams(planner={"load_scaling_down_sensitivity": 80})
    chosen = RetuneParams(planner=dict(planner_changes or {}))
    return RetuneDecision(
        decision_id=f"retune-0-w{window_index}",
        outcome=outcome,  # type: ignore[arg-type]
        window_index=window_index,
        window_start_s=0.0,
        window_end_s=60.0,
        as_of_s=120.0,
        launched_at_s=120.0,
        decided_at_s=125.0,
        num_requests=12,
        num_candidates=3,
        previous=previous,
        chosen=chosen if outcome in ("applied", "rolled_back", "dry_run") else previous,
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
            "parameters": _ROUTER_MAP["aggregated"],
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
    assert details.retune_events == []
    assert "retune_events" not in details.to_dict()


def test_on_tick_forwards_desired_router_map_and_records_router_policies(
    stub_controller,
):
    policies = [
        {
            "worker_type": "aggregated",
            "revision": 2,
            "parameters": _ROUTER_MAP["aggregated"],
        }
    ]
    with _adapter(_retune_only_config()) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            _TickResult(router_parameters=_ROUTER_MAP)
        )
        decision = adapter.on_tick(_snapshot(0, router_policies=policies))
        details = adapter.finalize()

    assert decision["router_parameters"] == _ROUTER_MAP
    assert decision["target_prefill"] is None and decision["target_decode"] is None
    (tick,) = details.ticks
    assert tick["router_policies"] == policies
    assert tick["router_policies"][0]["parameters"] is not policies[0]["parameters"]
    assert tick["effects"]["router_parameters"] == _ROUTER_MAP


@pytest.mark.parametrize("outcome", ["applied", "rolled_back"])
def test_actuated_planner_changes_apply_from_the_next_tick(
    stub_controller, monkeypatch, outcome
):
    def no_predictor(*args, **kwargs):
        raise AssertionError("no predictor field changed")

    monkeypatch.setattr(replay_adapter_module, "prepare_predictor", no_predictor)
    config = _load_scaling_config()
    with _adapter(config) as adapter:
        adapter.start()
        controller = stub_controller.instances[0]
        controller.results.append(
            _TickResult(
                decisions=[_decision(outcome, {"load_scaling_down_sensitivity": 70})]
            )
        )
        adapter.on_tick(_snapshot(0))
        assert config.load_scaling_down_sensitivity == 70
        adapter.on_tick(_snapshot(1))

    # The engine observed the old value in the decision tick and the new
    # value from the next tick on.
    assert [
        values["load_scaling_down_sensitivity"]
        for values in controller.seen_planner_values
    ] == [80, 70]


@pytest.mark.parametrize(
    "outcome", ["kept", "dry_run", "busy", "stale", "timeout", "failed"]
)
def test_non_actuated_outcomes_leave_the_planner_unchanged(stub_controller, outcome):
    config = _load_scaling_config()
    with _adapter(config) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            _TickResult(
                decisions=[_decision(outcome, {"load_scaling_down_sensitivity": 70})]
            )
        )
        adapter.on_tick(_snapshot(0))
        details = adapter.finalize()

    assert config.load_scaling_down_sensitivity == 80
    assert [event["outcome"] for event in details.retune_events] == [outcome]


def test_invalid_runtime_candidate_fails_the_replay(stub_controller, monkeypatch):
    monkeypatch.setattr(
        replay_adapter_module,
        "runtime_candidate_errors",
        lambda candidate, capabilities: ["min_gpu_budget exceeds the deployment"],
    )
    config = _load_scaling_config()
    with _adapter(config) as adapter:
        adapter.start()
        stub_controller.instances[0].results.append(
            _TickResult(
                decisions=[_decision("applied", {"load_scaling_down_sensitivity": 70})]
            )
        )
        with pytest.raises(ValueError, match="invalid planner configuration"):
            adapter.on_tick(_snapshot(0))

    assert config.load_scaling_down_sensitivity == 80


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
            _TickResult(
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

    # The tick capture carries the warmup length and no per-preset losses;
    # the retune events keep the full pre-sweep record.
    (tick,) = details.ticks
    (captured,) = tick["effects"]["retune_decisions"]
    assert captured["predictor_warmup"] == 3
    assert captured["predictor_presweep"] == {
        key: value for key, value in _PRESWEEP.items() if key != "losses"
    }
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
            _TickResult(
                decisions=[_decision("applied", {"load_scaling_down_sensitivity": 90})]
            )
        )
        adapter.on_tick(_snapshot(0))
        details = adapter.finalize()

    assert config.model_dump(mode="json", by_alias=True) != startup_dump
    digest = details.metadata["planner_config_digest"]
    assert digest == adapter._planner_metadata(startup_dump)["planner_config_digest"]
    assert digest != adapter._planner_metadata()["planner_config_digest"]


def test_retune_events_are_reported_without_detail_capture(stub_controller):
    with _adapter(_load_scaling_config(), capture_details=False) as adapter:
        adapter.start()
        controller = stub_controller.instances[0]
        controller.results.append(
            _TickResult(decisions=[_decision("kept", window_index=0)])
        )
        controller.results.append(
            _TickResult(decisions=[_decision("busy", window_index=1)])
        )
        adapter.on_tick(_snapshot(0))
        adapter.on_tick(_snapshot(1))
        details = adapter.finalize()

    assert details.ticks == []
    assert [event["decision_id"] for event in details.retune_events] == [
        "retune-0-w0",
        "retune-0-w1",
    ]
    assert details.to_dict()["retune_events"] == details.retune_events
