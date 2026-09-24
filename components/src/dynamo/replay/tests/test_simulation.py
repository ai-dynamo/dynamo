# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Tests for the transitional Dynamo Sweeper replay runner."""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)

from dynamo.replay import PlannerReplayDetails, ReplayReport
from dynamo.replay import api as replay_api
from dynamo.replay import run_trace_replay, simulation
from dynamo.router.simulation.config import RouterPredictionConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


class _FakeEngineArgs:
    def __init__(self, payload: str):
        self.payload = payload

    @classmethod
    def from_json(cls, payload: str):
        return cls(payload)


class _FakeRouterConfig:
    @classmethod
    def from_json(cls, payload: str):
        return json.loads(payload)


def _report(summary: dict, *, total_ticks: int | None = None) -> ReplayReport:
    planner = (
        None if total_ticks is None else PlannerReplayDetails(total_ticks=total_ticks)
    )
    return ReplayReport(
        summary=summary,
        per_request=None,
        coverage={},
        planner=planner,
    )


def _detailed_report(summary: dict) -> ReplayReport:
    return ReplayReport(
        summary=summary,
        per_request=[{"request_id": "request-1", "ttft_ms": 4.0}],
        coverage={"captured_request_count": 1},
        planner=None,
    )


def _agg_deployment() -> BackendDeploymentSpec:
    return BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args={"engine_type": "vllm", "max_num_seqs": 256},
        num_workers=3,
    )


def test_trace_runner_preserves_current_replay_arguments(monkeypatch) -> None:
    seen = {}

    def fake_run_trace_replay(**kwargs):
        seen.update(kwargs)
        return _report(
            {
                "output_throughput_tok_s": 42.0,
                "goodput_output_throughput_tok_s": 40.0,
            },
            total_ticks=7,
        )

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(simulation, "KvRouterConfig", _FakeRouterConfig)
    monkeypatch.setattr(simulation, "run_trace_replay", fake_run_trace_replay)
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "trace_path": "tiny.jsonl",
            "trace_format": "dynamo",
            "arrival_speedup_ratio": 2.0,
            "replay_concurrency": 8,
        },
        goal={
            "target": "goodput",
            "sla": {"ttft_ms": 100.0, "itl_ms": 20.0, "e2e_ms": None},
        },
        adapters={
            "dynamo.planner": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.planner",
                        kind="scaling_policy",
                        api_version=1,
                        config={"planner_config": {"mode": "agg"}},
                    ),
                )
            ),
            "dynamo.router": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.router",
                        kind="placement_policy",
                        api_version=1,
                        config={
                            "router_mode": "kv_router",
                            "router_config": {
                                "overlap_score_credit": 0.5,
                                "prefill_load_scale": 1.0,
                                "router_temperature": 0.0,
                            },
                        },
                    ),
                )
            ),
        },
    )

    report = simulation.DynamoReplayRunnerFactory().create(2).run(spec)

    assert seen["trace_files"] == "tiny.jsonl"
    assert seen["trace_format"] == "dynamo"
    assert seen["num_workers"] == 3
    assert seen["router_mode"] == "kv_router"
    assert seen["planner_config"] == {"mode": "agg"}
    assert seen["arrival_speedup_ratio"] == 2.0
    assert seen["replay_concurrency"] == 8
    assert seen["trace_block_size"] == 512
    assert seen["benchmark_granularity"] == 8
    assert seen["capture_per_request"] is False
    assert seen["capture_planner_details"] is False
    assert seen["sla_ttft_ms"] == 100.0
    assert seen["sla_itl_ms"] == 20.0
    assert seen["sla_e2e_ms"] is None
    assert report.metrics["planner_total_ticks"] == 7.0
    assert report.metadata["planner_total_ticks"] == 7


def test_trace_paths_only_workload_routes_to_trace_replay(monkeypatch) -> None:
    seen = {}

    def fake_run_trace_replay(**kwargs):
        seen.update(kwargs)
        return _report({"completed_requests": 2})

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(simulation, "run_trace_replay", fake_run_trace_replay)
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "trace_paths": ["first.jsonl", "second.jsonl"],
            "trace_format": "dynamo",
            "arrival_speedup_ratio": 2.0,
            "agentic_lanes": 4,
        },
        goal={"target": "throughput"},
    )

    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen["trace_files"] == ["first.jsonl", "second.jsonl"]
    assert seen["arrival_speedup_ratio"] == 2.0
    assert seen["agentic_lanes"] == 4
    assert report.metrics["completed_requests"] == 2.0


def test_trace_replay_rejects_boolean_agentic_lanes() -> None:
    with pytest.raises(TypeError, match="agentic_lanes must be an integer"):
        run_trace_replay("unused.jsonl", agentic_lanes=True)


def test_runner_captures_per_request_output_when_requested(monkeypatch) -> None:
    seen = {}

    def fake_run_trace_replay(**kwargs):
        seen.update(kwargs)
        return _detailed_report({"completed_requests": 1})

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(simulation, "run_trace_replay", fake_run_trace_replay)
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl", "trace_format": "dynamo"},
        goal={"target": "throughput"},
    )

    report = (
        simulation.DynamoReplayRunnerFactory()
        .create(0)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
            ),
        )
    )

    assert seen["capture_per_request"] is True
    assert report.metadata["native_report"]["per_request"] == [
        {"request_id": "request-1", "ttft_ms": 4.0}
    ]


def test_synthetic_disagg_preserves_request_count_and_load(monkeypatch) -> None:
    seen = {}

    def fake_run_synthetic_trace_replay(**kwargs):
        seen.update(kwargs)
        return _report({"output_throughput_tok_s": 99.0})

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation,
        "run_synthetic_trace_replay",
        fake_run_synthetic_trace_replay,
    )
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="sglang",
        backend_version="0.5.6",
        prefill_engine_args={"worker_type": "prefill"},
        decode_engine_args={"worker_type": "decode"},
        num_prefill_workers=2,
        num_decode_workers=4,
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={
            "trace_path": None,
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "concurrency": None,
            "request_rate": None,
            "turns_per_session": 2,
            "shared_prefix_ratio": 0.5,
            "num_prefix_groups": 4,
            "inter_turn_delay_ms": 12.0,
        },
        goal={"target": "throughput"},
        concurrency=32,
    )

    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen["input_tokens"] == 512
    assert seen["output_tokens"] == 128
    assert seen["request_count"] == 320
    assert seen["replay_concurrency"] == 32
    # Replay requires exactly one load controller. Closed-loop mode uses only
    # replay_concurrency; an arrival interval would make the request ambiguous.
    assert seen["arrival_interval_ms"] is None
    assert seen["num_prefill_workers"] == 2
    assert seen["num_decode_workers"] == 4
    assert seen["capture_per_request"] is False
    assert seen["capture_planner_details"] is False
    assert report.metrics == {
        "output_throughput_tok_s": 99.0,
        "power_w": None,
        "power_coverage": None,
    }


def test_synthetic_request_rate_preserves_open_loop_load(monkeypatch) -> None:
    seen = {}

    def fake_run_synthetic_trace_replay(**kwargs):
        seen.update(kwargs)
        return _report({"output_throughput_tok_s": 99.0})

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation,
        "run_synthetic_trace_replay",
        fake_run_synthetic_trace_replay,
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "trace_path": None,
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "concurrency": None,
            "request_rate": 20.0,
        },
        goal={"target": "throughput"},
    )

    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen["request_count"] == 200
    assert seen["replay_concurrency"] is None
    assert seen["arrival_interval_ms"] == 50.0
    assert report.metrics == {
        "output_throughput_tok_s": 99.0,
        "power_w": None,
        "power_coverage": None,
    }


@pytest.mark.parametrize("request_rate", [0.0, -1.0])
def test_synthetic_request_rate_must_be_positive(request_rate: float) -> None:
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "trace_path": None,
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "concurrency": None,
            "request_rate": request_rate,
        },
        goal={"target": "throughput"},
    )

    with pytest.raises(ValueError, match="positive request_rate"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


def test_direct_predict_resolves_kv_capacity_fraction(monkeypatch) -> None:
    class CapacityArgs:
        num_gpu_blocks = 100
        block_size = 16
        dp_size = 1

    monkeypatch.setattr(
        simulation.DynamoReplayRunner,
        "_engine_args",
        staticmethod(lambda _payload: CapacityArgs()),
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "isl": 100,
            "osl": 20,
            "kv_load_ratio": 0.5,
            "num_request_ratio": 10.0,
        },
        goal={"target": "throughput"},
    )

    runner = simulation.DynamoReplayRunnerFactory().create(0)
    assert runner._effective_in_flight_cap(spec) == 21
    assert runner._synthetic_kwargs(spec)["request_count"] == 210


def test_fixed_timing_keeps_aic_identity_out_of_runtime_args(monkeypatch) -> None:
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation,
        "resolve_aic_num_gpu_blocks",
        lambda payload: payload,
    )
    engine_args = simulation.DynamoReplayRunner._engine_args(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_backend_version": "0.11.0",
            "aic_system": "h200_sxm",
            "aic_model_path": "example/model",
            "aic_attention_dp_size": 2,
            "aic_pp_size": 1,
            "prefill_schedule_interval": 1,
            "prefill_decode_interval": 0,
            "aic_database_mode": "SILICON",
            "aic_strict_provenance": False,
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 1.0,
                "decode_ms": 1.0,
            },
        }
    )

    lowered = json.loads(engine_args.payload)
    assert "aic_backend" not in lowered
    assert "aic_backend_version" not in lowered
    assert "aic_system" not in lowered
    assert "aic_model_path" not in lowered
    assert "aic_attention_dp_size" not in lowered
    assert lowered["dp_size"] == 2
    assert "aic_pp_size" not in lowered
    assert "prefill_schedule_interval" not in lowered
    assert "prefill_decode_interval" not in lowered
    assert "aic_database_mode" not in lowered
    assert "aic_strict_provenance" not in lowered


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefill_schedule_interval", 2),
        ("prefill_schedule_interval", True),
        ("prefill_decode_interval", 1),
        ("prefill_decode_interval", False),
        ("aic_database_mode", "HYBRID"),
        ("aic_strict_provenance", True),
    ],
)
def test_legacy_replay_rejects_unrepresentable_settings(field, value):
    with pytest.raises(ValueError, match=f"legacy Dynamo replay requires {field}"):
        simulation.DynamoReplayRunner._engine_args({field: value})


@pytest.mark.parametrize(
    "extra",
    [
        {"pp": 2},
        {"database_mode": "HYBRID"},
        {"decoder_replay": True},
        {"unsupported_future_control": 7},
    ],
)
def test_legacy_replay_rejects_unrepresentable_canonical_aic_settings(extra):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    config = ForwardPassPerfModelConfig(
        model="example/model",
        system="h200_sxm",
        backend="vllm",
        worker_type="aggregated",
    ).to_dict()
    config.update(extra)
    with pytest.raises(ValueError, match="cannot represent AIC setting"):
        simulation.DynamoReplayRunner._engine_args(
            {
                "timing_model": {
                    "type": "external",
                    "provider": "aic",
                    "config": config,
                },
            }
        )


def test_factory_preserves_trtllm_disagg_gate() -> None:
    capabilities = simulation.DynamoReplayRunnerFactory().capabilities()

    assert capabilities.supports_backend_topology("trtllm", "agg")
    assert not capabilities.supports_backend_topology("trtllm", "disagg")
    assert capabilities.supports_disaggregated_attention_dp
    assert capabilities.supports_agentic_profile
    assert not capabilities.supports_agentic_host_offload


def test_factory_owns_replay_spec_abi_version(monkeypatch) -> None:
    seen = {}

    class SentinelCapabilities:
        def __init__(
            self,
            replay_spec_api_version=999,
            supported_backend_topologies=(),
            supported_hooks=(),
            supports_disaggregated_attention_dp=False,
            **features,
        ):
            seen["version"] = replay_spec_api_version
            seen[
                "supports_disaggregated_attention_dp"
            ] = supports_disaggregated_attention_dp
            self.replay_spec_api_version = replay_spec_api_version
            self.supported_backend_topologies = supported_backend_topologies
            self.supported_hooks = supported_hooks

    monkeypatch.setattr(simulation, "RunnerCapabilities", SentinelCapabilities)

    simulation.DynamoReplayRunnerFactory().capabilities()

    assert simulation._REPLAY_SPEC_API_VERSION == 1
    assert seen["version"] == 1
    assert seen["supports_disaggregated_attention_dp"] is True


@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_affinity_is_separate_from_kv_selection(mode) -> None:
    config = RouterPredictionConfig.model_validate(
        {"policy": "kv_router", "affinity": {"mode": mode, "ttl_seconds": 1.5}}
    )
    assert config.affinity.mode == mode
    assert config.affinity.ttl_seconds == 1.5
    assert config.overlap_score_credit == 1.0


@pytest.mark.parametrize("ttl", [True, "3600", 0, float("inf"), 31_536_001])
def test_affinity_rejects_invalid_ttl(ttl) -> None:
    with pytest.raises(ValueError):
        RouterPredictionConfig.model_validate(
            {"policy": "kv_router", "affinity": {"mode": "session", "ttl_seconds": ttl}}
        )


def test_affinity_rejects_round_robin() -> None:
    with pytest.raises(ValueError, match="requires policy='kv_router'"):
        RouterPredictionConfig.model_validate(
            {"policy": "round_robin", "affinity": {"mode": "session"}}
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"extra_engine_args": object()},
        {"planner_config": {}},
        {"max_sim_time_ms": 1},
        {"agentic_lanes": 2},
        {"num_workers": 2},
        {"arrival_speedup_ratio": 2},
        {"capture_per_request": True},
    ],
)
def test_canonical_entry_rejects_conflicting_legacy_arguments(extra) -> None:
    with pytest.raises(ValueError, match="legacy replay arguments"):
        run_trace_replay([], router_mode="kv_router", replay_spec_json="{}", **extra)


def test_canonical_entry_preserves_payload_and_native_resource_error(
    monkeypatch,
) -> None:
    payload = (
        '{"spec":{"version":1},"traffic":{"agentic_profile":{"duration_seconds":1.5}}}'
    )
    seen = {}

    def failed_native(*args, **kwargs):
        seen.update(kwargs)
        raise MemoryError("native report storage exhausted")

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", failed_native)
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_CORE_VERSION", "0.13.0", raising=False
    )
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", 1, raising=False
    )
    monkeypatch.setattr(replay_api, "version", lambda name: "0.13.0")
    with pytest.raises(MemoryError, match="native report storage"):
        run_trace_replay(
            [],
            router_mode="kv_router",
            replay_spec_json=payload,
            affinity={"mode": "session", "ttl_seconds": 1.5},
        )
    assert seen["replay_spec_json"] == payload
    assert json.loads(seen["affinity_json"])["ttl_seconds"] == 1.5


@pytest.mark.parametrize(
    "compiled,api",
    [(None, None), ("0.12.0", 1), ("0.13.0", True), ("0.13.0-dev.20260923", 1)],
)
def test_canonical_entry_requires_matching_native_runtime(monkeypatch, compiled, api):
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_CORE_VERSION", compiled, raising=False
    )
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", api, raising=False
    )
    monkeypatch.setattr(replay_api, "version", lambda name: "0.13.0")
    with pytest.raises(ValueError, match="matching AISimulate Python.*compiled core"):
        run_trace_replay([], router_mode="kv_router", replay_spec_json="{}")


def test_canonical_entry_accepts_equivalent_dev_version_spelling(monkeypatch):
    monkeypatch.setattr(
        replay_api._core,
        "AISIMULATE_CORE_VERSION",
        "0.13.0-dev.20260923",
        raising=False,
    )
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", 1, raising=False
    )
    monkeypatch.setattr(replay_api, "version", lambda name: "0.13.0.dev20260923")
    monkeypatch.setattr(
        replay_api, "_run_mocker_trace_replay", lambda *args, **kwargs: "{}"
    )
    assert run_trace_replay([], router_mode="kv_router", replay_spec_json="{}") == "{}"


def test_explicit_empty_profile_uses_canonical_path():
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(), workload={"agentic_profile": {}}, goal={}
    )
    assert simulation.DynamoReplayRunner._requires_canonical_replay(spec, None)


def test_goodput_goal_fails_closed_when_replay_omits_metric(monkeypatch) -> None:
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation,
        "run_trace_replay",
        lambda **kwargs: _report({"output_throughput_tok_s": 42.0}),
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl"},
        goal={
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 100.0, "itl_ms": 20.0},
        },
    )

    with pytest.raises(RuntimeError, match="did not emit goodput"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)
