# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Tests for the transitional Dynamo Sweeper replay runner."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate import _runtime
from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)

from dynamo.replay import PlannerReplayDetails, TelemetryOptions
from dynamo.replay import api as replay_api
from dynamo.replay import run_trace_replay, simulation
from dynamo.replay.config import lower_upstream_engine_args
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


def _fixed_args(backend="vllm", role="aggregated"):
    return {
        "engine_type": backend,
        "worker_type": role,
        "max_num_seqs": 256,
        "block_size": 16,
        "num_gpu_blocks": 4096,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }


def _engine_args(payload):
    return simulation.MockEngineArgs.from_json(
        json.dumps(lower_upstream_engine_args(payload))
    )


def _router_adapter(router_mode="kv_router", affinity=None, **config):
    return AdapterReplaySpec(
        runtime_hooks=(
            RuntimeHookSpec(
                "dynamo.router",
                "placement_policy",
                1,
                {
                    "router_mode": router_mode,
                    "router_config": config,
                    **({"affinity": affinity} if affinity is not None else {}),
                },
            ),
        )
    )


def _capture_native(monkeypatch, report):
    seen = {}

    def run_native(_trace_files, **kwargs):
        seen.update(kwargs)
        seen["payload"] = json.loads(kwargs["replay_spec_json"])
        return json.dumps({"lifecycle_operations": [], **report})

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", run_native)
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_CORE_VERSION", "0.13.0", raising=False
    )
    monkeypatch.setattr(
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", 2, raising=False
    )
    monkeypatch.setattr(replay_api, "version", lambda name: "0.13.0")
    return seen


def _agg_deployment() -> BackendDeploymentSpec:
    return BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args=_fixed_args(),
        num_workers=3,
        performance_model_metadata={
            "aggregated": {"config": {"model_path": "target-model"}}
        },
    )


def test_trace_runner_preserves_current_replay_arguments(monkeypatch) -> None:
    seen = _capture_native(
        monkeypatch,
        {"output_throughput_tok_s": 42.0, "goodput_output_throughput_tok_s": 40.0},
    )
    planner_calls = {}

    class Planner:
        def finalize(self, lifecycle):
            planner_calls["lifecycle"] = lifecycle
            return PlannerReplayDetails(total_ticks=7)

    planner = Planner()

    @contextmanager
    def scope(**kwargs):
        planner_calls.update(kwargs)
        yield planner

    monkeypatch.setattr(simulation, "_planner_replay_adapter", lambda: scope)
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
                        "dynamo.planner",
                        "scaling_policy",
                        1,
                        {"planner_config": {"mode": "agg"}},
                    ),
                )
            ),
            "dynamo.router": _router_adapter(overlap_score_credit=0.5),
        },
    )
    report = simulation.DynamoReplayRunnerFactory().create(2).run(spec)
    payload = seen["payload"]
    assert payload["traffic"]["trace_path"] == "tiny.jsonl"
    assert payload["traffic"]["trace_format"] == "dynamo"
    assert payload["traffic"]["arrival_speedup_ratio"] == 2.0
    assert payload["traffic"]["replay_concurrency"] == 8
    assert "trace_block_size" not in payload["traffic"]
    assert payload["spec"]["topology"]["workers"]["initial_workers"] == 3
    assert payload["spec"]["sla"] == {"ttft_ms": 100.0, "itl_ms": 20.0}
    assert payload["spec"]["record_per_request"] is False
    assert seen["router_mode"] == "kv_router"
    assert seen["router_config"].overlap_score_credit == 0.5
    assert seen["scaling_policy"] is planner
    assert planner_calls["planner_config_arg"] == '{"mode": "agg"}'
    assert planner_calls["benchmark_granularity"] == 8
    assert planner_calls["capture_details"] is False
    assert planner_calls["lifecycle"] == []
    assert report.metrics["planner_total_ticks"] == 7.0
    assert report.metadata["planner_total_ticks"] == 7


def test_trace_paths_only_workload_routes_to_trace_replay(monkeypatch) -> None:
    seen = _capture_native(monkeypatch, {"completed_requests": 2})
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
    assert seen["payload"]["traffic"]["trace_paths"] == ["first.jsonl", "second.jsonl"]
    assert seen["payload"]["traffic"]["arrival_speedup_ratio"] == 2.0
    assert seen["payload"]["traffic"]["agentic_lanes"] == 4
    assert seen["payload"]["traffic"]["source_type"] == "trace"
    assert report.metrics["completed_requests"] == 2.0


@pytest.mark.parametrize(
    ("nested_timestamp_basis", "resolved_timestamp_basis"),
    [
        (None, "relative"),
        ("auto", "relative"),
        ("absolute", "absolute"),
        ("relative", "relative"),
        (None, "not_applicable"),
    ],
)
def test_weka_runner_delegates_without_inventing_a_source_block_size(
    monkeypatch, nested_timestamp_basis, resolved_timestamp_basis
) -> None:
    seen = _capture_native(
        monkeypatch,
        {
            "completed_requests": 2,
            "agentic_graph": {"source_models": ["source-a", "source-b"]},
            "weka_nested_timestamp_basis": resolved_timestamp_basis,
        },
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "trace_path": "published-weka",
            "trace_format": "weka",
            "agentic_lanes": 1,
            "weka_nested_timestamp_basis": nested_timestamp_basis,
        },
        goal={"target": "throughput"},
    )
    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    traffic = seen["payload"]["traffic"]
    assert "trace_block_size" not in traffic
    assert traffic["agentic_lanes"] == 1
    assert traffic["execution_model"] == "target-model"
    assert traffic.get("weka_nested_timestamp_basis") == nested_timestamp_basis
    assert report.metadata["weka_nested_timestamp_basis"] == resolved_timestamp_basis
    assert report.metadata["agentic_graph"] == {
        "source_models": ["source-a", "source-b"]
    }
    assert report.metadata["agentic_model_projection"] == {
        "policy": "project_to_configured_target",
        "source_models": ["source-a", "source-b"],
        "target_model": "target-model",
    }
    assert "native_report" not in report.metadata


@pytest.mark.parametrize(
    ("metadata_config", "engine_args"),
    [
        pytest.param({"model_path": " target-model "}, {}, id="metadata-model-path"),
        pytest.param({"model": " target-model "}, {}, id="metadata-canonical-model"),
        pytest.param({}, {"aic_model_path": " target-model "}, id="engine-aic-path"),
        pytest.param(
            {},
            {"ais_perf_config": {"model": " target-model "}},
            id="engine-ais-perf-config",
        ),
    ],
)
def test_weka_runner_resolves_each_execution_target_model_source(
    monkeypatch, metadata_config, engine_args
) -> None:
    seen = _capture_native(monkeypatch, {"completed_requests": 1})
    args = {**_fixed_args(), **engine_args}
    if "ais_perf_config" in args:
        args.pop("timing_model")
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args=args,
        num_workers=1,
        performance_model_metadata={"aggregated": {"config": metadata_config}},
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "published-weka", "trace_format": "weka"},
        goal={"target": "throughput"},
    )
    simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert seen["payload"]["traffic"]["execution_model"] == "target-model"


def test_weka_runner_requires_a_configured_execution_target_model() -> None:
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args=_fixed_args(),
        num_workers=3,
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "published-weka", "trace_format": "weka"},
        goal={"target": "throughput"},
    )

    with pytest.raises(
        ValueError,
        match="agentic execution requires a configured target model",
    ):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


def test_dynamo_runner_defers_target_model_validation_until_trace_load(
    monkeypatch,
) -> None:
    seen = _capture_native(monkeypatch, {"completed_requests": 1})
    deployment = replace(_agg_deployment(), performance_model_metadata={})
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "standard.jsonl", "trace_format": "dynamo"},
        goal={"target": "throughput"},
    )
    simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert "execution_model" not in seen["payload"]["traffic"]


@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
def test_runner_forwards_and_retains_requested_telemetry(
    monkeypatch, router_mode
) -> None:
    telemetry = {
        "sample_interval_ms": 2500.0,
        "samples": [{"sample_ordinal": 0, "kind": "baseline", "sampled_at_ms": 0.0}],
    }
    seen = _capture_native(
        monkeypatch, {"completed_requests": 1, "telemetry": telemetry}
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl", "trace_format": "dynamo"},
        goal={"target": "throughput"},
        adapters={"dynamo.router": _router_adapter(router_mode=router_mode)},
    )
    report = (
        simulation.DynamoReplayRunnerFactory()
        .create(2)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(
                capture_telemetry=True, telemetry_sample_interval_ms=2500.0
            ),
        )
    )
    assert seen["router_mode"] == router_mode
    assert seen["capture_telemetry"] is True
    assert seen["telemetry_sample_interval_ms"] == 2500.0
    assert report.metadata["telemetry"] == telemetry
    assert "native_report" not in report.metadata


def test_trace_replay_rejects_boolean_agentic_lanes() -> None:
    with pytest.raises(TypeError, match="agentic_lanes must be an integer"):
        run_trace_replay("unused.jsonl", agentic_lanes=True)


def test_runner_captures_per_request_output_when_requested(monkeypatch) -> None:
    records = [{"request_id": "request-1", "ttft_ms": 4.0}]
    seen = _capture_native(
        monkeypatch, {"completed_requests": 1, "per_request": records}
    )
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
                include_raw_report=True, capture_per_request=True
            ),
        )
    )
    assert seen["payload"]["spec"]["record_per_request"] is True
    assert report.metadata["native_report"]["per_request"] == records


@pytest.mark.parametrize("mode", ["session", "sibling_group"])
@pytest.mark.parametrize("capture_telemetry", [False, True])
@pytest.mark.parametrize("capture_per_request", [False, True])
def test_affinity_changes_composition_without_changing_shared_pipeline(
    monkeypatch, mode, capture_telemetry, capture_per_request
):
    evidence = {"roles": {"aggregated": {"native_policy": "dynamo.SelectionCore"}}}
    telemetry = {"sample_interval_ms": 1000.0, "samples": []}
    seen = _capture_native(
        monkeypatch,
        {"completed_requests": 1, "routing_policy": evidence, "telemetry": telemetry},
    )
    base = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "trace.jsonl", "trace_format": "dynamo"},
        goal={},
        adapters={"dynamo.router": _router_adapter()},
    )
    requirements = ReplayOutputRequirements(
        capture_telemetry=capture_telemetry,
        capture_per_request=capture_per_request,
    )
    runner = simulation.DynamoReplayRunnerFactory().create(0)
    runner.run(base, output_requirements=requirements)
    payload_without_affinity = seen["payload"]
    assert seen["affinity_json"] is None
    affinity = {"mode": mode, "ttl_seconds": 3600}
    report = runner.run(
        replace(base, adapters={"dynamo.router": _router_adapter(affinity=affinity)}),
        output_requirements=requirements,
    )
    assert seen["payload"] == payload_without_affinity
    assert json.loads(seen["affinity_json"]) == affinity
    assert seen.get("capture_telemetry", False) is capture_telemetry
    assert seen["payload"]["spec"]["record_per_request"] is capture_per_request
    assert report.metadata["routing_policy"] == evidence
    if capture_telemetry:
        assert report.metadata["telemetry"] == telemetry
    else:
        assert "telemetry" not in report.metadata
    if capture_per_request:
        assert report.metadata["native_report"]["routing_policy"] == evidence


@pytest.mark.parametrize(
    "trace_format",
    [
        "mooncake",
        "mooncake-delta",
        "agentic_mooncake",
        "applied_compute_agentic",
        "dynamo",
        "weka",
    ],
)
def test_legacy_trace_formats_use_the_shared_traffic_driver(monkeypatch, trace_format):
    seen = _capture_native(monkeypatch, {"completed_requests": 1})
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "trace.jsonl", "trace_format": trace_format},
        goal={},
    )
    simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    traffic = seen["payload"]["traffic"]
    assert traffic["source_type"] == "trace"
    assert traffic["load_type"] == "trace_timestamps"
    assert traffic["trace_format"] == trace_format
    assert seen["payload"]["spec"]["requests"] == []
    assert "source_type" not in spec.workload


def test_legacy_identity_normalization_preserves_config_and_rejects_conflicts():
    identity = {
        "model": "target-model",
        "worker_type": "aggregated",
        "estimator_config": {"correction": {"enabled": False}},
    }
    deployment = replace(
        _agg_deployment(), agg_engine_args={"ais_perf_config": identity}
    )
    spec = ReplaySpec(
        backend_deployment=deployment, workload={"isl": 8, "osl": 2}, goal={}
    )
    normalized = simulation._normalize_legacy_spec(spec)
    assert normalized.backend_deployment.agg_engine_args["timing_model"] == {
        "type": "external",
        "provider": "aic",
        "config": identity,
    }
    assert spec.backend_deployment.agg_engine_args == {"ais_perf_config": identity}
    conflicting = replace(
        deployment,
        agg_engine_args={
            "ais_perf_config": identity,
            "timing_model": {"type": "fixed"},
        },
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        simulation._normalize_legacy_spec(replace(spec, backend_deployment=conflicting))


def test_legacy_disagg_model_projection_preserves_input_and_prefill_metadata():
    metadata = {"prefill": {"label": "keep", "config": {"dp_size": 2}}}
    deployment = replace(
        _agg_deployment(),
        deployment_mode="disagg",
        agg_engine_args=None,
        prefill_engine_args=_fixed_args(),
        decode_engine_args={"ais_perf_config": {"model": "target-model"}},
        performance_model_metadata=metadata,
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "legacy.jsonl", "trace_format": "agentic_mooncake"},
        goal={},
    )
    normalized = simulation._normalize_legacy_spec(spec)
    assert normalized.backend_deployment.performance_model_metadata["prefill"] == {
        "label": "keep",
        "config": {"dp_size": 2, "model": "target-model"},
    }
    assert metadata == {"prefill": {"label": "keep", "config": {"dp_size": 2}}}
    assert deployment.decode_engine_args == {
        "ais_perf_config": {"model": "target-model"}
    }
    assert "source_type" not in spec.workload


@pytest.mark.parametrize(
    "source_type,prefill_model",
    [(None, "different-model"), ("trace", None)],
    ids=["legacy-explicit-mismatch", "canonical-missing-prefill"],
)
def test_disagg_model_projection_keeps_shared_validation(source_type, prefill_model):
    metadata = {"decode": {"config": {"model": "target-model"}}}
    if prefill_model is not None:
        metadata["prefill"] = {"config": {"model": prefill_model}}
    deployment = replace(
        _agg_deployment(),
        deployment_mode="disagg",
        agg_engine_args=None,
        prefill_engine_args=_fixed_args(),
        decode_engine_args=_fixed_args(),
        num_prefill_workers=1,
        num_decode_workers=1,
        performance_model_metadata=metadata,
    )
    workload = {"trace_path": "legacy.jsonl", "trace_format": "agentic_mooncake"}
    if source_type is not None:
        workload.update(source_type=source_type, load_type="trace_timestamps")
    spec = ReplaySpec(backend_deployment=deployment, workload=workload, goal={})
    with pytest.raises(ValueError, match="same configured target model"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


def test_canonical_affinity_forwards_optional_telemetry_sinks(monkeypatch, tmp_path):
    seen = _capture_native(monkeypatch, {"completed_requests": 1})

    def callback(sample):
        return None

    path = tmp_path / "telemetry.jsonl"
    payload = '{"spec":{"version":1,"record_per_request":true}}'
    result = run_trace_replay(
        [],
        router_mode="kv_router",
        replay_spec_json=payload,
        affinity={"mode": "session"},
        telemetry_options=TelemetryOptions(callback=callback, jsonl_path=path),
    )
    assert json.loads(result)["completed_requests"] == 1
    assert seen["replay_spec_json"] == payload
    assert seen["capture_telemetry"] is False
    assert seen["telemetry_callback"] is callback
    assert seen["telemetry_jsonl_path"] == path


def test_synthetic_disagg_preserves_request_count_and_load(monkeypatch) -> None:
    seen = _capture_native(monkeypatch, {"output_throughput_tok_s": 99.0})
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="sglang",
        backend_version="0.5.6",
        prefill_engine_args=_fixed_args("sglang", "prefill"),
        decode_engine_args=_fixed_args("sglang", "decode"),
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
    traffic = seen["payload"]["traffic"]
    assert traffic["source_type"] == "synthetic-session"
    assert traffic["load_type"] == "concurrency"
    assert traffic["isl"] == 512 and traffic["osl"] == 128
    assert traffic["concurrency"] == 32 and traffic["num_request_ratio"] == 10.0
    assert "arrival_interval_ms" not in traffic and "request_rate" not in traffic
    assert traffic["turns_per_session"] == 2
    assert traffic["shared_prefix_ratio"] == 0.5 and traffic["num_prefix_groups"] == 4
    assert traffic["inter_turn_delay_ms"] == 12.0
    topology = seen["payload"]["spec"]["topology"]
    assert topology["prefill"]["initial_workers"] == 2
    assert topology["decode"]["initial_workers"] == 4
    assert report.metrics["output_throughput_tok_s"] == 99.0


def test_synthetic_request_rate_preserves_open_loop_load(monkeypatch) -> None:
    seen = _capture_native(monkeypatch, {"output_throughput_tok_s": 99.0})
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "request_rate": 20.0,
        },
        goal={"target": "throughput"},
    )
    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    traffic = seen["payload"]["traffic"]
    assert traffic["request_rate"] == 20.0 and traffic["num_request_ratio"] == 10.0
    assert traffic["load_type"] == "constant_rate"
    assert traffic.get("concurrency") is None
    assert seen["payload"]["spec"]["max_in_flight"] is None
    assert report.metrics["output_throughput_tok_s"] == 99.0


@pytest.mark.parametrize("request_rate", [0.0, -1.0])
def test_synthetic_request_rate_must_be_positive(
    monkeypatch, request_rate: float
) -> None:
    monkeypatch.setattr(
        simulation._DynamoReplayRuntime,
        "run_replay_json",
        lambda self, payload: _runtime.run_replay_json(payload),
    )
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "request_rate": request_rate,
        },
        goal={"target": "throughput"},
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="request_rate must be positive"
    ):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


def test_direct_predict_resolves_kv_capacity_fraction(monkeypatch) -> None:
    monkeypatch.setattr(
        simulation._DynamoReplayRuntime,
        "run_replay_json",
        lambda self, payload: _runtime.run_replay_json(payload),
    )
    deployment = replace(
        _agg_deployment(),
        agg_engine_args={**_fixed_args(), "num_gpu_blocks": 100, "block_size": 16},
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={
            "isl": 100,
            "osl": 20,
            "kv_load_ratio": 0.5,
            "num_request_ratio": 10.0,
        },
        goal={"target": "throughput"},
    )
    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == 210


def test_fixed_timing_keeps_aic_identity_out_of_runtime_args(monkeypatch) -> None:
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    engine_args = _engine_args(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_backend_version": "0.11.0",
            "aic_system": "h200_sxm",
            "aic_model_path": "example/model",
            "aic_attention_dp_size": 2,
            "aic_pp_size": 1,
            "num_gpu_blocks": 4096,
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


def test_factory_supports_native_trtllm_disagg() -> None:
    capabilities = simulation.DynamoReplayRunnerFactory().capabilities()

    assert capabilities.supports_backend_topology("trtllm", "agg")
    assert capabilities.supports_backend_topology("trtllm", "disagg")
    assert capabilities.supports_disaggregated_attention_dp


def test_factory_follows_engine_capabilities_with_adapter_constraints(
    monkeypatch,
) -> None:
    native = replace(
        simulation.EngineReplayRunnerFactory().capabilities(),
        supported_backend_topologies=(
            ("vllm", "agg"),
            ("trtllm", "disagg"),
            ("vllm", "afd"),
            ("future_backend", "agg"),
        ),
        supports_disaggregated_attention_dp=False,
    )
    monkeypatch.setattr(
        simulation.EngineReplayRunnerFactory, "capabilities", lambda self: native
    )

    capabilities = simulation.DynamoReplayRunnerFactory().capabilities()

    assert capabilities.supports_backend_topology("trtllm", "disagg")
    assert not capabilities.supports_backend_topology("sglang", "agg")
    assert not capabilities.supports_backend_topology("vllm", "afd")
    assert not capabilities.supports_backend_topology("future_backend", "agg")
    assert not capabilities.supports_disaggregated_attention_dp
    for provider, kind in (
        ("dynamo.router", "placement_policy"),
        ("dynamo.planner", "scaling_policy"),
    ):
        assert capabilities.supports_hook(
            RuntimeHookSpec(provider=provider, kind=kind, api_version=1, config={})
        )


def test_factory_owns_replay_spec_abi_version(monkeypatch) -> None:
    seen = {}

    class SentinelCapabilities:
        def __init__(
            self,
            replay_spec_api_version=999,
            supported_backend_topologies=(),
            supported_hooks=(),
            supports_disaggregated_attention_dp=False,
            **_kwargs,
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


def test_goodput_goal_fails_closed_when_replay_omits_metric(monkeypatch) -> None:
    _capture_native(monkeypatch, {"output_throughput_tok_s": 42.0})
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl"},
        goal={"target": "goodput_per_gpu", "sla": {"ttft_ms": 100.0, "itl_ms": 20.0}},
    )
    with pytest.raises(RuntimeError, match="did not emit goodput"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


def test_planner_bootstrap_preserves_each_canonical_role_identity():
    from types import SimpleNamespace

    from dynamo.replay.planner import _ais_session_kwargs

    prefill = {
        "model": "model-p",
        "system": "gpu-p",
        "backend": "vllm",
        "worker_type": "prefill",
        "systems_paths": ["custom-p"],
        "estimator_config": {"correction": {"enabled": False}},
    }
    decode = {
        "model": "model-d",
        "system": "gpu-d",
        "backend": "sglang",
        "worker_type": "decode",
        "systems_paths": ["custom-d"],
    }
    for config in (prefill, decode):
        args = SimpleNamespace(ais_perf_config=config)
        assert _ais_session_kwargs(None, args) == {"config": config}
        assert _ais_session_kwargs(
            config,
            SimpleNamespace(ais_perf_config=None, worker_type=config["worker_type"]),
        ) == {"config": config}


def test_public_prediction_bootstrap_prefers_canonical_worker_policy():
    from pathlib import Path

    import yaml
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig

    from dynamo.replay.planner import _ais_session_kwargs

    path = (
        Path(__file__).parent
        / "e2e/configs/unified_cli/predict/dynamo/08-synthetic-throughput-planner.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    raw.pop("planner")
    raw["engine"].update(estimation_mode="op_level", database_mode="SOL")
    raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "default"}
    deployment = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(raw)
    ).backend_deployment
    args = _engine_args(deployment.agg_engine_args)
    metadata = deployment.performance_model_metadata["aggregated"]["config"]
    assert metadata["model"] == raw["engine"]["model"]
    config = _ais_session_kwargs(metadata, args)["config"]
    assert config == args.ais_perf_config
    assert config["database_mode"] == "SOL"
    assert config["estimation_mode"] == "op_level"
    assert config["systems_paths"]


@pytest.mark.parametrize(
    "timing",
    [
        {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        {"type": "polynomial"},
    ],
)
def test_custom_timing_without_capacity_does_not_resolve_unused_model(
    monkeypatch, timing
):
    import aisimulate.capacity

    def unexpected_capacity_lookup(**kwargs):
        raise AssertionError("custom timing must not look up an unused model")

    monkeypatch.setattr(
        aisimulate.capacity, "estimate_num_gpu_blocks", unexpected_capacity_lookup
    )
    args = _engine_args(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_model_path": "/unused/model",
            "aic_system": "unused-gpu",
            "aic_tp_size": 2,
            "aic_attention_dp_size": 2,
            "timing_model": timing,
        }
    )
    assert args.num_gpu_blocks == 16384
    assert args.dp_size == 2
    assert args.ais_tp_size == 2
    assert args.ais_perf_config is None


@pytest.mark.parametrize(
    "timing",
    [
        {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        {"type": "polynomial"},
    ],
)
def test_compiled_custom_timing_consumes_capacity_only_fields(timing):
    from pathlib import Path

    import yaml
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig

    path = (
        Path(__file__).parent
        / "e2e/configs/unified_cli/predict/dynamo/07-synthetic-ais-router.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    raw.pop("router")
    raw["engine"]["backend_version"] = "current"
    worker = raw["engine"]["workers"]["aggregated"]
    worker["timing"] = timing
    worker["kv_cache"]["capacity"] = {
        "type": "default",
        "cuda_graph_reserved_bytes": 4096,
    }
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    payload = spec.backend_deployment.agg_engine_args
    assert payload["cuda_graph_reserved_bytes"] == 4096
    assert payload["num_gpu_blocks"] > 0
    args = _engine_args(payload)
    assert args.num_gpu_blocks == payload["num_gpu_blocks"]
    assert args.ais_perf_config is None
    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == raw["traffic"]["stop"]["requests"]


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
        {"execution_model": "example/model"},
        {"weka_nested_timestamp_basis": "relative"},
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
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", 2, raising=False
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
        replay_api._core, "AISIMULATE_REPLAY_API_VERSION", 2, raising=False
    )
    monkeypatch.setattr(replay_api, "version", lambda name: "0.13.0.dev20260923")
    monkeypatch.setattr(
        replay_api, "_run_mocker_trace_replay", lambda *args, **kwargs: "{}"
    )
    assert run_trace_replay([], router_mode="kv_router", replay_spec_json="{}") == "{}"


def test_explicit_empty_profile_keeps_the_shared_pipeline(monkeypatch):
    seen = _capture_native(monkeypatch, {"completed_requests": 1})
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_path": "unused.jsonl",
            "trace_format": "weka",
            "agentic_lanes": 1,
            "agentic_snapshot": {"seed": 42},
            "agentic_profile": {},
        },
        goal={},
    )
    simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert seen["payload"]["traffic"]["agentic_profile"] == {}


@pytest.mark.parametrize("include_raw_report", [False, True])
@pytest.mark.parametrize("has_goodput", [False, True])
def test_canonical_runner_enforces_goodput_contract(
    monkeypatch, include_raw_report, has_goodput
) -> None:
    closed = []
    metrics = {"output_throughput_tok_s": 42.0}
    if has_goodput:
        metrics["goodput_output_throughput_tok_s"] = 40.0

    class FakeCanonicalRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, spec, *, output_requirements):
            return simulation.ReplayReport(
                metrics=metrics,
                metadata={"native_report": {}},
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(simulation, "EngineReplayRunner", FakeCanonicalRunner)
    monkeypatch.setattr(simulation, "KvRouterConfig", _FakeRouterConfig)
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl"},
        goal={"target": "goodput", "sla": {"ttft_ms": 100.0}},
        adapters={
            "dynamo.router": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.router",
                        kind="placement_policy",
                        api_version=1,
                        config={
                            "router_mode": "kv_router",
                            "router_config": {},
                            "affinity": {"mode": "session"},
                        },
                    ),
                )
            ),
        },
    )
    runner = simulation.DynamoReplayRunnerFactory().create(0)
    requirements = ReplayOutputRequirements(include_raw_report=include_raw_report)
    if has_goodput:
        report = runner.run(spec, output_requirements=requirements)
        assert report.metrics["goodput_output_throughput_tok_s"] == 40.0
    else:
        with pytest.raises(RuntimeError, match="did not emit goodput"):
            runner.run(spec, output_requirements=requirements)
    assert closed == [True]
