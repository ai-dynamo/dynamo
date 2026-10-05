# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Tests for the transitional Dynamo Sweeper replay runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

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

from dynamo.replay import (
    PlannerReplayDetails,
    ReplayReport,
    ReplayTelemetryDetails,
    TelemetryOptions,
)
from dynamo.replay import api as replay_api
from dynamo.replay import config as replay_config
from dynamo.replay import run_trace_replay, simulation
from dynamo.replay.config import lower_upstream_engine_args

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


class _FakeEngineArgs:
    def __init__(self, payload: str):
        self.payload = payload
        values = json.loads(payload)
        self.ais_nextn = values.get("ais_nextn")
        self.ais_nextn_accept_rates = values.get("ais_nextn_accept_rates")
        self.ais_mtp_seed = values.get("ais_mtp_seed", 42)

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
        performance_model_metadata={
            "aggregated": {"config": {"model_path": "target-model"}}
        },
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
    assert seen["trace_block_size"] is None
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
    monkeypatch,
    nested_timestamp_basis,
    resolved_timestamp_basis,
) -> None:
    seen = {}

    def fake_native_replay(_trace_files, **kwargs):
        seen.update(kwargs)
        return _report(
            {
                "completed_requests": 2,
                "agentic_graph": {
                    "source_models": ["source-a", "source-b"],
                },
                "weka_nested_timestamp_basis": resolved_timestamp_basis,
            }
        )

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", fake_native_replay)
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

    assert seen["trace_block_size"] is None
    assert seen["agentic_lanes"] == 1
    assert seen["execution_model"] == "target-model"
    assert seen["weka_nested_timestamp_basis"] == nested_timestamp_basis
    assert report.metadata == {
        "agentic_qualification": "functional_only",
        "agentic_input_format": "weka",
        "agentic_lanes": 1,
        "weka_nested_timestamp_basis": resolved_timestamp_basis,
        "agentic_graph": {
            "source_models": ["source-a", "source-b"],
        },
        "agentic_model_projection": {
            "policy": "project_to_configured_target",
            "source_models": ["source-a", "source-b"],
            "target_model": "target-model",
        },
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
            {
                "timing_model": {
                    "type": "external",
                    "provider": "aic",
                    "config": {"model": " target-model "},
                }
            },
            id="engine-canonical-timing",
        ),
        pytest.param(
            {},
            {"ais_perf_config": {"model": " target-model "}},
            id="engine-ais-perf-config",
        ),
    ],
)
def test_weka_runner_resolves_each_execution_target_model_source(
    metadata_config, engine_args
) -> None:
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args={"engine_type": "vllm", **engine_args},
        num_workers=1,
        performance_model_metadata={"aggregated": {"config": metadata_config}},
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "published-weka", "trace_format": "weka"},
        goal={"target": "throughput"},
    )

    assert simulation.DynamoReplayRunner._execution_target_model(spec) == (
        "target-model"
    )


def test_weka_runner_requires_a_configured_execution_target_model() -> None:
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args={"engine_type": "vllm", "max_num_seqs": 256},
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
    seen = {}

    def fake_run_trace_replay(**kwargs):
        seen.update(kwargs)
        return _report({"completed_requests": 1})

    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args={"engine_type": "vllm", "max_num_seqs": 256},
        num_workers=1,
    )
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(simulation, "run_trace_replay", fake_run_trace_replay)
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "standard.jsonl", "trace_format": "dynamo"},
        goal={"target": "throughput"},
    )

    simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen["execution_model"] is None


def test_runner_forwards_and_retains_requested_telemetry(monkeypatch) -> None:
    seen = {}
    sample = {"sample_ordinal": 0, "kind": "baseline", "sampled_at_ms": 0.0}

    def fake_run_trace_replay(**kwargs):
        seen.update(kwargs)
        return ReplayReport(
            summary={"completed_requests": 1},
            per_request=None,
            coverage={},
            planner=None,
            telemetry=ReplayTelemetryDetails(
                sample_interval_ms=2_500.0,
                samples=[sample],
            ),
        )

    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(simulation, "run_trace_replay", fake_run_trace_replay)
    spec = ReplaySpec(
        backend_deployment=_agg_deployment(),
        workload={"trace_path": "tiny.jsonl", "trace_format": "dynamo"},
        goal={"target": "throughput"},
    )

    report = (
        simulation.DynamoReplayRunnerFactory()
        .create(2)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(
                capture_telemetry=True,
                telemetry_sample_interval_ms=2_500.0,
            ),
        )
    )

    assert seen["telemetry_options"] == TelemetryOptions(sample_interval_ms=2_500.0)
    assert "native_report" not in report.metadata
    assert report.metadata["telemetry"] == {
        "sample_interval_ms": 2_500.0,
        "samples": [sample],
    }


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
    engine_args = simulation.DynamoReplayRunner._engine_args(
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

    monkeypatch.setattr(simulation, "_DynamoRunnerCapabilities", SentinelCapabilities)

    simulation.DynamoReplayRunnerFactory().capabilities()

    assert simulation._REPLAY_SPEC_API_VERSION == 1
    assert seen["version"] == 1
    assert seen["supports_disaggregated_attention_dp"] is True


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
    args = simulation.DynamoReplayRunner._engine_args(deployment.agg_engine_args)
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
    args = simulation.DynamoReplayRunner._engine_args(
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
    args = simulation.DynamoReplayRunner._engine_args(payload)
    assert args.num_gpu_blocks == payload["num_gpu_blocks"]
    assert args.ais_perf_config is None
    report = simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["completed_requests"] == raw["traffic"]["stop"]["requests"]


def _mtp_timing():
    return {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "system": "test-system",
            "backend": "vllm",
            "worker_type": "aggregated",
            "estimation_mode": "op_level",
        },
    }


@pytest.fixture
def mtp_capacity_passthrough(monkeypatch):
    monkeypatch.setattr(
        replay_config, "materialize_aic_num_gpu_blocks", lambda raw: dict(raw)
    )


@pytest.mark.parametrize(
    "speculative_args",
    [
        {
            "speculation": {
                "kind": "mtp",
                "num_speculative_tokens": 3,
                "expected_accepted_tokens": 2.4,
                "seed": 42,
            }
        },
        {"aic_nextn": 3, "aic_nextn_accepted": 2.4, "aic_mtp_seed": 42},
        {"nextn": 3, "nextn_accepted": 2.4, "mtp_seed": 42},
    ],
)
def test_lowering_preserves_mtp_expected_acceptance(
    speculative_args, mtp_capacity_passthrough
) -> None:
    payload = {
        "engine_type": "vllm",
        "num_gpu_blocks": 100,
        "timing_model": _mtp_timing(),
        **speculative_args,
    }
    original = json.loads(json.dumps(payload))

    lowered = lower_upstream_engine_args(payload)

    assert lowered["ais_nextn"] == 3
    assert list(
        map(float, lowered["ais_nextn_accept_rates"].split(","))
    ) == pytest.approx([1.0, 1.0, 0.4])
    assert lowered["ais_mtp_seed"] == 42
    assert "speculation" not in lowered
    assert "nextn_accepted" not in lowered
    assert not any(key.startswith("aic_") for key in lowered)
    assert payload == original
    assert lower_upstream_engine_args(lowered) == lowered
    if "speculation" not in speculative_args:
        assert lowered["ais_perf_config"]["nextn"] == 3


def test_lowering_rejects_conflicting_mtp_acceptance() -> None:
    with pytest.raises(ValueError, match="both|combined|conflict"):
        lower_upstream_engine_args(
            {
                "engine_type": "vllm",
                "num_gpu_blocks": 100,
                "aic_nextn": 3,
                "aic_nextn_accepted": 2.4,
                "aic_nextn_accept_rates": "1,1,1",
            }
        )


@pytest.mark.parametrize("canonical_depth", [None, 2, 3])
def test_public_mtp_rejects_direct_canonical_config(canonical_depth) -> None:
    canonical = _mtp_timing()["config"]
    if canonical_depth is not None:
        canonical["speculation"] = {
            "kind": "mtp",
            "params": {"num_speculative_tokens": canonical_depth},
        }
    payload = {
        "engine_type": "vllm",
        "num_gpu_blocks": 100,
        "ais_perf_config": canonical,
        "speculation": {
            "kind": "mtp",
            "num_speculative_tokens": 2,
            "expected_accepted_tokens": 1.5,
        },
    }
    with pytest.raises(
        ValueError, match="speculation cannot be combined with ais_perf_config"
    ):
        lower_upstream_engine_args(payload)


@pytest.mark.parametrize("alias", ["aic_nextn", "nextn"])
@pytest.mark.parametrize("depth", [None, 0])
def test_disabled_speculation_needs_no_new_ais_api(monkeypatch, alias, depth) -> None:
    def unavailable(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(replay_config, "import_module", unavailable)
    payload = {"engine_type": "vllm", "timing_model": None, alias: depth}
    assert lower_upstream_engine_args(payload) == {
        "engine_type": "vllm",
        "timing_model": None,
    }
    assert alias in payload


@pytest.mark.parametrize("depth", [False, 0.0, "0", -1])
def test_disabled_speculation_does_not_hide_invalid_depth(depth) -> None:
    with pytest.raises(ValueError, match="integer|0..=5"):
        lower_upstream_engine_args({"engine_type": "vllm", "aic_nextn": depth})


def test_old_ais_exports_allow_import_and_non_speculative_replay() -> None:
    code = """
import sys
import aisimulate.runner
for name in ('normalize_mtp_engine_args', 'speculation_report_metadata'):
    if hasattr(aisimulate.runner, name):
        delattr(aisimulate.runner, name)
sys.modules['aisimulate.speculation'] = None
from dynamo.replay.config import lower_upstream_engine_args
from dynamo.replay.simulation import DynamoReplayRunnerFactory
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec
spec = ReplaySpec(
    backend_deployment=BackendDeploymentSpec(
        deployment_mode='agg', backend='vllm', backend_version='current',
        agg_engine_args={'engine_type':'vllm', 'num_gpu_blocks':64,
                         'block_size':16, 'timing_model':None, 'aic_nextn':0},
        num_workers=1),
    workload={'isl':16, 'osl':8, 'request_count':2, 'concurrency':1}, goal={})
runner = DynamoReplayRunnerFactory().create(0)
try:
    report = runner.run(spec)
    assert report.metrics['completed_requests'] == 2
    assert 'speculation' not in report.metadata
finally:
    runner.close()
try:
    lower_upstream_engine_args({'aic_nextn':2})
except RuntimeError as error:
    assert 'requirements.aisimulate.txt' in str(error)
else:
    raise AssertionError('active SD must require the matching source API')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("location", ["public", "nested", "timing", "canonical"])
@pytest.mark.parametrize("agentic", [False, True])
def test_ngram_is_rejected_before_either_runner_path(location, agentic) -> None:
    chosen = {
        "kind": "ngram",
        "num_speculative_tokens": 2,
        "acceptance_rates": [1, 0.5],
    }
    args = {"engine_type": "vllm", "num_gpu_blocks": 100}
    if location in {"public", "nested"}:
        args["speculation"] = chosen
        if location == "nested":
            args = {"rank": args, "dp_size": 1}
    else:
        canonical = {
            "speculation": {"kind": "ngram", "params": {"num_speculative_tokens": 2}}
        }
        if location == "timing":
            args["timing_model"] = {
                "type": "external",
                "provider": "aic",
                "config": canonical,
            }
        else:
            args["ais_perf_config"] = canonical
    spec = ReplaySpec(
        backend_deployment=replace(_agg_deployment(), agg_engine_args=args),
        workload={
            "trace_path": "unused.jsonl",
            "trace_format": "agentic_mooncake" if agentic else "mooncake",
        },
        goal={},
    )
    with pytest.raises(ValueError, match="only MTP"):
        simulation.DynamoReplayRunnerFactory().capabilities().require_compatible(spec)


def test_canonical_only_agentic_mtp_requires_authored_capacity() -> None:
    config = _mtp_timing()["config"]
    config["speculation"] = {"kind": "mtp", "params": {"num_speculative_tokens": 2}}
    spec = ReplaySpec(
        backend_deployment=replace(
            _agg_deployment(),
            agg_engine_args={
                "engine_type": "vllm",
                "ais_perf_config": config,
                "ais_nextn_accept_rates": "1,0.5",
            },
        ),
        workload={"trace_path": "unused.jsonl", "trace_format": "agentic_mooncake"},
        goal={},
    )
    with pytest.raises(ValueError, match="explicit fixed KV capacity"):
        simulation.DynamoReplayRunnerFactory().capabilities().require_compatible(spec)


def test_metadata_failure_precedes_native_execution(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation, "run_synthetic_trace_replay", lambda **kwargs: called.append(kwargs)
    )

    def invalid_metadata(spec, *, resolved_role_args):
        rank = resolved_role_args["aggregated"]["rank"]
        assert rank["aic_nextn"] == 2
        assert rank["aic_nextn_accept_rates"] == "1,0.5"
        raise ValueError("metadata cannot be resolved")

    monkeypatch.setattr(
        replay_config.speculation_api(), "speculation_report_metadata", invalid_metadata
    )
    spec = ReplaySpec(
        backend_deployment=replace(
            _agg_deployment(),
            agg_engine_args={
                "engine_type": "vllm",
                "num_gpu_blocks": 100,
                "aic_nextn": 2,
                "aic_nextn_accepted": 1.5,
                "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
            },
        ),
        workload={"isl": 16, "osl": 8, "request_count": 2, "concurrency": 1},
        goal={},
    )
    with pytest.raises(ValueError, match="metadata cannot be resolved"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    assert called == []


def test_ordinary_legacy_sampler_retains_default_acceptance() -> None:
    spec = ReplaySpec(
        backend_deployment=replace(
            _agg_deployment(),
            agg_engine_args={
                "engine_type": "vllm",
                "num_gpu_blocks": 100,
                "aic_nextn": 2,
                "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
            },
        ),
        workload={"isl": 16, "osl": 8, "request_count": 2, "concurrency": 1},
        goal={},
    )
    runner = simulation.DynamoReplayRunnerFactory().create(0)
    try:
        report = runner.run(spec)
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == 2
    assumptions = report.metadata["speculation"]["aggregated"]
    assert assumptions["expected_accepted_draft_tokens"] is None
    assert assumptions["conditional_acceptance_rates"] == "0.85,0.3"
    assert assumptions["cost_approximation"] == "fixed_timing"


def test_lowering_rejects_ngram_with_fixed_timing() -> None:
    with pytest.raises(ValueError, match="ngram draft scheduling"):
        lower_upstream_engine_args(
            {
                "engine_type": "vllm",
                "num_gpu_blocks": 100,
                "timing_model": {
                    "type": "fixed",
                    "prefill_ms": 1.0,
                    "decode_ms": 1.0,
                },
                "speculation": {
                    "kind": "ngram",
                    "num_speculative_tokens": 3,
                    "acceptance_rates": [1.0, 0.5, 0.2],
                },
            }
        )


def test_lowering_keeps_mtp_cost_in_canonical_timing(mtp_capacity_passthrough) -> None:
    payload = {
        "engine_type": "vllm",
        "num_gpu_blocks": 100,
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {
                "model": "test-model",
                "system": "test-system",
                "backend": "vllm",
                "worker_type": "aggregated",
            },
        },
        "speculation": {
            "kind": "mtp",
            "num_speculative_tokens": 3,
            "expected_accepted_tokens": 2.4,
            "seed": 42,
        },
    }

    lowered = lower_upstream_engine_args(payload)

    assert lowered["ais_perf_config"]["speculation"] == {
        "kind": "mtp",
        "params": {"num_speculative_tokens": 3},
    }
    assert lowered["ais_nextn"] == 3
    assert list(
        map(float, lowered["ais_nextn_accept_rates"].split(","))
    ) == pytest.approx([1.0, 1.0, 0.4])
    assert lowered["ais_mtp_seed"] == 42
    assert "timing_model" not in lowered
    assert "speculation" in payload


def test_mtp_report_keeps_acceptance_and_assumptions(
    monkeypatch, mtp_capacity_passthrough
) -> None:
    acceptance = {"sampled_draft_tokens": 24, "verification_steps": 10}
    monkeypatch.setattr(simulation, "MockEngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(
        simulation,
        "run_trace_replay",
        lambda **kwargs: _report(
            {"completed_requests": 2, "speculative_acceptance": acceptance}
        ),
    )
    deployment = replace(
        _agg_deployment(),
        agg_engine_args={
            "engine_type": "vllm",
            "num_gpu_blocks": 100,
            "timing_model": _mtp_timing(),
            "speculation": {
                "kind": "mtp",
                "num_speculative_tokens": 3,
                "expected_accepted_tokens": 2.4,
                "seed": 42,
            },
        },
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "tiny.jsonl", "trace_format": "agentic_mooncake"},
        goal={"target": "throughput"},
    )

    report = (
        simulation.DynamoReplayRunnerFactory()
        .create(0)
        .run(
            spec, output_requirements=ReplayOutputRequirements(include_raw_report=True)
        )
    )

    assert report.metadata["speculative_acceptance"] == acceptance
    assumed = report.metadata["speculation"]["aggregated"]
    assert assumed["resolved_method"] == "mtp"
    assert assumed["expected_accepted_draft_tokens"] == 2.4
    assert assumed["seed"] == 42
    assert assumed["qualification"] == "functional_only"
    assert report.metadata["native_report"]["speculation"]["aggregated"] == assumed
    assert (
        report.metadata["native_report"]["agentic_qualification"] == "functional_only"
    )
    assert (
        report.metadata["native_report"]["agentic_input_format"] == "agentic_mooncake"
    )


@pytest.mark.parametrize("supported", [False, True])
def test_factory_owns_mtp_capabilities(monkeypatch, supported) -> None:
    native = replace(
        simulation.EngineReplayRunnerFactory().capabilities(),
        supports_mtp_expected_acceptance=supported,
        supports_agentic_speculative_decoding=supported,
    )
    monkeypatch.setattr(
        simulation.EngineReplayRunnerFactory, "capabilities", lambda self: native
    )

    capabilities = simulation.DynamoReplayRunnerFactory().capabilities()

    assert capabilities.supports_mtp_expected_acceptance
    assert capabilities.supports_agentic_speculative_decoding
    assert capabilities.agentic_qualification == "functional_only"


def test_factory_accepts_agentic_mtp() -> None:
    deployment = replace(
        _agg_deployment(),
        agg_engine_args={
            "engine_type": "vllm",
            "num_gpu_blocks": 100,
            "timing_model": _mtp_timing(),
            "speculation": {
                "kind": "mtp",
                "num_speculative_tokens": 3,
                "expected_accepted_tokens": 2.4,
                "seed": 42,
            },
        },
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "tiny.jsonl", "trace_format": "agentic_mooncake"},
        goal={"target": "throughput"},
    )

    simulation.DynamoReplayRunnerFactory().capabilities().require_compatible(spec)


def test_factory_rejects_agentic_ngram() -> None:
    deployment = replace(
        _agg_deployment(),
        agg_engine_args={
            "engine_type": "vllm",
            "num_gpu_blocks": 100,
            "speculation": {
                "kind": "ngram",
                "num_speculative_tokens": 3,
                "acceptance_rates": [1.0, 0.5, 0.2],
            },
        },
    )
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={"trace_path": "tiny.jsonl", "trace_format": "agentic_mooncake"},
        goal={"target": "throughput"},
    )

    with pytest.raises(ValueError, match="ngram|MTP|mtp"):
        simulation.DynamoReplayRunnerFactory().capabilities().require_compatible(spec)


@pytest.mark.parametrize("canonical_mtp", [False, True])
def test_lowering_derives_scheduler_depth_from_canonical_cost(
    canonical_mtp, mtp_capacity_passthrough
) -> None:
    timing = _mtp_timing()
    if canonical_mtp:
        timing["config"]["speculation"] = {
            "kind": "mtp",
            "params": {"num_speculative_tokens": 2},
        }
    else:
        timing["config"]["nextn"] = 2
    lowered = lower_upstream_engine_args(
        {
            "engine_type": "vllm",
            "num_gpu_blocks": 100,
            "timing_model": timing,
            "aic_nextn_accept_rates": "1,0.5",
        }
    )
    assert lowered["ais_nextn"] == 2
    assert lowered["ais_nextn_accept_rates"] == "1,0.5"
    assert lowered["ais_perf_config"] == timing["config"]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("rates", [None, ""])
def test_canonical_mtp_requires_authored_acceptance_for_ordinary_replay(
    direct, rates
) -> None:
    timing = _mtp_timing()
    timing["config"]["speculation"] = {
        "kind": "mtp",
        "params": {"num_speculative_tokens": 2},
    }
    args = {
        "engine_type": "vllm",
        "num_gpu_blocks": 100,
        "ais_nextn_accept_rates": rates,
    }
    args["ais_perf_config" if direct else "timing_model"] = (
        timing["config"] if direct else timing
    )
    spec = ReplaySpec(
        backend_deployment=replace(_agg_deployment(), agg_engine_args=args),
        workload={"isl": 16, "osl": 8, "request_count": 1, "concurrency": 1},
        goal={},
    )
    with pytest.raises(ValueError, match="canonical MTP requires explicit acceptance"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)
    with pytest.raises(ValueError, match="canonical MTP requires explicit acceptance"):
        lower_upstream_engine_args(args)


def test_canonical_mtp_accepts_explicit_zero_mean(mtp_capacity_passthrough) -> None:
    timing = _mtp_timing()
    timing["config"]["speculation"] = {
        "kind": "mtp",
        "params": {"num_speculative_tokens": 2},
    }
    lowered = lower_upstream_engine_args(
        {"timing_model": timing, "num_gpu_blocks": 100, "aic_nextn_accepted": 0}
    )
    assert lowered["ais_nextn"] == 2
    assert list(map(float, lowered["ais_nextn_accept_rates"].split(","))) == [0, 0]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "field,authored,conflicting",
    [
        ("nextn", 2, 3),
        ("nextn_accept_rates", "1,0.5", "1,1"),
        ("mtp_seed", 42, 73),
    ],
)
def test_conflicting_native_speculation_aliases_fail_before_execution(
    nested, field, authored, conflicting
):
    rank = {
        "engine_type": "vllm",
        "ais_" + field: authored,
        "aic_" + field: conflicting,
    }
    spec = ReplaySpec(
        backend_deployment=replace(
            _agg_deployment(), agg_engine_args={"rank": rank} if nested else rank
        ),
        workload={},
        goal={},
    )
    with pytest.raises(ValueError, match=f"ais_{field} conflicts with aic_{field}"):
        simulation.DynamoReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize("canonical_mtp", [False, True])
@pytest.mark.parametrize("depth", [None, 0, 2, 3])
def test_canonical_cost_derives_native_depth_without_losing_identity(
    canonical_mtp, depth, mtp_capacity_passthrough
):
    config = _mtp_timing()["config"]
    if canonical_mtp:
        config["speculation"] = {"kind": "mtp", "params": {"num_speculative_tokens": 2}}
    else:
        config["nextn"] = 2
    args = {
        "engine_type": "vllm",
        "num_gpu_blocks": 100,
        "ais_perf_config": config,
        "ais_nextn": depth,
        "ais_nextn_accept_rates": "1,0.5",
        "ais_mtp_seed": 73,
    }
    if depth == 3:
        with pytest.raises(ValueError, match="speculative depth conflicts"):
            lower_upstream_engine_args(args)
    else:
        lowered = lower_upstream_engine_args(args)
        assert lowered["ais_nextn"] == 2
        assert lowered["ais_perf_config"] == config
        assert lowered["ais_nextn_accept_rates"] == "1,0.5"
        assert lowered["ais_mtp_seed"] == 73
    assert args["ais_nextn"] == depth


def test_ordinary_report_preserves_agentic_completion_evidence():
    evidence = {
        "agentic_play_outcomes": [{"play_id": "one", "status": "completed"}],
        "agentic_lifecycle_digest": "abc123",
        "agentic_lifecycle_event_count": 4,
        "agentic_graph": {"completed_requests": 2},
        "agentic_model_projection": {"target_model": "model"},
        "speculative_acceptance": {"decode_forwards": 5},
    }
    metrics, metadata = simulation.DynamoReplayRunner._normalize_report(
        _report({"completed_requests": 2, **evidence}), ReplayOutputRequirements()
    )
    assert metadata == evidence
    assert metrics["completed_requests"] == 2
