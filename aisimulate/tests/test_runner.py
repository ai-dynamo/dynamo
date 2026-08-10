# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-only implementation of the canonical Sweeper Runner contract."""

import json
import pickle

import pytest

import aisimulate
from aisimulate.runner import (
    EngineReplayRunner,
    EngineReplayRunnerFactory,
    InvalidRunnerError,
)
from aisimulate.sweeper import (
    AdapterReplaySpec,
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RuntimeHookSpec,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class RecordingRuntime:
    def __init__(self):
        self.execution_spec = None
        self.execution_spec_json = None

    def run_replay_json(self, execution_spec_json):
        self.execution_spec_json = execution_spec_json
        self.execution_spec = json.loads(execution_spec_json)
        return json.dumps(
            {
                "duration_ms": 4.0,
                "output_throughput_tok_s": 2000.0,
                "gpu_hours": 0.001,
                "mean_ttft_ms": 2.0,
                "mean_tpot_ms": 1.0,
                "mean_e2e_latency_ms": 4.0,
                "mean_output_token_throughput_per_user": 1000.0,
                "goodput_output_throughput_tok_s": 1500.0,
                "completed_requests": 1,
            }
        )


def _engine_args(*, role="aggregated", timing=None):
    return {
        "worker_type": role,
        "engine_type": "vllm",
        "aic_backend": "vllm",
        "aic_model_path": "test-model",
        "aic_system": "test-system",
        "aic_tp_size": 2,
        "aic_attention_dp_size": 1,
        "block_size": 4,
        "num_gpu_blocks": 16,
        "timing_model": timing
        or {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
    }


def _spec(*, deployment=None, workload=None, concurrency=None, adapters=None):
    return ReplaySpec(
        backend_deployment=deployment
        or BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=_engine_args(),
            num_workers=2,
        ),
        workload=workload
        or {"isl": 8, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
        goal={"target": "throughput"},
        concurrency=concurrency,
        adapters=adapters or {},
    )


def test_public_namespace_exports_engine_runner_contract():
    assert aisimulate.EngineReplayRunner is EngineReplayRunner
    assert aisimulate.EngineReplayRunnerFactory is EngineReplayRunnerFactory


def test_factory_is_pickleable_and_advertises_engine_only_capabilities():
    factory = pickle.loads(pickle.dumps(EngineReplayRunnerFactory()))
    capabilities = factory.capabilities()

    assert capabilities.supports_backend_topology("vllm", "agg")
    assert capabilities.supports_backend_topology("sglang", "disagg")
    assert not capabilities.supports_backend_topology("trtllm", "disagg")
    assert capabilities.supported_hooks == ()


def test_runner_lowers_canonical_spec_and_returns_replay_report():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=7)

    report = runner.run(_spec())

    assert isinstance(report, ReplayReport)
    assert report.metrics["output_throughput_tok_s"] == 2000.0
    assert report.metrics["mean_ttft_ms"] == 2.0
    assert report.metrics["mean_tpot_ms"] == 1.0
    assert report.metrics["mean_e2e_latency_ms"] == 4.0
    assert report.metrics["mean_output_token_throughput_per_user"] == 1000.0
    assert report.metrics["goodput_output_throughput_tok_s"] == 1500.0
    execution = runtime.execution_spec
    assert execution["topology"] == {
        "kind": "aggregated",
        "workers": {"initial_workers": 2, "startup_delay_ms": 0.0},
    }
    assert execution["engine"]["tensor_parallel_size"] == 2
    assert execution["engine"]["rank"]["backend"] == "vllm"
    assert execution["requests"][0]["input_tokens"] == 8
    assert execution["record_per_request"] is False
    assert isinstance(runtime.execution_spec_json, str)
    assert report.metadata == {}


def test_runner_lowers_sglang_with_prefix_caching_disabled():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.update(
        {
            "engine_type": "sglang",
            "aic_backend": "sglang",
            "block_size": 1,
            "enable_prefix_caching": False,
        }
    )
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="sglang",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    assert runtime.execution_spec["engine"]["rank"]["enable_prefix_caching"] is False


def test_runner_captures_requested_raw_and_per_request_report():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=7)

    report = runner.run(
        _spec(),
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )

    assert runtime.execution_spec["record_per_request"] is True
    assert report.metadata["native_report"]["completed_requests"] == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "must be a JSON string"),
        ("not-json", "returned invalid report JSON"),
        ("[]", "must be a JSON object"),
    ],
)
def test_runner_rejects_invalid_runtime_json_boundary_results(payload, message):
    class InvalidRuntime:
        def run_replay_json(self, execution_spec_json):
            assert isinstance(execution_spec_json, str)
            return payload

    with pytest.raises(InvalidRunnerError, match=message):
        EngineReplayRunnerFactory(runtime=InvalidRuntime()).create(0).run(_spec())


def test_runner_preserves_closed_loop_concurrency_in_execution_spec():
    runtime = RecordingRuntime()
    spec = _spec(
        workload={"isl": 4, "osl": 1, "concurrency": 3, "num_request_ratio": 2},
        concurrency=3,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    assert runtime.execution_spec["max_in_flight"] == 3
    assert len(runtime.execution_spec["requests"]) == 6
    assert {
        request["arrival_time_ms"] for request in runtime.execution_spec["requests"]
    } == {0.0}


def test_runner_lowers_disaggregated_grouped_engines():
    runtime = RecordingRuntime()
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="vllm",
        backend_version="test",
        prefill_engine_args=_engine_args(role="prefill"),
        decode_engine_args=_engine_args(role="decode"),
        num_prefill_workers=2,
        num_decode_workers=3,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    assert runtime.execution_spec["topology"]["kind"] == "disaggregated"
    assert runtime.execution_spec["topology"]["prefill"]["initial_workers"] == 2
    assert runtime.execution_spec["topology"]["decode"]["initial_workers"] == 3
    assert set(runtime.execution_spec["engine"]) == {"prefill", "decode"}


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_runner_rejects_disaggregated_attention_dp_before_runtime(role):
    runtime = RecordingRuntime()
    prefill_args = _engine_args(role="prefill")
    decode_args = _engine_args(role="decode")
    selected = prefill_args if role == "prefill" else decode_args
    selected["aic_attention_dp_size"] = 2
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="vllm",
        backend_version="test",
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=1,
        num_decode_workers=1,
    )

    with pytest.raises(ValueError, match=rf"{role} dp_size=1"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            _spec(deployment=deployment)
        )
    assert runtime.execution_spec is None


def test_runner_threads_canonical_backend_version_into_aic_timing():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        parallel_config={"tp": 2, "attention_dp": 1, "replicas": 2},
        agg_engine_args=engine_args,
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    timing = runtime.execution_spec["engine"]["rank"]["timing_model"]
    assert timing["config"]["backend_version"] == "0.11.1"


def test_runner_accepts_matching_backend_version_in_explicit_aic_timing():
    runtime = RecordingRuntime()
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 2,
            "attention_dp": 1,
            "backend_version": "0.11.1",
        },
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        agg_engine_args=_engine_args(timing=timing),
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    timing_config = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]
    assert timing_config["backend_version"] == "0.11.1"


def test_runner_rejects_conflicting_backend_version_in_explicit_aic_timing():
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 2,
            "attention_dp": 1,
            "backend_version": "0.10.0",
        },
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        agg_engine_args=_engine_args(timing=timing),
        num_workers=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"timing_model\.config\.backend_version='0\.10\.0' conflicts with "
            r"BackendDeploymentSpec backend_version='0\.11\.1'"
        ),
    ):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


def test_runner_rejects_parallel_config_that_conflicts_with_engine_args():
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        parallel_config={"tp": 4, "replicas": 2},
        agg_engine_args=_engine_args(),
        num_workers=2,
    )

    with pytest.raises(ValueError, match="parallel_config.tp=4 conflicts"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("turns_per_session", 2),
        ("shared_prefix_ratio", 0.5),
        ("num_prefix_groups", 2),
        ("inter_turn_delay_ms", 10.0),
    ],
)
def test_engine_runner_fails_closed_for_unimplemented_synthetic_shapes(field, value):
    workload = {
        "isl": 8,
        "osl": 2,
        "concurrency": 1,
        "num_request_ratio": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(workload=workload)
        )


def test_engine_runner_does_not_silently_parse_a_dynamo_trace_as_mooncake():
    with pytest.raises(ValueError, match="supports only format='mooncake'"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "trace_path": "unused.jsonl",
                    "trace_format": "dynamo",
                }
            )
        )


def test_engine_runner_rejects_dynamo_runtime_hooks():
    hook = RuntimeHookSpec(
        provider="dynamo.router",
        kind="placement_policy",
        api_version=1,
        config={"router_mode": "kv_router", "router_config": {}},
    )
    spec = _spec(
        adapters={
            "dynamo.router": AdapterReplaySpec(runtime_hooks=(hook,)),
        }
    )

    with pytest.raises(ValueError, match="does not support runtime hook"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(spec)


def test_runner_rejects_nested_backend_that_conflicts_with_deployment():
    engine_args = _engine_args()
    engine_args["rank"] = {
        "backend": "sglang",
        "block_size": 1,
        "num_gpu_blocks": 16,
        "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
    }
    for field in (
        "block_size",
        "num_gpu_blocks",
        "timing_model",
    ):
        engine_args.pop(field)

    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    with pytest.raises(ValueError, match="rank backend conflicts"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )
