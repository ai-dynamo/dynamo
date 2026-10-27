# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline dataset-to-Planner checks for measured timings and input equivalence."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aisimulate_core import _native
from aisimulate_core.datasets import Identity, configuration_id
from aisimulate_core.sdk import engine

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import EngineCapabilities

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


@pytest.fixture(params=["prefill", "decode"])
def dataset_profile(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    phase = request.param
    root = tmp_path / "dataset"
    root.mkdir()
    systems = tmp_path / "systems"
    systems.mkdir()
    hardware = Path(engine.__file__).parents[1] / "systems/h200_sxm.yaml"
    shutil.copyfile(hardware, systems / hardware.name)
    table = systems / "data/h200_sxm/gemm/vllm/1.0/fixture.parquet"
    table.parent.mkdir(parents=True)
    pq.write_table(pa.table({"latency": [2.0]}), table)
    basis = json.loads(
        _native.engine_spec_json_from_bincode(
            engine.compile_engine(
                "Qwen/Qwen3-0.6B",
                "h200_sxm",
                "vllm",
                "1.0",
                systems_path=str(systems),
                database_mode="SILICON",
                shared_layer=False,
                forward_model="op_level",
            )
        )
    )
    artifacts = []

    def retain(name, contents, role, runtime_path=None):
        (root / name).write_bytes(contents)
        artifact = dict(
            artifact_id=name,
            path=name,
            role=role,
            format="json",
            schema="test/v1",
            sha256=hashlib.sha256(contents).hexdigest(),
            bytes=len(contents),
            runtime_path=runtime_path,
        )
        artifacts.append(artifact)
        return artifact

    retain("hardware.yaml", hardware.read_bytes(), "hardware", hardware.name)
    retain(
        "table.parquet",
        table.read_bytes(),
        "operation-measurements",
        str(table.relative_to(systems)),
    )
    ops_key = "context_ops" if phase == "prefill" else "generation_ops"
    # Scopes cost 2 ms (4 ms for the longer prompt), each repeated three times.
    profile = dict(
        schema_version=2,
        basis=basis,
        worker_type=phase,
        ops=[
            {
                "MeasuredComposite": dict(
                    name="synthetic_scope",
                    system="h200_sxm",
                    backend="vllm",
                    version="1.0",
                    is_context=phase == "prefill",
                    weight_bytes=sum(
                        _native.weights_ops_json(json.dumps(basis[ops_key]))
                    ),
                    scale_factor=3.0,
                    cells=[
                        {"shape": [batch, length, 0], "latency_ms": latency}
                        for batch, length, latency in (
                            [(1, 4, 2.0), (1, 8, 4.0), (2, 4, 2.0)]
                            if phase == "prefill"
                            else [(1, 4, 2.0), (2, 4, 2.0)]
                        )
                    ],
                )
            }
        ],
        retained_files=[
            dict(path=a["runtime_path"], sha256=a["sha256"], size_bytes=a["bytes"])
            for a in artifacts
        ],
    )
    pin = retain("profile.json", json.dumps(profile).encode(), "execution-profile")[
        "sha256"
    ]
    config = dict(
        model="Qwen/Qwen3-0.6B",
        system="h200_sxm",
        backend="vllm",
        backend_version="1.0",
        worker_type=phase,
        estimation_mode="op_level",
        fallback_policy="deny",
        database_mode="SILICON",
        enable_shared_layer=False,
        estimator_config={
            "correction": {"enabled": False},
            "op_level": {"execution_profile": {"path": "profile.json", "sha256": pin}},
        },
    )
    retain("config.json", json.dumps(config).encode(), "predictor-config")
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model=config["model"],
        effective_settings={"worker_type": phase},
        unresolved_identity=None,
    )
    manifest = dict(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="execution-profile",
        configuration_id=configuration_id("op-latency", "execution-profile", identity),
        identity=identity.model_dump(),
        requested_settings=None,
        artifacts=artifacts,
        provenance={"source": "synthetic test, not silicon"},
        evidence_gaps=[],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(systems)
    planner = PlannerConfig(
        namespace="offline-profiles",
        mode=phase,
        backend="vllm",
        pre_deployment_sweeping_mode="none",
        ais_perf_model={
            "dataset_profiles": {
                phase: {"root": str(root), "manifest": "manifest.json"}
            }
        },
    )
    return planner, root, phase


def _model(config, phase):
    return PlannerEnginePerfModel(
        worker_type=phase,
        config=config,
        capabilities=EngineCapabilities(
            max_num_batched_tokens=8,
            max_num_seqs=2,
            max_kv_tokens=100,
        ),
    )


def _query(model, phase, *, tokens=8, **options):
    if phase == "prefill":
        return model.estimate_queued_prefill_time(
            [
                ForwardPassMetrics(
                    queued_requests=QueuedRequestMetrics(
                        num_prefill_requests=2,
                        sum_prefill_tokens=tokens,
                    )
                )
            ],
            max_num_batched_tokens=8,
            add_next_request=False,
            **options,
        )
    return model.estimate_scheduled_decode_itl(
        [
            ForwardPassMetrics(
                scheduled_requests=ScheduledRequestMetrics(
                    num_decode_requests=2,
                    sum_decode_kv_tokens=tokens,
                )
            )
        ],
        add_next_request=False,
        **options,
    )


def test_dataset_predicts_after_source_removal_and_capability_rebuild(dataset_profile):
    config, root, phase = dataset_profile
    restored = PlannerConfig.model_validate_json(config.model_dump_json())
    model = _model(restored, phase)
    assert model.has_sufficient_data()
    assert _query(model, phase) == pytest.approx(0.006)
    prepared = model._profile_directory.name
    shutil.rmtree(root)
    model.update_capabilities(replace(model._capabilities, max_kv_tokens=200))
    assert model._profile_directory.name == prepared
    assert _query(model, phase) == pytest.approx(0.006)
    # Capability limits and dependency defaults do not change the profile pin.
    assert (
        model._build_ais_config()["estimator_config"]["correction"]["enabled"] is False
    )


def test_profile_returns_unavailable_for_missing_shapes(dataset_profile):
    config, _, phase = dataset_profile
    model = _model(config, phase)
    assert _query(model, phase, tokens=7) is None
    assert _query(model, phase, tokens=12) is None


def test_profile_uses_existing_queue_and_decode_scaling(dataset_profile):
    config, _, phase = dataset_profile
    model = _model(config, phase)
    options = {"queue_scale": 0.5} if phase == "prefill" else {"decode_scale": 0.5}
    # Scaling two four-token requests to one still matches a measured cell.
    assert _query(model, phase, **options) == pytest.approx(0.006)
    if phase == "prefill":
        assert _query(model, phase, tokens=16, kv_hit_rate=0.5) == pytest.approx(0.006)


def test_profile_capacity_uses_existing_planner_workload(dataset_profile):
    config, _, phase = dataset_profile
    model = _model(config, phase)
    # Context at the midpoint of a one-token decode is exactly four past tokens.
    capacity = model.find_engine_capacity_rps(isl=4, osl=1)
    assert capacity is not None
    # Existing prefill capacity planning represents each candidate as one
    # prompt containing all tokens: 1x4 costs 6 ms; 1x8 costs 12 ms.
    expected = 1 / 0.006 if phase == "prefill" else 2 / 0.006
    assert capacity.rps == pytest.approx(expected)
    cached_isl = 8 if phase == "prefill" else 4
    cached = model.find_engine_capacity_rps(isl=cached_isl, osl=1, kv_hit_rate=0.5)
    assert cached is not None
    assert cached.rps == pytest.approx(capacity.rps)


def test_dataset_role_mismatch_and_tampering_fail(dataset_profile):
    config, root, phase = dataset_profile
    wrong = "decode" if phase == "prefill" else "prefill"
    source = config.ais_perf_model.dataset_profiles.pop(phase)
    config.ais_perf_model.dataset_profiles[wrong] = source
    with pytest.raises(ValueError, match="worker_type"):
        _model(config, wrong)
    config.ais_perf_model.dataset_profiles = {phase: source}
    (root / "profile.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        _model(config, phase)


def test_profile_estimates_next_request_with_existing_planner_logic(dataset_profile):
    config, _, phase = dataset_profile
    model = _model(config, phase)
    observed = ForwardPassMetrics(
        worker_id="worker",
        wall_time=0.006,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1 if phase == "prefill" else 0,
            sum_prefill_tokens=4 if phase == "prefill" else 0,
            num_decode_requests=1 if phase == "decode" else 0,
            sum_decode_kv_tokens=4 if phase == "decode" else 0,
        ),
    )
    model.add_observations({("worker", 0): observed})
    assert _query(model, phase) == pytest.approx(0.006)
    if phase == "prefill":
        estimate = model.estimate_queued_prefill_time(
            [observed], max_num_batched_tokens=8, add_next_request=True
        )
    else:
        estimate = model.estimate_scheduled_decode_itl(
            [observed], add_next_request=True
        )
    assert estimate == pytest.approx(0.006)


@pytest.mark.parametrize("role", ["prefill", "decode", "aggregated"])
def test_dataset_preserves_builtin_planner_estimates(tmp_path, monkeypatch, role):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    bundled = Path(engine.__file__).parents[1] / "systems"
    root = tmp_path / "dataset"
    systems = root / "systems"
    artifacts = []

    def retain(path, contents, kind, runtime_path=None):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        artifact = dict(
            artifact_id=path,
            path=path,
            role=kind,
            format=target.suffix.lstrip("."),
            schema="test/v1",
            sha256=hashlib.sha256(contents).hexdigest(),
            bytes=len(contents),
            rows=pq.ParquetFile(target).metadata.num_rows
            if target.suffix == ".parquet"
            else None,
            runtime_path=runtime_path,
        )
        artifacts.append(artifact)
        return artifact

    # Retain the published tables unchanged and compile their actual op graphs.
    files = [
        bundled / "b200_sxm.yaml",
        *sorted((bundled / "data/b200_sxm").glob("*/vllm/0.24.0/*")),
    ]
    for source in files:
        relative = str(source.relative_to(bundled))
        retain(
            "systems/" + relative,
            source.read_bytes(),
            "hardware" if source.suffix == ".yaml" else "operation-measurements",
            relative,
        )
    basis = json.loads(
        _native.engine_spec_json_from_bincode(
            engine.compile_engine(
                "Qwen/Qwen3-0.6B",
                "b200_sxm",
                "vllm",
                "0.24.0",
                systems_path=str(systems),
                database_mode="SILICON",
                shared_layer=False,
                forward_model="op_level",
            )
        )
    )
    profile = dict(
        schema_version=2,
        basis=basis,
        worker_type=role,
        ops=basis["generation_ops" if role == "decode" else "context_ops"],
        retained_files=[
            dict(path=a["runtime_path"], sha256=a["sha256"], size_bytes=a["bytes"])
            for a in artifacts
        ],
    )
    if role == "aggregated":
        profile["generation_ops"] = basis["generation_ops"]
    pin = retain("profile.json", json.dumps(profile).encode(), "execution-profile")[
        "sha256"
    ]
    config = dict(
        model="Qwen/Qwen3-0.6B",
        system="b200_sxm",
        backend="vllm",
        backend_version="0.24.0",
        worker_type=role,
        estimation_mode="op_level",
        fallback_policy="deny",
        database_mode="SILICON",
        enable_shared_layer=False,
        estimator_config={"correction": {"enabled": False}},
    )
    external = config | {
        "estimator_config": config["estimator_config"]
        | {"op_level": {"execution_profile": {"path": "profile.json", "sha256": pin}}}
    }
    retain("config.json", json.dumps(external).encode(), "predictor-config")
    identity = Identity(
        hardware=config["system"],
        framework=config["backend"],
        framework_version=config["backend_version"],
        operation_family=None,
        model=config["model"],
        effective_settings={"worker_type": role},
        unresolved_identity=None,
    )
    manifest = dict(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="execution-profile",
        configuration_id=configuration_id("op-latency", "execution-profile", identity),
        identity=identity.model_dump(),
        requested_settings=None,
        artifacts=artifacts,
        provenance={"source": "bundled operation tables"},
        evidence_gaps=[],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    models = [
        PlannerEnginePerfModel(
            worker_type=role,
            config=PlannerConfig(
                namespace="input-equivalence",
                mode="agg" if role == "aggregated" else role,
                backend="vllm",
                pre_deployment_sweeping_mode="none",
                ais_perf_model=spec,
            ),
            capabilities=EngineCapabilities(
                max_num_batched_tokens=128, max_num_seqs=4, max_kv_tokens=4096
            ),
        )
        for spec in [
            {"roles": {role: config}},
            {
                "dataset_profiles": {
                    role: {"root": str(root), "manifest": "manifest.json"}
                }
            },
        ]
    ]
    assert all(model.has_sufficient_data() for model in models)
    # Preparation owns its snapshot, including across an aggregated model rebuild.
    shutil.rmtree(root)
    for model in models:
        model.update_capabilities(replace(model._capabilities, max_kv_tokens=8192))
    metrics = [
        ForwardPassMetrics(
            queued_requests=QueuedRequestMetrics(
                num_prefill_requests=2, sum_prefill_tokens=256
            ),
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=2 if role == "aggregated" else 0,
                sum_prefill_tokens=128 if role == "aggregated" else 0,
                num_decode_requests=2 if role != "prefill" else 0,
                sum_decode_kv_tokens=256 if role != "prefill" else 0,
            ),
        )
    ]
    if role != "decode":
        estimates = [
            model.estimate_queued_prefill_time(
                metrics, max_num_batched_tokens=128, add_next_request=False
            )
            for model in models
        ]
        assert estimates[0] is not None and estimates[0] > 0
        assert estimates[0] == estimates[1]
    if role != "prefill":
        estimates = [
            model.estimate_scheduled_decode_itl(metrics, add_next_request=False)
            for model in models
        ]
        assert estimates[0] is not None and estimates[0] > 0
        assert estimates[0] == estimates[1]
    capacities = [model.find_engine_capacity_rps(isl=64, osl=2) for model in models]
    assert capacities[0] is not None and capacities[0].rps > 0
    assert capacities[0] == capacities[1]
