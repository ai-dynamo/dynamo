# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline dataset-to-Planner checks with synthetic, hand-derived timings."""

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
    # 2 ms per synthetic scope, repeated three times: exactly 6 ms.
    profile = dict(
        schema_version=1,
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
                        {
                            "shape": [batch, 4 if phase == "prefill" else 5, 0],
                            "latency_ms": 2.0,
                        }
                        for batch in (1, 2)
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


def test_profile_capacity_preserves_batch_geometry(dataset_profile):
    config, _, phase = dataset_profile
    model = _model(config, phase)
    # Context at the midpoint of a one-token decode is exactly four past tokens.
    capacity = model.find_engine_capacity_rps(isl=4, osl=1)
    assert capacity is not None
    assert capacity.rps == pytest.approx(2 / 0.006)
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
