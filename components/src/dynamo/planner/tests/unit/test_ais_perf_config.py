# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical AIS configuration survives Planner input, runtime binding and reload."""

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamo.planner.config.planner_config import AISPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import EngineCapabilities

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _role_config(**overrides):
    return {
        "model": "test-model",
        "system": "test-system",
        "backend": "vllm",
        **overrides,
    }


def _planner(role_config, **overrides):
    return PlannerConfig(
        namespace="test-ais-config",
        mode="decode",
        optimization_target="sla",
        ais_perf_model={"roles": {"decode": role_config}},
        **overrides,
    )


def _limits():
    return EngineCapabilities(
        max_num_batched_tokens=4096, max_num_seqs=128, max_kv_tokens=100000
    )


def test_canonical_defaults_and_complete_controls_roundtrip(tmp_path):
    roots = [str(tmp_path / "second"), str(tmp_path / "first")]
    for root in roots:
        Path(root).mkdir()
    controls = {
        "features": {"attention_kv_weight": 0.75},
        "fpm_regression": {
            "sampling": {"bins_per_axis": [2, 8], "max_observations": 20},
            "fit": {"singular_ridge_scale": 0.0001},
        },
        "correction": {
            "sampling": {"bins_per_axis": [5, 3]},
            "factor_bounds": {"min": 0.7, "max": None},
        },
    }
    config = _planner(_role_config(systems_paths=roots, estimator_config=controls))
    role = config.ais_perf_model.roles["decode"]
    assert role["estimation_mode"] == "auto"
    assert role["fallback_policy"] == "deny"
    assert role["worker_type"] == "decode"
    assert role["estimator_config"] == controls
    assert role["systems_paths"] == roots
    serialized = config.model_dump(mode="json")
    assert "aic_perf_model" not in serialized
    assert (
        PlannerConfig.model_validate(serialized).ais_perf_model == config.ais_perf_model
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"worker_type": "prefill"},
        {"typo_config": 1},
        {"estimator_config": {"regression_typo": {}}},
        {"estimator_config": {"fpm_regression": {"min_observations": 0}}},
    ],
)
def test_invalid_canonical_fields_rejected_at_config_boundary(changes):
    with pytest.raises(ValidationError):
        _planner(_role_config(**changes))


@pytest.mark.parametrize("canonical", [False, True])
def test_retired_perf_model_field_is_rejected(canonical):
    values = {"aic_perf_model": None}
    if canonical:
        values["ais_perf_model"] = {"roles": {"decode": _role_config()}}
    with pytest.raises(ValidationError, match="aic_perf_model is no longer supported"):
        PlannerConfig(mode="decode", **values)


def test_runtime_options_preserve_explicit_controls_and_cache_identity():
    config = _planner(
        _role_config(
            estimation_mode="fpm_regression",
            estimator_config={
                "fpm_regression": {"sampling": {"bins_per_axis": [2, 8]}},
                "correction": {"factor_bounds": {"min": 0.75}},
            },
        ),
        max_num_fpm_samples=16,
        fpm_sample_bucket_size=16,
    )
    original = deepcopy(config.ais_perf_model)
    model = PlannerEnginePerfModel(
        worker_type="decode", config=config, capabilities=_limits()
    )
    payload = model._build_ais_config()
    assert payload["estimator_config"]["fpm_regression"]["sampling"][
        "bins_per_axis"
    ] == [2, 8]
    assert payload["estimator_config"]["correction"]["sampling"]["bins_per_axis"] == [
        4,
        4,
    ]
    assert payload["estimator_config"]["correction"]["max_num_tokens"] == 4096
    assert config.ais_perf_model == original
    first_key = model._model_key()
    config.ais_perf_model.roles["decode"]["systems_paths"] = ["/different"]
    assert model._model_key() != first_key
    config.ais_perf_model = original
    config.ais_perf_model.roles["decode"]["estimator_config"]["features"] = {
        "ffn_token_weight": 0.5
    }
    assert model._model_key() != first_key


@pytest.mark.parametrize("role", ["prefill", "decode", "aggregated"])
def test_cold_regression_has_traceable_role_identity(role):
    config = PlannerConfig.model_construct(
        namespace="worker-group-a", model_name="actual-model", backend="vllm"
    )
    model = PlannerEnginePerfModel(
        worker_type=role, config=config, capabilities=_limits()
    )
    payload = model._build_ais_config()
    assert payload["model"] == "actual-model"
    assert payload["worker_type"] == role
    assert payload["estimation_mode"] == "fpm_regression"
    assert payload["fallback_policy"] == "deny"
    assert model._engine_diagnostics()["readiness"] == "insufficient_data"


def test_worker_capabilities_cannot_silently_override_canonical_identity():
    config = _planner(_role_config(estimation_mode="fpm_regression", kv_block_size=16))
    limits = _limits()
    limits.kv_cache_block_size = 32
    with pytest.raises(ValueError, match="kv_block_size conflicts"):
        PlannerEnginePerfModel(worker_type="decode", config=config, capabilities=limits)


def test_spec_rejects_unknown_role():
    with pytest.raises(ValidationError):
        AISPerfModelSpec(roles={"agg": _role_config()})


def test_legacy_identity_shape_is_rejected():
    with pytest.raises(ValidationError, match="hf_id"):
        AISPerfModelSpec.model_validate(
            {
                "hf_id": "test",
                "system": "h200_sxm",
                "backend": "vllm",
                "decode_pick": {"tp": 2, "dp": 2},
            }
        )
