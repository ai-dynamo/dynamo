# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler DGD materialization boundaries."""

from __future__ import annotations

import copy

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

try:
    from dynamo.profiler.utils import dgd_materialization
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


def test_materialize_dgd_applies_transforms_once_in_fixed_order(monkeypatch) -> None:
    blueprint = {
        "spec": {
            "services": {
                "Worker": {
                    "extraPodSpec": {
                        "mainContainer": {"args": ["base"]},
                    }
                }
            }
        }
    }
    original = copy.deepcopy(blueprint)
    events: list[str] = []

    def _append_step(config: dict, step: str) -> dict:
        result = copy.deepcopy(config)
        result["spec"]["services"]["Worker"]["extraPodSpec"]["mainContainer"][
            "args"
        ].append(step)
        events.append(step)
        return result

    monkeypatch.setattr(
        dgd_materialization,
        "apply_dgd_overrides",
        lambda config, _override: _append_step(config, "override"),
    )

    class _Modifier:
        @staticmethod
        def apply_model_runtime_constraints(config, _model):
            return _append_step(config, "runtime")

    monkeypatch.setattr(
        dgd_materialization,
        "CONFIG_MODIFIERS",
        {"test-backend": _Modifier},
    )
    monkeypatch.setattr(
        dgd_materialization,
        "inject_tolerations_into_dgd",
        lambda config, _tolerations: _append_step(config, "tolerations"),
    )

    materialized = materialize_dgd(
        blueprint,
        purpose=DGDMaterializationPurpose.BENCHMARK_CANDIDATE,
        override={"spec": {}},
        tolerations=[{"key": "gpu"}],
        runtime_backend="test-backend",
        model_name_or_path="test/model",
    )

    assert events == ["override", "runtime", "tolerations"]
    assert materialized["spec"]["services"]["Worker"]["extraPodSpec"]["mainContainer"][
        "args"
    ] == ["base", "override", "runtime", "tolerations"]
    assert blueprint == original


def test_materialize_dgd_copies_blueprint_without_transforms() -> None:
    blueprint = {"spec": {"services": {"Worker": {"replicas": 1}}}}

    materialized = materialize_dgd(
        blueprint,
        purpose=DGDMaterializationPurpose.INTERPOLATION,
    )

    assert materialized == blueprint
    assert materialized is not blueprint
    assert materialized["spec"] is not blueprint["spec"]


def test_materialize_dgd_only_changes_last_document(monkeypatch) -> None:
    config_map = {"apiVersion": "v1", "kind": "ConfigMap", "data": {"key": "value"}}
    dgd = {"apiVersion": "nvidia.com/v1alpha1", "kind": "DynamoGraphDeployment"}
    final_config = [config_map, dgd]

    def _apply_override(config: dict, _override: dict) -> dict:
        result = copy.deepcopy(config)
        result["metadata"] = {"name": "materialized"}
        return result

    monkeypatch.setattr(
        dgd_materialization,
        "apply_dgd_overrides",
        _apply_override,
    )

    materialized = materialize_dgd(
        final_config,
        purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
        override={"metadata": {"name": "materialized"}},
    )

    assert materialized == [
        config_map,
        {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "materialized"},
        },
    ]
    assert materialized is not final_config
    assert materialized[0] is not config_map
    assert final_config == [config_map, dgd]


def test_materialize_dgd_rejects_non_object_dgd() -> None:
    with pytest.raises(TypeError, match="final output DGD blueprint must be an object"):
        materialize_dgd(
            [{"kind": "ConfigMap"}, "not-an-object"],
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
        )
