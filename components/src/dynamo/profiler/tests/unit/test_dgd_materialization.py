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


def _component(name: str, component_type: str) -> dict:
    return {
        "name": name,
        "type": component_type,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "args": ["--model", "Qwen/Qwen3-0.6B"],
                    }
                ]
            }
        },
    }


@pytest.mark.parametrize(
    ("backend", "topology", "workers"),
    [
        ("vllm", "aggregate", [("worker", "worker")]),
        ("vllm", "disaggregated", [("decode", "decode"), ("prefill", "prefill")]),
        ("sglang", "aggregate", [("decode", "worker")]),
        (
            "sglang",
            "disaggregated",
            [("decode", "decode"), ("prefill", "prefill")],
        ),
    ],
)
def test_explicit_trust_remote_code_targets_selected_worker_roles(
    backend: str,
    topology: str,
    workers: list[tuple[str, str]],
) -> None:
    blueprint = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "components": [
                _component("Frontend", "frontend"),
                *(_component(name, component_type) for name, component_type in workers),
            ]
        },
    }

    materialized = materialize_dgd(
        blueprint,
        purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
        runtime_backend=backend,
        trust_remote_code=True,
    )

    components = {
        component["name"]: component for component in materialized["spec"]["components"]
    }
    for name, _ in workers:
        args = components[name]["podTemplate"]["spec"]["containers"][0]["args"]
        assert args.count("--trust-remote-code") == 1, topology
    frontend_args = components["Frontend"]["podTemplate"]["spec"]["containers"][0][
        "args"
    ]
    assert "--trust-remote-code" not in frontend_args


def test_materialize_dgd_applies_transforms_once_in_fixed_order(monkeypatch) -> None:
    blueprint = {
        "spec": {
            "components": [
                {
                    "name": "Worker",
                    "type": "worker",
                    "podTemplate": {
                        "spec": {"containers": [{"name": "main", "args": ["base"]}]}
                    },
                }
            ]
        }
    }
    original = copy.deepcopy(blueprint)
    events: list[str] = []

    def _append_step(config: dict, step: str) -> dict:
        result = copy.deepcopy(config)
        result["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0][
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
    assert materialized["spec"]["components"][0]["podTemplate"]["spec"]["containers"][
        0
    ]["args"] == ["base", "override", "runtime", "tolerations"]
    assert blueprint == original


def test_materialize_dgd_copies_blueprint_without_transforms() -> None:
    blueprint = {
        "spec": {"components": [{"name": "Worker", "type": "worker", "replicas": 1}]}
    }

    materialized = materialize_dgd(
        blueprint,
        purpose=DGDMaterializationPurpose.INTERPOLATION,
    )

    assert materialized == blueprint
    assert materialized is not blueprint
    assert materialized["spec"] is not blueprint["spec"]


def test_materialize_remote_model_preserves_explicit_frontend_cli(
    monkeypatch,
) -> None:
    from dynamo.profiler.utils.dgd_template import load_dgd_template

    blueprint = load_dgd_template("vllm", "agg")
    monkeypatch.setattr(dgd_materialization, "model_has_auto_map", lambda _model: False)

    materialized = materialize_dgd(
        blueprint,
        purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
        runtime_backend="vllm",
        model_name_or_path="Qwen/Qwen3-0.6B",
    )

    frontend = next(
        component
        for component in materialized["spec"]["components"]
        if component["name"] == "Frontend"
    )
    main = next(
        container
        for container in frontend["podTemplate"]["spec"]["containers"]
        if container["name"] == "main"
    )
    assert main["command"] == ["python3"]
    assert main["args"] == ["-m", "dynamo.frontend"]


def test_materialize_dgd_only_changes_last_document(monkeypatch) -> None:
    config_map = {"apiVersion": "v1", "kind": "ConfigMap", "data": {"key": "value"}}
    dgd = {"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment"}
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
            "apiVersion": "nvidia.com/v1beta1",
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


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (["python3", "-m", "dynamo.mocker"], []),
        (["sh", "-c"], ["python3 -m dynamo.mocker --model-path test/model"]),
    ],
)
def test_mocker_detection_matches_discrete_command_tokens(command, args) -> None:
    assert dgd_materialization._invokes_mocker(command, args)


def test_mocker_detection_rejects_substring_matches() -> None:
    args = ["-m", "dynamo.worker", "--model", "org/dynamo.mocker-model"]
    assert not dgd_materialization._invokes_mocker(["python3"], args)
