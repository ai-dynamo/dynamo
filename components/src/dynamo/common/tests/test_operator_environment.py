# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same contract cases are rendered by the operator's Go test suite."""

import json
from pathlib import Path

import pytest

from dynamo.common.operator_environment import component_backend, owned_environment

pytestmark = [pytest.mark.gpu_0, pytest.mark.unit, pytest.mark.pre_merge]


def operator_environment_cases():
    fixture = (
        Path(__file__).resolve().parents[5]
        / "deploy/operator/internal/dynamo/testdata/operator_environment.json"
    )
    with fixture.open() as stream:
        return json.load(stream)


@pytest.mark.parametrize(
    "case", operator_environment_cases(), ids=lambda case: case["name"]
)
def test_operator_environment_contract(case):
    main = {
        "name": "main",
        "image": "engine:1.8.0",
        "command": ["python3", "-m", "dynamo.vllm"],
        **case.get("main", {}),
    }
    component = {"name": "worker", "type": "worker", **case.get("component", {})}
    if not component.get("roles"):
        pod = component.setdefault("podTemplate", {}).setdefault("spec", {})
        pod["containers"] = [main]
    dgd = {"spec": case.get("graph", {})}
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            component_backend(dgd, component)
        return

    assert component_backend(dgd, component) == case["backend"]
    if component.get("roles"):
        role = next(role for role in component["roles"] if role["name"] == case["role"])
        component["podTemplate"] = role["podTemplate"]
    bindings, omitted = owned_environment(dgd, component)
    assert bindings == case["bindings"]
    assert omitted == set(case.get("omitted", []))
    assert not (set(case.get("preserved", [])) & (bindings.keys() | omitted))
