# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    StageContract,
    Workflow,
    WorkflowValidationError,
    compile_workflow,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def _workflow() -> Workflow:
    workflow = Workflow("portable-plan")
    text = workflow.input("text", type="text")
    contract = {
        "id": "normalize",
        "inputs": {"text": {"type": "text"}},
        "outputs": {"normalized": {"type": "text"}},
    }
    stage = workflow.stage(
        "normalize",
        StageContract.from_dict(contract),
        text=text,
    )
    workflow.output("text", stage.normalized)
    return workflow


def test_execution_plan_round_trip_contains_only_portable_bindings() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.local(normalize="normalizer"))

    restored = ExecutionPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.to_json() == plan.to_json()
    assert restored.to_dict()["bindings"] == {
        "normalize": {"kind": "local", "runner_key": "normalizer"}
    }
    assert restored.to_dict()["edges"] == [
        {
            "source": {"input": "text"},
            "target": {"stage": "normalize", "port": "text"},
            "carrier": "local",
        }
    ]


def test_execution_plan_rejects_invalid_physical_edges() -> None:
    plan = compile_workflow(
        _workflow(), DeploymentSpec.local(normalize="normalizer")
    ).to_dict()
    plan["edges"][0]["source"] = {"input": "other"}

    with pytest.raises(WorkflowValidationError, match="does not match WorkflowIR"):
        ExecutionPlan.from_dict(plan)


def test_execution_plan_json_rejects_duplicate_keys() -> None:
    plan = compile_workflow(
        _workflow(), DeploymentSpec.local(normalize="normalizer")
    ).to_dict()
    serialized = json.dumps(plan).replace(
        '"version": 0', '"version": 0, "version": 0', 1
    )

    with pytest.raises(WorkflowValidationError, match="duplicate JSON key"):
        ExecutionPlan.from_json(serialized)
