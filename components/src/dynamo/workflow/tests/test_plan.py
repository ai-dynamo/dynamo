# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    EdgePlan,
    ExecutionPlan,
    LocalBinding,
    StageContract,
    ValueRef,
    ValueSpec,
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
    workflow = Workflow("physical-plan")
    text = workflow.input("text", ValueSpec(type="text"))
    stage = workflow.stage(
        "normalize",
        StageContract(
            id="normalize",
            inputs={"text": ValueSpec(type="text")},
            outputs={"normalized": ValueSpec(type="text")},
        ),
        text=text,
    )
    workflow.output("text", stage.normalized)
    return workflow


def test_execution_plan_contains_only_in_memory_decisions() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.local(normalize="normalizer"))

    assert plan.bindings == {"normalize": LocalBinding(runner_key="normalizer")}
    assert plan.edges == (
        EdgePlan(
            source=ValueRef.for_input("text"),
            target_stage="normalize",
            target_port="text",
            carrier="local",
        ),
    )


def test_compilation_defaults_to_stage_id_local_bindings() -> None:
    plan = compile_workflow(_workflow())

    assert plan.bindings == {"normalize": LocalBinding(runner_key="normalize")}
    assert all(edge.carrier == "local" for edge in plan.edges)


def test_execution_plan_rejects_invalid_physical_edges() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.local(normalize="normalizer"))

    with pytest.raises(WorkflowValidationError, match="does not match"):
        ExecutionPlan(
            workflow=plan.workflow,
            bindings=plan.bindings,
            edges=(
                EdgePlan(
                    source=ValueRef.for_input("other"),
                    target_stage="normalize",
                    target_port="text",
                    carrier="local",
                ),
            ),
        )
