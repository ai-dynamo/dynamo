# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

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
    text = workflow.input("text", type="text")
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
            definition=plan.definition,
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


def test_handler_plan_contains_bindings_without_graph_edges() -> None:
    workflow = Workflow("imperative-plan")
    echo = StageContract(
        id="echo",
        inputs={"text": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow.use("echo", echo)

    @workflow.handler(
        inputs={"text": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )
    async def run(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"text": inputs["text"]}

    plan = compile_workflow(workflow)

    assert plan.bindings == {"echo": LocalBinding(runner_key="echo")}
    assert plan.stage_contracts == {"echo": echo}
    assert plan.edges == ()
