# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.experimental.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    InlineBinding,
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


def _workflow(stage_id: str = "normalize") -> Workflow:
    workflow = Workflow("physical-plan")
    text = workflow.input("text")
    stage = workflow.stage(
        stage_id,
        StageContract(
            id="normalize",
            inputs={"text"},
            outputs={"normalized"},
        ),
        text=text,
    )
    workflow.output("text", stage.normalized)
    return workflow


def test_execution_plan_contains_only_in_memory_decisions() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    assert plan.workflow == _workflow().build()
    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalizer")}


def test_compilation_defaults_to_stage_id_inline_bindings() -> None:
    plan = compile_workflow(_workflow())

    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalize")}


def test_default_compilation_supports_stage_named_cls() -> None:
    plan = compile_workflow(_workflow("cls"))

    assert plan.bindings == {"cls": InlineBinding(runner_key="cls")}


def test_execution_plan_rejects_missing_stage_bindings() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    with pytest.raises(WorkflowValidationError, match=r"missing=\['normalize'\]"):
        ExecutionPlan(
            workflow=plan.workflow,
            bindings={},
        )
