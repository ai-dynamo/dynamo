# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
    StageContract,
    StreamSpec,
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
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    assert plan.workflow == _workflow().build()
    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalizer")}


def test_compilation_defaults_to_stage_id_inline_bindings() -> None:
    plan = compile_workflow(_workflow())

    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalize")}


def test_compilation_rejects_declared_stream_execution() -> None:
    chunks = StreamSpec(item=ValueSpec(type="json"))
    workflow = Workflow("stream-plan")
    source = workflow.input("chunks", chunks)
    stage = workflow.stage(
        "stream",
        StageContract(
            id="stream",
            inputs={"chunks": chunks},
            outputs={"chunks": chunks},
        ),
        chunks=source,
    )
    workflow.output("chunks", stage.chunks)

    with pytest.raises(WorkflowValidationError, match="not supported"):
        compile_workflow(workflow)


def test_execution_plan_rejects_missing_stage_bindings() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    with pytest.raises(WorkflowValidationError, match=r"missing=\['normalize'\]"):
        ExecutionPlan(
            workflow=plan.workflow,
            bindings={},
        )


def test_remote_plan_records_remote_binding() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec.remote(normalize="workflows.normalize.generate"),
    )

    assert plan.remote
    assert plan.bindings == {"normalize": RemoteBinding("workflows.normalize.generate")}


def test_mixed_placement_records_each_stage_binding() -> None:
    contract = StageContract(
        id="text-stage",
        inputs={"text": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow = Workflow("mixed-placement")
    value = workflow.input("text", ValueSpec(type="text"))
    first = workflow.stage("first", contract, text=value)
    second = workflow.stage("second", contract, text=first.text)
    workflow.output("text", second.text)

    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "first": InlineBinding("first"),
                "second": RemoteBinding("workflows.second.generate"),
            }
        ),
    )

    assert plan.bindings == {
        "first": InlineBinding("first"),
        "second": RemoteBinding("workflows.second.generate"),
    }


@pytest.mark.parametrize(
    "value_spec",
    [
        ValueSpec(type="tensor"),
        ValueSpec(type="image"),
        ValueSpec(type="object", class_id="opaque.Value"),
    ],
    ids=["tensor", "image", "object"],
)
def test_remote_rich_value_cannot_cross_a_process_boundary(
    value_spec: ValueSpec,
) -> None:
    contract = StageContract(
        id="rich-stage",
        inputs={"value": value_spec},
        outputs={"result": ValueSpec(type="json")},
    )
    workflow = Workflow("remote-rich-value")
    value = workflow.input("value", value_spec)
    result = workflow.stage("stage", contract, value=value)
    workflow.output("result", result.result)
    deployment = DeploymentSpec.remote(stage="workflows.stage.generate")

    with pytest.raises(
        WorkflowValidationError, match="cannot cross a process boundary"
    ):
        compile_workflow(workflow, deployment)

    with pytest.raises(
        WorkflowValidationError, match="cannot cross a process boundary"
    ):
        ExecutionPlan(
            workflow=workflow.build(),
            bindings=deployment.bindings,
        )


def test_remote_rich_workflow_output_cannot_cross_a_process_boundary() -> None:
    rich_value = ValueSpec(type="object", class_id="opaque.Value")
    contract = StageContract(
        id="rich-output-stage",
        inputs={"request": ValueSpec(type="json")},
        outputs={"value": rich_value},
    )
    workflow = Workflow("remote-rich-output")
    request = workflow.input("request", ValueSpec(type="json"))
    result = workflow.stage("stage", contract, request=request)
    workflow.output("value", result.value)

    with pytest.raises(
        WorkflowValidationError, match="cannot cross a process boundary"
    ):
        compile_workflow(
            workflow,
            DeploymentSpec.remote(stage="workflows.stage.generate"),
        )


def test_remote_endpoint_id_is_a_stable_discovery_identity() -> None:
    assert RemoteBinding("namespace.component.endpoint").endpoint_id == (
        "namespace.component.endpoint"
    )

    with pytest.raises(WorkflowValidationError, match="namespace.component.endpoint"):
        RemoteBinding("component.endpoint")
