# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    EdgePlan,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
    StageContract,
    StreamSpec,
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
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalizer")}
    assert plan.edges == (
        EdgePlan(
            source=ValueRef.for_input("text"),
            target_stage="normalize",
            target_port="text",
            carrier="in_process",
        ),
    )


def test_compilation_defaults_to_stage_id_inline_bindings() -> None:
    plan = compile_workflow(_workflow())

    assert plan.bindings == {"normalize": InlineBinding(runner_key="normalize")}
    assert all(edge.carrier == "in_process" for edge in plan.edges)


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


def test_execution_plan_rejects_invalid_physical_edges() -> None:
    plan = compile_workflow(_workflow(), DeploymentSpec.inline(normalize="normalizer"))

    with pytest.raises(WorkflowValidationError, match="does not match"):
        ExecutionPlan(
            workflow=plan.workflow,
            bindings=plan.bindings,
            edges=(
                EdgePlan(
                    source=ValueRef.for_input("other"),
                    target_stage="normalize",
                    target_port="text",
                    carrier="in_process",
                ),
            ),
        )


def test_remote_plan_selects_inline_carrier() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec.remote(normalize="workflows.normalize.generate"),
    )

    assert plan.remote
    assert plan.bindings == {"normalize": RemoteBinding("workflows.normalize.generate")}
    assert plan.edges[0].carrier == "inline"


def test_mixed_placement_is_rejected() -> None:
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

    with pytest.raises(WorkflowValidationError, match="homogeneous"):
        compile_workflow(
            workflow,
            DeploymentSpec(
                {
                    "first": InlineBinding("first"),
                    "second": RemoteBinding("workflows.second.generate"),
                }
            ),
        )


@pytest.mark.parametrize(
    "value_spec",
    [
        ValueSpec(type="tensor"),
        ValueSpec(type="image"),
        ValueSpec(type="object", class_id="opaque.Value"),
    ],
    ids=["tensor", "image", "object"],
)
def test_remote_rich_value_requires_a_transport_carrier(
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

    with pytest.raises(WorkflowValidationError, match="cannot carry value type"):
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
