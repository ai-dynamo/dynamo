# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
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
    workflow = Workflow("physical-plan")
    text = workflow.input("text")
    stage = workflow.stage(
        "normalize",
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
        inputs={"text"},
        outputs={"text"},
    )
    workflow = Workflow("mixed-placement")
    value = workflow.input("text")
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


def test_remote_planning_does_not_assume_a_runtime_value_type() -> None:
    contract = StageContract(
        id="opaque-stage",
        inputs={"value"},
        outputs={"result"},
    )
    workflow = Workflow("remote-opaque-value")
    value = workflow.input("value")
    result = workflow.stage("stage", contract, value=value)
    workflow.output("result", result.result)

    with pytest.raises(WorkflowValidationError, match="no common declared carrier"):
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


def test_declared_nixl_carrier_lowers_tensor_fanout_per_consumer() -> None:
    tensor = ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 8))
    encoder_contract = StageContract(
        id="encoder",
        inputs={"request": ValueSpec(type="json")},
        outputs={"embedding": tensor},
    )
    classifier_contract = StageContract(
        id="classifier",
        inputs={"embedding": tensor},
        outputs={"scores": ValueSpec(type="json")},
    )
    generator_contract = StageContract(
        id="generator",
        inputs={"embedding": tensor},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow = Workflow("nixl-fanout")
    request = workflow.input("request", type="json")
    encoder = workflow.stage("encoder", encoder_contract, request=request)
    classifier = workflow.stage(
        "classifier", classifier_contract, embedding=encoder.embedding
    )
    generator = workflow.stage(
        "generator", generator_contract, embedding=encoder.embedding
    )
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)

    plan = compile_workflow(
        workflow,
        DeploymentSpec.remote(
            tensor_carrier="nixl",
            encoder="workflows.encoder.generate",
            classifier="workflows.classifier.generate",
            generator="workflows.generator.generate",
        ),
    )
    assert plan.bindings["encoder"].tensor_carrier == "nixl"
    assert {edge.transfer_id: edge.carrier for edge in plan.edges} == {
        "encoder.request": "inline",
        "classifier.embedding": "nixl",
        "generator.embedding": "nixl",
    }
