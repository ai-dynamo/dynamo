# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from dynamo.workflow import (
    StageContract,
    StageIR,
    StageRef,
    ValueRef,
    ValueSpec,
    Workflow,
    WorkflowHandler,
    WorkflowIR,
    WorkflowValidationError,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def _contracts():
    embedding = ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))
    encoder = StageContract(
        id="encoder",
        inputs={"image": ValueSpec(type="image", mode="RGB")},
        outputs={"embedding": embedding},
    )
    classifier = StageContract(
        id="classifier",
        inputs={"embedding": embedding},
        outputs={"scores": ValueSpec(type="json")},
    )
    generator = StageContract(
        id="generator",
        inputs={
            "embedding": embedding,
            "prompt": ValueSpec(type="text"),
        },
        outputs={"text": ValueSpec(type="text")},
    )
    return encoder, classifier, generator


def _workflow() -> WorkflowIR:
    encoder_contract, classifier_contract, generator_contract = _contracts()

    class EncoderWorker:
        contract = encoder_contract

    workflow = Workflow("vision-response")
    image = workflow.input("image", type="image", mode="RGB")
    prompt = workflow.input("prompt", type="text")
    encoder = workflow.stage("encoder", EncoderWorker, image=image)
    classifier = workflow.stage(
        "classifier",
        classifier_contract,
        embedding=encoder.embedding,
    )
    generator = workflow.stage(
        "generator",
        generator_contract,
        embedding=encoder.embedding,
        prompt=prompt,
    )
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)
    return workflow.build()


def test_builds_fanout_workflow_from_declared_output_attributes():
    workflow = _workflow()

    assert [stage.id for stage in workflow.stages] == [
        "encoder",
        "classifier",
        "generator",
    ]
    assert workflow.output_spec("scores") == ValueSpec(type="json")
    encoder_ref = workflow.stages[1].inputs["embedding"]
    assert encoder_ref == ValueRef.for_stage_output("encoder", "embedding")


def test_ir_keeps_complete_contracts_inline() -> None:
    workflow = Workflow("echo-flow")
    value = workflow.input("text", type="text")
    echo = workflow.stage(
        "echo",
        StageContract(
            id="echo-contract",
            inputs={"text": ValueSpec(type="text")},
            outputs={"text": ValueSpec(type="text")},
        ),
        text=value,
    )
    workflow.output("text", echo.text)

    workflow_ir = workflow.build()
    assert workflow_ir.inputs == {"text": ValueSpec(type="text")}
    assert workflow_ir.outputs == {"text": ValueRef.for_stage_output("echo", "text")}
    assert workflow_ir.stages == (
        StageIR(
            id="echo",
            contract=StageContract(
                id="echo-contract",
                inputs={"text": ValueSpec(type="text")},
                outputs={"text": ValueSpec(type="text")},
            ),
            inputs={"text": ValueRef.for_input("text")},
        ),
    )


def test_output_method_handles_attribute_collisions_and_invalid_names():
    contract = StageContract(
        id="producer",
        outputs={
            "output": ValueSpec(type="text"),
            "hyphen-name": ValueSpec(type="text"),
            "text": ValueSpec(type="text"),
        },
    )
    workflow = Workflow("fallback")
    seed = workflow.add_input("seed", ValueSpec(type="text"))
    passthrough = StageContract(
        id="passthrough",
        inputs={"seed": ValueSpec(type="text")},
        outputs=contract.outputs,
    )
    producer = workflow.add_stage("producer", passthrough, inputs={"seed": seed})

    assert producer.text == producer.output("text")
    assert producer.output("output").output_name == "output"
    assert producer.output("hyphen-name").output_name == "hyphen-name"
    with pytest.raises(WorkflowValidationError, match="no output"):
        producer.output("missing")


def test_contracts_are_deeply_immutable():
    contract = StageContract(id="source", outputs={"text": ValueSpec(type="text")})

    with pytest.raises(TypeError):
        contract.outputs["other"] = ValueSpec(type="text")
    with pytest.raises(FrozenInstanceError):
        contract.id = "changed"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": []},
        {"type": "meaning"},
        {"type": "text", "dtype": "float32"},
        {"type": "tensor", "dtype": []},
        {"type": "tensor", "dtype": "fp8"},
        {"type": "tensor", "shape": "dynamic"},
        {"type": "tensor", "shape": [-1, 4]},
        {"type": "image", "mode": 123},
        {"type": "image", "mode": float("nan")},
        {"type": "image", "mode": "\ud800"},
        {"type": "object"},
        {"type": "object", "class_id": 123},
        {"type": "object", "class_id": "\ud800"},
    ],
)
def test_rejects_invalid_value_specs(spec: dict[str, Any]) -> None:
    with pytest.raises(WorkflowValidationError):
        ValueSpec(**spec)


def test_rejects_incompatible_edge_and_foreign_reference():
    consumer = StageContract(
        id="consumer",
        inputs={"value": ValueSpec(type="tensor", dtype="float32", shape=(4,))},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow = Workflow("incompatible")
    value = workflow.add_input(
        "value", ValueSpec(type="tensor", dtype="float16", shape=(4,))
    )
    other = Workflow("other")
    foreign = other.add_input("value", ValueSpec(type="tensor"))

    with pytest.raises(WorkflowValidationError, match="incompatible"):
        workflow.add_stage("consumer", consumer, inputs={"value": value})
    with pytest.raises(WorkflowValidationError, match="different workflow"):
        workflow.add_stage("consumer", consumer, inputs={"value": foreign})


def test_rejects_missing_inputs_and_conflicting_contract_ids():
    first = StageContract(
        id="shared",
        inputs={"value": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )
    conflicting = StageContract(
        id="shared",
        inputs={"value": ValueSpec(type="text")},
        outputs={"json": ValueSpec(type="json")},
    )
    workflow = Workflow("contracts")
    value = workflow.add_input("value", ValueSpec(type="text"))
    first_stage = workflow.add_stage("first", first, inputs={"value": value})

    with pytest.raises(WorkflowValidationError, match="missing"):
        workflow.add_stage("missing", first, inputs={})
    with pytest.raises(WorkflowValidationError, match="conflicting schemas"):
        workflow.add_stage("second", conflicting, inputs={"value": first_stage.text})


def test_ir_rejects_conflicting_inline_contracts() -> None:
    workflow = _workflow()
    stages = list(workflow.stages)
    generator = stages[2]
    stages[2] = StageIR(
        id=generator.id,
        contract=StageContract(
            id="classifier",
            inputs=generator.contract.inputs,
            outputs=generator.contract.outputs,
        ),
        inputs=generator.inputs,
    )

    with pytest.raises(WorkflowValidationError, match="conflicting schemas"):
        WorkflowIR(
            name=workflow.name,
            inputs=workflow.inputs,
            stages=tuple(stages),
            outputs=workflow.outputs,
        )


def test_ir_rejects_cycles_unreachable_and_dead_stages():
    text = ValueSpec(type="text")
    contract = StageContract(id="node", inputs={"value": text}, outputs={"value": text})
    cycle = (
        StageIR(
            id="a",
            contract=contract,
            inputs={"value": ValueRef.for_stage_output("b", "value")},
        ),
        StageIR(
            id="b",
            contract=contract,
            inputs={"value": ValueRef.for_stage_output("a", "value")},
        ),
    )
    with pytest.raises(WorkflowValidationError, match="cycle"):
        WorkflowIR(
            name="cycle",
            inputs={"seed": text},
            stages=cycle,
            outputs={"value": ValueRef.for_stage_output("a", "value")},
        )

    source = StageContract(id="source", outputs={"value": text})
    with pytest.raises(WorkflowValidationError, match="not reachable"):
        WorkflowIR(
            name="unreachable",
            inputs={"seed": text},
            stages=(StageIR(id="source", contract=source),),
            outputs={"value": ValueRef.for_stage_output("source", "value")},
        )

    dead_stage = StageIR(
        id="dead", contract=contract, inputs={"value": ValueRef.for_input("seed")}
    )
    with pytest.raises(WorkflowValidationError, match="do not contribute"):
        WorkflowIR(
            name="dead",
            inputs={"seed": text},
            stages=(dead_stage,),
            outputs={"value": ValueRef.for_input("seed")},
        )


def test_ir_rejects_unknown_references():
    workflow = _workflow()
    outputs = dict(workflow.outputs)
    outputs["text"] = ValueRef.for_stage_output("missing", "text")
    with pytest.raises(WorkflowValidationError, match="unknown stage"):
        WorkflowIR(
            name=workflow.name,
            inputs=workflow.inputs,
            stages=workflow.stages,
            outputs=outputs,
        )


def test_builds_imperative_handler_with_fixed_stage_catalog() -> None:
    contract = StageContract(
        id="echo",
        inputs={"text": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow = Workflow("imperative")
    echo = workflow.use("echo", contract)

    @workflow.handler(
        inputs={"request": ValueSpec(type="json")},
        outputs={"result": ValueSpec(type="text")},
    )
    async def run(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"result": inputs["request"]["text"]}

    definition = workflow.build()

    assert isinstance(definition, WorkflowHandler)
    assert isinstance(echo, StageRef)
    assert echo.id == "echo"
    assert definition.inputs == {"request": ValueSpec(type="json")}
    assert definition.outputs == {"result": ValueSpec(type="text")}
    assert definition.stages == {"echo": contract}
    assert definition.callback is run


def test_handler_catalog_accepts_use_after_decorator() -> None:
    contract = StageContract(id="source", outputs={"text": ValueSpec(type="text")})
    workflow = Workflow("handler-first")

    @workflow.handler(inputs={}, outputs={"text": ValueSpec(type="text")})
    async def run(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"text": "ready"}

    source = workflow.use("source", contract)
    definition = workflow.build()

    assert isinstance(definition, WorkflowHandler)
    assert source.contract is contract
    assert definition.stages == {"source": contract}


def test_rejects_mixed_graph_and_handler_authoring() -> None:
    contract = StageContract(id="source", outputs={"text": ValueSpec(type="text")})

    graph = Workflow("graph")
    graph.input("request", type="json")
    with pytest.raises(WorkflowValidationError, match="cannot mix"):
        graph.use("source", contract)

    handler = Workflow("handler")
    handler.use("source", contract)
    with pytest.raises(WorkflowValidationError, match="cannot mix"):
        handler.input("request", type="json")


def test_rejects_missing_duplicate_and_synchronous_handlers() -> None:
    workflow = Workflow("missing")
    workflow.use(
        "source",
        StageContract(id="source", outputs={"text": ValueSpec(type="text")}),
    )
    with pytest.raises(WorkflowValidationError, match="requires a handler"):
        workflow.build()

    duplicate = Workflow("duplicate")

    @duplicate.handler(inputs={}, outputs={"text": ValueSpec(type="text")})
    async def first(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"text": "first"}

    with pytest.raises(WorkflowValidationError, match="already has a handler"):
        duplicate.handler(inputs={}, outputs={"text": ValueSpec(type="text")})

    synchronous = Workflow("synchronous")

    @synchronous.handler(inputs={}, outputs={"text": ValueSpec(type="text")})
    def sync_handler(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"text": "invalid"}

    with pytest.raises(WorkflowValidationError, match="async function"):
        synchronous.build()


def test_handler_definition_is_deeply_immutable() -> None:
    workflow = Workflow("immutable-handler")

    @workflow.handler(inputs={}, outputs={"text": ValueSpec(type="text")})
    async def run(inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        return {"text": "ready"}

    definition = workflow.build()
    assert isinstance(definition, WorkflowHandler)
    with pytest.raises(TypeError):
        definition.outputs["other"] = ValueSpec(type="text")
    with pytest.raises(TypeError):
        definition.stages["other"] = StageContract(
            id="other", outputs={"text": ValueSpec(type="text")}
        )
