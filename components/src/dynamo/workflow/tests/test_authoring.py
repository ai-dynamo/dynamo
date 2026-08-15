# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from dynamo.workflow import (
    StageContract,
    StageIR,
    StreamSpec,
    ValueRef,
    ValueSpec,
    Workflow,
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
    image = workflow.input("image", ValueSpec(type="image", mode="RGB"))
    prompt = workflow.input("prompt", ValueSpec(type="text"))
    encoder = workflow.stage("encoder", EncoderWorker.contract, image=image)
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


def test_stage_requires_contract_and_accepts_mapping_expansion() -> None:
    contract = StageContract(
        id="hyphenated-input",
        inputs={"input-value": ValueSpec(type="text")},
        outputs={"text": ValueSpec(type="text")},
    )

    class ContractProvider:
        contract = StageContract(
            id="provider",
            outputs={"text": ValueSpec(type="text")},
        )

    workflow = Workflow("explicit-contract")
    value = workflow.input("value", ValueSpec(type="text"))
    stage = workflow.stage(
        "producer",
        contract,
        **{"input-value": value},
    )
    workflow.output("text", stage.text)

    assert workflow.build().stages[0].inputs == {"input-value": value}
    with pytest.raises(WorkflowValidationError, match="must use StageContract"):
        workflow.stage("implicit", ContractProvider, **{"input-value": value})


def test_ir_keeps_complete_contracts_inline() -> None:
    workflow = Workflow("echo-flow")
    value = workflow.input("text", ValueSpec(type="text"))
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
    seed = workflow.input("seed", ValueSpec(type="text"))
    passthrough = StageContract(
        id="passthrough",
        inputs={"seed": ValueSpec(type="text")},
        outputs=contract.outputs,
    )
    producer = workflow.stage("producer", passthrough, seed=seed)

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


def test_stream_specs_are_typed_and_connect_only_to_compatible_streams() -> None:
    chunks = StreamSpec(item=ValueSpec(type="json"))
    workflow = Workflow("stream-flow")
    source = workflow.input("chunks", chunks)
    consumer = workflow.stage(
        "consumer",
        StageContract(
            id="stream-consumer",
            inputs={"chunks": chunks},
            outputs={"chunks": chunks},
        ),
        chunks=source,
    )
    workflow.output("chunks", consumer.chunks)

    assert workflow.build().output_spec("chunks") == chunks

    with pytest.raises(WorkflowValidationError, match="stream output"):
        workflow.stage(
            "value-consumer",
            StageContract(
                id="value-consumer",
                inputs={"value": ValueSpec(type="json")},
                outputs={"value": ValueSpec(type="json")},
            ),
            value=consumer.chunks,
        )


def test_stream_specs_require_value_items() -> None:
    with pytest.raises(WorkflowValidationError, match="stream items"):
        StreamSpec(item="json")


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
    value = workflow.input(
        "value", ValueSpec(type="tensor", dtype="float16", shape=(4,))
    )
    other = Workflow("other")
    foreign = other.input("value", ValueSpec(type="tensor"))

    with pytest.raises(WorkflowValidationError, match="incompatible"):
        workflow.stage("consumer", consumer, value=value)
    with pytest.raises(WorkflowValidationError, match="different workflow"):
        workflow.stage("consumer", consumer, value=foreign)


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
    value = workflow.input("value", ValueSpec(type="text"))
    first_stage = workflow.stage("first", first, value=value)

    with pytest.raises(WorkflowValidationError, match="missing"):
        workflow.stage("missing", first)
    with pytest.raises(WorkflowValidationError, match="conflicting schemas"):
        workflow.stage("second", conflicting, value=first_stage.text)


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
