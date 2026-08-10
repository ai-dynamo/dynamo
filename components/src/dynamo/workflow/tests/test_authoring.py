# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import FrozenInstanceError

import pytest

from dynamo.workflow import (
    StageContract,
    StageIR,
    ValueRef,
    ValueSpec,
    Workflow,
    WorkflowBuilder,
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
    assert encoder_ref.to_dict() == {"stage": "encoder", "output": "embedding"}


def test_json_is_deterministic_and_round_trips_unicode():
    workflow = _workflow()
    encoded = workflow.to_json()

    assert encoded == workflow.to_json()
    assert WorkflowIR.from_json(encoded).to_dict() == workflow.to_dict()
    assert WorkflowIR.from_json(
        encoded.replace("vision-response", "视觉-workflow")
    ).name == ("视觉-workflow")


def test_output_method_handles_attribute_collisions_and_invalid_names():
    contract = StageContract(
        id="producer",
        outputs={
            "output": ValueSpec(type="text"),
            "hyphen-name": ValueSpec(type="text"),
            "text": ValueSpec(type="text"),
        },
    )
    workflow = WorkflowBuilder("fallback")
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
        {"type": "meaning"},
        {"type": "text", "dtype": "float32"},
        {"type": "tensor", "dtype": "fp8"},
        {"type": "tensor", "shape": [-1, 4]},
        {"type": "object"},
    ],
)
def test_rejects_invalid_value_specs(spec):
    with pytest.raises(WorkflowValidationError):
        ValueSpec.from_dict(spec)


def test_rejects_incompatible_edge_and_foreign_reference():
    consumer = StageContract(
        id="consumer",
        inputs={"value": ValueSpec(type="tensor", dtype="float32", shape=(4,))},
        outputs={"text": ValueSpec(type="text")},
    )
    workflow = WorkflowBuilder("incompatible")
    value = workflow.add_input(
        "value", ValueSpec(type="tensor", dtype="float16", shape=(4,))
    )
    other = WorkflowBuilder("other")
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
    workflow = WorkflowBuilder("contracts")
    value = workflow.add_input("value", ValueSpec(type="text"))
    first_stage = workflow.add_stage("first", first, inputs={"value": value})

    with pytest.raises(WorkflowValidationError, match="missing"):
        workflow.add_stage("missing", first, inputs={})
    with pytest.raises(WorkflowValidationError, match="conflicting schemas"):
        workflow.add_stage("second", conflicting, inputs={"value": first_stage.text})


def test_parser_rejects_unknown_fields_and_duplicate_json_keys():
    data = _workflow().to_dict()
    data["placement"] = {"gpu": 0}
    with pytest.raises(WorkflowValidationError, match="unknown fields"):
        WorkflowIR.from_dict(data)
    with pytest.raises(WorkflowValidationError, match="duplicate JSON key"):
        WorkflowIR.from_json('{"schema":"one","schema":"two"}')


def test_parser_rejects_cycles_unreachable_and_dead_stages():
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


def test_parser_rejects_unknown_references_and_schema_version():
    data = _workflow().to_dict()
    data["outputs"]["text"] = {"stage": "missing", "output": "text"}
    with pytest.raises(WorkflowValidationError, match="unknown stage"):
        WorkflowIR.from_dict(data)

    data = _workflow().to_dict()
    data["version"] = 1
    with pytest.raises(WorkflowValidationError, match="unsupported workflow version"):
        WorkflowIR.from_dict(data)

    malformed = json.loads(_workflow().to_json())
    malformed["stages"][0]["contract"]["outputs"]["embedding"]["extra"] = "x"
    with pytest.raises(WorkflowValidationError, match="unknown fields"):
        WorkflowIR.from_dict(malformed)
