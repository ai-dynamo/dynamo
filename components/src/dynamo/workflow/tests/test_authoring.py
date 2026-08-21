# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle
from dataclasses import FrozenInstanceError

import pytest

from dynamo.workflow import (
    StageContract,
    StageIR,
    ValueRef,
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


def _contracts() -> tuple[StageContract, StageContract, StageContract]:
    encoder = StageContract(
        id="encoder",
        inputs={"image"},
        outputs={"embedding"},
    )
    classifier = StageContract(
        id="classifier",
        inputs={"embedding"},
        outputs={"scores"},
    )
    generator = StageContract(
        id="generator",
        inputs={"embedding", "prompt"},
        outputs={"text"},
    )
    return encoder, classifier, generator


def _workflow() -> WorkflowIR:
    encoder_contract, classifier_contract, generator_contract = _contracts()
    workflow = Workflow("vision-response")
    image = workflow.input("image")
    prompt = workflow.input("prompt")
    encoder = workflow.stage("encoder", encoder_contract, image=image)
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


def test_builds_fanout_workflow_from_declared_output_attributes() -> None:
    workflow = _workflow()

    assert [stage.id for stage in workflow.stages] == [
        "encoder",
        "classifier",
        "generator",
    ]
    assert workflow.inputs == frozenset({"image", "prompt"})
    assert workflow.stages[1].inputs["embedding"] == ValueRef.for_stage_output(
        "encoder", "embedding"
    )


def test_stage_requires_contract_and_accepts_mapping_expansion() -> None:
    contract = StageContract(
        id="hyphenated-input",
        inputs={"input-value"},
        outputs={"text"},
    )

    class ContractProvider:
        contract = StageContract(
            id="provider", inputs={"input-value"}, outputs={"text"}
        )

    workflow = Workflow("explicit-contract")
    value = workflow.input("value")
    stage = workflow.stage("producer", contract, **{"input-value": value})
    workflow.output("text", stage.text)

    assert workflow.build().stages[0].inputs == {"input-value": value}
    with pytest.raises(WorkflowValidationError, match="must use StageContract"):
        workflow.stage("implicit", ContractProvider, **{"input-value": value})


def test_ir_keeps_complete_name_only_contracts_inline() -> None:
    workflow = Workflow("echo-flow")
    value = workflow.input("text")
    echo = workflow.stage(
        "echo",
        StageContract(id="echo-contract", inputs={"text"}, outputs={"text"}),
        text=value,
    )
    workflow.output("text", echo.text)

    workflow_ir = workflow.build()
    assert workflow_ir.inputs == frozenset({"text"})
    assert workflow_ir.outputs == {"text": ValueRef.for_stage_output("echo", "text")}
    assert workflow_ir.stages == (
        StageIR(
            id="echo",
            contract=StageContract(
                id="echo-contract", inputs={"text"}, outputs={"text"}
            ),
            inputs={"text": ValueRef.for_input("text")},
        ),
    )


def test_stage_ir_inputs_must_match_its_contract() -> None:
    contract = StageContract(id="consumer", inputs={"value"}, outputs={"value"})

    with pytest.raises(WorkflowValidationError, match=r"missing=\['value'\]"):
        StageIR(id="consumer", contract=contract)
    with pytest.raises(WorkflowValidationError, match=r"extra=\['extra'\]"):
        StageIR(
            id="consumer",
            contract=contract,
            inputs={
                "value": ValueRef.for_input("value"),
                "extra": ValueRef.for_input("value"),
            },
        )


def test_contracts_require_name_sets_and_are_immutable() -> None:
    contract = StageContract(id="source", inputs={"request"}, outputs={"text"})

    assert contract.inputs == frozenset({"request"})
    assert contract.outputs == frozenset({"text"})
    with pytest.raises(FrozenInstanceError):
        contract.id = "changed"
    with pytest.raises(WorkflowValidationError, match="set of names"):
        StageContract(id="mapping", outputs={"text": object()})
    with pytest.raises(WorkflowValidationError, match="non-empty string"):
        StageContract(id="empty-name", inputs={"request"}, outputs={""})
    with pytest.raises(WorkflowValidationError, match="at least one input"):
        StageContract(id="no-input", outputs={"text"})
    with pytest.raises(WorkflowValidationError, match="at least one output"):
        StageContract(id="no-output", inputs={"request"})


def test_names_use_a_portable_ascii_grammar() -> None:
    with pytest.raises(WorkflowValidationError, match="letters, digits"):
        Workflow("é-stage")


def test_workflow_inputs_require_name_sets() -> None:
    with pytest.raises(WorkflowValidationError, match="set of names"):
        WorkflowIR(
            name="mapping-inputs",
            inputs={"value": object()},
            stages=(),
            outputs={"value": ValueRef.for_input("value")},
        )


def test_output_method_handles_attribute_collisions_and_invalid_names() -> None:
    workflow = Workflow("fallback")
    seed = workflow.input("seed")
    producer = workflow.stage(
        "producer",
        StageContract(
            id="producer",
            inputs={"seed"},
            outputs={"output", "hyphen-name", "text"},
        ),
        seed=seed,
    )

    assert producer.output_names == ("hyphen-name", "output", "text")
    assert producer.text == producer.output("text")
    assert producer.output("output").output_name == "output"
    assert producer.output("hyphen-name").output_name == "hyphen-name"
    with pytest.raises(WorkflowValidationError, match="no output"):
        producer.output("missing")


def test_stage_handle_supports_copy_and_pickle() -> None:
    workflow = Workflow("copy-handle")
    seed = workflow.input("seed")
    handle = workflow.stage(
        "producer",
        StageContract(id="producer", inputs={"seed"}, outputs={"text"}),
        seed=seed,
    )

    copies = (
        copy.copy(handle),
        copy.deepcopy(handle),
        pickle.loads(pickle.dumps(handle)),
    )
    assert all(cloned.text == handle.text for cloned in copies)


def test_rejects_foreign_reference() -> None:
    consumer = StageContract(id="consumer", inputs={"value"}, outputs={"value"})
    workflow = Workflow("consumer")
    other = Workflow("other")
    foreign = other.input("value")

    with pytest.raises(WorkflowValidationError, match="different workflow"):
        workflow.stage("consumer", consumer, value=foreign)


def test_rejects_missing_inputs_and_conflicting_contract_ids() -> None:
    first = StageContract(id="shared", inputs={"value"}, outputs={"text"})
    conflicting = StageContract(id="shared", inputs={"value"}, outputs={"json"})
    workflow = Workflow("contracts")
    value = workflow.input("value")
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


def test_ir_rejects_cycles_and_dead_stages() -> None:
    contract = StageContract(id="node", inputs={"value"}, outputs={"value"})
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
            inputs={"seed"},
            stages=cycle,
            outputs={"value": ValueRef.for_stage_output("a", "value")},
        )

    dead_stage = StageIR(
        id="dead",
        contract=contract,
        inputs={"value": ValueRef.for_input("seed")},
    )
    with pytest.raises(WorkflowValidationError, match="do not contribute"):
        WorkflowIR(
            name="dead",
            inputs={"seed"},
            stages=(dead_stage,),
            outputs={"value": ValueRef.for_input("seed")},
        )


def test_ir_rejects_unknown_references() -> None:
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
