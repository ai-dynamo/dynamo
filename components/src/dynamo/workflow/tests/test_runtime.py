# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    StageContext,
    StageContract,
    StageRunner,
    ValueSpec,
    Workflow,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowValidationError,
    compile_workflow,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


EMBEDDING = ValueSpec(type="object", class_id="embedding")
ENCODER = StageContract(
    id="encoder-worker",
    inputs={"text": ValueSpec(type="text")},
    outputs={"embedding": EMBEDDING},
)
CLASSIFIER = StageContract(
    id="classifier-worker",
    inputs={"embedding": EMBEDDING},
    outputs={"scores": ValueSpec(type="json")},
)
GENERATOR = StageContract(
    id="generator-worker",
    inputs={"embedding": EMBEDDING},
    outputs={"text": ValueSpec(type="text")},
)


def _workflow() -> Workflow:
    workflow = Workflow("local-execution")
    text = workflow.input("text", ValueSpec(type="text"))
    encoder = workflow.stage("encoder", _Encoder.contract, text=text)
    classifier = workflow.stage(
        "classifier", _Classifier.contract, embedding=encoder.embedding
    )
    generator = workflow.stage(
        "generator", _Generator.contract, embedding=encoder.embedding
    )
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)
    return workflow


async def _compile_local(
    workflow: Workflow, **runners: StageRunner
) -> WorkflowExecutor:
    plan = compile_workflow(
        workflow,
        DeploymentSpec.local(**{stage_id: stage_id for stage_id in runners}),
    )
    return await WorkflowExecutor.bind(plan, local_runners=runners)


@dataclass
class _Encoder:
    contract = ENCODER
    embedding: object

    async def run(self, inputs, context: StageContext):
        assert context.stage_id == "encoder"
        return {"embedding": self.embedding}


@dataclass
class _Classifier:
    contract = CLASSIFIER
    expected: object

    async def run(self, inputs, context: StageContext):
        assert inputs["embedding"] is self.expected
        return {"scores": {"class-a": 0.75, "class-b": 0.25}}


@dataclass
class _Generator:
    contract = GENERATOR
    expected: object

    async def run(self, inputs, context: StageContext):
        assert inputs["embedding"] is self.expected
        return {"text": "generated"}


async def test_concise_compile_and_run_preserve_fanout_value_identity():
    embedding = object()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_Classifier(embedding),
        generator=_Generator(embedding),
    )

    result = await plan.run({"text": "hello"}, attempt_id="request-1")

    assert result == {
        "scores": {"class-a": 0.75, "class-b": 0.25},
        "text": "generated",
    }


class _BranchBarrier:
    def __init__(self, count: int) -> None:
        self._remaining = count
        self.open = asyncio.Event()

    async def enter(self) -> None:
        self._remaining -= 1
        if self._remaining == 0:
            self.open.set()
        await self.open.wait()


@dataclass
class _BarrierClassifier:
    contract = CLASSIFIER
    barrier: _BranchBarrier

    async def run(self, inputs, context: StageContext):
        await self.barrier.enter()
        return {"scores": {"ok": True}}


@dataclass
class _BarrierGenerator:
    contract = GENERATOR
    barrier: _BranchBarrier

    async def run(self, inputs, context: StageContext):
        await self.barrier.enter()
        return {"text": "joined"}


async def test_independent_branches_run_concurrently_before_join():
    embedding = object()
    barrier = _BranchBarrier(2)
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_BarrierClassifier(barrier),
        generator=_BarrierGenerator(barrier),
    )

    assert await plan.run({"text": "hello"}) == {
        "scores": {"ok": True},
        "text": "joined",
    }


class WorkerFailure(RuntimeError):
    pass


@dataclass
class _FailingClassifier:
    contract = CLASSIFIER
    barrier: _BranchBarrier

    async def run(self, inputs, context: StageContext):
        await self.barrier.enter()
        raise WorkerFailure("classifier failed")


@dataclass
class _CancelledGenerator:
    contract = GENERATOR
    barrier: _BranchBarrier
    cancelled: asyncio.Event

    async def run(self, inputs, context: StageContext):
        await self.barrier.enter()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def test_first_worker_failure_cancels_and_awaits_sibling():
    embedding = object()
    barrier = _BranchBarrier(2)
    cancelled = asyncio.Event()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_FailingClassifier(barrier),
        generator=_CancelledGenerator(barrier, cancelled),
    )

    with pytest.raises(WorkerFailure, match="classifier failed"):
        await plan.run({"text": "hello"})

    assert cancelled.is_set()


@dataclass
class _BlockingClassifier:
    contract = CLASSIFIER
    started: asyncio.Event
    cancelled: asyncio.Event

    async def run(self, inputs, context: StageContext):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert context.cancelled
            self.cancelled.set()
            raise


async def test_timeout_cancels_and_awaits_running_stages():
    embedding = object()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_BlockingClassifier(started, cancelled),
        generator=_Generator(embedding),
    )

    with pytest.raises(asyncio.TimeoutError):
        await plan.run({"text": "hello"}, timeout=0.01)

    assert started.is_set()
    assert cancelled.is_set()


async def test_caller_cancellation_cleans_up_running_stages():
    embedding = object()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_BlockingClassifier(started, cancelled),
        generator=_Generator(embedding),
    )
    task = asyncio.create_task(plan.run({"text": "hello"}))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


async def test_compile_requires_exact_bindings_and_matching_contracts():
    with pytest.raises(WorkflowValidationError, match="missing"):
        compile_workflow(_workflow(), DeploymentSpec.local(encoder="encoder"))

    wrong = SimpleNamespace(contract=CLASSIFIER, run=_Generator(object()).run)
    with pytest.raises(WorkflowValidationError, match="does not match"):
        await WorkflowExecutor.bind(
            compile_workflow(
                _workflow(),
                DeploymentSpec.local(
                    encoder="encoder",
                    classifier="classifier",
                    generator="generator",
                ),
            ),
            local_runners={
                "encoder": _Encoder(object()),
                "classifier": _Classifier(object()),
                "generator": wrong,
            },
        )


async def test_runtime_rejects_bad_inputs_and_worker_outputs():
    embedding = object()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_Classifier(embedding),
        generator=_Generator(embedding),
    )
    with pytest.raises(WorkflowExecutionError, match="must be text"):
        await plan.run({"text": b"not text"})
    with pytest.raises(WorkflowExecutionError, match="extra"):
        await plan.run({"text": "hello", "extra": "value"})

    class BadGenerator:
        contract = GENERATOR

        async def run(self, inputs, context):
            return {"wrong": "value"}

    bad_plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_Classifier(embedding),
        generator=BadGenerator(),
    )
    with pytest.raises(WorkflowExecutionError, match="outputs differ"):
        await bad_plan.run({"text": "hello"})


async def test_runtime_enforces_json_data_model() -> None:
    workflow = Workflow("json-values")
    value = workflow.input("value", ValueSpec(type="json"))
    workflow.output("value", value)
    plan = await _compile_local(workflow)

    shared = [1, 2]
    valid = {"none": None, "bool": True, "number": 1.5, "shared": [shared, shared]}
    assert await plan.run({"value": valid}) == {"value": valid}

    cyclic: list[object] = []
    cyclic.append(cyclic)
    invalid_values: list[object] = [
        (1, 2),
        {1: "non-string key"},
        float("nan"),
        float("inf"),
        cyclic,
    ]
    for invalid in invalid_values:
        with pytest.raises(WorkflowExecutionError, match="JSON data model"):
            await plan.run({"value": invalid})


async def test_tensor_and_image_constraints_are_checked_without_framework_imports():
    tensor_contract = StageContract(
        id="tensor",
        inputs={
            "tensor": ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))
        },
        outputs={"image": ValueSpec(type="image", mode="RGB")},
    )
    workflow = Workflow("runtime-types")
    tensor = workflow.input(
        "tensor", ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))
    )
    stage = workflow.stage("convert", tensor_contract, tensor=tensor)
    workflow.output("image", stage.image)

    class Converter:
        contract = tensor_contract

        async def run(self, inputs, context):
            return {"image": SimpleNamespace(mode="RGB", size=(10, 10))}

    plan = await _compile_local(workflow, convert=Converter())
    value = SimpleNamespace(dtype="float32", shape=(2, 4))

    assert (await plan.run({"tensor": value}))["image"].mode == "RGB"
    with pytest.raises(WorkflowExecutionError, match="shape"):
        await plan.run({"tensor": SimpleNamespace(dtype="float32", shape=(2, 3))})
