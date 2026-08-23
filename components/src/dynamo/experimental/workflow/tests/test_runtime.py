# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dynamo.experimental.workflow import (
    DeploymentSpec,
    StageContext,
    StageContract,
    StageRunner,
    Workflow,
    WorkflowExecutionError,
    WorkflowOrchestrator,
    WorkflowValidationError,
    compile_workflow,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


ENCODER = StageContract(
    id="encoder-worker",
    inputs={"text"},
    outputs={"embedding"},
)
CLASSIFIER = StageContract(
    id="classifier-worker",
    inputs={"embedding"},
    outputs={"scores"},
)
GENERATOR = StageContract(
    id="generator-worker",
    inputs={"embedding"},
    outputs={"text"},
)


def _workflow() -> Workflow:
    workflow = Workflow("local-execution")
    text = workflow.input("text")
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
) -> WorkflowOrchestrator:
    plan = compile_workflow(
        workflow,
        DeploymentSpec.inline(**{stage_id: stage_id for stage_id in runners}),
    )
    return await WorkflowOrchestrator.bind(plan, inline_runners=runners)


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
        compile_workflow(_workflow(), DeploymentSpec.inline(encoder="encoder"))

    wrong = SimpleNamespace(contract=CLASSIFIER, run=_Generator(object()).run)
    with pytest.raises(WorkflowValidationError, match="does not match"):
        await WorkflowOrchestrator.bind(
            compile_workflow(
                _workflow(),
                DeploymentSpec.inline(
                    encoder="encoder",
                    classifier="classifier",
                    generator="generator",
                ),
            ),
            inline_runners={
                "encoder": _Encoder(object()),
                "classifier": _Classifier(object()),
                "generator": wrong,
            },
        )


async def test_runtime_accepts_opaque_values_and_rejects_bad_outputs():
    embedding = object()
    plan = await _compile_local(
        _workflow(),
        encoder=_Encoder(embedding),
        classifier=_Classifier(embedding),
        generator=_Generator(embedding),
    )
    assert await plan.run({"text": object()}) == {
        "scores": {"class-a": 0.75, "class-b": 0.25},
        "text": "generated",
    }
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
