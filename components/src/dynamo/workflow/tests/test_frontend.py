# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import ModuleType

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowOrchestrator,
    WorkflowTokenEngine,
    WorkflowValidationError,
    compile_workflow,
    load_workflow_orchestrator,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


CONTRACT = StageContract(
    id="token-stage",
    inputs={"request": ValueSpec(type="json")},
    outputs={"chunk": ValueSpec(type="json")},
)


class _Runner:
    contract = CONTRACT

    async def run(self, inputs, context):
        return {
            "chunk": {
                "token_ids": [inputs["request"]["token_ids"][0] + 1],
                "index": 0,
                "finish_reason": "stop",
            }
        }


async def _orchestrator(runner=None) -> WorkflowOrchestrator:
    workflow = Workflow("frontend-workflow")
    request = workflow.input("request", ValueSpec(type="json"))
    stage = workflow.stage("generate", CONTRACT, request=request)
    workflow.output("chunk", stage.chunk)
    return await WorkflowOrchestrator.bind(
        compile_workflow(workflow, DeploymentSpec.inline(generate="generate")),
        inline_runners={"generate": runner or _Runner()},
    )


class _Context:
    def __init__(self) -> None:
        self.stopped = asyncio.Event()

    def id(self):
        return "request-7"

    async def async_killed_or_stopped(self):
        await self.stopped.wait()
        return True


class _FutureContext(_Context):
    def async_killed_or_stopped(self):
        return asyncio.get_running_loop().create_future()


@pytest.mark.parametrize("context", [_Context(), _FutureContext()])
async def test_token_engine_runs_fixed_request_and_chunk_abi(context) -> None:
    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(await _orchestrator()).generate(
            {"token_ids": [41]}, context
        )
    ]

    assert chunks == [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]


async def test_loader_invokes_trusted_provider_with_runtime_only(monkeypatch) -> None:
    orchestrator = await _orchestrator()
    module = ModuleType("test_workflow_provider")
    seen = []

    async def create(runtime):
        seen.append(runtime)
        return orchestrator

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = object()

    loaded = await load_workflow_orchestrator("test_workflow_provider:create", runtime)

    assert loaded is orchestrator
    assert seen == [runtime]


async def test_frontend_rejects_noncanonical_workflow_abi() -> None:
    workflow = Workflow("wrong-frontend-abi")
    request = workflow.input("payload", ValueSpec(type="json"))
    stage = workflow.stage("generate", CONTRACT, request=request)
    workflow.output("chunk", stage.chunk)
    orchestrator = await WorkflowOrchestrator.bind(
        compile_workflow(workflow), inline_runners={"generate": _Runner()}
    )

    with pytest.raises(WorkflowValidationError, match="request: json"):
        WorkflowTokenEngine(orchestrator)


async def test_token_engine_cancels_workflow_when_frontend_context_stops() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingRunner:
        contract = CONTRACT

        async def run(self, inputs, context):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert context.cancelled
                cancelled.set()
                raise

    context = _Context()

    async def stop():
        await started.wait()
        context.stopped.set()

    stop_task = asyncio.create_task(stop())
    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(
            await _orchestrator(BlockingRunner())
        ).generate({"token_ids": [1]}, context)
    ]
    await stop_task

    assert chunks == []
    assert cancelled.is_set()
