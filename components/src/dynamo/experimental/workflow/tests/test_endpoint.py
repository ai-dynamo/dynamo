# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dynamo.experimental.workflow import (
    DeploymentSpec,
    StageContract,
    Workflow,
    WorkflowEndpointHandler,
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


CONTRACT = StageContract(
    id="token-stage",
    inputs={"request"},
    outputs={"chunk"},
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
    workflow = Workflow("endpoint-workflow")
    request = workflow.input("request")
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


@pytest.mark.parametrize("context_type", [_Context, _FutureContext])
async def test_endpoint_handler_runs_fixed_request_and_chunk_abi(
    context_type,
) -> None:
    context = context_type()
    chunks = [
        chunk
        async for chunk in WorkflowEndpointHandler(await _orchestrator()).generate(
            {"token_ids": [41]}, context
        )
    ]

    assert chunks == [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]


async def test_endpoint_handler_rejects_noncanonical_workflow_abi() -> None:
    workflow = Workflow("wrong-endpoint-abi")
    request = workflow.input("payload")
    stage = workflow.stage("generate", CONTRACT, request=request)
    workflow.output("chunk", stage.chunk)
    orchestrator = await WorkflowOrchestrator.bind(
        compile_workflow(workflow), inline_runners={"generate": _Runner()}
    )

    with pytest.raises(WorkflowValidationError, match="'request' input"):
        WorkflowEndpointHandler(orchestrator)


async def test_endpoint_handler_cancels_workflow_when_context_stops() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingRunner:
        contract = CONTRACT

        async def run(self, inputs, context):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    context = _Context()

    async def stop():
        await started.wait()
        context.stopped.set()

    stop_task = asyncio.create_task(stop())
    chunks = [
        chunk
        async for chunk in WorkflowEndpointHandler(
            await _orchestrator(BlockingRunner())
        ).generate({"token_ids": [1]}, context)
    ]
    await stop_task

    assert chunks == []
    assert cancelled.is_set()
