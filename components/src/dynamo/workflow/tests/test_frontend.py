# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowExecutor,
    WorkflowFrontendApplication,
    WorkflowTokenEngine,
    compile_workflow,
    load_workflow_frontend_application,
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


async def _application(**overrides) -> WorkflowFrontendApplication:
    workflow = Workflow("frontend-workflow")
    request = workflow.input("request", type="json")
    stage = workflow.stage("generate", CONTRACT, request=request)
    workflow.output("chunk", stage.chunk)
    executor = await WorkflowExecutor.bind(
        compile_workflow(workflow, DeploymentSpec.local(generate="generate")),
        local_runners={"generate": _Runner()},
    )
    values = {"executor": executor, "model_path": "org/model"}
    values.update(overrides)
    return WorkflowFrontendApplication(**values)


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


async def test_token_engine_adapts_preprocessed_request_to_one_workflow_chunk() -> None:
    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(await _application()).generate(
            {"token_ids": [41]}, _Context()
        )
    ]

    assert chunks == [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]


async def test_token_engine_accepts_future_shaped_rust_cancellation() -> None:
    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(await _application()).generate(
            {"token_ids": [41]}, _FutureContext()
        )
    ]

    assert chunks == [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]


async def test_frontend_application_supports_explicit_boundary_adapters() -> None:
    application = await _application(
        request_adapter=lambda request: {
            "request": {"token_ids": [request["token_ids"][-1]]}
        },
        result_adapter=lambda result: {
            **result["chunk"],
            "engine_data": {"source": "workflow"},
        },
    )

    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(application).generate(
            {"token_ids": [1, 8]}, _Context()
        )
    ]

    assert chunks[0]["token_ids"] == [9]
    assert chunks[0]["engine_data"] == {"source": "workflow"}


async def test_frontend_application_accepts_a_custom_chat_template_path() -> None:
    application = await _application(
        custom_template_path=Path("templates/vision.jinja")
    )

    assert application.custom_template_path == Path("templates/vision.jinja")

    with pytest.raises(TypeError, match="pathlib.Path"):
        await _application(custom_template_path="templates/vision.jinja")


async def test_loader_invokes_trusted_async_provider_with_runtime_and_config(
    monkeypatch,
) -> None:
    application = await _application(model_name="served-workflow")
    module = ModuleType("test_workflow_provider")
    seen = []

    async def create(runtime, config):
        seen.append((runtime, config))
        return application

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = object()
    config = SimpleNamespace()

    loaded = await load_workflow_frontend_application(
        "test_workflow_provider:create", runtime, config
    )

    assert loaded is application
    assert seen == [(runtime, config)]


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

    workflow = Workflow("frontend-cancellation")
    request = workflow.input("request", type="json")
    stage = workflow.stage("generate", CONTRACT, request=request)
    workflow.output("chunk", stage.chunk)
    executor = await WorkflowExecutor.bind(
        compile_workflow(workflow, DeploymentSpec.local(generate="generate")),
        local_runners={"generate": BlockingRunner()},
    )
    context = _Context()

    async def stop():
        await started.wait()
        context.stopped.set()

    stop_task = asyncio.create_task(stop())
    chunks = [
        chunk
        async for chunk in WorkflowTokenEngine(
            WorkflowFrontendApplication(executor, "org/model")
        ).generate({"token_ids": [1]}, context)
    ]
    await stop_task

    assert chunks == []
    assert cancelled.is_set()
