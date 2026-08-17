# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("torch", reason="the vLLM workflow components require PyTorch")

from dynamo.workflow import (  # noqa: E402
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    WorkflowOrchestrator,
)
from dynamo.workflow.remote import StageResponseEnvelope  # noqa: E402
from examples.custom_backend.user_ensemble.remote import (  # noqa: E402
    classifier_worker as classifier_worker_module,
)
from examples.custom_backend.user_ensemble.remote import (  # noqa: E402
    response_worker as response_worker_module,
)
from examples.custom_backend.user_ensemble.remote.bindings import (  # noqa: E402
    CLASSIFIER_ENDPOINT,
    GENERATOR_ENDPOINT,
    RESPONSE_ENDPOINT,
    RESPONSE_PLACEMENT_ENV,
    compile_remote_workflow,
)
from examples.custom_backend.user_ensemble.remote.provider import (  # noqa: E402
    provide_workflow,
)
from examples.custom_backend.user_ensemble.stages import (  # noqa: E402
    EnsembleResponseStage,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeEndpoint:
    def __init__(self) -> None:
        self.handler: Any = None

    async def serve_endpoint(self, handler: Any) -> None:
        self.handler = handler


class _FakeRuntime:
    def __init__(self) -> None:
        self.endpoint_ids: list[str] = []
        self.created_endpoint = _FakeEndpoint()

    def endpoint(self, endpoint_id: str) -> _FakeEndpoint:
        self.endpoint_ids.append(endpoint_id)
        return self.created_endpoint


class _Barrier:
    def __init__(self) -> None:
        self.arrivals: set[str] = set()
        self.ready = asyncio.Event()

    async def arrive(self, name: str) -> None:
        self.arrivals.add(name)
        if self.arrivals == {"classifier", "generator"}:
            self.ready.set()
        await asyncio.wait_for(self.ready.wait(), timeout=1.0)


class _Client:
    def __init__(self, name: str, barrier: _Barrier) -> None:
        self.name = name
        self.barrier = barrier
        self.request: Mapping[str, Any] | None = None

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> Any:
        del context
        assert annotated is False
        self.request = request
        if self.name in {"classifier", "generator"}:
            await self.barrier.arrive(self.name)

        async def stream():
            if self.name == "generator":
                yield {"token_ids": [4, 2], "index": 0, "finish_reason": "stop"}
                return
            if self.name == "response":
                completion = dict(request["inputs"]["completion"])
                completion["engine_data"] = {
                    "ensemble": {"classifier_scores": dict(request["inputs"]["scores"])}
                }
                outputs = {"chunk": completion}
            else:
                outputs = {"scores": {"relevant": 0.75, "not_relevant": 0.25}}
            yield StageResponseEnvelope(
                stage_id=request["stage"],
                contract_id=request["contract"],
                attempt_id=request["attempt"],
                invocation_id=request["invocation"],
                outputs=outputs,
            ).to_dict()

        return stream()


class _ClientEndpoint:
    def __init__(self, client: _Client) -> None:
        self._client = client

    async def client(self) -> _Client:
        return self._client


class _ClientRuntime:
    def __init__(self, clients: Mapping[str, _Client]) -> None:
        self._clients = clients

    def endpoint(self, endpoint_id: str) -> _ClientEndpoint:
        return _ClientEndpoint(self._clients[endpoint_id])


def test_remote_plan_uses_only_inline_json_edges() -> None:
    plan = compile_remote_workflow()

    assert type(plan.bindings["classifier"]) is RemoteBinding
    assert isinstance(plan.bindings["generator"], GenerateEndpointBinding)
    assert plan.bindings["response"] == InlineBinding("response")
    assert plan.bindings["classifier"].endpoint_id == CLASSIFIER_ENDPOINT
    assert plan.bindings["generator"].endpoint_id == GENERATOR_ENDPOINT
    assert {edge.transfer_id: edge.carrier for edge in plan.edges} == {
        "classifier.request": "inline",
        "generator.request": "inline",
        "response.completion": "inline",
        "response.scores": "inline",
    }


def test_response_stage_can_bind_to_remote_endpoint() -> None:
    plan = compile_remote_workflow(response_placement="remote")

    assert plan.bindings["response"] == RemoteBinding(RESPONSE_ENDPOINT)
    assert {edge.transfer_id: edge.carrier for edge in plan.edges} == {
        "classifier.request": "inline",
        "generator.request": "inline",
        "response.completion": "inline",
        "response.scores": "inline",
    }


def test_response_placement_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="response_placement"):
        compile_remote_workflow(response_placement="elsewhere")  # type: ignore[arg-type]


async def test_frontend_provider_binds_remote_plan_and_inline_response() -> None:
    orchestrator = object.__new__(WorkflowOrchestrator)
    bind = AsyncMock(return_value=orchestrator)
    runtime = object()

    with patch.dict(os.environ, {RESPONSE_PLACEMENT_ENV: "inline"}), patch(
        "examples.custom_backend.user_ensemble.remote.provider.WorkflowOrchestrator.bind",
        bind,
    ):
        provided = await provide_workflow(runtime)

    assert bind.await_args.args == (compile_remote_workflow(),)
    assert bind.await_args.kwargs["runtime"] is runtime
    response = bind.await_args.kwargs["inline_runners"]["response"]
    assert response.contract.id == "ensemble-response"
    assert provided is orchestrator


async def test_frontend_provider_binds_remote_response_without_inline_runner() -> None:
    orchestrator = object.__new__(WorkflowOrchestrator)
    bind = AsyncMock(return_value=orchestrator)
    runtime = object()

    with patch.dict(os.environ, {RESPONSE_PLACEMENT_ENV: "remote"}), patch(
        "examples.custom_backend.user_ensemble.remote.provider.WorkflowOrchestrator.bind",
        bind,
    ):
        provided = await provide_workflow(runtime)

    assert bind.await_args.args == (
        compile_remote_workflow(response_placement="remote"),
    )
    assert bind.await_args.kwargs == {
        "runtime": runtime,
        "inline_runners": {},
    }
    assert provided is orchestrator


async def test_classifier_worker_uses_json_remote_stage_protocol() -> None:
    runtime = _FakeRuntime()

    await classifier_worker_module.classifier_worker.__wrapped__(runtime)

    assert runtime.endpoint_ids == [CLASSIFIER_ENDPOINT]
    assert runtime.created_endpoint.handler is not None


async def test_response_worker_uses_json_remote_stage_protocol() -> None:
    runtime = _FakeRuntime()

    await response_worker_module.response_worker.__wrapped__(runtime)

    assert runtime.endpoint_ids == [RESPONSE_ENDPOINT]
    assert runtime.created_endpoint.handler is not None


async def test_orchestrator_fans_out_request_concurrently_and_joins() -> None:
    barrier = _Barrier()
    classifier = _Client("classifier", barrier)
    generator = _Client("generator", barrier)
    orchestrator = await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=_ClientRuntime(
            {
                CLASSIFIER_ENDPOINT: classifier,
                GENERATOR_ENDPOINT: generator,
            }
        ),
        inline_runners={"response": EnsembleResponseStage()},
    )
    request = {
        "token_ids": [1, 2],
        "output_options": {},
        "multi_modal_data": {"image_url": [{"Url": "image"}]},
    }

    result = await orchestrator.run({"request": request})

    assert barrier.arrivals == {"classifier", "generator"}
    assert generator.request == request
    assert classifier.request is not None
    assert classifier.request["inputs"]["request"] == request
    assert result["chunk"]["token_ids"] == [4, 2]
    assert result["chunk"]["engine_data"]["ensemble"]["classifier_scores"] == {
        "relevant": 0.75,
        "not_relevant": 0.25,
    }


async def test_orchestrator_can_join_through_remote_response() -> None:
    barrier = _Barrier()
    classifier = _Client("classifier", barrier)
    generator = _Client("generator", barrier)
    response = _Client("response", barrier)
    orchestrator = await WorkflowOrchestrator.bind(
        compile_remote_workflow(response_placement="remote"),
        runtime=_ClientRuntime(
            {
                CLASSIFIER_ENDPOINT: classifier,
                GENERATOR_ENDPOINT: generator,
                RESPONSE_ENDPOINT: response,
            }
        ),
    )
    request = {
        "token_ids": [1, 2],
        "output_options": {},
        "multi_modal_data": {"image_url": [{"Url": "image"}]},
    }

    result = await orchestrator.run({"request": request})

    assert barrier.arrivals == {"classifier", "generator"}
    assert response.request is not None
    assert response.request["inputs"]["completion"]["token_ids"] == [4, 2]
    assert response.request["inputs"]["scores"] == {
        "relevant": 0.75,
        "not_relevant": 0.25,
    }
    assert result["chunk"]["engine_data"]["ensemble"]["classifier_scores"] == {
        "relevant": 0.75,
        "not_relevant": 0.25,
    }
