# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.experimental.workflow import (
    DeploymentSpec,
    GenerateEndpointBinding,
    StageContext,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowExecutionError,
    WorkflowValidationError,
    compile_workflow,
)
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.generate import GenerateEndpointInvoker, collect_generation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


GENERATOR = StageContract(
    id="generator",
    inputs={"request": ValueSpec(type="json")},
    outputs={"completion": ValueSpec(type="json")},
)


def _workflow(generator_contract: StageContract = GENERATOR) -> Workflow:
    workflow = Workflow("request-generator")
    request = workflow.input("request", ValueSpec(type="json"))
    generator = workflow.stage(
        "generator",
        generator_contract,
        request=request,
    )
    workflow.output("completion", generator.completion)
    return workflow


def test_generate_binding_compiles_for_request_only_contract() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec(
            {"generator": GenerateEndpointBinding("models.decoder.generate")}
        ),
    )

    assert plan.remote
    assert plan.bindings == {
        "generator": GenerateEndpointBinding("models.decoder.generate")
    }


def test_generate_binding_rejects_a_non_generate_stage_contract() -> None:
    incompatible = StageContract(
        id="generator",
        inputs=GENERATOR.inputs,
        outputs={"completion": ValueSpec(type="text")},
    )

    with pytest.raises(WorkflowValidationError, match="stage output"):
        compile_workflow(
            _workflow(incompatible),
            DeploymentSpec(
                {"generator": GenerateEndpointBinding("models.decoder.generate")}
            ),
        )


class _Client:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = responses
        self.request: Mapping[str, Any] | None = None
        self.context: Any = None
        self.stream_closed = False

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(
        self, request: Mapping[str, Any], *, annotated: bool, context: Any = None
    ) -> Any:
        assert annotated is False
        self.request = request
        self.context = context

        async def stream():
            try:
                for response in self.responses:
                    yield response
            finally:
                self.stream_closed = True

        return stream()


class _Endpoint:
    def __init__(self, client: _Client) -> None:
        self._client = client

    async def client(self) -> _Client:
        return self._client


class _Runtime:
    def __init__(self, clients: Mapping[str, _Client]) -> None:
        self._clients = clients

    def endpoint(self, endpoint_id: str) -> _Endpoint:
        return _Endpoint(self._clients[endpoint_id])


def _context(request_context: Any = None) -> StageContext:
    return StageContext(
        workflow_name="request-generator",
        stage_id="generator",
        attempt_id="request-1",
        invocation_id="request-1:generator",
        deadline=None,
        _cancelled=asyncio.Event(),
        request_context=request_context,
    )


class _TransportContext:
    def __init__(self) -> None:
        self.stopped = False

    def stop_generating(self) -> None:
        self.stopped = True


class _ParentContext:
    def __init__(self) -> None:
        self.child = _TransportContext()

    def detached(self, context_id: str) -> _TransportContext:
        assert context_id == "request-1:generator"
        return self.child


def _request() -> dict[str, Any]:
    return {
        "token_ids": [1, 2],
        "sampling_options": {"n": 1},
        "output_options": {},
        "multi_modal_data": {"image_url": [{"Url": "data:image/jpeg;base64,AA=="}]},
        "multi_modal_uuids": ["image-1"],
        "mm_processor_kwargs": {"max_pixels": 1024},
        "mm_routing_info": {"mm_hashes": ["hash-1"]},
    }


async def test_generate_invoker_forwards_multimodal_request_unchanged() -> None:
    transport = _Client(
        [
            {"token_ids": [7], "index": 0},
            {"token_ids": [8, 9], "index": 0, "finish_reason": "stop"},
        ]
    )
    request = _request()

    stream = await GenerateEndpointInvoker(transport).open(
        "generator",
        {"request": request},
        _context(),
    )
    completion = await collect_generation(stream, "generator")
    await stream.aclose()

    assert completion == {
        "token_ids": [7, 8, 9],
        "index": 0,
        "finish_reason": "stop",
    }
    assert transport.request == request


async def test_generate_invoker_accepts_null_n_as_the_frontend_default() -> None:
    transport = _Client([{"token_ids": [42], "index": 0, "finish_reason": "stop"}])

    result = await GenerateEndpointInvoker(transport).run(
        "generator",
        GENERATOR,
        {
            "request": {
                "token_ids": [1, 2],
                "sampling_options": {"n": None},
                "output_options": {},
            }
        },
        _context(),
    )

    assert result["completion"]["token_ids"] == [42]


async def test_generate_invoker_cancels_owned_stream_on_collection_error() -> None:
    transport = _Client(
        [
            {"token_ids": [42], "index": 0, "finish_reason": "stop"},
            {"token_ids": [43], "index": 0},
        ]
    )
    parent = _ParentContext()

    with pytest.raises(WorkflowExecutionError, match="after terminal"):
        await GenerateEndpointInvoker(transport).run(
            "generator",
            GENERATOR,
            {"request": {"token_ids": [1], "output_options": {}}},
            _context(parent),
        )

    assert parent.child.stopped
    assert transport.stream_closed


async def test_dispatcher_binds_generate_protocol_for_stock_endpoint() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec(
            {"generator": GenerateEndpointBinding("models.decoder.generate")}
        ),
    )
    generator_client = _Client(
        [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]
    )
    dispatcher = await StageDispatcher.bind(
        plan,
        runtime=_Runtime({"models.decoder.generate": generator_client}),
    )
    request = _request()

    result = await dispatcher.call(
        "generator",
        GENERATOR,
        {"request": request},
        _context(),
    )

    assert result["completion"]["token_ids"] == [42]
    assert generator_client.request == request


@pytest.mark.parametrize(
    "request_value, message",
    [
        ({"sampling_options": {"n": 2}}, "requires n=1"),
        ({"output_options": {"logprobs": 0}}, "does not support logprobs"),
    ],
)
async def test_generate_invoker_rejects_unsupported_frontend_options(
    request_value: Mapping[str, Any], message: str
) -> None:
    with pytest.raises(WorkflowExecutionError, match=message):
        await GenerateEndpointInvoker(_Client([])).run(
            "generator",
            GENERATOR,
            {"request": request_value},
            _context(),
        )
