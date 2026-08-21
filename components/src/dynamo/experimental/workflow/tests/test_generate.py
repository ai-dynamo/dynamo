# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.experimental.workflow import (
    DeploymentSpec,
    GenerateEndpointBinding,
    NixlTensorFanout,
    NixlTensorRef,
    RemoteBinding,
    StageContext,
    StageContract,
    Workflow,
    WorkflowExecutionError,
    WorkflowOrchestrator,
    WorkflowValidationError,
    compile_workflow,
)
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.generate import GenerateEndpointInvoker, collect_generation
from dynamo.experimental.workflow.remote import NixlCarriedValue

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


GENERATOR = StageContract(
    id="generator",
    inputs={"request"},
    outputs={"completion"},
)


def _workflow(generator_contract: StageContract = GENERATOR) -> Workflow:
    workflow = Workflow("request-generator")
    request = workflow.input("request")
    generator = workflow.stage(
        "generator",
        generator_contract,
        request=request,
    )
    workflow.output(
        "completion", generator.output(sorted(generator_contract.outputs)[0])
    )
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
        outputs={"chunk"},
    )

    with pytest.raises(WorkflowValidationError, match="stage output"):
        compile_workflow(
            _workflow(incompatible),
            DeploymentSpec(
                {"generator": GenerateEndpointBinding("models.decoder.generate")}
            ),
        )


def test_generate_binding_accepts_external_encoder_ports_with_nixl() -> None:
    external = StageContract(
        id="external-generator",
        inputs={"request", "encoder_features", "encoder_metadata"},
        outputs={"completion"},
    )
    workflow = Workflow("external-generator")
    request = workflow.input("request")
    features = workflow.input("encoder_features")
    metadata = workflow.input("encoder_metadata")
    completion = workflow.stage(
        "generator",
        external,
        request=request,
        encoder_features=features,
        encoder_metadata=metadata,
    )
    workflow.output("completion", completion.completion)

    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "generator": GenerateEndpointBinding(
                    "models.decoder.generate", tensor_carrier="nixl"
                )
            }
        ),
    )

    assert plan.bindings["generator"].tensor_carrier == "nixl"


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


async def test_generate_invoker_adapts_nixl_features_for_stock_vllm() -> None:
    transport = _Client([{"token_ids": [7], "index": 0, "finish_reason": "stop"}])
    request = _request()
    reference = NixlTensorRef(
        transfer_id="generator.encoder_features",
        lease_id="lease-1",
        shape=(3, 4),
        dtype="float16",
        device="cpu",
        rdma_metadata={"opaque": True},
    ).to_dict()

    result = await GenerateEndpointInvoker(transport).run(
        "generator",
        StageContract(
            id="external-generator",
            inputs={"request", "encoder_features", "encoder_metadata"},
            outputs={"completion"},
        ),
        {
            "request": request,
            "encoder_features": NixlCarriedValue(reference),
            "encoder_metadata": {"row_splits": [0, 3], "image_token_id": 99},
        },
        _context(),
    )

    assert result["completion"]["token_ids"] == [7]
    assert transport.request["encoder_result"]["features"] == reference
    assert "multi_modal_data" not in transport.request
    assert "multi_modal_uuids" not in transport.request


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


async def test_dispatcher_passes_remote_tensor_reference_to_stock_vllm() -> None:
    encoder_contract = StageContract(
        id="encoder", inputs={"request"}, outputs={"encoder_features"}
    )
    generator_contract = StageContract(
        id="external-generator",
        inputs={"request", "encoder_features", "encoder_metadata"},
        outputs={"completion"},
    )
    workflow = Workflow("external-encoder-generate")
    request = workflow.input("request")
    metadata = workflow.input("encoder_metadata")
    encoder = workflow.stage("encoder", encoder_contract, request=request)
    generator = workflow.stage(
        "generator",
        generator_contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=metadata,
    )
    workflow.output("completion", generator.completion)
    encoder_endpoint = "models.encoder.generate"
    generator_endpoint = "models.decoder.generate"
    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "encoder": RemoteBinding(encoder_endpoint, tensor_carrier="nixl"),
                "generator": GenerateEndpointBinding(
                    generator_endpoint, tensor_carrier="nixl"
                ),
            }
        ),
    )

    class EncoderInvoker:
        async def run(self, stage_id, contract, inputs, context, output_transfers):
            transfer_ids = output_transfers["encoder_features"]
            fanout = NixlTensorFanout(
                {
                    transfer_id: NixlTensorRef(
                        transfer_id=transfer_id,
                        lease_id="lease-1",
                        shape=(3, 4),
                        dtype="float16",
                        device="cpu",
                        rdma_metadata={"opaque": True},
                    )
                    for transfer_id in transfer_ids
                }
            )
            return {"encoder_features": NixlCarriedValue(fanout.to_dict())}

    generator_client = _Client(
        [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]
    )
    dispatcher = StageDispatcher(
        plan,
        {},
        {
            encoder_endpoint: EncoderInvoker(),
            generator_endpoint: GenerateEndpointInvoker(generator_client),
        },
    )
    orchestrator = WorkflowOrchestrator(plan, dispatcher)

    result = await orchestrator.run(
        {
            "request": _request(),
            "encoder_metadata": {"row_splits": [0, 3], "image_token_id": 99},
        }
    )

    assert result["completion"]["token_ids"] == [42]
    assert (
        generator_client.request["encoder_result"]["features"]["transfer_id"]
        == "generator.encoder_features"
    )


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
