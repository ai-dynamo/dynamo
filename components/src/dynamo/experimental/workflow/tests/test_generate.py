# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.common.external_encoder import ExternalEncoderResult
from dynamo.experimental.workflow import (
    DeploymentSpec,
    GenerateEndpointBinding,
    NixlTensorRef,
    RemoteBinding,
    StageContext,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowExecutionError,
    WorkflowValidationError,
    compile_workflow,
)
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.generate import GenerateEndpointClient

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


TENSOR = ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))
ENCODER = StageContract(
    id="encoder",
    inputs={"request": ValueSpec(type="json")},
    outputs={"encoder_features": TENSOR, "encoder_metadata": ValueSpec(type="json")},
)
GENERATOR = StageContract(
    id="generator",
    inputs={
        "request": ValueSpec(type="json"),
        "encoder_features": TENSOR,
        "encoder_metadata": ValueSpec(type="json"),
    },
    outputs={"chunk": ValueSpec(type="json")},
)


def _workflow(generator_contract: StageContract = GENERATOR) -> Workflow:
    workflow = Workflow("external-encoder")
    request = workflow.input("request", ValueSpec(type="json"))
    encoder = workflow.stage("encoder", ENCODER, request=request)
    generator = workflow.stage(
        "generator",
        generator_contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=encoder.encoder_metadata,
    )
    workflow.output("chunk", generator.chunk)
    return workflow


def test_generate_binding_compiles_with_remote_stage_protocols() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    "workflows.encoder.generate", tensor_carrier="nixl"
                ),
                "generator": GenerateEndpointBinding("models.decoder.generate"),
            }
        ),
    )

    assert plan.remote
    assert {edge.transfer_id: edge.carrier for edge in plan.edges} == {
        "encoder.request": "inline",
        "generator.request": "inline",
        "generator.encoder_features": "nixl",
        "generator.encoder_metadata": "inline",
    }


def test_generate_binding_rejects_a_non_generate_stage_contract() -> None:
    incompatible = StageContract(
        id="generator",
        inputs=GENERATOR.inputs,
        outputs={"chunk": ValueSpec(type="text")},
    )
    with pytest.raises(WorkflowValidationError, match="stage output"):
        compile_workflow(
            _workflow(incompatible),
            DeploymentSpec(
                {
                    "encoder": RemoteBinding(
                        "workflows.encoder.generate", tensor_carrier="nixl"
                    ),
                    "generator": GenerateEndpointBinding("models.decoder.generate"),
                }
            ),
        )


def _reference() -> NixlTensorRef:
    return NixlTensorRef(
        transfer_id="generator.encoder_features",
        lease_id="lease-1",
        shape=(2, 4),
        dtype="float32",
        device="cuda:0",
        rdma_metadata={"opaque": "read"},
    )


def test_external_encoder_result_round_trips_strictly() -> None:
    value = ExternalEncoderResult.from_parts(
        _reference().to_dict(), {"row_splits": [0, 2], "image_token_id": 151655}
    )

    assert ExternalEncoderResult.from_dict(value.to_dict()) == value
    bad = value.to_dict()
    bad["row_splits"] = [0, 1]
    with pytest.raises(ValueError, match="packed feature rows"):
        ExternalEncoderResult.from_dict(bad)


class _Client:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = responses
        self.request: Mapping[str, Any] | None = None
        self.context: Any = None

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(
        self, request: Mapping[str, Any], *, annotated: bool, context: Any = None
    ) -> Any:
        assert annotated is False
        self.request = request
        self.context = context

        async def stream():
            for response in self.responses:
                yield response

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


def _context() -> StageContext:
    return StageContext(
        workflow_name="external-encoder",
        stage_id="generator",
        attempt_id="request-1",
        invocation_id="request-1:generator",
        deadline=None,
        _cancelled=asyncio.Event(),
    )


async def test_generate_client_packs_encoder_result_and_folds_token_deltas() -> None:
    transport = _Client(
        [
            {"token_ids": [7], "index": 0},
            {"token_ids": [8, 9], "index": 0, "finish_reason": "stop"},
        ]
    )
    result = await GenerateEndpointClient(transport).run(
        "generator",
        GENERATOR,
        {
            "request": {
                "token_ids": [1, 2],
                "sampling_options": {"n": 1},
                "output_options": {},
                "multi_modal_data": {"image_url": [{"Url": "ignored"}]},
                "multi_modal_uuids": ["ignored-uuid"],
                "mm_processor_kwargs": {"max_pixels": 1024},
                "mm_routing_info": {"mm_hashes": ["hash"]},
            },
            "encoder_features": _reference().to_dict(),
            "encoder_metadata": {
                "row_splits": [0, 2],
                "image_token_id": 151655,
            },
        },
        _context(),
        {},
    )

    assert result == {
        "chunk": {"token_ids": [7, 8, 9], "index": 0, "finish_reason": "stop"}
    }
    assert transport.request is not None
    encoder_result = ExternalEncoderResult.from_dict(
        transport.request["encoder_result"]
    )
    assert encoder_result.features == _reference().to_dict()
    assert "multi_modal_data" not in transport.request
    assert "multi_modal_uuids" not in transport.request
    assert "mm_processor_kwargs" not in transport.request
    assert "mm_routing_info" not in transport.request


async def test_generate_client_accepts_null_n_as_the_frontend_default() -> None:
    transport = _Client([{"token_ids": [42], "index": 0, "finish_reason": "stop"}])

    result = await GenerateEndpointClient(transport).run(
        "generator",
        GENERATOR,
        {
            "request": {
                "token_ids": [1, 2],
                "sampling_options": {"n": None},
                "output_options": {},
            },
            "encoder_features": _reference().to_dict(),
            "encoder_metadata": {
                "row_splits": [0, 2],
                "image_token_id": 151655,
            },
        },
        _context(),
        {},
    )

    assert result["chunk"]["token_ids"] == [42]


async def test_dispatcher_binds_generate_protocol_for_stock_endpoint() -> None:
    plan = compile_workflow(
        _workflow(),
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    "workflows.encoder.generate", tensor_carrier="nixl"
                ),
                "generator": GenerateEndpointBinding("models.decoder.generate"),
            }
        ),
    )
    generator_client = _Client(
        [{"token_ids": [42], "index": 0, "finish_reason": "stop"}]
    )
    dispatcher = await StageDispatcher.bind(
        plan,
        runtime=_Runtime(
            {
                "workflows.encoder.generate": _Client([]),
                "models.decoder.generate": generator_client,
            }
        ),
    )

    result = await dispatcher.call(
        "generator",
        GENERATOR,
        {
            "request": {"token_ids": [1], "output_options": {}},
            "encoder_features": _reference().to_dict(),
            "encoder_metadata": {
                "row_splits": [0, 2],
                "image_token_id": 151655,
            },
        },
        _context(),
    )

    assert result["chunk"]["token_ids"] == [42]
    assert generator_client.request is not None
    assert "encoder_result" in generator_client.request


@pytest.mark.parametrize(
    "request_value, message",
    [
        ({"sampling_options": {"n": 2}}, "requires n=1"),
        ({"output_options": {"logprobs": 0}}, "does not support logprobs"),
    ],
)
async def test_generate_client_rejects_unsupported_frontend_options(
    request_value: Mapping[str, Any], message: str
) -> None:
    with pytest.raises(WorkflowExecutionError, match=message):
        await GenerateEndpointClient(_Client([])).run(
            "generator",
            GENERATOR,
            {
                "request": request_value,
                "encoder_features": _reference().to_dict(),
                "encoder_metadata": {
                    "row_splits": [0, 2],
                    "image_token_id": 151655,
                },
            },
            _context(),
            {},
        )
