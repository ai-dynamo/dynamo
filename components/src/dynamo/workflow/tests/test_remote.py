# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from dynamo.workflow import (
    DeploymentSpec,
    InlineBinding,
    NixlTensorFanout,
    NixlTensorRef,
    RemoteBinding,
    StageContext,
    StageContract,
    Workflow,
    WorkflowExecutionError,
    WorkflowOrchestrator,
    compile_workflow,
)
from dynamo.workflow.remote import (
    STAGE_REQUEST_SCHEMA,
    NixlCarriedValue,
    RemoteStageClient,
    RemoteStageServer,
    StageRequestEnvelope,
    StageResponseEnvelope,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


CONTRACT = StageContract(
    id="normalize",
    inputs={"text"},
    outputs={"normalized"},
)


def _context(request_context=None):
    return StageContext(
        workflow_name="remote-wire",
        stage_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        deadline=None,
        _cancelled=asyncio.Event(),
        request_context=request_context,
    )


def _request(inputs, *, carriers=None, transfers=None):
    return StageRequestEnvelope(
        inputs=inputs,
        input_carriers={} if carriers is None else carriers,
        output_transfers={} if transfers is None else transfers,
    ).to_dict()


def _response(outputs, *, carriers=None):
    return StageResponseEnvelope(
        outputs=outputs,
        output_carriers={} if carriers is None else carriers,
    ).to_dict()


def test_carrier_envelope_round_trip_is_strict_and_versioned() -> None:
    envelope = StageRequestEnvelope(
        inputs={"text": "HELLO"},
        input_carriers={},
        output_transfers={},
    )

    assert StageRequestEnvelope.from_dict(envelope.to_dict()) == envelope
    bad = envelope.to_dict()
    bad["extra"] = True
    with pytest.raises(WorkflowExecutionError, match="unknown fields"):
        StageRequestEnvelope.from_dict(bad)

    bad = envelope.to_dict()
    bad["schema"] = f"{STAGE_REQUEST_SCHEMA}.future"
    with pytest.raises(WorkflowExecutionError, match="unsupported.*schema"):
        StageRequestEnvelope.from_dict(bad)

    bad = envelope.to_dict()
    bad["version"] = 0
    with pytest.raises(WorkflowExecutionError, match="unsupported.*version"):
        StageRequestEnvelope.from_dict(bad)


class _Client:
    def __init__(self, responses) -> None:
        self.responses = responses
        self.request = None
        self.context = None

    async def round_robin(self, request, *, annotated, context=None):
        assert annotated is False
        self.request = request
        self.context = context

        async def stream():
            for response in self.responses:
                yield response

        return stream()


class _ChildContext:
    def __init__(self, context_id: str) -> None:
        self.context_id = context_id
        self._stopped = asyncio.Event()

    def stop_generating(self) -> None:
        self._stopped.set()

    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    def is_killed(self) -> bool:
        return False

    def id(self) -> str:
        return self.context_id

    async def async_killed_or_stopped(self) -> bool:
        await self._stopped.wait()
        return True


class _ParentContext:
    def __init__(self) -> None:
        self.children: list[_ChildContext] = []

    def detached(self, context_id: str) -> _ChildContext:
        child = _ChildContext(context_id)
        self.children.append(child)
        return child


async def test_remote_client_wraps_ordinary_values_for_the_endpoint() -> None:
    transport = _Client([_response({"normalized": "hello"})])

    result = await RemoteStageClient(transport).run(
        "normalize", CONTRACT, {"text": "HELLO"}, _context(), {}
    )

    assert result == {"normalized": "hello"}
    assert StageRequestEnvelope.from_dict(transport.request).inputs == {"text": "HELLO"}


async def test_remote_client_tags_only_internal_carried_values() -> None:
    reference = {"transfer_id": "normalize.text", "opaque": True}
    transport = _Client(
        [_response({"normalized": reference}, carriers={"normalized": "nixl"})]
    )

    result = await RemoteStageClient(transport).run(
        "normalize",
        CONTRACT,
        {"text": NixlCarriedValue(reference)},
        _context(),
        {},
    )

    request = StageRequestEnvelope.from_dict(transport.request)
    assert request.inputs == {"text": reference}
    assert request.input_carriers == {"text": "nixl"}
    assert result == {"normalized": NixlCarriedValue(reference)}


async def test_remote_client_creates_invocation_scoped_transport_context() -> None:
    transport = _Client([_response({"normalized": "hello"})])
    parent = _ParentContext()

    await RemoteStageClient(transport).run(
        "normalize",
        CONTRACT,
        {"text": "HELLO"},
        _context(request_context=parent),
        {},
    )

    assert [child.context_id for child in parent.children] == ["request-1:normalize"]
    assert transport.context is parent.children[0]


async def test_remote_client_enforces_one_response_mapping() -> None:
    with pytest.raises(WorkflowExecutionError, match="no response mapping"):
        await RemoteStageClient(_Client([])).run(
            "normalize", CONTRACT, {"text": "HELLO"}, _context(), {}
        )

    response = _response({"normalized": "hello"})
    parent = _ParentContext()
    with pytest.raises(WorkflowExecutionError, match="multiple response mappings"):
        await RemoteStageClient(_Client([response, response])).run(
            "normalize",
            CONTRACT,
            {"text": "HELLO"},
            _context(request_context=parent),
            {},
        )
    assert parent.children[0].is_stopped()


async def test_remote_client_adds_stage_context_to_transport_failures() -> None:
    class FailingClient:
        async def round_robin(self, request, *, annotated, context=None):
            raise TypeError("unsupported payload")

    with pytest.raises(
        WorkflowExecutionError,
        match="remote stage 'normalize'.*transport boundary",
    ) as error:
        await RemoteStageClient(FailingClient()).run(
            "normalize", CONTRACT, {"text": object()}, _context(), {}
        )

    assert isinstance(error.value.__cause__, TypeError)


class _Runner:
    contract = CONTRACT

    async def run(self, inputs, context):
        assert context.workflow_name is None
        assert context.stage_id == "normalize"
        assert context.attempt_id == "request-1:normalize"
        return {"normalized": inputs["text"].strip().lower()}


async def test_remote_server_runs_unary_stage_over_streaming_endpoint() -> None:
    context = _ChildContext("request-1:normalize")

    responses = [
        response
        async for response in RemoteStageServer("normalize", _Runner()).generate(
            _request({"text": " HELLO "}), context=context
        )
    ]

    assert dict(StageResponseEnvelope.from_dict(responses[0]).outputs) == {
        "normalized": "hello"
    }


async def test_ordinary_mapping_cannot_collide_with_carrier_metadata() -> None:
    value = {
        "schema": STAGE_REQUEST_SCHEMA,
        "version": 1,
        "input_carriers": {"value": "nixl"},
    }

    class OpaqueRunner:
        contract = StageContract(id="opaque", inputs={"value"}, outputs={"result"})

        async def run(self, inputs, context):
            return {"result": inputs["value"]}

    response = (
        await RemoteStageServer("opaque", OpaqueRunner())
        .generate(_request({"value": value}))
        .__anext__()
    )

    assert StageResponseEnvelope.from_dict(response).outputs["result"] == value


async def test_remote_server_cancels_runner_when_transport_stops() -> None:
    class BlockingRunner:
        contract = CONTRACT

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def run(self, inputs, context):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert context.cancelled
                self.cancelled.set()
                raise

    runner = BlockingRunner()
    context = _ChildContext("request-1:normalize")
    response = asyncio.create_task(
        RemoteStageServer("normalize", runner)
        .generate(_request({"text": "hello"}), context=context)
        .__anext__()
    )
    await runner.started.wait()
    context.stop_generating()

    with pytest.raises(asyncio.CancelledError):
        await response
    assert runner.cancelled.is_set()


class _TensorCarrier:
    def __init__(self) -> None:
        self.exports: list[str] = []

    def can_export(self, value):
        return isinstance(value, torch.Tensor)

    async def import_tensor(self, reference):
        assert reference == {"remote": "reference"}
        return torch.ones((2, 4), dtype=torch.float32)

    async def export_tensor(self, tensor, transfer_id):
        return (await self.export_tensor_fanout(tensor, (transfer_id,)))[transfer_id]

    async def export_tensor_fanout(self, tensor, transfer_ids):
        self.exports.extend(transfer_ids)
        return {
            transfer_id: NixlTensorRef(
                transfer_id=transfer_id,
                lease_id=f"lease-{transfer_id}",
                shape=tuple(tensor.shape),
                dtype="float32",
                device="cpu",
                rdma_metadata={"opaque": transfer_id},
            ).to_dict()
            for transfer_id in transfer_ids
        }


async def test_server_selects_nixl_for_top_level_tensor_at_runtime() -> None:
    class TensorRunner:
        contract = StageContract(
            id="tensor", inputs={"tensor"}, outputs={"tensor", "metadata"}
        )

        async def run(self, inputs, context):
            return {"tensor": inputs["tensor"] * 2, "metadata": {"ok": True}}

    carrier = _TensorCarrier()
    response = (
        await RemoteStageServer("tensor", TensorRunner(), carrier)
        .generate(
            _request(
                {"tensor": {"remote": "reference"}},
                carriers={"tensor": "nixl"},
                transfers={"tensor": ("classifier.tensor", "generator.tensor")},
            )
        )
        .__anext__()
    )
    envelope = StageResponseEnvelope.from_dict(response)
    fanout = NixlTensorFanout.from_dict(envelope.outputs["tensor"])

    assert set(fanout.transfers) == {"classifier.tensor", "generator.tensor"}
    assert envelope.output_carriers == {"tensor": "nixl"}
    assert envelope.outputs["metadata"] == {"ok": True}
    assert carrier.exports == ["classifier.tensor", "generator.tensor"]


async def test_tensor_output_requires_remote_nixl_consumers() -> None:
    class TensorRunner:
        contract = StageContract(id="tensor", inputs={"value"}, outputs={"tensor"})

        async def run(self, inputs, context):
            return {"tensor": torch.ones(1)}

    with pytest.raises(WorkflowExecutionError, match="no NIXL consumer transfers"):
        await RemoteStageServer("tensor", TensorRunner(), _TensorCarrier()).generate(
            _request({"value": "ordinary"})
        ).__anext__()


async def test_nested_tensor_is_not_implicitly_carried() -> None:
    nested = {"tensor": torch.ones(1)}

    class NestedRunner:
        contract = StageContract(id="nested", inputs={"value"}, outputs={"result"})

        async def run(self, inputs, context):
            return {"result": nested}

    response = (
        await RemoteStageServer("nested", NestedRunner(), _TensorCarrier())
        .generate(_request({"value": "ordinary"}))
        .__anext__()
    )
    envelope = StageResponseEnvelope.from_dict(response)

    assert envelope.outputs["result"] is nested
    assert envelope.output_carriers == {}


TEXT_ENCODER = StageContract(id="text-encoder", inputs={"text"}, outputs={"tokens"})
TEXT_GENERATOR = StageContract(id="text-generator", inputs={"tokens"}, outputs={"text"})
RESPONSE = StageContract(id="response", inputs={"text"}, outputs={"chunk"})


class _TextEncoder:
    contract = TEXT_ENCODER

    async def run(self, inputs: Mapping[str, Any], context: StageContext):
        return {"tokens": inputs["text"].lower().split()}


class _TextGenerator:
    contract = TEXT_GENERATOR

    async def run(self, inputs: Mapping[str, Any], context: StageContext):
        return {"text": " ".join(reversed(inputs["tokens"]))}


class _Response:
    contract = RESPONSE

    async def run(self, inputs: Mapping[str, Any], context: StageContext):
        return {"chunk": {"text": inputs["text"]}}


class _LoopbackClient:
    def __init__(self, server: RemoteStageServer) -> None:
        self._server = server

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(self, request, *, annotated, context=None):
        assert annotated is False
        return self._server.generate(request, context)


class _Endpoint:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def client(self) -> Any:
        return self._client


class _Runtime:
    def __init__(self, clients: Mapping[str, Any]) -> None:
        self._clients = clients

    def endpoint(self, endpoint_id: str) -> _Endpoint:
        return _Endpoint(self._clients[endpoint_id])


async def test_ordinary_values_work_on_nixl_capable_remote_bindings() -> None:
    workflow = Workflow("remote-text")
    text = workflow.input("text")
    encoder = workflow.stage("encoder", TEXT_ENCODER, text=text)
    generator = workflow.stage("generator", TEXT_GENERATOR, tokens=encoder.tokens)
    response = workflow.stage("response", RESPONSE, text=generator.text)
    workflow.output("chunk", response.chunk)
    endpoints = {
        "encoder": "workflows.encoder.generate",
        "generator": "workflows.generator.generate",
    }
    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "encoder": RemoteBinding(endpoints["encoder"], tensor_carrier="nixl"),
                "generator": RemoteBinding(
                    endpoints["generator"], tensor_carrier="nixl"
                ),
                "response": InlineBinding("response"),
            }
        ),
    )
    clients = {
        endpoints["encoder"]: _LoopbackClient(
            RemoteStageServer("encoder", _TextEncoder(), _TensorCarrier())
        ),
        endpoints["generator"]: _LoopbackClient(
            RemoteStageServer("generator", _TextGenerator(), _TensorCarrier())
        ),
    }
    orchestrator = await WorkflowOrchestrator.bind(
        plan,
        runtime=_Runtime(clients),
        inline_runners={"response": _Response()},
    )

    assert await orchestrator.run({"text": "Hello Dynamo"}) == {
        "chunk": {"text": "dynamo hello"}
    }


async def test_tensor_server_imports_input_and_exports_per_consumer() -> None:
    class TensorRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor"},
            outputs={"tensor"},
        )

        async def run(self, inputs, context):
            return {"tensor": inputs["tensor"] * 2}

    class Carrier:
        def __init__(self):
            self.exports = []
            self.released = []

        def can_export(self, value):
            return isinstance(value, torch.Tensor)

        async def import_tensor(self, reference):
            assert reference == {"remote": "reference"}
            return torch.ones((2, 4), dtype=torch.float32)

        async def export_tensor(self, tensor, transfer_id):
            return (await self.export_tensor_fanout(tensor, (transfer_id,)))[
                transfer_id
            ]

        async def export_tensor_fanout(self, tensor, transfer_ids):
            assert torch.equal(tensor, torch.full((2, 4), 2.0))
            self.exports.extend(transfer_ids)
            return {
                transfer_id: NixlTensorRef(
                    transfer_id=transfer_id,
                    lease_id=f"lease-{transfer_id}",
                    shape=tuple(tensor.shape),
                    dtype="float32",
                    device="cpu",
                    rdma_metadata={"opaque": transfer_id},
                ).to_dict()
                for transfer_id in transfer_ids
            }

        def release_imported_tensor(self, tensor):
            self.released.append(tensor)

    carrier = Carrier()
    request = StageRequestEnvelope(
        inputs={"tensor": {"remote": "reference"}},
        input_carriers={"tensor": "nixl"},
        output_transfers={"tensor": ("classifier.tensor", "generator.tensor")},
    )

    responses = [
        response
        async for response in RemoteStageServer(
            "tensor", TensorRunner(), carrier
        ).generate(request.to_dict())
    ]
    outputs = StageResponseEnvelope.from_dict(responses[0]).outputs
    fanout = NixlTensorFanout.from_dict(outputs["tensor"])

    assert set(fanout.transfers) == {"classifier.tensor", "generator.tensor"}
    assert carrier.exports == ["classifier.tensor", "generator.tensor"]
    assert len(carrier.released) == 1
    assert torch.equal(carrier.released[0], torch.ones((2, 4)))


async def test_tensor_server_releases_borrowed_input_after_runner_failure() -> None:
    class FailingRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor"},
            outputs={"result"},
        )

        async def run(self, inputs, context):
            raise RuntimeError("classifier failed")

    class Carrier:
        def __init__(self):
            self.tensor = torch.ones((2, 4), dtype=torch.float32)
            self.released = []

        def can_export(self, value):
            return isinstance(value, torch.Tensor)

        async def import_tensor(self, reference):
            return self.tensor

        async def export_tensor(self, tensor, transfer_id):
            raise AssertionError("no tensor output is declared")

        async def export_tensor_fanout(self, tensor, transfer_ids):
            raise AssertionError("no tensor output is declared")

        def release_imported_tensor(self, tensor):
            self.released.append(tensor)

    carrier = Carrier()
    request = StageRequestEnvelope(
        inputs={"tensor": {"remote": "reference"}},
        input_carriers={"tensor": "nixl"},
        output_transfers={},
    )

    with pytest.raises(RuntimeError, match="classifier failed"):
        await RemoteStageServer("tensor", FailingRunner(), carrier).generate(
            request.to_dict()
        ).__anext__()

    assert carrier.released == [carrier.tensor]


async def test_tensor_import_is_cancelled_when_transport_stops() -> None:
    import_cancelled = asyncio.Event()
    import_started = asyncio.Event()

    class TensorRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor"},
            outputs={"result"},
        )

        async def run(self, inputs, context):
            raise AssertionError("runner must not start before its tensor arrives")

    class BlockingCarrier:
        def can_export(self, value):
            return False

        async def import_tensor(self, reference):
            try:
                import_started.set()
                await asyncio.Event().wait()
            finally:
                import_cancelled.set()

        async def export_tensor(self, tensor, transfer_id):
            raise AssertionError("no tensor output is declared")

        async def export_tensor_fanout(self, tensor, transfer_ids):
            raise AssertionError("no tensor output is declared")

    request = StageRequestEnvelope(
        inputs={"tensor": {"remote": "reference"}},
        input_carriers={"tensor": "nixl"},
        output_transfers={},
    )
    transport_context = _ChildContext("request-1:tensor")
    response = asyncio.create_task(
        RemoteStageServer("tensor", TensorRunner(), BlockingCarrier())
        .generate(request.to_dict(), context=transport_context)
        .__anext__()
    )
    await import_started.wait()
    transport_context.stop_generating()

    with pytest.raises(asyncio.CancelledError):
        await response
    assert import_cancelled.is_set()
