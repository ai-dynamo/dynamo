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
from dynamo.workflow.remote import RemoteStageClient, RemoteStageServer

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


def _context(timeout=None, request_context=None):
    loop = asyncio.get_running_loop()
    return StageContext(
        workflow_name="remote-wire",
        stage_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        deadline=None if timeout is None else loop.time() + timeout,
        _cancelled=asyncio.Event(),
        request_context=request_context,
    )


def test_request_envelope_round_trip_is_strict_and_versioned() -> None:
    envelope = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        timeout_seconds=1.5,
        inputs={"text": "HELLO"},
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
    bad["version"] = 0.0
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


async def test_remote_client_sends_inputs_and_accepts_one_response_mapping() -> None:
    transport = _Client([{"normalized": "hello"}])

    result = await RemoteStageClient(transport).run(
        "normalize", CONTRACT, {"text": "HELLO"}, _context(timeout=1.0), {}
    )

    assert result == {"normalized": "hello"}
    assert transport.request == {"text": "HELLO"}


async def test_remote_client_creates_an_invocation_scoped_transport_context() -> None:
    transport = _Client([{"normalized": "hello"}])
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


async def test_remote_client_rejects_missing_or_duplicate_response_mapping() -> None:
    client = RemoteStageClient(_Client([]))
    with pytest.raises(WorkflowExecutionError, match="no terminal response"):
        await client.run("normalize", CONTRACT, {"text": "HELLO"}, _context(), {})

    response = {"normalized": "hello"}
    client = RemoteStageClient(_Client([response, response]))
    with pytest.raises(WorkflowExecutionError, match="multiple terminal responses"):
        await client.run("normalize", CONTRACT, {"text": "HELLO"}, _context(), {})


async def test_remote_client_adds_stage_context_to_transport_failures() -> None:
    class FailingClient:
        async def round_robin(self, request, *, annotated, context=None):
            raise TypeError("unsupported payload")

    with pytest.raises(
        WorkflowExecutionError,
        match="remote stage 'normalize'.*transport boundary",
    ) as error:
        await RemoteStageClient(FailingClient()).run(
            "normalize", CONTRACT, {"text": object()}, _context()
        )

    assert isinstance(error.value.__cause__, TypeError)


class _Runner:
    contract = CONTRACT

    async def run(self, inputs, context):
        assert context.workflow_name is None
        assert context.stage_id == "normalize"
        assert context.attempt_id == "request-1:normalize"
        assert context.invocation_id == "request-1:normalize"
        assert context.deadline is None
        return {"normalized": inputs["text"].strip().lower()}


async def test_remote_server_validates_and_runs_stage_contract() -> None:
    request = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        timeout_seconds=None,
        inputs={"text": " HELLO "},
        output_transfers={},
    )

    responses = [
        response
        async for response in RemoteStageServer("normalize", _Runner()).generate(
            {"text": " HELLO "}, context=transport_context
        )
    ]

    assert len(responses) == 1
    assert StageResponseEnvelope.from_dict(responses[0]).outputs == {
        "normalized": "hello"
    }


async def test_remote_server_enforces_deadline() -> None:
    class BlockingRunner:
        contract = CONTRACT

        async def run(self, inputs, context):
            await asyncio.Event().wait()

    request = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        timeout_seconds=0.01,
        inputs={"text": "hello"},
        output_transfers={},
    )

    with pytest.raises(asyncio.TimeoutError):
        await RemoteStageServer("normalize", BlockingRunner()).generate(
            request.to_dict()
        ).__anext__()


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
    transport_context = _ChildContext("request-1:normalize")
    request = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        timeout_seconds=None,
        inputs={"text": "hello"},
        output_transfers={},
    )
    response = asyncio.create_task(
        RemoteStageServer("normalize", runner)
        .generate({"text": "hello"}, context=transport_context)
        .__anext__()
    )
    await runner.started.wait()

    transport_context.stop_generating()

    with pytest.raises(asyncio.CancelledError):
        await response
    assert runner.cancelled.is_set()


async def test_remote_server_accepts_opaque_values_without_type_declarations() -> None:
    value = object()

    class OpaqueRunner:
        contract = StageContract(
            id="opaque",
            inputs={"value"},
            outputs={"result"},
        )

        async def run(self, inputs, context):
            return {"result": inputs["value"]}

    response = (
        await RemoteStageServer("opaque", OpaqueRunner())
        .generate({"value": value})
        .__anext__()
    )

    assert response["result"] is value


TEXT_ENCODER = StageContract(
    id="text-encoder",
    inputs={"text"},
    outputs={"tokens"},
)
KEYWORD_CLASSIFIER = StageContract(
    id="keyword-classifier",
    inputs={"tokens"},
    outputs={"scores"},
)
TEXT_GENERATOR = StageContract(
    id="text-generator",
    inputs={"tokens"},
    outputs={"text"},
)
RESPONSE = StageContract(
    id="response",
    inputs={"scores", "text"},
    outputs={"chunk"},
)


class _TextEncoder:
    contract = TEXT_ENCODER

    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        self.calls += 1
        return {"tokens": inputs["text"].lower().split()}


class _KeywordClassifier:
    contract = KEYWORD_CLASSIFIER

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        tokens = inputs["tokens"]
        workflow_hits = sum(token == "workflow" for token in tokens)
        score = workflow_hits / max(1, len(tokens))
        return {"scores": {"workflow": score, "other": 1.0 - score}}


class _TextGenerator:
    contract = TEXT_GENERATOR

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"text": " ".join(reversed(inputs["tokens"]))}


class _Response:
    contract = RESPONSE

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"chunk": {"text": inputs["text"], "scores": inputs["scores"]}}


class _LoopbackClient:
    def __init__(self, server: RemoteStageServer) -> None:
        self._server = server

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> Any:
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
        self.endpoint_ids: list[str] = []

    def endpoint(self, endpoint_id: str) -> _Endpoint:
        self.endpoint_ids.append(endpoint_id)
        return _Endpoint(self._clients[endpoint_id])


async def test_three_remote_stages_fan_out_and_join_through_direct_mappings() -> None:
    workflow = Workflow("remote-text-fanout")
    text = workflow.input("text")
    encoder = workflow.stage("encoder", TEXT_ENCODER, text=text)
    classifier = workflow.stage("classifier", KEYWORD_CLASSIFIER, tokens=encoder.tokens)
    generator = workflow.stage("generator", TEXT_GENERATOR, tokens=encoder.tokens)
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)

    endpoint_ids = {
        "encoder": "workflows.encoder.generate",
        "classifier": "workflows.classifier.generate",
        "generator": "workflows.generator.generate",
    }
    encoder_runner = _TextEncoder()
    runners = {
        "encoder": encoder_runner,
        "classifier": _KeywordClassifier(),
        "generator": _TextGenerator(),
    }
    plan = compile_workflow(workflow, DeploymentSpec.remote(**endpoint_ids))
    clients = {
        endpoint_ids[stage_id]: _LoopbackClient(RemoteStageServer(stage_id, runner))
        for stage_id, runner in runners.items()
    }
    runtime = _Runtime(clients)
    executor = await WorkflowOrchestrator.bind(plan, runtime=runtime)

    result = await executor.run(
        {"text": "Dynamo workflow runs across processes"},
        attempt_id="remote-example-1",
    )

    assert result == {
        "scores": {"workflow": 0.2, "other": 0.8},
        "text": "processes across runs workflow dynamo",
    }
    assert encoder_runner.calls == 1
    assert runtime.endpoint_ids == sorted(endpoint_ids.values())


async def test_remote_branches_join_in_an_inline_response_stage() -> None:
    workflow = Workflow("mixed-text-fanout")
    text = workflow.input("text")
    encoder = workflow.stage("encoder", TEXT_ENCODER, text=text)
    classifier = workflow.stage("classifier", KEYWORD_CLASSIFIER, tokens=encoder.tokens)
    generator = workflow.stage("generator", TEXT_GENERATOR, tokens=encoder.tokens)
    response = workflow.stage(
        "response",
        RESPONSE,
        scores=classifier.scores,
        text=generator.text,
    )
    workflow.output("chunk", response.chunk)

    endpoint_ids = {
        "encoder": "workflows.encoder.generate",
        "classifier": "workflows.classifier.generate",
        "generator": "workflows.generator.generate",
    }
    runners = {
        "encoder": _TextEncoder(),
        "classifier": _KeywordClassifier(),
        "generator": _TextGenerator(),
    }
    clients = {
        endpoint_ids[stage_id]: _LoopbackClient(RemoteStageServer(stage_id, runner))
        for stage_id, runner in runners.items()
    }
    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                **{
                    stage_id: RemoteBinding(endpoint_id)
                    for stage_id, endpoint_id in endpoint_ids.items()
                },
                "response": InlineBinding("response"),
            }
        ),
    )
    orchestrator = await WorkflowOrchestrator.bind(
        plan,
        runtime=_Runtime(clients),
        inline_runners={"response": _Response()},
    )

    result = await orchestrator.run({"text": "Dynamo workflow"})

    assert result == {
        "chunk": {
            "text": "workflow dynamo",
            "scores": {"workflow": 0.5, "other": 0.5},
        }
    }


async def test_tensor_server_imports_input_and_exports_per_consumer() -> None:
    tensor_spec = ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))

    class TensorRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor": tensor_spec},
            outputs={"tensor": tensor_spec},
        )

        async def run(self, inputs, context):
            return {"tensor": inputs["tensor"] * 2}

    class Carrier:
        def __init__(self):
            self.exports = []

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

    carrier = Carrier()
    request = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="tensor",
        contract_id="tensor",
        attempt_id="request-1",
        invocation_id="request-1:tensor",
        timeout_seconds=None,
        inputs={"tensor": {"remote": "reference"}},
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


async def test_tensor_import_is_bounded_by_remote_stage_deadline() -> None:
    tensor_spec = ValueSpec(type="tensor", dtype="float32", shape=("dynamic", 4))
    import_cancelled = asyncio.Event()

    class TensorRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor": tensor_spec},
            outputs={"result": ValueSpec(type="json")},
        )

        async def run(self, inputs, context):
            raise AssertionError("runner must not start before its tensor arrives")

    class BlockingCarrier:
        async def import_tensor(self, reference):
            try:
                await asyncio.Event().wait()
            finally:
                import_cancelled.set()

        async def export_tensor(self, tensor, transfer_id):
            raise AssertionError("no tensor output is declared")

        async def export_tensor_fanout(self, tensor, transfer_ids):
            raise AssertionError("no tensor output is declared")

    request = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="tensor",
        contract_id="tensor",
        attempt_id="request-1",
        invocation_id="request-1:tensor",
        timeout_seconds=0.01,
        inputs={"tensor": {"remote": "reference"}},
        output_transfers={},
    )

    with pytest.raises(asyncio.TimeoutError):
        await RemoteStageServer("tensor", TensorRunner(), BlockingCarrier()).generate(
            request.to_dict()
        ).__anext__()
    assert import_cancelled.is_set()
