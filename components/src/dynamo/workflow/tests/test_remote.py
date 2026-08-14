# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.workflow import (
    DeploymentSpec,
    StageContext,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowValidationError,
    compile_workflow,
)
from dynamo.workflow.remote import (
    STAGE_REQUEST_SCHEMA,
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
    inputs={"text": ValueSpec(type="text")},
    outputs={"normalized": ValueSpec(type="text")},
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


async def test_remote_client_sends_identity_and_accepts_one_terminal_response() -> None:
    response = StageResponseEnvelope(
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        outputs={"normalized": "hello"},
    )
    transport = _Client([response.to_dict()])

    result = await RemoteStageClient(transport).run(
        "normalize", CONTRACT, {"text": "HELLO"}, _context(timeout=1.0)
    )

    assert result == {"normalized": "hello"}
    assert transport.request is not None
    assert transport.request["attempt"] == "request-1"
    assert transport.request["invocation"] == "request-1:normalize"
    assert 0 < transport.request["timeout_seconds"] <= 1.0


async def test_remote_client_creates_an_invocation_scoped_transport_context() -> None:
    response = StageResponseEnvelope(
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        outputs={"normalized": "hello"},
    )
    transport = _Client([response.to_dict()])
    parent = _ParentContext()

    await RemoteStageClient(transport).run(
        "normalize",
        CONTRACT,
        {"text": "HELLO"},
        _context(request_context=parent),
    )

    assert [child.context_id for child in parent.children] == ["request-1:normalize"]
    assert transport.context is parent.children[0]


async def test_remote_client_rejects_missing_or_duplicate_terminal_response() -> None:
    client = RemoteStageClient(_Client([]))
    with pytest.raises(WorkflowExecutionError, match="no terminal response"):
        await client.run("normalize", CONTRACT, {"text": "HELLO"}, _context())

    response = StageResponseEnvelope(
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        invocation_id="request-1:normalize",
        outputs={"normalized": "hello"},
    ).to_dict()
    client = RemoteStageClient(_Client([response, response]))
    with pytest.raises(WorkflowExecutionError, match="multiple terminal responses"):
        await client.run("normalize", CONTRACT, {"text": "HELLO"}, _context())


class _Runner:
    contract = CONTRACT

    async def run(self, inputs, context):
        assert context.workflow_name == "remote-wire"
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
    )

    responses = [
        response
        async for response in RemoteStageServer("normalize", _Runner()).generate(
            request.to_dict()
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
    )
    response = asyncio.create_task(
        RemoteStageServer("normalize", runner)
        .generate(request.to_dict(), transport_context)
        .__anext__()
    )
    await runner.started.wait()

    transport_context.stop_generating()

    with pytest.raises(asyncio.CancelledError):
        await response
    assert runner.cancelled.is_set()


def test_inline_server_rejects_undeclared_tensor_fallback() -> None:
    class TensorRunner:
        contract = StageContract(
            id="tensor",
            inputs={"tensor": ValueSpec(type="tensor")},
            outputs={"result": ValueSpec(type="json")},
        )

        async def run(self, inputs, context):
            return {"result": {}}

    with pytest.raises(WorkflowValidationError, match="does not support.*tensor"):
        RemoteStageServer("tensor", TensorRunner())


TOKENS = ValueSpec(type="json")
TEXT_ENCODER = StageContract(
    id="text-encoder",
    inputs={"text": ValueSpec(type="text")},
    outputs={"tokens": TOKENS},
)
KEYWORD_CLASSIFIER = StageContract(
    id="keyword-classifier",
    inputs={"tokens": TOKENS},
    outputs={"scores": ValueSpec(type="json")},
)
TEXT_GENERATOR = StageContract(
    id="text-generator",
    inputs={"tokens": TOKENS},
    outputs={"text": ValueSpec(type="text")},
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


async def test_three_remote_stages_fan_out_and_join_through_envelopes() -> None:
    workflow = Workflow("remote-text-fanout")
    text = workflow.input("text", ValueSpec(type="text"))
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
    executor = await WorkflowExecutor.bind(plan, runtime=runtime)

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
