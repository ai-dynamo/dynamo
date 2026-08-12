# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dynamo.experimental.workflow import (
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
from dynamo.experimental.workflow.remote import (
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


def _stage():
    workflow = Workflow("remote-wire")
    text = workflow.input("text", type="text")
    stage = workflow.stage("normalize", CONTRACT, text=text)
    workflow.output("text", stage.normalized)
    return workflow.build().stages[0]


def _context(timeout=None):
    loop = asyncio.get_running_loop()
    return StageContext(
        workflow_name="remote-wire",
        stage_id="normalize",
        attempt_id="request-1",
        deadline=None if timeout is None else loop.time() + timeout,
        _cancelled=asyncio.Event(),
    )


def test_request_envelope_round_trip_is_strict_and_versioned() -> None:
    envelope = StageRequestEnvelope(
        workflow_name="remote-wire",
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
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

    async def round_robin(self, request, *, annotated):
        assert annotated is False
        self.request = request

        async def stream():
            for response in self.responses:
                yield response

        return stream()


async def test_remote_client_sends_identity_and_accepts_one_terminal_response() -> None:
    response = StageResponseEnvelope(
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        outputs={"normalized": "hello"},
    )
    transport = _Client([response.to_dict()])

    result = await RemoteStageClient(transport).run(
        _stage(), {"text": "HELLO"}, _context(timeout=1.0)
    )

    assert result == {"normalized": "hello"}
    assert transport.request["attempt"] == "request-1"
    assert 0 < transport.request["timeout_seconds"] <= 1.0


async def test_remote_client_rejects_missing_or_duplicate_terminal_response() -> None:
    client = RemoteStageClient(_Client([]))
    with pytest.raises(WorkflowExecutionError, match="no terminal response"):
        await client.run(_stage(), {"text": "HELLO"}, _context())

    response = StageResponseEnvelope(
        stage_id="normalize",
        contract_id="normalize",
        attempt_id="request-1",
        outputs={"normalized": "hello"},
    ).to_dict()
    client = RemoteStageClient(_Client([response, response]))
    with pytest.raises(WorkflowExecutionError, match="multiple terminal responses"):
        await client.run(_stage(), {"text": "HELLO"}, _context())


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
        timeout_seconds=0.01,
        inputs={"text": "hello"},
    )

    with pytest.raises(asyncio.TimeoutError):
        await RemoteStageServer("normalize", BlockingRunner()).generate(
            request.to_dict()
        ).__anext__()


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

    async def round_robin(self, request: Mapping[str, Any], *, annotated: bool) -> Any:
        assert annotated is False
        return self._server.generate(request)


async def test_three_remote_stages_fan_out_and_join_through_envelopes() -> None:
    workflow = Workflow("remote-text-fanout")
    text = workflow.input("text", type="text")
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
        endpoint_ids[stage_id]: RemoteStageClient(
            _LoopbackClient(RemoteStageServer(stage_id, runner))
        )
        for stage_id, runner in runners.items()
    }
    executor = WorkflowExecutor(plan, remote_clients=clients)

    result = await executor.run(
        {"text": "Dynamo workflow runs across processes"},
        attempt_id="remote-example-1",
    )

    assert result == {
        "scores": {"workflow": 0.2, "other": 0.8},
        "text": "processes across runs workflow dynamo",
    }
    assert encoder_runner.calls == 1
