# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import asdict

import httpx
import pytest
from test_core import request

from dynamo.compaction.clients import (
    CappedTransport,
    Profile,
    SDKModels,
    make_client,
    selection_rows,
)
from dynamo.compaction.coordinator import Counter, Pipeline
from dynamo.compaction.fixtures import sdk_response
from dynamo.compaction.helper import execute, process
from dynamo.compaction.protocol import Invalid, canonical, strict_json

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def profile(**overrides):
    return Profile(
        **{
            "decision_model": "cpu-fixture-decision",
            "generation_model": "cpu-fixture-generation",
            "counter": Counter("fixture_bytes", len, False),
            "max_input_tokens": 100000,
            "min_probability": 0.9,
            "min_confidence": 0.9,
            **overrides,
        }
    )


def packet():
    req = request()
    return canonical(
        {
            "version": 1,
            "snapshot_sha256": req.snapshot_sha256,
            "session_id": req.session_id,
            "units": [asdict(unit) for unit in req.units],
            "context_units": [],
            "budget": {"max_output_tokens": 4096},
        }
    ).encode()


def body():
    return sdk_response(
        httpx.Request(
            "POST",
            "http://fixture.invalid/v1/decisions",
            json={"model": "cpu-fixture-decision", "questions": [{"name": "a"}]},
        )
    ).json()


def test_helper_uses_real_sdk_decisions_and_chat_serialization():
    seen = []

    def handler(req):
        assert req.headers["x-dynamo-session-id"] == "cpu-session"
        seen.append((req.url.path, strict_json(req.content)))
        return sdk_response(req)

    async def run():
        async with make_client(
            "http://fixture.invalid/v1", httpx.MockTransport(handler)
        ) as client:
            models = SDKModels(client, profile(), fixture=True)
            return await process(packet(), Pipeline(models, profile().counter))

    result = asyncio.run(run())
    assert result["status"] == "accepted", result
    assert result["retained_ids"] == ("u",)
    assert [path for path, _ in seen] == [
        "/v1/decisions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert set(seen[0][1]) == {"model", "input", "questions"}
    assert seen[0][1]["questions"][0]["choices"][0]["value"] == "keep"
    assert "Do not delete files." in seen[0][1]["input"]
    assert result["qualified_counter"] is False


@pytest.mark.parametrize("mode", ["fixture", "fixture-sdk"])
def test_cpu_helper_execution_modes(mode):
    result = asyncio.run(execute(packet(), mode))
    assert result["status"] == "accepted", result
    assert result["mode"] == "fixture"


def test_model_cli_has_no_unqualified_live_path():
    assert (
        asyncio.run(execute(packet(), "model"))["reason"] == "unqualified_model_profile"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.update(model="other"),
        lambda b: b.update(answers=[]),
        lambda b: b["answers"][0].update(name="u"),
        lambda b: b["answers"][0].update(confidence=True),
        lambda b: b["answers"][0].update(choice="keep"),
        lambda b: b["answers"][0]["probabilities"][0].update(value=True),
        lambda b: b["answers"][0]["probabilities"][0].update(probability=0.4),
        lambda b: b["usage"].update(input_tokens=True),
        lambda b: b["usage"].update(total_tokens=2),
        lambda b: b["usage"]["input_tokens_details"].update(cached_tokens=None),
    ],
)
def test_invalid_decision_responses_reject(mutation):
    response = body()
    mutation(response)
    with pytest.raises(Invalid):
        selection_rows(response, request().units[1:], profile())


def test_refusal_and_low_confidence_preserve_evidence():
    response = body()
    response["answers"][0] = {"type": "refusal", "name": "a"}
    assert (
        selection_rows(response, request().units[1:], profile())["selections"][0][
            "action"
        ]
        == "unknown"
    )
    response = body()
    response["answers"][0]["confidence"] = 0.5
    assert (
        selection_rows(response, request().units[1:], profile())["selections"][0][
            "action"
        ]
        == "keep"
    )


@pytest.mark.parametrize("status", [429, 500])
def test_errors_have_no_hidden_retries(status):
    calls = []

    async def run():
        def handler(req):
            calls.append(req)
            return httpx.Response(
                status, json={"error": {"message": "redacted", "type": "error"}}
            )

        async with make_client(
            "http://fixture.invalid/v1", httpx.MockTransport(handler)
        ) as client:
            return await Pipeline(
                SDKModels(client, profile(), fixture=True), profile().counter
            ).run(request())

    result = asyncio.run(run())
    assert result.reason == "decision_transport_failure"
    assert len(calls) == 1


def test_response_cap_rejects_before_sdk_parse():
    async def run():
        transport = CappedTransport(
            httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 11)), 10
        )
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("http://fixture.invalid")

    with pytest.raises(Invalid, match="bytes"):
        asyncio.run(run())


def test_question_and_context_limits_reject_before_network():
    async def run():
        async with make_client(
            "http://fixture.invalid/v1",
            httpx.MockTransport(lambda _: pytest.fail("network must not run")),
        ) as client:
            models = SDKModels(client, profile(max_input_tokens=1), fixture=True)
            with pytest.raises(Invalid, match="context"):
                await models.select(
                    request().units[1:], request().units, request().session_id
                )
            with pytest.raises(Invalid, match="question"):
                await models.select(
                    request().units * 33, request().units, request().session_id
                )

    asyncio.run(run())


@pytest.mark.parametrize("credential", [None, "", False, 123])
def test_local_client_requires_explicit_nonempty_credential(credential):
    with pytest.raises(Invalid, match="credentials"):
        make_client(
            "http://127.0.0.1:8000/v1",
            httpx.MockTransport(sdk_response),
            api_key=credential,
        )
