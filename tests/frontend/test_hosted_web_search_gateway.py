# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""HTTP contract tests for the opt-in hosted-search gateway; no model required."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from openai import AsyncOpenAI

from examples.deployments.hosted_web_search.gateway import Config, create_app

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


def message(text="Dynamo uses Rust. [1]"):
    """Construct a completed assistant output item."""
    return {
        "type": "message",
        "id": "msg_answer",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        ],
    }


def function(name, call_id="call_search", namespace=None):
    """Construct a model-selected function without invoking it."""
    item = {
        "type": "function_call",
        "id": "fc_" + call_id,
        "call_id": call_id,
        "name": name,
        "arguments": '{"query":"Dynamo frontend integration"}',
        "status": "completed",
    }
    if namespace:
        item["namespace"] = namespace
    return item


def turn(output, tokens=5, status="completed"):
    """Return model usage with an output list for one inference round."""
    return {
        "model": "test-model",
        "status": status,
        "output": output,
        "incomplete_details": {"reason": "max_output_tokens"}
        if status == "incomplete"
        else None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": tokens,
            "total_tokens": 10 + tokens,
        },
    }


@asynccontextmanager
async def gateway(**limits):
    """Run real HTTP peers on ephemeral ports and record all dispatched work."""
    state = {
        "model": [],
        "search": [],
        "turns": [],
        "headers": [],
        "results": {
            "results": [
                {
                    "url": "https://github.com/ai-dynamo/dynamo",
                    "title": "Dynamo",
                    "snippet": "Distributed inference with a Rust frontend.",
                }
            ]
        },
        "started": asyncio.Event(),
        "cancelled": asyncio.Event(),
    }

    async def model(request):
        """Consume a scripted turn or select the gateway's private search function."""
        body = await request.json()
        state["model"].append(body)
        state["headers"].append(dict(request.headers))
        if state["turns"]:
            return web.json_response(state["turns"].pop(0))
        if len(state["model"]) == 1:
            name = next(
                t["name"]
                for t in body["tools"]
                if t.get("description", "").startswith("Search the web")
            )
            return web.json_response(turn([function(name)]))
        return web.json_response(turn([message("🦀 Dynamo uses Rust. [1]")], tokens=7))

    async def provider(request):
        """Record search requests and allow deterministic failure or cancellation."""
        state["search"].append(await request.json())
        state["headers"].append(dict(request.headers))
        state["started"].set()
        if state.get("wait"):
            try:
                await asyncio.Event().wait()
            finally:
                state["cancelled"].set()
        if state.get("status"):
            return web.Response(status=state["status"], text="SECRET_PROVIDER_ERROR")
        return web.json_response(state["results"])

    backend = web.Application()
    backend.router.add_post("/v1/responses", model)
    search = web.Application()
    search.router.add_post("/search", provider)
    async with TestServer(
        backend, handler_cancellation=True
    ) as model_server, TestServer(search, handler_cancellation=True) as search_server:
        config = Config(
            str(model_server.make_url("/")).rstrip("/"),
            str(search_server.make_url("/search")),
            **limits,
        )
        async with TestClient(
            TestServer(create_app(config), handler_cancellation=True)
        ) as client:
            yield client, state


def request(**options):
    """Build a minimal Responses request accepting hosted search."""
    return {
        "model": "test-model",
        "input": "Where is the frontend integrated in Dynamo?",
        "tools": [{"type": "web_search"}],
        **options,
    }


def events(text):
    """Decode complete SSE data lines into event objects."""
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", [None, "auto", "required", {"type": "web_search"}])
async def test_search_lifecycle(stream, choice, tmp_path):
    """Search, resume inference, cite Unicode text, and preserve public response identity."""
    payload = request(
        stream=stream, include=["web_search_call.action.sources"], max_output_tokens=20
    )
    if choice is not None:
        payload["tool_choice"] = choice
    async with gateway() as (client, state):
        response = await client.post("/v1/responses", json=payload)
        assert response.status == 200
        wire = await response.text()
        if stream:
            emitted = events(wire)
            assert [e["sequence_number"] for e in emitted] == list(range(len(emitted)))
            assert emitted[0]["type"] == "response.created"
            assert emitted[-1]["type"] == "response.completed"
            result = emitted[-1]["response"]
            assert emitted[0]["response"]["id"] == result["id"]
            kinds = [e["type"] for e in emitted]
            assert kinds.index("response.web_search_call.completed") < kinds.index(
                "response.output_text.delta"
            )
            assert [
                e["output_index"]
                for e in emitted
                if e["type"] == "response.output_item.done"
            ] == [0, 1]
        else:
            result = json.loads(wire)
        assert "dynamo_web_search" not in wire
        assert result["status"] == "completed"
        assert result["usage"]["total_tokens"] == 32
        assert result["usage"]["output_tokens"] == 12
        search, answer = result["output"]
        assert search["type"] == "web_search_call" and search["status"] == "completed"
        assert (
            search["action"]["sources"][0]["url"]
            == state["results"]["results"][0]["url"]
        )
        part = answer["content"][0]
        citation = part["annotations"][0]
        assert part["text"][citation["start_index"] : citation["end_index"]] == "[1]"
        assert citation["url"] == search["action"]["sources"][0]["url"]
        assert [r["max_output_tokens"] for r in state["model"]] == [20, 15]
        assert state["search"] == [
            {"query": "Dynamo frontend integration", "max_results": 5}
        ]
        replay = state["model"][1]["input"][-1]
        assert replay["type"] == "function_call_output"
        assert json.loads(replay["output"])["results"][0]["citation"] == "[1]"
        (tmp_path / "request.json").write_text(json.dumps(payload, indent=2))
        (tmp_path / "response.txt").write_text(wire)


@pytest.mark.asyncio
async def test_collision_and_mixed_client_functions():
    """Return client functions unchanged, even when their name matches the default alias."""
    tools = [
        {
            "type": "function",
            "name": "dynamo_web_search",
            "parameters": {"type": "object"},
        },
        {
            "type": "namespace",
            "name": "crm",
            "tools": [{"type": "function", "name": "lookup"}],
        },
        {"type": "web_search"},
    ]
    async with gateway() as (client, state):
        external = function("lookup", "client_call", namespace="crm")
        state["turns"] = [turn([function("dynamo_web_search_"), external])]
        result = await (
            await client.post("/v1/responses", json=request(tools=tools))
        ).json()
        assert result["output"][1] == external
        assert len(state["search"]) == 1 and len(state["model"]) == 1
        assert state["model"][0]["tools"][:2] == tools[:2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"tool_choice": "none"},
        {
            "tool_choice": {"type": "function", "name": "client"},
            "tools": [{"type": "function", "name": "client"}, {"type": "web_search"}],
        },
    ],
)
async def test_disallowed_search_is_not_executed(options):
    """Fail a backend choice violation before contacting the provider."""
    async with gateway() as (client, state):
        response = await client.post("/v1/responses", json=request(**options))
        assert response.status == 502
        assert not state["search"]


@pytest.mark.asyncio
async def test_none_returns_text_without_search():
    """Honor an explicit no-tools choice and avoid inventing citations."""
    async with gateway() as (client, state):
        state["turns"] = [turn([message()])]
        response = await client.post("/v1/responses", json=request(tool_choice="none"))
        assert response.status == 200 and not state["search"]
        assert (await response.json())["output"][0]["content"][0]["annotations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"tools": [{"type": "file_search"}]},
        {"tools": [{"type": "web_search", "filters": {}}]},
        {"previous_response_id": "resp_old"},
        {"store": True},
        {"background": True},
        {"tool_choice": {"type": "allowed_tools", "tools": []}},
        {"max_output_tokens": 0},
        {"max_tool_calls": 999},
        {"input": [{"type": "web_search_call"}]},
    ],
)
async def test_unsupported_requests_fail_before_dispatch(options):
    """Reject unsupported contracts before starting model or provider work."""
    async with gateway() as (client, state):
        response = await client.post("/v1/responses", json=request(**options))
        assert response.status == 400
        assert not state["model"] and not state["search"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["http", "schema", "size", "timeout"])
async def test_provider_failure(stream, failure):
    """Return sanitized failures and never emit successful completion after an error."""
    limits = {"timeout": 0.15} if failure == "timeout" else {}
    async with gateway(**limits) as (client, state):
        if failure == "http":
            state["status"] = 503
        elif failure == "schema":
            state["results"] = {
                "results": [{"url": "file:///secret", "title": "bad", "snippet": "bad"}]
            }
        elif failure == "size":
            state["results"] = {"results": [], "padding": "x" * 70_000}
        else:
            state["wait"] = True
        response = await client.post("/v1/responses", json=request(stream=stream))
        wire = await response.text()
        assert "SECRET_PROVIDER_ERROR" not in wire
        if stream:
            emitted = events(wire)
            assert emitted[-1]["type"] == "response.failed"
            assert emitted[-1]["response"]["output"][0]["status"] == "failed"
            assert not any(e["type"] == "response.completed" for e in emitted)
        else:
            assert response.status == (504 if failure == "timeout" else 502)


@pytest.mark.asyncio
async def test_disconnect_cancels_search():
    """Closing the client stream cancels the outstanding provider HTTP request."""
    async with gateway(timeout=5) as (client, state):
        state["wait"] = True
        response = await client.post("/v1/responses", json=request(stream=True))
        await asyncio.wait_for(state["started"].wait(), 2)
        response.close()
        await asyncio.wait_for(state["cancelled"].wait(), 2)
        assert len(state["model"]) == 1


@pytest.mark.asyncio
async def test_limits_and_partial_failure():
    """A later call-limit error retains completed searches in the terminal event."""
    async with gateway(max_searches=1) as (client, state):
        state["turns"] = [
            turn([function("dynamo_web_search")]),
            turn([function("dynamo_web_search", "second")]),
        ]
        response = await client.post("/v1/responses", json=request(stream=True))
        emitted = events(await response.text())
        assert emitted[-1]["type"] == "response.failed"
        assert emitted[-1]["response"]["output"][0]["status"] == "completed"
        assert len(state["search"]) == 1


@pytest.mark.asyncio
async def test_token_exhaustion_is_incomplete():
    """Stop without another model turn when the aggregate output budget is spent."""
    async with gateway() as (client, state):
        response = await client.post("/v1/responses", json=request(max_output_tokens=5))
        result = await response.json()
        assert result["status"] == "incomplete"
        assert result["incomplete_details"] == {"reason": "max_output_tokens"}
        assert len(state["model"]) == 1


@pytest.mark.asyncio
async def test_credentials_are_operator_owned():
    """Never forward client authorization; separate inference and provider credentials."""
    async with gateway(dynamo_token="model-secret", search_token="search-secret") as (
        client,
        state,
    ):
        response = await client.post(
            "/v1/responses",
            json=request(),
            headers={"Authorization": "Bearer client-secret"},
        )
        assert response.status == 200
        assert [h.get("Authorization") for h in state["headers"]] == [
            "Bearer model-secret",
            "Bearer search-secret",
            "Bearer model-secret",
        ]
        assert "secret" not in await response.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        [{"type": "message", "id": "x", "role": "assistant", "content": [{}]}],
        [function("dynamo_web_search"), function("dynamo_web_search")],
    ],
)
async def test_malformed_model_output_is_not_executed(output):
    """Malformed content and duplicate call IDs produce a controlled upstream error."""
    async with gateway() as (client, state):
        state["turns"] = [turn(copy.deepcopy(output))]
        response = await client.post("/v1/responses", json=request())
        assert response.status == 502 and not state["search"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["required", {"type": "web_search"}])
async def test_forced_choice_requires_a_call(choice):
    """Do not report success when the model ignores a forced search choice."""
    async with gateway() as (client, state):
        state["turns"] = [turn([message()])]
        response = await client.post("/v1/responses", json=request(tool_choice=choice))
        assert response.status == 502 and not state["search"]


@pytest.mark.asyncio
async def test_intermediate_reasoning_and_empty_results():
    """Preserve reasoning before a search and avoid citations when no sources exist."""
    reasoning = {
        "type": "reasoning",
        "id": "rs_first",
        "summary": [{"type": "summary_text", "text": "Need a source."}],
    }
    async with gateway() as (client, state):
        state["results"] = {"results": []}
        state["turns"] = [turn([reasoning, function("dynamo_web_search")])]
        response = await client.post("/v1/responses", json=request())
        result = await response.json()
        assert response.status == 200
        assert result["output"][0] == reasoning
        assert result["output"][1]["type"] == "web_search_call"
        assert result["output"][2]["content"][0]["annotations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["parallel", "unknown", "truncated"])
async def test_invalid_model_selection_is_not_executed(kind):
    """Enforce advertised tools, parallel-call policy, and complete search arguments."""
    async with gateway() as (client, state):
        output = [function("missing" if kind == "unknown" else "dynamo_web_search")]
        if kind == "parallel":
            output.append(function("dynamo_web_search", "other"))
        state["turns"] = [
            turn(output, status="incomplete" if kind == "truncated" else "completed")
        ]
        response = await client.post(
            "/v1/responses", json=request(parallel_tool_calls=False)
        )
        assert response.status == 502 and not state["search"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_client(stream):
    """Parse JSON and SSE with the OpenAI SDK's strict response validation."""
    async with gateway() as (client, state):
        async with AsyncOpenAI(
            api_key="unused",
            base_url=str(client.make_url("/v1")),
            _strict_response_validation=True,
        ) as sdk:
            response = await sdk.responses.create(
                **request(tool_choice="required", stream=stream)
            )
            if stream:
                completed = []
                async for event in response:
                    if event.type == "response.completed":
                        completed.append(event.response)
                assert len(completed) == 1
                response = completed[0]
            assert response.status == "completed"
            assert response.output[0].type == "web_search_call"
            assert response.output_text == "🦀 Dynamo uses Rust. [1]"
            assert len(state["search"]) == 1
