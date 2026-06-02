# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt-enhanced visual generation clients."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

import pytest
from aiohttp import web

from dynamo.common.multimodal import (
    EnhancerMode,
    T2IEnhancedClient,
    T2VEnhancedClient,
    T2VEnhancedClientError,
)

LLM_URL = "http://llm.test"
T2V_URL = "http://t2v.test"
T2I_URL = "http://t2i.test"
LLM_MODEL = "Qwen/Qwen3-0.6B"
T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
T2I_MODEL = "black-forest-labs/FLUX.1-dev"
LLM_PATH = "/v1/chat/completions"
T2V_PATH = "/v1/videos"
T2I_PATH = "/v1/images/generations"

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]


def _llm_ok_payload(rewritten: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": rewritten}}
        ],
    }


def _t2v_ok_payload(url: str) -> dict:
    return {"data": [{"url": url}]}


def _t2i_ok_payload(*, url: str | None = None, b64_json: str | None = None) -> dict:
    data: dict[str, str] = {}
    if url is not None:
        data["url"] = url
    if b64_json is not None:
        data["b64_json"] = b64_json
    return {"created": 123, "data": [data]}


class _MockOpenAIServer:
    def __init__(self) -> None:
        self._app = web.Application()
        self._app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(self._app)
        self._site: web.TCPSite | None = None
        self._responses: dict[str, deque[tuple[int, Any, str | None]]] = defaultdict(
            deque
        )
        self.requests: list[dict[str, Any]] = []
        self.url: str | None = None

    async def __aenter__(self) -> "_MockOpenAIServer":
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        assert self._site._server is not None
        sock = self._site._server.sockets[0]
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._runner.cleanup()

    def respond_json(self, path: str, payload: Any, status: int = 200) -> None:
        self._responses[path].append((status, payload, None))

    def respond_text(self, path: str, body: str, status: int = 200) -> None:
        self._responses[path].append((status, None, body))

    def requests_for(self, path: str) -> list[dict[str, Any]]:
        return [request for request in self.requests if request["path"] == path]

    async def _handle(self, request: web.Request) -> web.Response:
        try:
            request_json = await request.json()
        except json.JSONDecodeError:
            request_json = None

        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "json": request_json,
            }
        )

        if not self._responses[request.path]:
            return web.Response(status=404, text=f"unexpected request: {request.path}")

        status, payload, body = self._responses[request.path].popleft()
        if payload is not None:
            return web.json_response(payload, status=status)
        return web.Response(status=status, text=body)


async def test_generate_runs_enhancer_then_t2v():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(
            LLM_PATH,
            _llm_ok_payload("a cinematic skyline at golden hour"),
        )
        server.respond_json(T2V_PATH, _t2v_ok_payload("file:///tmp/result.mp4"))

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            result = await client.generate("a skyline at golden hour")

        assert result.url == "file:///tmp/result.mp4"
        assert result.b64_json is None
        assert result.rewritten_prompt == "a cinematic skyline at golden hour"
        assert result.timings.enhance_ms > 0
        assert result.timings.t2v_ms > 0
        assert result.timings.e2e_ms >= result.timings.enhance_ms

        # T2V was called with the rewritten prompt, not the user input.
        t2v_calls = server.requests_for(T2V_PATH)
        assert t2v_calls, "expected /v1/videos call"
        assert t2v_calls[0]["json"]["prompt"] == "a cinematic skyline at golden hour"


async def test_generate_runs_enhancer_then_t2i():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(
            LLM_PATH,
            _llm_ok_payload("a polished studio render of a copper robot"),
        )
        server.respond_json(
            T2I_PATH,
            _t2i_ok_payload(url="file:///tmp/result.png"),
        )

        async with T2IEnhancedClient(
            llm_url=server.url,
            t2i_url=server.url,
            llm_model=LLM_MODEL,
            t2i_model=T2I_MODEL,
        ) as client:
            result = await client.generate(
                "a copper robot",
                size="1024x1024",
                response_format="url",
            )

        assert result.url == "file:///tmp/result.png"
        assert result.b64_json is None
        assert result.rewritten_prompt == "a polished studio render of a copper robot"
        assert result.timings.enhance_ms > 0
        assert result.timings.t2i_ms > 0

        t2i_calls = server.requests_for(T2I_PATH)
        assert t2i_calls, "expected /v1/images/generations call"
        assert t2i_calls[0]["json"]["model"] == T2I_MODEL
        assert (
            t2i_calls[0]["json"]["prompt"]
            == "a polished studio render of a copper robot"
        )
        assert t2i_calls[0]["json"]["size"] == "1024x1024"


async def test_t2i_supports_b64_json_response():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(
            T2I_PATH,
            _t2i_ok_payload(b64_json="aW1hZ2UtYnl0ZXM="),
        )

        async with T2IEnhancedClient(
            llm_url=server.url,
            t2i_url=server.url,
            llm_model=LLM_MODEL,
            t2i_model=T2I_MODEL,
        ) as client:
            result = await client.generate(
                "raw image prompt",
                enhancer=EnhancerMode.OFF,
                response_format="b64_json",
                nvext={"num_inference_steps": 15},
            )

        assert result.url is None
        assert result.b64_json == "aW1hZ2UtYnl0ZXM="
        assert result.rewritten_prompt is None

        t2i_calls = server.requests_for(T2I_PATH)
        body = t2i_calls[0]["json"]
        assert body["prompt"] == "raw image prompt"
        assert body["response_format"] == "b64_json"
        assert body["nvext"] == {"num_inference_steps": 15}


async def test_enhancer_off_skips_llm_call():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(T2V_PATH, _t2v_ok_payload("file:///tmp/skipped.mp4"))

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            result = await client.generate("raw user prompt", enhancer=EnhancerMode.OFF)

        assert result.url == "file:///tmp/skipped.mp4"
        assert result.rewritten_prompt is None
        assert result.timings.enhance_ms == 0

        assert not server.requests_for(LLM_PATH)


async def test_enhancer_off_string_alias():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(T2V_PATH, _t2v_ok_payload("file:///tmp/off.mp4"))

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            result = await client.generate("anything", enhancer="off")

        assert result.rewritten_prompt is None


async def test_default_enhancer_off_applies_when_no_override():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(T2V_PATH, _t2v_ok_payload("file:///tmp/default-off.mp4"))

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
            default_enhancer=EnhancerMode.OFF,
        ) as client:
            result = await client.generate("anything")

        assert result.rewritten_prompt is None


async def test_llm_5xx_raises_client_error():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_text(LLM_PATH, "internal error", status=500)

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            with pytest.raises(T2VEnhancedClientError) as exc_info:
                await client.generate("anything")

        assert "HTTP 500" in str(exc_info.value)


async def test_t2v_5xx_raises_client_error():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(LLM_PATH, _llm_ok_payload("rewritten"))
        server.respond_text(T2V_PATH, "bad gateway", status=502)

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            with pytest.raises(T2VEnhancedClientError) as exc_info:
                await client.generate("anything")

        assert "HTTP 502" in str(exc_info.value)


async def test_t2v_extra_kwargs_are_forwarded():
    async with _MockOpenAIServer() as server:
        assert server.url is not None
        server.respond_json(T2V_PATH, _t2v_ok_payload("file:///tmp/forward.mp4"))

        async with T2VEnhancedClient(
            llm_url=server.url,
            t2v_url=server.url,
            llm_model=LLM_MODEL,
            t2v_model=T2V_MODEL,
        ) as client:
            await client.generate(
                "anything",
                enhancer=EnhancerMode.OFF,
                size="832x480",
                num_inference_steps=50,
                num_frames=33,
            )

        t2v_calls = server.requests_for(T2V_PATH)
        body = t2v_calls[0]["json"]
        assert body["size"] == "832x480"
        assert body["num_inference_steps"] == 50
        assert body["num_frames"] == 33


async def test_client_must_be_entered_when_session_not_provided():
    client = T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    )
    with pytest.raises(T2VEnhancedClientError):
        await client.generate("anything", enhancer=EnhancerMode.OFF)
