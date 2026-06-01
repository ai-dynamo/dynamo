# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for T2VEnhancedClient."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from aiohttp import web
import pytest

from dynamo.common.clients import (
    EnhancerMode,
    T2VEnhancedClient,
    T2VEnhancedClientError,
)


LLM_URL = "http://llm.test"
T2V_URL = "http://t2v.test"
LLM_MODEL = "Qwen/Qwen3-0.6B"
T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LLM_PATH = "/v1/chat/completions"
T2V_PATH = "/v1/videos"


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
        except Exception:
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


@pytest.mark.asyncio
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
        assert result.rewritten_prompt == "a cinematic skyline at golden hour"
        assert result.timings.enhance_ms > 0
        assert result.timings.t2v_ms > 0
        assert result.timings.e2e_ms >= result.timings.enhance_ms

        # T2V was called with the rewritten prompt, not the user input.
        t2v_calls = server.requests_for(T2V_PATH)
        assert t2v_calls, "expected /v1/videos call"
        assert t2v_calls[0]["json"]["prompt"] == "a cinematic skyline at golden hour"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_client_must_be_entered_when_session_not_provided():
    client = T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    )
    with pytest.raises(T2VEnhancedClientError):
        await client.generate("anything", enhancer=EnhancerMode.OFF)
