# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the custom-encoder-only HTTP service."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from aiohttp.test_utils import TestClient, TestServer

from examples.custom_encoder.benchmark.encoder_only_server import create_app

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


class _FakeEncoder:
    def __init__(self) -> None:
        self.raws: list[list[str]] = []
        self.failures_remaining = 0
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.shutdown_calls = 0

    async def encode(self, raws: list[str]) -> list[object]:
        self.raws.append(raws)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("encoder failed")
        return [object() for _ in raws]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _payload(*images: str, stream: bool = False) -> dict:
    content = [{"type": "text", "text": "encode this image"}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image}} for image in images
    )
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


@asynccontextmanager
async def _client(encoder: _FakeEncoder) -> AsyncIterator[TestClient]:
    client = TestClient(TestServer(create_app(encoder, model="default-model")))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def test_nonstreaming_request_runs_one_encoder_call() -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        health = await client.get("/health")
        response = await client.post("/v1/chat/completions", json=_payload(_IMAGE))

        assert health.status == 200
        assert await health.json() == {"status": "ready"}
        assert response.status == 200
        body = await response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert body["choices"][0]["message"]["content"] == "ok"
        assert body["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "total_tokens": 1,
        }
        assert encoder.raws == [[_IMAGE]]

    assert encoder.shutdown_calls == 1


async def test_request_larger_than_aiohttp_default_is_accepted() -> None:
    encoder = _FakeEncoder()
    image = "data:image/png;base64," + base64.b64encode(b"\0" * 800_000).decode()
    body = io.BytesIO(json.dumps(_payload(image)).encode())
    async with _client(encoder) as client:
        response = await client.post(
            "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status == 200
    assert encoder.raws == [[image]]


async def test_stream_starts_after_encoder_and_reports_requested_usage() -> None:
    encoder = _FakeEncoder()
    encoder.started = asyncio.Event()
    encoder.release = asyncio.Event()
    payload = _payload(_IMAGE, stream=True)
    payload["stream_options"] = {"include_usage": True}

    async with _client(encoder) as client:
        response_task = asyncio.create_task(
            client.post("/v1/chat/completions", json=payload)
        )
        await encoder.started.wait()
        assert not response_task.done()

        encoder.release.set()
        response = await response_task
        events = [
            line.removeprefix("data: ")
            for line in (await response.text()).splitlines()
            if line.startswith("data: ")
        ]

    assert response.status == 200
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"
    assert chunks[0]["choices"][0]["finish_reason"] == "stop"
    assert chunks[1]["choices"] == []
    assert chunks[1]["usage"]["completion_tokens"] == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(), "exactly one image_url"),
        (_payload(_IMAGE, _IMAGE), "exactly one image_url"),
        (_payload("https://example.com/image.png"), "inline base64 image"),
        ({**_payload(_IMAGE), "stream": "yes"}, "'stream' must be a boolean"),
    ],
)
async def test_invalid_requests_return_openai_error(
    payload: dict, message: str
) -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        body = await response.json()

    assert response.status == 400
    assert message in body["error"]["message"]
    assert body["error"]["type"] == "invalid_request_error"
    assert encoder.raws == []


async def test_malformed_json_returns_openai_error() -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        response = await client.post(
            "/v1/chat/completions",
            data="{",
            headers={"Content-Type": "application/json"},
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["type"] == "invalid_request_error"


async def test_encoder_failure_does_not_stop_later_requests() -> None:
    encoder = _FakeEncoder()
    encoder.failures_remaining = 1
    async with _client(encoder) as client:
        failed = await client.post("/v1/chat/completions", json=_payload(_IMAGE))
        failed_body = await failed.json()
        succeeded = await client.post("/v1/chat/completions", json=_payload(_IMAGE))

    assert failed.status == 500
    assert failed_body["error"]["type"] == "server_error"
    assert succeeded.status == 200
    assert len(encoder.raws) == 2
