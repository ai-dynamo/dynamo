# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the custom-encoder-only HTTP service."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from examples.custom_encoder.benchmark.encoder_only_server import (
    DUMMY_RESPONSE,
    create_app,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_JPEG = b"\xff\xd8jpeg-bytes\xff\xd9"


class _FakeEncoder:
    def __init__(self) -> None:
        self.raws: list[list[bytes]] = []
        self.failures_remaining = 0
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.shutdown_calls = 0

    async def encode(self, raws: list[bytes]) -> list[object]:
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


def _form(
    image: bytes = _JPEG,
    *,
    field_name: str = "image",
    content_type: str = "image/jpeg",
    extra: bool = False,
) -> FormData:
    form = FormData()
    form.add_field(
        field_name,
        io.BytesIO(image),
        filename="image.jpg",
        content_type=content_type,
    )
    if extra:
        form.add_field("metadata", "unexpected")
    return form


@asynccontextmanager
async def _client(encoder: _FakeEncoder) -> AsyncIterator[TestClient]:
    client = TestClient(TestServer(create_app(encoder)))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def test_multipart_request_forwards_raw_jpeg_and_returns_ten_bytes() -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        health = await client.get("/health")
        response = await client.post("/encode", data=_form())
        body = await response.read()

        assert health.status == 200
        assert await health.json() == {"status": "ready"}
        assert response.status == 200
        assert response.content_type == "text/plain"
        assert body == DUMMY_RESPONSE == b"encoder-ok"
        assert len(body) == 10
        assert encoder.raws == [[_JPEG]]

    assert encoder.shutdown_calls == 1


async def test_request_larger_than_aiohttp_default_is_accepted() -> None:
    encoder = _FakeEncoder()
    image = b"x" * 2_000_000
    async with _client(encoder) as client:
        response = await client.post("/encode", data=_form(image))

    assert response.status == 200
    assert encoder.raws == [[image]]


@pytest.mark.parametrize(
    ("form", "message"),
    [
        (_form(field_name="photo"), "field named 'image'"),
        (_form(content_type="image/png"), "must use image/jpeg"),
        (_form(image=b""), "must not be empty"),
        (_form(extra=True), "exactly one multipart field"),
    ],
)
async def test_invalid_multipart_requests_return_plain_errors(
    form: FormData, message: str
) -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        response = await client.post("/encode", data=form)
        body = await response.text()

    assert response.status == 400
    assert response.content_type == "text/plain"
    assert message in body
    assert encoder.raws == []


async def test_non_multipart_request_is_rejected() -> None:
    encoder = _FakeEncoder()
    async with _client(encoder) as client:
        response = await client.post("/encode", data=_JPEG)
        body = await response.text()

    assert response.status == 400
    assert "multipart/form-data" in body
    assert encoder.raws == []


async def test_encoder_failure_does_not_stop_later_requests() -> None:
    encoder = _FakeEncoder()
    encoder.failures_remaining = 1
    async with _client(encoder) as client:
        failed = await client.post("/encode", data=_form())
        failed_body = await failed.text()
        succeeded = await client.post("/encode", data=_form())

    assert failed.status == 500
    assert failed_body == "encoder failed"
    assert succeeded.status == 200
    assert len(encoder.raws) == 2
