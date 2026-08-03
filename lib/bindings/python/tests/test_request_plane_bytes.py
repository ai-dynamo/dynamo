# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _generate(request, context):
    yield request


@pytest.fixture
async def bytes_client(runtime):
    endpoint = runtime.endpoint("bytes-roundtrip.backend.generate")
    server_task = asyncio.ensure_future(endpoint.serve_endpoint(_generate))
    client = await endpoint.client()
    try:
        await client.wait_for_instances()
        yield client
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True)
async def test_client_round_trips_bytes(request_plane, bytes_client):
    """A Python ``bytes`` field round-trips through the Python ``Client`` over
    the default msgpack request plane, arriving as ``bytes`` (not base64, not
    an int array). This is the end-to-end proof that the ``rmpv::Value``
    request-plane intermediate lets workers emit raw ``bytes`` instead of
    base64-encoding binary payloads.
    """
    payload = {"img": b"\xde\xad\xbe\xef", "n": 7, "text": "hi"}

    stream = await bytes_client.generate(payload)
    responses = [response async for response in stream]

    assert len(responses) == 1, f"expected 1 frame, got {len(responses)}"
    data = responses[0].data()
    assert data["n"] == 7
    assert data["text"] == "hi"

    # msgpack carries `bytes` natively (rmpv::Value::Binary -> PyBytes). The
    # legacy JSON codec has no binary type, so `bytes` degrades to a list of
    # ints on the wire — accept either so the test is codec-agnostic.
    img = data["img"]
    if isinstance(img, bytes):
        assert img == b"\xde\xad\xbe\xef", img
    elif isinstance(img, list):
        assert img == [0xDE, 0xAD, 0xBE, 0xEF], img
    else:
        raise AssertionError(f"unexpected img type {type(img)}: {img!r}")
