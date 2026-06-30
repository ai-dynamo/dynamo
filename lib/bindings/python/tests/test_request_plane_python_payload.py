# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _generate(request, context):
    assert dict(context.metadata.items()) == request.get("expected_metadata", {})
    if request["kind"] == "error":
        raise ValueError("direct adapter test error")
    if request["kind"] == "malformed":
        yield object()
        return

    yield request["payload"]
    yield {
        "_dynamo_annotated": True,
        "data": {"annotated": request["payload"]},
        "id": "chunk-2",
        "event": "delta",
        "comment": ["direct", "python"],
    }


@pytest.fixture
async def request_plane_client(runtime):
    endpoint = runtime.endpoint("direct-python-msgpack.backend.generate")
    health_payload = {
        "kind": "normal",
        "payload": {"health": True},
        "expected_metadata": {},
    }
    server_task = asyncio.ensure_future(
        endpoint.serve_endpoint(_generate, health_check_payload=health_payload)
    )
    client = await endpoint.client()
    await client.wait_for_instances()

    yield client

    server_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await server_task


@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_python_request_plane_plain_annotated_error_and_malformed_frames(
    request_plane_client,
):
    payload = {
        "text": "hello 中",
        "tokens": [1, 2, 65535],
        "nested": {"stream": True, "temperature": 0.25},
        "nullable": None,
    }
    request = {
        "kind": "normal",
        "payload": payload,
        "expected_metadata": {"trace": "adapter-test"},
    }

    from dynamo.runtime import Context

    context = Context()
    context.metadata["trace"] = "adapter-test"
    stream = await request_plane_client.generate(request, context=context)
    responses = [response async for response in stream]

    assert len(responses) == 2
    assert responses[0].data() == payload
    assert responses[1].data() == {"annotated": payload}
    assert responses[1].id() == "chunk-2"
    assert responses[1].event() == "delta"
    assert responses[1].comments() == ["direct", "python"]

    for kind, message in [
        ("error", "direct adapter test error"),
        ("malformed", "failed serializing Python response"),
    ]:
        stream = await request_plane_client.generate(
            {"kind": kind, "expected_metadata": {}}
        )
        with pytest.raises(ValueError, match=message):
            async for _ in stream:
                pass
