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
    if request["kind"] == "explicit-none":
        yield {
            "_dynamo_annotated": True,
            "data": {"nested": {"value": 42}},
            "id": None,
            "event": None,
            "comment": None,
            "error": None,
        }
        return
    if request["kind"] == "reused-mutable":
        shared = {"sequence": 0}
        for sequence in range(64):
            shared["sequence"] = sequence
            yield shared
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

    # instance_tcp_addresses() only includes instances whose transport is TCP,
    # so an instance may be absent from the dict on other request planes
    # (e.g. NATS). Only assert on the entries that are present: they must be a
    # subset of instance_ids() and each address must be a "host:port/..."
    # string. On the TCP request plane every instance has one.
    tcp_addresses = client.instance_tcp_addresses()
    assert set(tcp_addresses) <= set(client.instance_ids())
    for address in tcp_addresses.values():
        assert ":" in address

    yield client

    server_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await server_task


@pytest.mark.asyncio
@pytest.mark.timeout(30)
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

    stream = await request_plane_client.generate(
        {"kind": "explicit-none", "expected_metadata": {}}
    )
    explicit_none_responses = [response async for response in stream]
    assert len(explicit_none_responses) == 1
    assert explicit_none_responses[0].data() == {"nested": {"value": 42}}
    assert explicit_none_responses[0].id() is None
    assert explicit_none_responses[0].event() is None
    assert explicit_none_responses[0].comments() is None

    stream = await request_plane_client.generate(
        {"kind": "reused-mutable", "expected_metadata": {}}
    )
    reused_mutable_responses = [response.data() async for response in stream]
    assert reused_mutable_responses == [
        {"sequence": sequence} for sequence in range(64)
    ]

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
