# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import enum

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


class _Color(enum.IntEnum):
    RED = 7


class _Label(str):
    pass


def _types_response():
    """One value of each type handlers commonly yield, plus subclasses and
    out-of-i64 ints that the encoder must hand to its generic fallback."""
    return {
        "none": None,
        "bools": [True, False],
        "ints": [
            0,
            1,
            -1,
            127,
            128,
            65536,
            -(2**31),
            2**63 - 1,
            -(2**63),
            2**64 - 1,
        ],
        "floats": [0.0, -0.25, 1e300],
        "text": "héllo 中",
        "tuple": (1, "two", None, (3.5,)),
        "nested": {"empty_list": [], "empty_dict": {}, "rows": [{"id": 1, "ok": True}]},
        "int_subclass": _Color.RED,
        "str_subclass": _Label("label"),
    }


async def _generate(request, context):
    assert dict(context.metadata.items()) == request.get("expected_metadata", {})
    if request["kind"] == "types":
        yield _types_response()
        return
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
    try:
        await client.wait_for_instances()
        yield client
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True)
async def test_client_instances_snapshot(request_plane, request_plane_client):
    """Client.instances() exposes the endpoint's instances with transport
    details on each request plane: "tcp" carries a "host:port/..." address,
    "nats" carries a "nats_tcp" subject address."""
    expected_kind = "nats_tcp" if request_plane == "nats" else "tcp"

    instances = request_plane_client.instances()
    assert {i.instance_id for i in instances} == set(
        request_plane_client.instance_ids()
    )
    for instance in instances:
        assert instance.namespace == "direct-python-msgpack"
        assert instance.component == "backend"
        assert instance.endpoint == "generate"
        assert instance.device_type in (None, "cpu", "cuda")
        assert instance.transport.kind == expected_kind
        assert instance.transport.address  # populated on both planes


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


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_python_request_plane_response_value_types(request_plane_client):
    """Response values keep their meaning through the response encoder,
    including bool vs int, tuples, subclasses and ints outside i64."""
    stream = await request_plane_client.generate(
        {"kind": "types", "expected_metadata": {}}
    )
    responses = [response.data() async for response in stream]

    expected = _types_response()
    expected["tuple"] = [1, "two", None, [3.5]]
    expected["int_subclass"] = 7
    expected["str_subclass"] = "label"
    assert responses == [expected]

    data = responses[0]
    assert [type(value) for value in data["bools"]] == [bool, bool]
    assert all(type(value) is int for value in data["ints"])
    assert type(data["nested"]["rows"][0]["ok"]) is bool
    assert type(data["int_subclass"]) is int
