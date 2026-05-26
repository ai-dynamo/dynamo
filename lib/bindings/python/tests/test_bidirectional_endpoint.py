# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the bidirectional Python engine registration API.

Coverage of the round-trip path (client driving multi-frame input,
asserting multi-frame output, exercising cancellation, exercising
input-stream-end-is-not-cancellation) requires a Python client surface
for `PushRouter::<serde_json::Value, _>::generate(ManyIn<_>)` which is
not exposed today. Those checks live as TODOs below and are covered by
the corresponding Rust-side bidirectional E2E test in
`lib/runtime/tests/bidirectional_e2e.rs` (which drives the bidirectional
engine through the same PushRouter path with a Rust engine).
"""

import asyncio

import pytest

from dynamo._core import PyAsyncRequestStream

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_py_async_request_stream_symbol_exists():
    """Smoke check that the inbound-iterator pyclass is reachable from Python."""
    assert PyAsyncRequestStream is not None
    # Class is constructed Rust-side only; instantiating from Python without
    # an mpsc backing channel is intentionally not supported.
    with pytest.raises(TypeError):
        PyAsyncRequestStream()


def test_serve_bidirectional_endpoint_method_exists():
    """`Endpoint.serve_bidirectional_endpoint` is exposed on the pyclass."""
    from dynamo._core import Endpoint

    assert hasattr(
        Endpoint, "serve_bidirectional_endpoint"
    ), "Endpoint.serve_bidirectional_endpoint should be exposed by the bindings"


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_serve_bidirectional_endpoint_starts(temp_file_store, runtime):
    """
    Register a bidirectional engine and verify the worker boots without
    raising. The serve coroutine blocks until shutdown, so we run it
    briefly under wait_for and expect a timeout, which confirms the
    handler attached cleanly to the request-plane server.
    """
    handler_invocations = []

    async def generate(request_stream, context):
        # Pulls frames from the inbound iterator until the caller closes
        # the input half, then exits naturally. Real engines would keep
        # yielding response chunks past input end; this skeleton just
        # records that it was invoked.
        handler_invocations.append(True)
        try:
            async for frame in request_stream:
                yield {"echo": frame}
        except StopAsyncIteration:
            return

    endpoint = runtime.endpoint("test_bidi_register.backend.generate")

    async def init_server():
        await endpoint.serve_bidirectional_endpoint(generate)

    server_task = asyncio.create_task(init_server())

    # serve_bidirectional_endpoint blocks indefinitely on success; a
    # timeout here means the worker has booted and is waiting for work.
    # A returned coroutine means setup failed before reaching the serve
    # loop, which surfaces as an early exception we want to bubble up.
    #
    # `asyncio.shield` prevents `wait_for` from cancelling `server_task`
    # when the timeout fires; without it the task would always be in the
    # cancelled state by the time the assertion below runs.
    try:
        await asyncio.wait_for(asyncio.shield(server_task), timeout=0.5)
    except asyncio.TimeoutError:
        pass

    assert not server_task.done(), (
        f"serve_bidirectional_endpoint exited unexpectedly: "
        f"{server_task.exception()!r}"
    )

    server_task.cancel()
    try:
        await server_task
    except (asyncio.CancelledError, Exception):
        pass

    # Handler was never invoked (no caller drove a request), but the
    # registration round-trip wired through the bindings → Rust adapter →
    # Ingress::for_engine path without raising.
    assert (
        handler_invocations == []
    ), "no client was driven; handler should not have been called"


# TODO: end-to-end round-trip tests (multi-frame echo, cancellation,
# input-stream-end-is-not-cancellation) need a Python client surface for
# bidirectional `PushRouter::<T, U>::generate(ManyIn<T>)`. Tracked under
# the follow-up to expose a `Client.generate_bidirectional(async_iter)`
# method; until then, the Rust-side test in
# `lib/runtime/tests/bidirectional_e2e.rs` exercises the same PushRouter
# code path through a Rust engine.
