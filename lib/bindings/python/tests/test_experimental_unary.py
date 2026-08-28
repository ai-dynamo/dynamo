# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from dynamo._core import Context
from dynamo.experimental.endpoint import UnaryClient, serve_unary_endpoint
from dynamo.experimental.llm import LLMUnaryClient

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.forked,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.core,
]


@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_unary_adapters_round_trip_over_tcp(runtime: Any) -> None:
    endpoint = runtime.endpoint(f"experimental-{uuid4().hex}.backend.generate")

    async def handler(request: dict[str, Any], *, context: Any) -> dict[str, Any]:
        return {"value": request["value"], "request_id": context.id()}

    server_task = asyncio.create_task(serve_unary_endpoint(endpoint, handler))
    try:
        client = await endpoint.client()
        await client.wait_for_instances()

        response = await UnaryClient(client).complete({"value": "hello"})

        assert response["value"] == "hello"
        assert response["request_id"]
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_unary_client_propagates_cancellation_to_nested_endpoint(
    runtime: Any,
) -> None:
    suffix = uuid4().hex
    inner_endpoint = runtime.endpoint(f"experimental-{suffix}.inner.generate")
    outer_endpoint = runtime.endpoint(f"experimental-{suffix}.outer.generate")
    inner_started = asyncio.Event()
    inner_stopped = asyncio.Event()

    async def inner_handler(request: Any, *, context: Context) -> Any:
        del request
        inner_started.set()
        await context.async_killed_or_stopped()
        inner_stopped.set()
        raise asyncio.CancelledError

    inner_server = asyncio.create_task(
        serve_unary_endpoint(inner_endpoint, inner_handler)
    )
    outer_server: asyncio.Task[None] | None = None
    call: asyncio.Task[Any] | None = None
    try:
        inner_client = await inner_endpoint.client()
        await inner_client.wait_for_instances()

        async def outer_handler(request: Any, *, context: Context) -> Any:
            return await UnaryClient(inner_client).complete(request, context=context)

        outer_server = asyncio.create_task(
            serve_unary_endpoint(outer_endpoint, outer_handler)
        )
        outer_client = await outer_endpoint.client()
        await outer_client.wait_for_instances()

        context = Context()
        call = asyncio.create_task(
            UnaryClient(outer_client).complete({"value": "hello"}, context=context)
        )
        await asyncio.wait_for(inner_started.wait(), timeout=2)

        context.stop_generating()

        await asyncio.wait_for(inner_stopped.wait(), timeout=2)
    finally:
        if call is not None:
            call.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError, ValueError):
                await call
        for server_task in (outer_server, inner_server):
            if server_task is not None:
                server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await server_task


@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_llm_unary_client_collects_a_generate_endpoint(runtime: Any) -> None:
    endpoint = runtime.endpoint(f"experimental-{uuid4().hex}.backend.generate")

    async def generate(
        request: dict[str, Any], *, context: Any
    ) -> AsyncIterator[dict[str, Any]]:
        del request, context
        yield {"token_ids": [7], "index": 0}
        yield {
            "token_ids": [8, 9],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {"completion_tokens": 3},
        }

    server_task = asyncio.create_task(endpoint.serve_endpoint(generate))
    try:
        client = await endpoint.client()
        await client.wait_for_instances()

        response = await LLMUnaryClient(client).complete(
            {
                "token_ids": [1, 2],
                "sampling_options": {"n": 1},
                "output_options": {},
            }
        )

        assert response == {
            "token_ids": [7, 8, 9],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {"completion_tokens": 3},
        }
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
