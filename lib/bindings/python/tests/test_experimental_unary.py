# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from dynamo.experimental.endpoint import UnaryClient, serve_unary_endpoint
from dynamo.experimental.llm import LLMUnaryClient

pytestmark = [
    pytest.mark.asyncio,
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
