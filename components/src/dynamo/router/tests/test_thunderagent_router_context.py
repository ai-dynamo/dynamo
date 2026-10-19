# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trace context forwarding in the custom Python router."""

from unittest.mock import AsyncMock

import pytest

from dynamo.thunderagent_router.__main__ import ThunderAgentRouterHandler

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.asyncio
async def test_passthrough_forwards_context_to_kv_router() -> None:
    handler = ThunderAgentRouterHandler(runtime=None, config=None)  # type: ignore[arg-type]
    handler._scheduler = object()
    handler._kv_router = AsyncMock()

    context = object()

    async def responses():
        yield {"token_ids": [4], "finish_reason": "stop"}

    handler._kv_router.generate_from_request.return_value = responses()
    request = {"model": "test-model", "token_ids": [1, 2, 3]}

    results = [output async for output in handler.generate(request, context=context)]

    assert results == [{"token_ids": [4], "finish_reason": "stop"}]
    handler._kv_router.generate_from_request.assert_awaited_once()
    _, kwargs = handler._kv_router.generate_from_request.await_args
    assert kwargs == {"context": context}
