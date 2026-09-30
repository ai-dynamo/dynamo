# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        start_profile=AsyncMock(),
        stop_profile=AsyncMock(),
    )
    return handler


@pytest.mark.asyncio
async def test_start_profile_calls_engine_with_prefix():
    handler = _make_handler()

    result = await handler.start_profile({"profile_prefix": "prefix"})

    assert result["status"] == "ok"
    handler.engine_client.start_profile.assert_awaited_once_with(
        profile_prefix="prefix"
    )


@pytest.mark.asyncio
async def test_start_profile_without_prefix_passes_none():
    handler = _make_handler()

    result = await handler.start_profile({})

    assert result["status"] == "ok"
    handler.engine_client.start_profile.assert_awaited_once_with(profile_prefix=None)


@pytest.mark.asyncio
async def test_stop_profile_calls_engine():
    handler = _make_handler()

    result = await handler.stop_profile({})

    assert result["status"] == "ok"
    handler.engine_client.stop_profile.assert_awaited_once_with()
