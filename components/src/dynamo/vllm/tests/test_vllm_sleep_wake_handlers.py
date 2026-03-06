# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

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
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._sleep_wake_lock = asyncio.Lock()
    handler._sleeping_tags = set()
    return handler


@pytest.mark.asyncio
async def test_wake_up_before_sleep_is_noop():
    handler = _make_handler()

    result = await handler.wake_up({})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_sleep_and_wake_are_tag_aware_and_resume_on_full_restore():
    handler = _make_handler()

    first_sleep = await handler.sleep({"level": 2})
    partial_wake = await handler.wake_up({"tags": ["weights"]})
    final_wake = await handler.wake_up({"tags": ["kv_cache"]})

    assert first_sleep["status"] == "ok"
    assert partial_wake["status"] == "ok"
    assert final_wake["status"] == "ok"

    handler.engine_client.pause_generation.assert_awaited_once()
    handler.engine_client.sleep.assert_awaited_once_with(2)
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    assert handler.engine_client.wake_up.await_args_list == [
        call(["weights"]),
        call(["kv_cache"]),
    ]
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_sleep_is_idempotent():
    handler = _make_handler()

    first_sleep = await handler.sleep({"level": 1})
    second_sleep = await handler.sleep({"level": 1})

    assert first_sleep["status"] == "ok"
    assert second_sleep["status"] == "ok"
    handler.engine_client.sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_sleep_returns_error_for_unexpected_unregister_failure():
    handler = _make_handler()
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery backend down")
    )

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    handler.engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_sleep_ignores_benign_unregister_failure():
    handler = _make_handler()
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("endpoint already absent")
    )

    result = await handler.sleep({"level": 1})

    assert result["status"] == "ok"
    handler.engine_client.pause_generation.assert_awaited_once()
    handler.engine_client.sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_wake_up_returns_error_for_unexpected_register_failure():
    handler = _make_handler()
    handler._sleeping_tags = {"weights"}
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "error"
    handler.engine_client.wake_up.assert_awaited_once_with(["weights"])
    handler.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_wake_up_ignores_benign_register_failure():
    handler = _make_handler()
    handler._sleeping_tags = {"weights"}
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("already registered")
    )

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_awaited_once_with(["weights"])
    handler.engine_client.resume_generation.assert_awaited_once()
