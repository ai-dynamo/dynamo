# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            pause_generation=AsyncMock(),
            release_memory_occupation=AsyncMock(),
            resume_memory_occupation=AsyncMock(),
            continue_generation=AsyncMock(),
        )
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._memory_occupation_lock = asyncio.Lock()
    handler._released_memory_tags = set()
    handler._memory_serving_active = True
    return handler


@pytest.mark.asyncio
async def test_resume_before_release_is_noop():
    handler = _make_handler()

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    handler.engine.tokenizer_manager.resume_memory_occupation.assert_not_awaited()
    handler.engine.tokenizer_manager.continue_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_and_resume_are_idempotent():
    handler = _make_handler()

    first_release = await handler.release_memory_occupation({})
    second_release = await handler.release_memory_occupation({})

    first_resume = await handler.resume_memory_occupation({})
    second_resume = await handler.resume_memory_occupation({})

    assert first_release["status"] == "ok"
    assert second_release["status"] == "ok"
    assert first_resume["status"] == "ok"
    assert second_resume["status"] == "ok"

    handler.engine.tokenizer_manager.pause_generation.assert_awaited_once()
    handler.engine.tokenizer_manager.release_memory_occupation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine.tokenizer_manager.resume_memory_occupation.assert_awaited_once()
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_resume_keeps_worker_paused_until_all_tags_resumed():
    handler = _make_handler()

    await handler.release_memory_occupation({"tags": ["weights", "kv_cache"]})
    partial_resume = await handler.resume_memory_occupation({"tags": ["weights"]})

    assert partial_resume["status"] == "ok"
    handler.engine.tokenizer_manager.continue_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    final_resume = await handler.resume_memory_occupation({"tags": ["kv_cache"]})

    assert final_resume["status"] == "ok"
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_retries_continue_when_memory_is_already_restored():
    handler = _make_handler()
    handler._memory_serving_active = False

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    handler.engine.tokenizer_manager.resume_memory_occupation.assert_not_awaited()
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
