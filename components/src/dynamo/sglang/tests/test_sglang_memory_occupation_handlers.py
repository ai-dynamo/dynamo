# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture
def handler():
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            is_pause=False,
            pause_generation=AsyncMock(),
            release_memory_occupation=AsyncMock(),
            resume_memory_occupation=AsyncMock(),
            continue_generation=AsyncMock(),
            auto_create_handle_loop=MagicMock(),
            rid_to_state={},
            flush_cache=AsyncMock(return_value=SimpleNamespace(success=True)),
            clear_hicache_storage=AsyncMock(return_value=SimpleNamespace(success=True)),
            server_args=SimpleNamespace(hicache_storage_backend=None),
        )
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._engine_route_lock = asyncio.Lock()
    return handler


def _registered_engine_routes(handler, configured_routes=None):
    registered = {}

    class Runtime:
        def register_engine_route(self, path, route_handler):
            registered[path] = route_handler

    handler.config = SimpleNamespace(
        dynamo_args=SimpleNamespace(engine_routes=configured_routes or [])
    )
    handler.register_engine_routes(Runtime())
    return registered


def _make_native_manager_methods_routable(manager):
    mocks = {}
    for method_name in (
        "pause_generation",
        "continue_generation",
        "release_memory_occupation",
        "resume_memory_occupation",
    ):
        method_mock = getattr(manager, method_name)
        mocks[method_name] = method_mock

        async def route_method(_method_mock=method_mock, **kwargs):
            return await _method_mock(**kwargs)

        setattr(manager, method_name, route_method)
    return mocks


@pytest.mark.asyncio
async def test_native_memory_routes_follow_sglang_pause_state(handler):
    manager = handler.engine.tokenizer_manager
    method_mocks = _make_native_manager_methods_routable(manager)

    async def pause_generation(**_kwargs):
        manager.is_pause = True

    async def continue_generation(**_kwargs):
        manager.is_pause = False

    method_mocks["pause_generation"].side_effect = pause_generation
    method_mocks["continue_generation"].side_effect = continue_generation
    routes = _registered_engine_routes(handler)

    assert {
        "pause_generation",
        "continue_generation",
        "release_memory_occupation",
        "resume_memory_occupation",
    }.issubset(routes)

    await routes["pause_generation"]({})
    await routes["release_memory_occupation"]({"tags": ["weights", "kv_cache"]})
    await routes["resume_memory_occupation"]({"tags": ["weights"]})

    assert manager.is_pause is True
    assert handler.generate_endpoint.unregister_endpoint_instance.await_count == 3
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    await routes["resume_memory_occupation"]({"tags": ["kv_cache"]})

    assert handler.generate_endpoint.unregister_endpoint_instance.await_count == 4
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    await routes["continue_generation"]({})

    assert manager.is_pause is False
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_configured_engine_route_overrides_default_and_syncs_discovery(handler):
    custom_pause_mock = AsyncMock(return_value={"custom": True})

    async def custom_pause():
        return await custom_pause_mock()

    handler.engine.custom_pause = custom_pause
    handler.engine.tokenizer_manager.is_pause = True
    method_mocks = _make_native_manager_methods_routable(
        handler.engine.tokenizer_manager
    )
    routes = _registered_engine_routes(handler, ["pause_generation=custom_pause"])

    result = await routes["pause_generation"]({})

    assert result == {"custom": True}
    custom_pause_mock.assert_awaited_once_with()
    method_mocks["pause_generation"].assert_not_awaited()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_kv_blocks_flushes_sglang_cache(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="none"
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "success", "message": "KV cache cleared"}]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_clears_configured_sglang_external_cache(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="nixl"
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "success", "message": "KV cache cleared"}]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_clear_kv_blocks_rejects_active_sglang_requests(handler):
    handler.engine.tokenizer_manager.rid_to_state = {"request-1": object()}

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [
        {
            "status": "error",
            "message": "Cannot clear KV cache while requests are active",
        }
    ]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_not_called()
    handler.engine.tokenizer_manager.flush_cache.assert_not_awaited()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_returns_error_without_engine(handler):
    handler.engine = None

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [
        {
            "status": "error",
            "message": "KV cache clear not supported on this worker",
        }
    ]


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_flush_failure(handler):
    handler.engine.tokenizer_manager.flush_cache = AsyncMock(
        return_value=SimpleNamespace(success=False, message="cache busy")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "cache busy"}]
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_sglang_external_cache_failure(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="nixl"
    )
    handler.engine.tokenizer_manager.clear_hicache_storage = AsyncMock(
        return_value=SimpleNamespace(success=False, message="storage busy")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "storage busy"}]
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_flush_exception(handler):
    handler.engine.tokenizer_manager.flush_cache = AsyncMock(
        side_effect=RuntimeError("flush crashed")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "flush crashed"}]
