# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.request_handlers.handler_base import HandlerBase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _TestWorkerHandler(HandlerBase):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._sleep_wake_lock = asyncio.Lock()
    handler._inflight_lock = asyncio.Lock()
    handler._inflight_requests = 0
    handler._no_inflight_requests = asyncio.Event()
    handler._no_inflight_requests.set()
    handler._memory_released = False
    handler._reject_new_requests = False
    handler._wait_for_inflight_requests = AsyncMock()
    handler._call_collective_rpc = AsyncMock()
    handler._split_memory_tags = MagicMock(return_value=(["kv_cache"], False))
    return handler


@pytest.mark.asyncio
async def test_resume_before_release_is_noop():
    handler = _make_handler()

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    assert result["message"] == "Memory already resumed"
    handler._call_collective_rpc.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_request_started_respects_reject_flag():
    handler = _make_handler()
    handler._reject_new_requests = True

    started = await handler._mark_request_started()

    assert started is False
    assert handler._inflight_requests == 0


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
    assert second_release["message"] == "Memory already released"
    assert second_resume["message"] == "Memory already resumed"

    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler._wait_for_inflight_requests.assert_awaited_once_with(30.0)
    handler._call_collective_rpc.assert_any_await("sleep", ["kv_cache"])

    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    handler._call_collective_rpc.assert_any_await("wakeup", ["kv_cache"])


@pytest.mark.asyncio
async def test_release_uses_tags_from_request_body():
    handler = _make_handler()

    result = await handler.release_memory_occupation({"tags": ["kv_cache"]})

    assert result["status"] == "ok"
    handler._split_memory_tags.assert_called_once_with(["kv_cache"])
    handler._call_collective_rpc.assert_awaited_once_with("sleep", ["kv_cache"])


@pytest.mark.asyncio
async def test_release_returns_error_for_invalid_timeout():
    handler = _make_handler()

    result = await handler.release_memory_occupation({"timeout_s": "bad-timeout"})

    assert result["status"] == "error"
    assert "timeout_s" in result["message"]
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_awaited()
    handler._wait_for_inflight_requests.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_returns_error_for_unregister_failure():
    handler = _make_handler()
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery backend down")
    )

    result = await handler.release_memory_occupation({})

    assert result["status"] == "error"
    assert handler._reject_new_requests is False
    handler._wait_for_inflight_requests.assert_not_awaited()
    handler._call_collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_returns_error_for_register_failure():
    handler = _make_handler()
    handler._memory_released = True
    handler._reject_new_requests = True
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "error"
    handler._call_collective_rpc.assert_awaited_once_with("wakeup", ["kv_cache"])
    assert handler._memory_released is True
    assert handler._reject_new_requests is True


@pytest.mark.asyncio
async def test_resume_uses_configured_gms_lock_mode(monkeypatch):
    handler = _make_handler()
    handler._memory_released = True
    handler._split_memory_tags = MagicMock(return_value=([], True))

    from gpu_memory_service.common.types import RequestedLockType
    from gpu_memory_service.integrations import trtllm as trtllm_integration

    manager = MagicMock()
    manager.is_unmapped = True
    monkeypatch.setattr(HandlerBase, "_get_gms_manager", staticmethod(lambda: manager))
    monkeypatch.setattr(
        trtllm_integration,
        "get_gms_lock_mode",
        lambda: RequestedLockType.RW_OR_RO,
    )

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    manager.connect.assert_called_once_with(RequestedLockType.RW_OR_RO)
    manager.remap_all_vas.assert_called_once_with()


@pytest.mark.asyncio
async def test_release_uses_local_fallback_when_collective_rpc_is_unsupported():
    handler = _make_handler()
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError(
            "Executor type <class '...'> does not support collective RPC."
        )
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=True)
    handler._call_local_virtual_memory_method = MagicMock()

    result = await handler.release_memory_occupation({"tags": ["kv_cache"]})

    assert result["status"] == "ok"
    assert "skipped_tags" not in result
    handler._call_local_virtual_memory_method.assert_called_once_with(
        "sleep", ["kv_cache"]
    )


@pytest.mark.asyncio
async def test_release_skips_kv_cache_when_collective_rpc_is_unsupported_in_multi_rank():
    handler = _make_handler()
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError(
            "Executor type <class '...'> does not support collective RPC."
        )
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=False)
    handler._call_local_virtual_memory_method = MagicMock()

    result = await handler.release_memory_occupation({"tags": ["kv_cache"]})

    assert result["status"] == "ok"
    assert result["skipped_tags"] == ["kv_cache"]
    handler._call_local_virtual_memory_method.assert_not_called()


@pytest.mark.asyncio
async def test_resume_uses_local_fallback_when_collective_rpc_is_unsupported():
    handler = _make_handler()
    handler._memory_released = True
    handler._reject_new_requests = True
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError(
            "Executor type <class '...'> does not support collective RPC."
        )
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=True)
    handler._call_local_virtual_memory_method = MagicMock()

    result = await handler.resume_memory_occupation({"tags": ["kv_cache"]})

    assert result["status"] == "ok"
    assert "skipped_tags" not in result
    handler._call_local_virtual_memory_method.assert_called_once_with(
        "wakeup", ["kv_cache"]
    )
