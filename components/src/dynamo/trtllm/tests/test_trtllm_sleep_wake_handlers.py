# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM sleep/wake handler logic.

These tests cover the in-flight tracking, reject-flag, and sleep/wake round-trips
defined in HandlerBase without requiring a real GPU or TRT-LLM engine.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping: CUDA not available (tensorrt_llm import requires GPU).",
        allow_module_level=True,
    )

from dynamo.trtllm.request_handlers.handler_base import HandlerBase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


class _ConcreteHandler(HandlerBase):
    async def generate(self, request, context):
        yield {}


def _make_handler(*, split_tags_return=None) -> _ConcreteHandler:
    """Create a HandlerBase subclass with all external deps mocked out."""
    handler = _ConcreteHandler.__new__(_ConcreteHandler)
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
    if split_tags_return is None:
        split_tags_return = (["kv_cache"], False)
    handler._split_memory_tags = MagicMock(return_value=split_tags_return)
    return handler


# ---------------------------------------------------------------------------
# In-flight tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_request_started_respects_reject_flag():
    handler = _make_handler()
    await handler._set_reject_new_requests(True)
    assert not await handler._mark_request_started()
    assert handler._inflight_requests == 0


@pytest.mark.asyncio
async def test_mark_request_started_and_finished():
    handler = _make_handler()
    assert await handler._mark_request_started()
    assert handler._inflight_requests == 1
    assert not handler._no_inflight_requests.is_set()
    await handler._mark_request_finished()
    assert handler._inflight_requests == 0
    assert handler._no_inflight_requests.is_set()


@pytest.mark.asyncio
async def test_mark_request_finished_is_idempotent():
    handler = _make_handler()
    # Extra call when count is already 0 must not underflow
    await handler._mark_request_finished()
    assert handler._inflight_requests == 0


# ---------------------------------------------------------------------------
# release_memory_occupation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_is_noop_when_already_released():
    handler = _make_handler()
    handler._memory_released = True
    result = await handler.release_memory_occupation({})
    assert result["status"] == "ok"
    assert "already released" in result["message"]
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()


@pytest.mark.asyncio
async def test_release_returns_error_for_invalid_timeout():
    handler = _make_handler()
    result = await handler.release_memory_occupation({"timeout_s": -1})
    assert result["status"] == "error"
    assert "timeout_s" in result["message"]


@pytest.mark.asyncio
async def test_release_returns_error_for_non_numeric_timeout():
    handler = _make_handler()
    result = await handler.release_memory_occupation({"timeout_s": "bad"})
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_release_calls_collective_rpc_for_kv_cache():
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    result = await handler.release_memory_occupation({})
    assert result["status"] == "ok"
    handler._call_collective_rpc.assert_awaited_once_with("sleep", ["kv_cache"])
    assert handler._memory_released


@pytest.mark.asyncio
async def test_release_uses_local_fallback_when_collective_rpc_unsupported():
    """Single-rank executor: falls back to local VMM ops when collective RPC raises."""
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError("does not support collective rpc")
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=True)
    handler._call_local_virtual_memory_method = MagicMock()

    result = await handler.release_memory_occupation({})

    assert result["status"] == "ok"
    handler._call_local_virtual_memory_method.assert_called_once_with(
        "sleep", ["kv_cache"]
    )
    assert "skipped_tags" not in result


@pytest.mark.asyncio
async def test_release_skips_kv_cache_when_collective_rpc_unsupported_multi_rank():
    """Multi-rank executor: kv_cache sleep is skipped and reported."""
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError("does not support collective rpc")
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=False)

    result = await handler.release_memory_occupation({})

    assert result["status"] == "ok"
    assert "kv_cache" in result.get("skipped_tags", [])


@pytest.mark.asyncio
async def test_release_unregisters_endpoint_and_restores_on_error():
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    handler._call_collective_rpc = AsyncMock(side_effect=RuntimeError("engine error"))

    result = await handler.release_memory_occupation({})

    assert result["status"] == "error"
    handler.generate_endpoint.unregister_endpoint_instance.assert_called_once()
    handler.generate_endpoint.register_endpoint_instance.assert_called_once()
    assert not handler._memory_released
    assert not handler._reject_new_requests


# ---------------------------------------------------------------------------
# resume_memory_occupation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_is_noop_when_not_released():
    handler = _make_handler()
    result = await handler.resume_memory_occupation({})
    assert result["status"] == "ok"
    assert "already resumed" in result["message"]
    handler.generate_endpoint.register_endpoint_instance.assert_not_called()


@pytest.mark.asyncio
async def test_release_and_resume_round_trip():
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    release = await handler.release_memory_occupation({})
    assert release["status"] == "ok"
    assert handler._memory_released

    resume = await handler.resume_memory_occupation({})
    assert resume["status"] == "ok"
    assert not handler._memory_released
    assert not handler._reject_new_requests
    handler.generate_endpoint.register_endpoint_instance.assert_called_once()


@pytest.mark.asyncio
async def test_resume_uses_local_fallback_when_collective_rpc_unsupported():
    handler = _make_handler(split_tags_return=(["kv_cache"], False))
    handler._memory_released = True
    handler._call_collective_rpc = AsyncMock(
        side_effect=RuntimeError("does not support collective rpc")
    )
    handler._can_use_local_kv_sleep_fallback = MagicMock(return_value=True)
    handler._call_local_virtual_memory_method = MagicMock()

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    handler._call_local_virtual_memory_method.assert_called_once_with(
        "wakeup", ["kv_cache"]
    )


@pytest.mark.asyncio
async def test_resume_weights_via_gms_manager():
    """When GMS is initialised, resume reconnects and remaps the weights manager."""
    handler = _make_handler(split_tags_return=([], True))  # weights only
    handler._memory_released = True

    mock_manager = MagicMock()
    mock_manager.is_unmapped = True

    with (
        patch(
            "dynamo.trtllm.request_handlers.handler_base.HandlerBase._get_gms_manager",
            return_value=mock_manager,
        ),
        patch(
            "dynamo.trtllm.request_handlers.handler_base.HandlerBase._get_trtllm_gms_lock_mode",
            return_value=MagicMock(),
        ),
    ):
        result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    mock_manager.connect.assert_called_once()
    mock_manager.remap_all_vas.assert_called_once()


# ---------------------------------------------------------------------------
# generate_locally inflight guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_locally_rejected_when_sleeping():
    handler = _make_handler()
    handler._reject_new_requests = True

    chunks = []
    ctx = MagicMock()
    async for chunk in handler.generate_locally({"token_ids": []}, ctx):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "error" in str(chunks[0].get("finish_reason", ""))
