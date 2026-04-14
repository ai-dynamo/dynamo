# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM sleep/wake handler logic.

Tests cover in-flight tracking, reject-flag, and sleep/wake delegation to
TRTLLMEngineQuiesceController without requiring a real GPU or TRT-LLM engine.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


def _make_handler() -> _ConcreteHandler:
    """Create a HandlerBase subclass with mocked quiesce controller and endpoint."""
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
    handler._reject_new_requests = False
    # Mock the quiesce controller that release/resume delegate to
    handler._quiesce_controller = MagicMock()
    handler._quiesce_controller.is_quiesced = False
    handler._quiesce_controller.quiesce = AsyncMock(return_value=True)
    handler._quiesce_controller.resume = AsyncMock(return_value=True)
    handler._quiesce_controller.mark_resumed = MagicMock()
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
async def test_release_is_noop_when_already_quiesced():
    handler = _make_handler()
    handler._quiesce_controller.is_quiesced = True
    result = await handler.release_memory_occupation({})
    assert result["status"] == "ok"
    assert "already released" in result["message"]
    handler._quiesce_controller.quiesce.assert_not_called()


@pytest.mark.asyncio
async def test_release_returns_error_for_non_numeric_timeout():
    handler = _make_handler()
    result = await handler.release_memory_occupation({"timeout_s": "bad"})
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_release_delegates_to_quiesce_controller():
    handler = _make_handler()
    result = await handler.release_memory_occupation({})
    assert result["status"] == "ok"
    handler._quiesce_controller.quiesce.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_release_passes_tags_to_controller():
    handler = _make_handler()
    result = await handler.release_memory_occupation({"tags": ["weights"]})
    assert result["status"] == "ok"
    handler._quiesce_controller.quiesce.assert_awaited_once_with(["weights"])


@pytest.mark.asyncio
async def test_release_unregisters_endpoint_and_restores_on_error():
    handler = _make_handler()
    handler._quiesce_controller.quiesce = AsyncMock(
        side_effect=RuntimeError("engine error")
    )
    result = await handler.release_memory_occupation({})
    assert result["status"] == "error"
    handler.generate_endpoint.unregister_endpoint_instance.assert_called_once()
    handler.generate_endpoint.register_endpoint_instance.assert_called_once()
    assert not handler._reject_new_requests


# ---------------------------------------------------------------------------
# resume_memory_occupation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_is_noop_when_not_quiesced():
    handler = _make_handler()
    result = await handler.resume_memory_occupation({})
    assert result["status"] == "ok"
    assert "already resumed" in result["message"]
    handler._quiesce_controller.resume.assert_not_called()


@pytest.mark.asyncio
async def test_release_and_resume_round_trip():
    handler = _make_handler()
    release = await handler.release_memory_occupation({})
    assert release["status"] == "ok"

    # After release, controller reports quiesced
    handler._quiesce_controller.is_quiesced = True

    resume = await handler.resume_memory_occupation({})
    assert resume["status"] == "ok"
    handler._quiesce_controller.resume.assert_awaited_once()
    handler._quiesce_controller.mark_resumed.assert_called_once()
    assert not handler._reject_new_requests
    handler.generate_endpoint.register_endpoint_instance.assert_called_once()


@pytest.mark.asyncio
async def test_resume_passes_tags_to_controller():
    handler = _make_handler()
    handler._quiesce_controller.is_quiesced = True
    result = await handler.resume_memory_occupation({"tags": ["kv_cache"]})
    assert result["status"] == "ok"
    handler._quiesce_controller.resume.assert_awaited_once_with(["kv_cache"])


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
