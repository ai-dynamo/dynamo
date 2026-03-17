# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang.request_handlers.handler_base import (
    DEFAULT_MEMORY_OCCUPATION_TAGS,
    BaseWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _stub_sglang_io_struct(monkeypatch):
    """Keep unit tests independent from CUDA-only sglang imports."""

    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class _Req:
        def __init__(self, tags=None):
            self.tags = tags

    io_struct.PauseGenerationReqInput = _Req
    io_struct.ReleaseMemoryOccupationReqInput = _Req
    io_struct.ResumeMemoryOccupationReqInput = _Req
    io_struct.ContinueGenerationReqInput = _Req

    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)


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
    handler._memory_released = False
    return handler


@pytest.mark.asyncio
async def test_resume_before_release_is_noop():
    handler = _make_handler()

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    assert result["message"] == "Memory already resumed"
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
    assert second_release["message"] == "Memory already released"
    assert second_resume["message"] == "Memory already resumed"
    assert DEFAULT_MEMORY_OCCUPATION_TAGS == ["kv_cache", "weights"]

    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == DEFAULT_MEMORY_OCCUPATION_TAGS
    assert resume_req.tags == DEFAULT_MEMORY_OCCUPATION_TAGS

    handler.engine.tokenizer_manager.pause_generation.assert_awaited_once()
    handler.engine.tokenizer_manager.release_memory_occupation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine.tokenizer_manager.resume_memory_occupation.assert_awaited_once()
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_uses_default_tags_even_when_request_specifies_subset():
    handler = _make_handler()

    await handler.release_memory_occupation({"tags": ["weights"]})
    resume_result = await handler.resume_memory_occupation({"tags": ["weights"]})

    assert resume_result["status"] == "ok"
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert resume_req.tags == DEFAULT_MEMORY_OCCUPATION_TAGS
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_with_no_sleeping_state_is_noop():
    handler = _make_handler()

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "ok"
    assert result["message"] == "Memory already resumed"
    handler.engine.tokenizer_manager.resume_memory_occupation.assert_not_awaited()
    handler.engine.tokenizer_manager.continue_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_returns_error_when_worker_has_no_tokenizer_manager():
    handler = _make_handler()
    handler.engine = None

    result = await handler.release_memory_occupation({})

    assert result == {
        "status": "error",
        "message": "memory control not supported on this worker",
    }
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_returns_error_when_worker_has_no_tokenizer_manager():
    handler = _make_handler()
    handler.engine = None

    result = await handler.resume_memory_occupation({})

    assert result == {
        "status": "error",
        "message": "memory control not supported on this worker",
    }
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()
