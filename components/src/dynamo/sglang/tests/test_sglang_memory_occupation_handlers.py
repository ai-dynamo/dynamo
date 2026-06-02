# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang.request_handlers.handler_base import (
    BaseWorkerHandler,
    SGLangEngineQuiesceController,
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


@pytest.fixture
def handler():
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
    handler._quiesce_controller = SGLangEngineQuiesceController(handler.engine)
    handler._quiesce_lock = asyncio.Lock()
    return handler


@pytest.mark.asyncio
async def test_release_and_resume_are_idempotent(handler):
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

    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags is None
    assert resume_req.tags is None

    handler.engine.tokenizer_manager.pause_generation.assert_awaited_once()
    handler.engine.tokenizer_manager.release_memory_occupation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine.tokenizer_manager.resume_memory_occupation.assert_awaited_once()
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_occupation_handlers_forward_tags_exactly(handler):
    await handler.release_memory_occupation({"tags": []})
    resume_result = await handler.resume_memory_occupation({"tags": []})

    assert resume_result["status"] == "ok"
    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == []
    assert resume_req.tags == []

    handler.engine.tokenizer_manager.pause_generation.reset_mock()
    handler.engine.tokenizer_manager.release_memory_occupation.reset_mock()
    handler.engine.tokenizer_manager.resume_memory_occupation.reset_mock()
    handler.engine.tokenizer_manager.continue_generation.reset_mock()
    handler.generate_endpoint.unregister_endpoint_instance.reset_mock()
    handler.generate_endpoint.register_endpoint_instance.reset_mock()

    await handler.release_memory_occupation({"tags": ["weights"]})
    resume_result = await handler.resume_memory_occupation({})

    assert resume_result["status"] == "ok"
    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == ["weights"]
    assert resume_req.tags is None
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_recovers_generation_pause_after_failed_release_rollback(handler):
    manager = handler.engine.tokenizer_manager
    manager.release_memory_occupation = AsyncMock(
        side_effect=RuntimeError("release failed")
    )
    failed_continue = AsyncMock(side_effect=RuntimeError("continue failed"))
    manager.continue_generation = failed_continue

    release_result = await handler.release_memory_occupation({})

    assert release_result["status"] == "error"
    assert handler._quiesce_controller.is_quiesced is False
    assert handler._quiesce_controller.needs_resume_recovery is True
    failed_continue.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    manager.continue_generation = AsyncMock()
    resume_result = await handler.resume_memory_occupation({})

    assert resume_result["status"] == "ok"
    manager.resume_memory_occupation.assert_not_awaited()
    manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    assert handler._quiesce_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_release_reregisters_after_clean_quiesce_rollback(handler):
    manager = handler.engine.tokenizer_manager
    manager.release_memory_occupation = AsyncMock(
        side_effect=RuntimeError("release failed")
    )

    release_result = await handler.release_memory_occupation({})

    # Rollback continued generation cleanly, so the engine is serving-safe again.
    assert release_result["status"] == "error"
    assert handler._quiesce_controller.is_quiesced is False
    assert handler._quiesce_controller.needs_resume_recovery is False
    manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    # The endpoint must rejoin the routing pool since resume will early-return.
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "endpoint_method"),
    [
        ("release_memory_occupation", "unregister_endpoint_instance"),
        ("resume_memory_occupation", "register_endpoint_instance"),
    ],
)
async def test_memory_control_returns_error_without_quiesce_controller(
    handler,
    method_name,
    endpoint_method,
):
    handler.engine = None
    handler._quiesce_controller = None

    result = await getattr(handler, method_name)({})

    assert result == {
        "status": "error",
        "message": "memory control not supported on this worker",
    }
    getattr(handler.generate_endpoint, endpoint_method).assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_keeps_quiesced_state_when_register_fails(handler):
    await handler.release_memory_occupation({})
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "error"
    assert handler._quiesce_controller is not None
    assert handler._quiesce_controller.is_quiesced is True
