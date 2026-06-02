# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "Skipping: CUDA not available (tensorrt_llm import requires GPU).",
        allow_module_level=True,
    )
pytest.importorskip(
    "dynamo._core",
    reason="dynamo Rust Python bindings are required for TrtllmLLMEngine",
)
pytest.importorskip(
    "dynamo.nixl_connect",
    reason="NIXL bindings are required to import TrtllmLLMEngine",
)
pytest.importorskip("tensorrt_llm")

from dynamo.trtllm.llm_engine import TrtllmLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _make_engine() -> TrtllmLLMEngine:
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    engine._quiesce_lock = asyncio.Lock()
    engine._inflight_lock = asyncio.Lock()
    engine._inflight_requests = 0
    engine._no_inflight_requests = asyncio.Event()
    engine._no_inflight_requests.set()
    engine._reject_new_requests = False
    engine._resume_recovery_required = False

    controller = MagicMock()
    controller.is_quiesced = False
    controller.needs_resume_recovery = False

    async def _quiesce(tags=None):
        controller.is_quiesced = True
        return True

    async def _resume(tags=None):
        return True

    def _mark_resumed():
        controller.is_quiesced = False

    controller.quiesce = AsyncMock(side_effect=_quiesce)
    controller.resume = AsyncMock(side_effect=_resume)
    controller.mark_resumed = MagicMock(side_effect=_mark_resumed)
    engine._quiesce_controller = controller
    return engine


@pytest.mark.asyncio
async def test_engine_controls_expose_trtllm_management_capabilities():
    controls = _make_engine().supported_controls()

    assert controls == {
        "release_memory_occupation",
        "resume_memory_occupation",
    }


@pytest.mark.asyncio
async def test_release_and_resume_delegate_to_quiesce_controller():
    engine = _make_engine()

    release_result = await engine.engine_control(
        "release_memory_occupation", {"tags": ["kv_cache"]}
    )
    resume_result = await engine.engine_control(
        "resume_memory_occupation", {"tags": ["kv_cache"]}
    )

    assert release_result["status"] == "ok"
    assert resume_result["status"] == "ok"
    engine._quiesce_controller.quiesce.assert_awaited_once_with(["kv_cache"])
    engine._quiesce_controller.resume.assert_awaited_once_with(["kv_cache"])
    engine._quiesce_controller.mark_resumed.assert_called_once_with()
    assert engine._reject_new_requests is False


@pytest.mark.asyncio
async def test_release_rejects_new_requests_until_resume():
    engine = _make_engine()

    await engine.release_memory_occupation({})

    assert await engine._mark_request_started() is False

    await engine.resume_memory_occupation({})
    assert await engine._mark_request_started() is True
    await engine._mark_request_finished()


@pytest.mark.asyncio
async def test_resume_clears_reject_after_failed_quiesce():
    engine = _make_engine()
    engine._quiesce_controller.quiesce = AsyncMock(
        side_effect=RuntimeError("sleep failed")
    )

    release = await engine.release_memory_occupation({})

    assert release["status"] == "error"
    assert engine._reject_new_requests is True
    assert engine._resume_recovery_required is True
    assert await engine._mark_request_started() is False

    resume = await engine.resume_memory_occupation({})

    assert resume["status"] == "ok"
    engine._quiesce_controller.resume.assert_not_awaited()
    engine._quiesce_controller.mark_resumed.assert_called_once_with()
    assert engine._reject_new_requests is False
    assert engine._resume_recovery_required is False


@pytest.mark.asyncio
async def test_resume_recovers_partial_quiesce_before_accepting_requests():
    engine = _make_engine()

    async def _quiesce_failed(tags=None):
        engine._quiesce_controller.needs_resume_recovery = True
        raise RuntimeError("weights failed")

    async def _resume(tags=None):
        engine._quiesce_controller.needs_resume_recovery = False
        return True

    engine._quiesce_controller.quiesce = AsyncMock(side_effect=_quiesce_failed)
    engine._quiesce_controller.resume = AsyncMock(side_effect=_resume)

    release = await engine.release_memory_occupation({})

    assert release["status"] == "error"
    assert engine._reject_new_requests is True
    assert engine._resume_recovery_required is True

    resume = await engine.resume_memory_occupation({})

    assert resume["status"] == "ok"
    engine._quiesce_controller.resume.assert_awaited_once_with(None)
    engine._quiesce_controller.mark_resumed.assert_called_once_with()
    assert engine._reject_new_requests is False
    assert engine._resume_recovery_required is False


@pytest.mark.asyncio
async def test_generate_returns_error_chunk_when_rejecting_requests():
    engine = _make_engine()
    await engine._set_reject_new_requests(True)

    chunks = []
    async for chunk in engine.generate({"token_ids": []}, MagicMock()):
        chunks.append(chunk)

    assert chunks == [
        {
            "finish_reason": "error",
            "token_ids": [],
            "index": 0,
            "completion_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    ]
