# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sglang")


class _Req:
    def __init__(self, tags=None, **kwargs):
        self.tags = tags
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_io_struct_stub() -> types.ModuleType:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    io_struct.PauseGenerationReqInput = _Req
    io_struct.ReleaseMemoryOccupationReqInput = _Req
    io_struct.ResumeMemoryOccupationReqInput = _Req
    io_struct.ContinueGenerationReqInput = _Req
    io_struct.UpdateWeightFromDiskReqInput = _Req
    io_struct.UpdateWeightsFromTensorReqInput = _Req
    io_struct.UpdateWeightsFromDistributedReqInput = _Req
    io_struct.UpdateWeightsFromIPCReqInput = _Req
    io_struct.UpdateWeightVersionReqInput = _Req
    return io_struct


try:
    import sglang.srt.managers.io_struct  # noqa: F401
except ImportError:
    managers = types.ModuleType("sglang.srt.managers")
    managers.__path__ = []
    sys.modules.setdefault("sglang.srt.managers", managers)
    sys.modules.setdefault("sglang.srt.managers.io_struct", _make_io_struct_stub())

from dynamo.sglang.llm_engine import SglangLLMEngine  # noqa: E402
from dynamo.sglang.pause import SGLangEnginePauseController  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.post_merge,
]


@pytest.fixture(autouse=True)
def _stub_sglang_io_struct(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers.io_struct", _make_io_struct_stub()
    )
    monkeypatch.setattr("dynamo.sglang.pause.PauseGenerationReqInput", _Req)
    monkeypatch.setattr("dynamo.sglang.pause.ReleaseMemoryOccupationReqInput", _Req)
    monkeypatch.setattr("dynamo.sglang.pause.ResumeMemoryOccupationReqInput", _Req)
    monkeypatch.setattr("dynamo.sglang.pause.ContinueGenerationReqInput", _Req)
    monkeypatch.setattr("dynamo.sglang.llm_engine.UpdateWeightFromDiskReqInput", _Req)
    monkeypatch.setattr(
        "dynamo.sglang.llm_engine.UpdateWeightsFromTensorReqInput", _Req
    )
    monkeypatch.setattr(
        "dynamo.sglang.llm_engine.UpdateWeightsFromDistributedReqInput", _Req
    )
    monkeypatch.setattr("dynamo.sglang.llm_engine.UpdateWeightsFromIPCReqInput", _Req)
    monkeypatch.setattr("dynamo.sglang.llm_engine.UpdateWeightVersionReqInput", _Req)


def _make_engine() -> SglangLLMEngine:
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    tokenizer_manager = SimpleNamespace(
        pause_generation=AsyncMock(),
        release_memory_occupation=AsyncMock(),
        resume_memory_occupation=AsyncMock(),
        continue_generation=AsyncMock(),
        start_profile=AsyncMock(),
        stop_profile=AsyncMock(),
        update_weights_from_disk=AsyncMock(return_value=(True, "loaded", 3)),
        update_weights_from_tensor=AsyncMock(return_value=(True, "loaded")),
        update_weights_from_distributed=AsyncMock(return_value=(True, "loaded")),
        update_weights_from_ipc=AsyncMock(return_value=(True, "loaded")),
        initial_weights_loaded=False,
        abort_request=MagicMock(),
        server_args=SimpleNamespace(weight_version=None),
    )
    engine.engine = SimpleNamespace(tokenizer_manager=tokenizer_manager)
    engine._pause_controller = SGLangEnginePauseController(engine.engine)
    engine._pause_lock = asyncio.Lock()
    return engine


@pytest.mark.asyncio
async def test_engine_controls_expose_sglang_management_capabilities():
    controls = _make_engine().supported_controls()

    assert controls == {
        "start_profile",
        "stop_profile",
        "release_memory_occupation",
        "resume_memory_occupation",
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_distributed",
        "update_weights_from_ipc",
        "update_weight_version",
    }


@pytest.mark.asyncio
async def test_memory_routes_delegate_to_tokenizer_manager():
    engine = _make_engine()

    release_result = await engine.engine_control(
        "release_memory_occupation", {"tags": ["weights"]}
    )
    resume_result = await engine.engine_control(
        "resume_memory_occupation", {"tags": ["weights"]}
    )

    assert release_result["status"] == "ok"
    assert resume_result["status"] == "ok"

    release_req = (
        engine.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        engine.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == ["weights"]
    assert resume_req.tags == ["weights"]
    engine.engine.tokenizer_manager.pause_generation.assert_awaited_once()
    engine.engine.tokenizer_manager.continue_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_recovers_generation_pause_after_failed_release_rollback():
    engine = _make_engine()
    manager = engine.engine.tokenizer_manager
    manager.release_memory_occupation = AsyncMock(
        side_effect=RuntimeError("release failed")
    )
    failed_continue = AsyncMock(side_effect=RuntimeError("continue failed"))
    manager.continue_generation = failed_continue

    release_result = await engine.release_memory_occupation({})

    assert release_result["status"] == "error"
    assert engine._pause_controller.is_paused is False
    assert engine._pause_controller.needs_resume_recovery is True
    failed_continue.assert_awaited_once()

    manager.continue_generation = AsyncMock()
    resume_result = await engine.resume_memory_occupation({})

    assert resume_result["status"] == "ok"
    manager.resume_memory_occupation.assert_not_awaited()
    manager.continue_generation.assert_awaited_once()
    assert engine._pause_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_update_weights_from_disk_forwards_request_body():
    engine = _make_engine()

    result = await engine.update_weights_from_disk({"model_path": "/weights"})

    assert result == {
        "success": True,
        "message": "loaded",
        "num_paused_requests": 3,
    }
    (
        req,
        request_context,
    ) = engine.engine.tokenizer_manager.update_weights_from_disk.await_args.args
    assert req.model_path == "/weights"
    assert request_context is None


@pytest.mark.asyncio
async def test_update_weight_version_sets_server_args_and_aborts_requests():
    engine = _make_engine()

    result = await engine.update_weight_version(
        {"new_version": "v2", "abort_all_requests": True}
    )

    assert result == {
        "success": True,
        "message": "Weight version updated to v2",
        "new_version": "v2",
    }
    assert engine.engine.tokenizer_manager.server_args.weight_version == "v2"
    engine.engine.tokenizer_manager.abort_request.assert_called_once_with(
        abort_all=True
    )
