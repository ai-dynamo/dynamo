# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.v1.engine.async_llm")
pytest.importorskip("vllm.usage.usage_lib")

from dynamo.vllm.handlers import VllmEngineQuiesceController  # noqa: E402
from dynamo.vllm.llm_engine import VllmLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_engine(include_scale: bool = False) -> VllmLLMEngine:
    engine = VllmLLMEngine.__new__(VllmLLMEngine)
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
        start_profile=AsyncMock(),
        stop_profile=AsyncMock(),
    )
    if include_scale:
        engine_client.scale_elastic_ep = AsyncMock()

    engine.engine_client = engine_client
    engine._quiesce_controller = VllmEngineQuiesceController(engine_client)
    engine._quiesce_lock = asyncio.Lock()
    engine._scale_ep_lock = asyncio.Lock()
    return engine


@pytest.mark.asyncio
async def test_engine_controls_expose_vllm_management_capabilities():
    controls = _make_engine().supported_controls()

    assert controls == {"start_profile", "stop_profile", "sleep", "wake_up"}

    scaled_controls = _make_engine(include_scale=True).supported_controls()
    assert "scale_elastic_ep" in scaled_controls


@pytest.mark.asyncio
async def test_sleep_and_wake_delegate_to_engine_client():
    engine = _make_engine()

    sleep_result = await engine.engine_control("sleep", {"level": 2})
    wake_result = await engine.engine_control("wake_up", {"tags": ["weights"]})

    assert sleep_result["status"] == "ok"
    assert wake_result["status"] == "ok"
    engine.engine_client.pause_generation.assert_awaited_once()
    engine.engine_client.sleep.assert_awaited_once_with(2)
    engine.engine_client.wake_up.assert_awaited_once_with(["weights"])
    engine.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_wake_up_recovers_generation_pause_after_failed_sleep_rollback():
    engine = _make_engine()
    engine.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    failed_resume = AsyncMock(side_effect=RuntimeError("resume failed"))
    engine.engine_client.resume_generation = failed_resume

    sleep_result = await engine.sleep({"level": 1})

    assert sleep_result["status"] == "error"
    assert engine._quiesce_controller.is_quiesced is False
    assert engine._quiesce_controller.needs_resume_recovery is True
    failed_resume.assert_awaited_once()

    engine.engine_client.resume_generation = AsyncMock()
    wake_result = await engine.wake_up({})

    assert wake_result["status"] == "ok"
    engine.engine_client.wake_up.assert_not_awaited()
    engine.engine_client.resume_generation.assert_awaited_once()
    assert engine._quiesce_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_profile_routes_delegate_to_engine_client():
    engine = _make_engine()

    start_result = await engine.engine_control(
        "start_profile", {"profile_prefix": "pref"}
    )
    stop_result = await engine.engine_control("stop_profile", {})

    assert start_result["status"] == "ok"
    assert stop_result["status"] == "ok"
    engine.engine_client.start_profile.assert_awaited_once_with(profile_prefix="pref")
    engine.engine_client.stop_profile.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_scale_elastic_ep_validates_required_size_before_ray_import():
    engine = _make_engine(include_scale=True)

    result = await engine.engine_control("scale_elastic_ep", {})

    assert result == {
        "status": "error",
        "message": "Missing required field: new_data_parallel_size",
    }
    engine.engine_client.scale_elastic_ep.assert_not_awaited()
