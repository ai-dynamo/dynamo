# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.v1.engine.async_llm")

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
async def test_engine_routes_expose_vllm_management_callbacks():
    routes = await _make_engine().engine_routes()

    assert set(routes) == {"start_profile", "stop_profile", "sleep", "wake_up"}

    scaled_routes = await _make_engine(include_scale=True).engine_routes()
    assert "scale_elastic_ep" in scaled_routes


@pytest.mark.asyncio
async def test_sleep_and_wake_delegate_to_engine_client():
    engine = _make_engine()

    sleep_result = await engine.sleep({"level": 2})
    wake_result = await engine.wake_up({"tags": ["weights"]})

    assert sleep_result["status"] == "ok"
    assert wake_result["status"] == "ok"
    engine.engine_client.pause_generation.assert_awaited_once()
    engine.engine_client.sleep.assert_awaited_once_with(2)
    engine.engine_client.wake_up.assert_awaited_once_with(["weights"])
    engine.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_profile_routes_delegate_to_engine_client():
    engine = _make_engine()

    start_result = await engine.start_profile({"profile_prefix": "pref"})
    stop_result = await engine.stop_profile({})

    assert start_result["status"] == "ok"
    assert stop_result["status"] == "ok"
    engine.engine_client.start_profile.assert_awaited_once_with(profile_prefix="pref")
    engine.engine_client.stop_profile.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_scale_elastic_ep_validates_required_size_before_ray_import():
    engine = _make_engine(include_scale=True)

    result = await engine.scale_elastic_ep({})

    assert result == {
        "status": "error",
        "message": "Missing required field: new_data_parallel_size",
    }
    engine.engine_client.scale_elastic_ep.assert_not_awaited()
