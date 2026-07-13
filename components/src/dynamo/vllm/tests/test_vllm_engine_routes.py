# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.v1.engine.async_llm")
pytest.importorskip("vllm.usage.usage_lib")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.vllm.handlers import VllmEnginePauseController  # noqa: E402
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
        is_sleeping=AsyncMock(return_value=False),
        is_paused=AsyncMock(return_value=False),
        resume_generation=AsyncMock(),
        start_profile=AsyncMock(),
        stop_profile=AsyncMock(),
        reset_prefix_cache=AsyncMock(return_value=True),
    )
    if include_scale:
        engine_client.scale_elastic_ep = AsyncMock()

    engine.engine_client = engine_client
    engine._pause_controller = VllmEnginePauseController(engine_client)
    engine._pause_lock = asyncio.Lock()
    engine._scale_ep_lock = asyncio.Lock()
    engine._scale_ep_in_progress = False
    return engine


@pytest.mark.parametrize(
    "mode",
    [
        DisaggregationMode.AGGREGATED,
        DisaggregationMode.PREFILL,
        DisaggregationMode.DECODE,
    ],
)
def test_engine_controls_expose_vllm_management_capabilities(mode):
    engine = _make_engine()
    engine.disaggregation_mode = mode
    controls = engine.supported_controls()

    assert controls == {
        "start_profile",
        "stop_profile",
        "sleep",
        "wake_up",
        "clear_kv_blocks",
    }

    scaled_controls = _make_engine(include_scale=True).supported_controls()
    assert "scale_elastic_ep" in scaled_controls


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_result", [True, None])
async def test_clear_kv_blocks_resets_prefix_and_connector_cache(reset_result):
    engine = _make_engine()
    engine.engine_client.reset_prefix_cache.return_value = reset_result

    result = await engine.engine_control("clear_kv_blocks", {})

    assert result == {"status": "success", "message": "KV cache cleared"}
    engine.engine_client.reset_prefix_cache.assert_awaited_once_with(
        reset_connector=True
    )


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_reset_failure():
    engine = _make_engine()
    engine.engine_client.reset_prefix_cache.return_value = False

    result = await engine.engine_control("clear_kv_blocks", {"ignored": True})

    assert result == {"status": "error", "message": "KV cache reset failed"}


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_engine_exception():
    engine = _make_engine()
    engine.engine_client.reset_prefix_cache.side_effect = RuntimeError("cache busy")

    result = await engine.engine_control("clear_kv_blocks", {})

    assert result == {"status": "error", "message": "cache busy"}


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_unavailable_engine():
    engine = _make_engine()
    engine.engine_client = None

    result = await engine.engine_control("clear_kv_blocks", {})

    assert result == {"status": "error", "message": "Engine is not running"}


@pytest.mark.asyncio
async def test_unsupported_engine_control_retains_error_contract():
    result = await _make_engine().engine_control("not_supported", {})

    assert result == {
        "status": "error",
        "message": "unsupported engine control: not_supported",
    }


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
    engine.engine_client.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_sleep_rollback_wakes_if_scheduler_is_still_paused():
    engine = _make_engine()
    engine.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    engine.engine_client.is_paused.side_effect = [True, False]

    sleep_result = await engine.sleep({"level": 1})

    assert sleep_result["status"] == "error"
    assert engine._pause_controller.is_paused is False
    assert engine._pause_controller.needs_resume_recovery is False
    engine.engine_client.wake_up.assert_awaited_once_with()
    engine.engine_client.resume_generation.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_scale_elastic_ep_rejects_size_below_two():
    engine = _make_engine(include_scale=True)

    result = await engine.engine_control(
        "scale_elastic_ep", {"new_data_parallel_size": 1}
    )

    assert result["status"] == "error"
    assert ">= 2" in result["message"]
    engine.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_elastic_ep_rejects_non_integer():
    engine = _make_engine(include_scale=True)

    result = await engine.engine_control(
        "scale_elastic_ep", {"new_data_parallel_size": "four"}
    )

    assert result["status"] == "error"
    assert "must be an integer" in result["message"]
    engine.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_elastic_ep_rejects_concurrent_call():
    engine = _make_engine(include_scale=True)
    # Simulate a scale already running: the second caller must be rejected
    # rather than queued (a queued caller GCs the first caller's TCPStore).
    engine._scale_ep_in_progress = True

    result = await engine.engine_control(
        "scale_elastic_ep", {"new_data_parallel_size": 4}
    )

    assert result["status"] == "error"
    assert "already in progress" in result["message"]
    engine.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_elastic_ep_patches_and_restores_ray_list_nodes(monkeypatch):
    engine = _make_engine(include_scale=True)

    # The success path imports ray and swaps ray.util.state.list_nodes for a
    # GCS-backed shim (works around ray --minimal lacking the dashboard HTTP
    # server). Inject fakes so the path runs without a real cluster and assert
    # the original list_nodes is restored afterward.
    def sentinel_list_nodes(**kw):
        return []

    fake_state = SimpleNamespace(list_nodes=sentinel_list_nodes)
    fake_util = SimpleNamespace(state=fake_state)
    fake_ray = SimpleNamespace(nodes=lambda: [], util=fake_util)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.util", fake_util)
    monkeypatch.setitem(sys.modules, "ray.util.state", fake_state)

    result = await engine.engine_control(
        "scale_elastic_ep", {"new_data_parallel_size": 4}
    )

    assert result == {
        "status": "ok",
        "message": "Scaled to data_parallel_size=4",
        "new_data_parallel_size": 4,
    }
    engine.engine_client.scale_elastic_ep.assert_awaited_once_with(4)
    # Patch must be restored even on the success path.
    assert fake_state.list_nodes is sentinel_list_nodes
    assert engine._scale_ep_in_progress is False
