# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")
pytest.importorskip("vllm.v1.engine.exceptions")

from dynamo.vllm.handlers import (  # noqa: E402
    BaseWorkerHandler,
    VllmEnginePauseController,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._pause_controller = VllmEnginePauseController(handler.engine_client)
    handler._pause_lock = asyncio.Lock()
    return handler


@pytest.mark.asyncio
async def test_wake_up_before_sleep_is_noop():
    handler = _make_handler()

    result = await handler.wake_up({})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_sleep_and_wake_are_idempotent():
    handler = _make_handler()

    first_sleep = await handler.sleep({"level": 2})
    second_sleep = await handler.sleep({"level": 2})
    first_wake = await handler.wake_up({})
    second_wake = await handler.wake_up({})

    assert first_sleep["status"] == "ok"
    assert second_sleep["status"] == "ok"
    assert first_wake["status"] == "ok"
    assert second_wake["status"] == "ok"

    handler.engine_client.pause_generation.assert_awaited_once()
    handler.engine_client.sleep.assert_awaited_once_with(2)
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_without_level_uses_vllm_default_sleep():
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(None)

    assert changed is True
    engine_client.pause_generation.assert_awaited_once()
    engine_client.sleep.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_quiesce_can_skip_cache_clear():
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(1, clear_cache=False)

    assert changed is True
    engine_client.pause_generation.assert_awaited_once_with(clear_cache=False)
    engine_client.sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_quiesce_uses_no_clear_sleep_utility_when_available():
    engine_core = SimpleNamespace(call_utility_async=AsyncMock())
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
        engine_core=engine_core,
    )
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(1, clear_cache=False)

    assert changed is True
    engine_client.pause_generation.assert_awaited_once_with(clear_cache=False)
    engine_core.call_utility_async.assert_awaited_once_with(
        "gms_sleep_no_clear", 1, "abort"
    )
    engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_up_passes_explicit_tags_from_request():
    handler = _make_handler()
    await handler._pause_controller.pause(1)

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_awaited_once_with(["weights"])
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_wake_up_recovers_generation_pause_after_failed_sleep_rollback():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    failed_resume = AsyncMock(side_effect=RuntimeError("resume failed"))
    handler.engine_client.resume_generation = failed_resume

    sleep_result = await handler.sleep({"level": 1})

    assert sleep_result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is True
    failed_resume.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine_client.resume_generation = AsyncMock()
    wake_result = await handler.wake_up({})

    assert wake_result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    assert handler._pause_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_sleep_reregisters_after_clean_pause_rollback():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))

    result = await handler.sleep({"level": 1})

    # Rollback resumed generation cleanly, so the engine is serving-safe again.
    assert result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is False
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    # The endpoint must rejoin the routing pool since wake_up will early-return.
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_returns_error_for_unregister_failure():
    handler = _make_handler()
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery backend down")
    )

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    handler.engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_up_returns_error_for_register_failure():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.wake_up({})

    assert result["status"] == "error"
    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.engine_client.resume_generation.assert_awaited_once()
    assert handler._pause_controller.is_paused is True


@pytest.mark.asyncio
async def test_sleep_with_handoff_releases_failover_lock():
    handler = _make_handler()
    lock = SimpleNamespace(release=AsyncMock())
    handler._gms_failover_lock = lock

    result = await handler.sleep({"level": 1, "release_failover_lock": True})

    assert result["status"] == "ok"
    assert result["failover_lock_released"] is True
    handler.engine_client.pause_generation.assert_awaited_once_with(clear_cache=False)
    handler.engine_client.sleep.assert_awaited_once_with(1)
    lock.release.assert_awaited_once()
    assert handler._gms_failover_lock is None
