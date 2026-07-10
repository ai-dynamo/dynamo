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
async def test_sleep_and_wake_manage_versioned_generate_alias():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    assert (await handler.sleep({"level": 1}))["status"] == "ok"
    assert (await handler.wake_up({}))["status"] == "ok"

    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    versioned_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    versioned_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_alias_unregister_failure_rolls_back_every_endpoint():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(
            side_effect=RuntimeError("alias discovery unavailable")
        ),
        register_endpoint_instance=AsyncMock(),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    # Converging to the published state must reconcile every endpoint, including
    # the endpoint whose unregister call failed with an unknown remote outcome.
    versioned_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_sleep_rollback_publication_recovers_via_wake_while_awake():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(
            side_effect=[RuntimeError("alias register failed"), None]
        ),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    sleep_result = await handler.sleep({"level": 1})
    assert sleep_result["status"] == "error"
    assert handler._pause_controller.is_paused is False

    # The engine is already awake, but endpoint publication still needs repair.
    wake_result = await handler.wake_up({})

    assert wake_result["status"] == "ok"
    assert versioned_endpoint.register_endpoint_instance.await_count == 2
    handler.engine_client.wake_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_alias_registration_failure_fails_closed_then_recovers():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(
            side_effect=[RuntimeError("alias register failed"), None]
        ),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)
    assert (await handler.sleep({"level": 1}))["status"] == "ok"

    first_wake = await handler.wake_up({})

    assert first_wake["status"] == "error"
    # A partial publish is compensated by unpublishing the complete endpoint
    # set, leaving the awake engine safely unreachable until a retry converges.
    assert handler.generate_endpoint.unregister_endpoint_instance.await_count == 2
    assert versioned_endpoint.unregister_endpoint_instance.await_count == 2
    assert handler._pause_controller.is_paused is False

    second_wake = await handler.wake_up({})

    assert second_wake["status"] == "ok"
    assert handler.generate_endpoint.register_endpoint_instance.await_count == 2
    assert versioned_endpoint.register_endpoint_instance.await_count == 2


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
    # Native wake already completed; only publication remains pending.
    assert handler._pause_controller.is_paused is False

    handler.generate_endpoint.register_endpoint_instance = AsyncMock()
    retry = await handler.wake_up({})

    assert retry["status"] == "ok"
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    handler.engine_client.wake_up.assert_awaited_once_with()
