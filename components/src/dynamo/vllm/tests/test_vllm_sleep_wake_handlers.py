# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

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
async def test_snapshot_wraps_sleep_wake_with_communicator_checkpoint_calls():
    engine_client = AsyncMock()
    controller = VllmEnginePauseController(
        engine_client,
        prepare_for_process_checkpoint=True,
    )

    assert await controller.pause(None) is True
    assert await controller.resume() is True

    assert engine_client.mock_calls == [
        call.pause_generation(),
        call.sleep(),
        call.checkpoint_prepare(),
        call.checkpoint_restore(),
        call.wake_up(),
        call.resume_generation(),
    ]


@pytest.mark.asyncio
async def test_snapshot_restore_failure_is_fail_closed():
    """A failed communicator restore must not be replayed.

    The restored process keeps the engine paused and lets the exception
    propagate, so it dies rather than resuming on half-rebuilt communicators.
    """
    engine_client = AsyncMock()
    engine_client.checkpoint_restore.side_effect = RuntimeError("restore failed")
    controller = VllmEnginePauseController(
        engine_client,
        prepare_for_process_checkpoint=True,
    )
    assert await controller.pause(None) is True

    with pytest.raises(RuntimeError, match="restore failed"):
        await controller.resume()

    engine_client.checkpoint_restore.assert_awaited_once()
    # Never reaches wake_up, so memory is not remapped and scheduling stays
    # stopped; the controller is still paused because only mark_resumed()
    # clears that, and the caller never gets there.
    engine_client.wake_up.assert_not_awaited()
    engine_client.resume_generation.assert_not_awaited()
    assert controller.is_paused is True


@pytest.mark.asyncio
async def test_snapshot_prepare_failure_leaves_controller_unpaused():
    """A failed prepare must not record a pause it did not complete, or the
    next resume would issue a checkpoint_restore with no matching prepare."""
    engine_client = AsyncMock()
    engine_client.checkpoint_prepare.side_effect = RuntimeError("prepare failed")
    controller = VllmEnginePauseController(
        engine_client,
        prepare_for_process_checkpoint=True,
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        await controller.pause(None)

    assert controller.is_paused is False


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
@pytest.mark.parametrize(
    "cancel", [False, True], ids=["sleep-error", "sleep-cancelled"]
)
async def test_unknown_sleep_outcome_keeps_worker_out_of_discovery(cancel):
    handler = _make_handler()
    sleep_entered = asyncio.Event()

    async def native_sleep(level):
        sleep_entered.set()
        if cancel:
            await asyncio.Future()
        raise RuntimeError("native sleep failed")

    handler.engine_client.sleep.side_effect = native_sleep
    task = asyncio.create_task(handler.sleep({"level": 1}))
    try:
        await asyncio.wait_for(sleep_entered.wait(), timeout=2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            result = await asyncio.wait_for(task, timeout=2)
            assert result["status"] == "error"
            assert "native sleep failed" in result["message"]
            sleep_message = result["message"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()
    handler.engine_client.resume_generation.assert_not_awaited()

    if not cancel:
        assert "restart the worker" in sleep_message

    for operation in (handler.wake_up, handler.resume_generation, handler.sleep):
        result = await operation({})
        assert result["status"] == "error"
        assert "restart the worker" in result["message"]

    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.engine_client.sleep.assert_awaited_once_with(1)
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


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
