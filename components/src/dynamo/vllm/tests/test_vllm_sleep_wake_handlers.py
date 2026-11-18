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
        # vLLM >= 0.32 returns whether every sleeping tag is awake again.
        wake_up=AsyncMock(return_value=True),
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
    engine_client.wake_up.return_value = True
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
async def test_partial_wake_up_resumes_only_when_fully_awake():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True])
    await handler._pause_controller.pause(2)

    first = await handler.wake_up({"tags": ["weights"]})

    # KV cache still asleep: generation stays paused, worker stays unregistered.
    assert first == {
        "status": "ok",
        "message": "Engine partially awake",
        "is_sleeping": True,
    }
    assert handler._pause_controller.is_paused is True
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    second = await handler.wake_up({"tags": ["kv_cache"]})

    assert second == {"status": "ok", "message": "Engine woke"}
    assert handler.engine_client.wake_up.await_args_list == [
        call(["weights"]),
        call(["kv_cache"]),
    ]
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    assert handler._pause_controller.is_paused is False


@pytest.mark.asyncio
async def test_partial_wake_up_asks_is_sleeping_when_wake_up_returns_none():
    """vLLM < 0.32 returns nothing from wake_up."""
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(return_value=None)
    handler.engine_client.is_sleeping = AsyncMock(side_effect=[True, False])
    await handler._pause_controller.pause(2)

    first = await handler.wake_up({"tags": ["weights"]})

    assert first["message"] == "Engine partially awake"
    handler.engine_client.resume_generation.assert_not_awaited()

    second = await handler.wake_up({"tags": ["kv_cache"]})

    assert second["message"] == "Engine woke"
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_generation_route_leaves_sleeping_engine_paused():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True])
    await handler._pause_controller.pause(2)
    await handler.wake_up({"tags": ["weights"]})
    handler._paused = True  # RL pause_generation before a weight update

    result = await handler.resume_generation({})

    assert result["status"] == "ok"
    assert result["is_sleeping"] is True
    assert handler._paused is False
    handler.engine_client.resume_generation.assert_not_awaited()

    await handler.wake_up({"tags": ["kv_cache"]})

    handler.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_after_partial_wake_up_sleeps_the_woken_memory():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True])
    await handler.sleep({"level": 2})
    # vLLM drops "scheduling" before waking memory.
    await handler.wake_up({"tags": ["weights", "scheduling"]})

    # Level 0 only pauses scheduling; it would release nothing.
    assert (await handler.sleep({"level": 0}))["status"] == "error"
    result = await handler.sleep({"level": 2})

    assert result == {"status": "ok", "message": "Engine slept (level=2)"}
    assert handler.engine_client.sleep.await_args_list == [call(2), call(2)]
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    # Asleep again: the next sleep is a no-op, a full wake resumes once.
    assert (await handler.sleep({"level": 2}))["message"] == "Engine already sleeping"
    assert (await handler.wake_up({}))["message"] == "Engine woke"
    handler.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_after_kv_cache_only_wake_up_reports_error():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(return_value=False)
    await handler.sleep({"level": 2})
    await handler.wake_up({"tags": ["kv_cache"]})
    # vLLM ignores a wake whose tags are not all asleep, so this wakes nothing.
    await handler.wake_up({"tags": ["weights", "kv_cache"]})

    result = await handler.sleep({"level": 2})

    # vLLM ignores sleep while the weights sleep; don't report the KV cache freed.
    assert result["status"] == "error"
    handler.engine_client.sleep.assert_awaited_once_with(2)
    assert handler._pause_controller.is_paused is True


@pytest.mark.asyncio
async def test_partial_wake_state_does_not_leak_into_the_next_sleep():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True, False])
    await handler.sleep({"level": 2})
    await handler.wake_up({"tags": ["weights"]})
    await handler.wake_up({"tags": ["kv_cache"]})

    await handler.sleep({"level": 2})
    await handler.wake_up({"tags": ["kv_cache"]})
    result = await handler.sleep({"level": 2})

    # The weights woken in the first cycle are asleep again in this one.
    assert result["status"] == "error"
    assert handler.engine_client.sleep.await_args_list == [call(2), call(2)]


@pytest.mark.asyncio
async def test_failed_sleep_after_partial_wake_up_requires_a_full_wake():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True, False])
    handler.engine_client.sleep = AsyncMock(
        side_effect=[None, RuntimeError("boom"), None]
    )
    await handler.sleep({"level": 2})
    await handler.wake_up({"tags": ["weights"]})
    assert (await handler.sleep({"level": 2}))["status"] == "error"

    # vLLM may now hold unmapped weights it records as awake: a partial wake
    # could resume on them, and a level 2 retry would read buffers from them.
    assert (await handler.sleep({"level": 2}))["status"] == "error"
    assert (await handler.wake_up({"tags": ["kv_cache"]}))["status"] == "error"
    assert handler.engine_client.sleep.await_count == 2
    assert (await handler.wake_up({}))["message"] == "Engine woke"
    assert handler.engine_client.wake_up.await_args_list == [call(["weights"]), call()]

    # The next cycle wakes partially again.
    await handler.sleep({"level": 2})
    assert (await handler.wake_up({"tags": ["weights"]}))["is_sleeping"] is True
    handler.engine_client.resume_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_after_failed_register_does_not_resleep_an_awake_engine():
    handler = _make_handler()
    handler.engine_client.wake_up = AsyncMock(side_effect=[False, True])
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )
    await handler.sleep({"level": 1})
    await handler.wake_up({"tags": ["weights"]})
    assert (await handler.wake_up({"tags": ["kv_cache"]}))["status"] == "error"

    # vLLM is fully awake; a failed sleep from here could not be woken back.
    result = await handler.sleep({"level": 1})

    assert result["message"] == "Engine already sleeping"
    handler.engine_client.sleep.assert_awaited_once_with(1)


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
