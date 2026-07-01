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


class _RecordingEngineClient:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.pause_generation = AsyncMock(side_effect=self._record("pause_generation"))
        self.sleep = AsyncMock(side_effect=self._record("sleep"))
        self.wake_up = AsyncMock(side_effect=self._record("wake_up"))
        self.resume_generation = AsyncMock(
            side_effect=self._record("resume_generation")
        )
        self.collective_rpc = AsyncMock(side_effect=self._record_collective_rpc)

    def _record(self, name: str):
        async def record(*args, **kwargs):
            self.calls.append((name, args or kwargs))

        return record

    async def _record_collective_rpc(self, method: str, **kwargs):
        self.calls.append((method, kwargs))


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
        collective_rpc=AsyncMock(),
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._pause_controller = VllmEnginePauseController(handler.engine_client)
    handler._pause_lock = asyncio.Lock()
    return handler


def _make_recording_engine_client() -> _RecordingEngineClient:
    return _RecordingEngineClient()


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
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("sleep",), {"kwargs": {"level": 2}}),
        (("wake_up",), {"kwargs": {}}),
    ]
    handler.engine_client.sleep.assert_not_awaited()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_without_level_uses_vllm_default_sleep():
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
        collective_rpc=AsyncMock(),
    )
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(None)

    assert changed is True
    engine_client.pause_generation.assert_awaited_once()
    assert engine_client.collective_rpc.await_args_list == [
        (("sleep",), {"kwargs": {}}),
    ]
    engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_quiesces_generation_before_worker_sleep():
    engine_client = _make_recording_engine_client()
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(1)

    assert changed is True
    assert engine_client.calls == [
        ("pause_generation", {}),
        ("sleep", {"kwargs": {"level": 1}}),
    ]


@pytest.mark.asyncio
async def test_resume_wakes_workers_before_generation():
    engine_client = _make_recording_engine_client()
    controller = VllmEnginePauseController(engine_client)
    await controller.pause(1)
    engine_client.calls.clear()

    changed = await controller.resume()

    assert changed is True
    assert engine_client.calls == [
        ("wake_up", {"kwargs": {}}),
        ("resume_generation", {}),
    ]


@pytest.mark.asyncio
async def test_pause_keeps_generation_paused_after_worker_sleep_failure():
    engine_client = _make_recording_engine_client()

    async def fail_sleep(method: str, **kwargs):
        engine_client.calls.append((method, kwargs))
        if method == "sleep":
            raise RuntimeError("sleep failed")

    engine_client.collective_rpc = AsyncMock(side_effect=fail_sleep)
    controller = VllmEnginePauseController(engine_client)

    with pytest.raises(RuntimeError, match="sleep failed"):
        await controller.pause(1)

    assert engine_client.calls == [
        ("pause_generation", {}),
        ("sleep", {"kwargs": {"level": 1}}),
    ]
    assert controller.needs_resume_recovery is True


@pytest.mark.asyncio
async def test_resume_propagates_worker_wake_failure():
    engine_client = _make_recording_engine_client()
    controller = VllmEnginePauseController(engine_client)
    await controller.pause(1)
    engine_client.calls.clear()

    async def fail_restore(method: str, **kwargs):
        engine_client.calls.append((method, kwargs))
        if method == "wake_up":
            raise RuntimeError("restore failed")

    engine_client.collective_rpc = AsyncMock(side_effect=fail_restore)

    with pytest.raises(RuntimeError, match="restore failed"):
        await controller.resume()

    assert engine_client.calls == [
        ("wake_up", {"kwargs": {}}),
    ]
    assert controller.needs_resume_recovery is True


@pytest.mark.asyncio
async def test_wake_up_passes_explicit_tags_from_request():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.engine_client.collective_rpc.reset_mock()

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("wake_up",), {"kwargs": {"tags": ["weights"]}}),
    ]
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_wake_up_recovers_generation_pause_after_failed_sleep_rollback():
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock(
        side_effect=RuntimeError("sleep failed")
    )
    failed_resume = AsyncMock(side_effect=RuntimeError("resume failed"))
    handler.engine_client.resume_generation = failed_resume

    sleep_result = await handler.sleep({"level": 1})

    assert sleep_result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is True
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("sleep",), {"kwargs": {"level": 1}}),
    ]
    failed_resume.assert_not_awaited()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine_client.collective_rpc = AsyncMock()
    handler.engine_client.resume_generation = AsyncMock()
    wake_result = await handler.wake_up({})

    assert wake_result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("wake_up",), {"kwargs": {}}),
    ]
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    assert handler._pause_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_sleep_stays_unregistered_until_recovery():
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock(
        side_effect=RuntimeError("sleep failed")
    )

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is True
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("sleep",), {"kwargs": {"level": 1}}),
    ]
    handler.engine_client.resume_generation.assert_not_awaited()
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
    handler.engine_client.collective_rpc.assert_not_awaited()
    handler.engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_up_returns_error_for_register_failure():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.engine_client.collective_rpc.reset_mock()
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.wake_up({})

    assert result["status"] == "error"
    handler.engine_client.wake_up.assert_not_awaited()
    assert handler.engine_client.collective_rpc.await_args_list == [
        (("wake_up",), {"kwargs": {}}),
    ]
    handler.engine_client.resume_generation.assert_awaited_once()
    assert handler._pause_controller.is_paused is True
