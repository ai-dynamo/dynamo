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
from dynamo.vllm.idle_sleep import IdleSleepMonitor, IdleSleepState  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        with self._track_request_activity():
            yield {}


def _make_handler(with_idle_monitor: bool = True) -> _TestWorkerHandler:
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
    if with_idle_monitor:
        handler.idle_monitor = IdleSleepMonitor(
            sleep_fn=handler._idle_sleep,
            timeout=60.0,
            level=1,
            escalate_fn=handler._escalate_sleep,
            escalate_timeout=60.0,
            escalate_level=2,
        )
    return handler


@pytest.mark.asyncio
async def test_idle_sleep_reuses_sleep_route_path():
    handler = _make_handler()

    result = await handler._idle_sleep(1)

    assert result["status"] == "ok"
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.engine_client.pause_generation.assert_awaited_once()
    handler.engine_client.sleep.assert_awaited_once_with(1)
    # The sleep path reports the transition back to the monitor.
    assert handler.idle_monitor.state is IdleSleepState.ASLEEP
    assert handler.idle_monitor.current_level == 1


@pytest.mark.asyncio
async def test_external_sleep_and_wake_update_idle_monitor():
    handler = _make_handler()

    await handler.sleep({"level": 2})
    assert handler.idle_monitor.state is IdleSleepState.ASLEEP
    assert handler.idle_monitor.current_level == 2

    await handler.wake_up({})
    assert handler.idle_monitor.state is IdleSleepState.WARM
    assert handler.idle_monitor.current_level is None


@pytest.mark.asyncio
async def test_escalate_sleep_wakes_and_resleeps_deeper():
    handler = _make_handler()
    await handler.sleep({"level": 1})
    handler.engine_client.sleep.reset_mock()

    result = await handler._escalate_sleep(1, 2)

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.engine_client.sleep.assert_awaited_once_with(2)
    # Discovery is not rejoined: escalation happens while unregistered.
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()
    assert handler._pause_controller.is_paused is True


@pytest.mark.asyncio
async def test_escalate_sleep_requires_sleeping_engine():
    handler = _make_handler()

    result = await handler._escalate_sleep(1, 2)

    assert result["status"] == "error"
    handler.engine_client.wake_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_escalate_sleep_restores_previous_level_on_failure():
    handler = _make_handler()
    await handler.sleep({"level": 1})
    handler.engine_client.sleep = AsyncMock(
        side_effect=[RuntimeError("escalate failed"), None]
    )

    result = await handler._escalate_sleep(1, 2)

    assert result["status"] == "error"
    assert handler.engine_client.sleep.await_args_list[0].args == (2,)
    assert handler.engine_client.sleep.await_args_list[1].args == (1,)
    assert handler._pause_controller.is_paused is True


@pytest.mark.asyncio
async def test_generate_tracks_in_flight_requests():
    handler = _make_handler()
    seen_in_flight = []

    async def consume():
        async for _ in handler.generate({}, SimpleNamespace(id=lambda: "r1")):
            seen_in_flight.append(handler.idle_monitor.in_flight)

    await consume()

    assert seen_in_flight == [1]
    assert handler.idle_monitor.in_flight == 0


@pytest.mark.asyncio
async def test_generate_without_idle_monitor_is_noop():
    handler = _make_handler(with_idle_monitor=False)

    chunks = [
        chunk async for chunk in handler.generate({}, SimpleNamespace(id=lambda: "r1"))
    ]

    assert chunks == [{}]
    assert handler.idle_monitor is None
