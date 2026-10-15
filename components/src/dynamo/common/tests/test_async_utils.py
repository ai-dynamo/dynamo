# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Event

import pytest

from dynamo.common.async_utils import run_async_to_completion, run_sync_in_thread

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.asyncio
async def test_run_sync_in_thread_joins_worker_before_propagating_cancellation():
    started = Event()
    release = Event()
    finished = Event()

    def operation() -> None:
        started.set()
        release.wait()
        finished.set()

    tasks_before = asyncio.all_tasks()
    task = asyncio.create_task(run_sync_in_thread(operation))
    assert await asyncio.to_thread(started.wait, 1)
    assert asyncio.all_tasks() - tasks_before == {task}

    try:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_run_async_to_completion_joins_task_before_propagating_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def operation() -> None:
        started.set()
        await release.wait()
        finished.set()

    task = asyncio.create_task(run_async_to_completion(operation()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_run_async_to_completion_cancels_work_before_admission():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(
        run_async_to_completion(operation(), complete_on_cancel=lambda: False)
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_run_sync_in_thread_reports_failure_after_cancellation():
    started = Event()
    release = Event()
    contexts = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    def operation() -> None:
        started.set()
        release.wait()
        raise RuntimeError("delayed thread failure")

    try:
        task = asyncio.create_task(run_sync_in_thread(operation))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        loop.set_exception_handler(previous_handler)
        release.set()

    assert len(contexts) == 1
    assert contexts[0]["message"] == (
        "Thread operation failed after its caller was cancelled"
    )
    assert str(contexts[0]["exception"]) == "delayed thread failure"


@pytest.mark.asyncio
async def test_run_async_to_completion_reports_failure_after_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()
    contexts = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    async def operation() -> None:
        started.set()
        await release.wait()
        raise RuntimeError("delayed async failure")

    try:
        task = asyncio.create_task(run_async_to_completion(operation()))
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        loop.set_exception_handler(previous_handler)
        release.set()

    assert len(contexts) == 1
    assert contexts[0]["message"] == (
        "Async operation failed after its caller was cancelled"
    )
    assert str(contexts[0]["exception"]) == "delayed async failure"
