# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Event

import pytest

from dynamo.common.async_utils import run_sync_in_thread

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

    task = asyncio.create_task(run_sync_in_thread(operation))
    assert await asyncio.to_thread(started.wait, 1)

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
