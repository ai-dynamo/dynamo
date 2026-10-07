# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.vllm.lora_state import LoRAState
from dynamo.vllm.runtime_lora_cache import cache_reservation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_cache_reservation_release_survives_repeated_cancellation(tmp_path):
    state = LoRAState()
    settings = SimpleNamespace(max_cache_bytes=1024, max_download_bytes=256)
    entered = asyncio.Event()

    async def hold_reservation():
        async with cache_reservation(state, tmp_path, settings):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_reservation())
    await asyncio.wait_for(entered.wait(), timeout=1)
    guard = state.runtime_cache_guard
    assert guard is not None
    await guard.acquire()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    guard.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.runtime_cache_reserved_bytes == 0
