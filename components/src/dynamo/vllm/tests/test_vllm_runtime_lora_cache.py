# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.llm import HttpError
from dynamo.vllm.lora_state import LoRAState
from dynamo.vllm.runtime_lora_cache import _cache_size, cache_reservation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_cache_scan_tolerates_disappearing_staging_file(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    vanished = staging / "temporary"
    vanished.write_bytes(bytes(4))
    (tmp_path / "retained").write_bytes(bytes(8))
    original = Path.lstat

    def remove_before_stat(path, *args, **kwargs):
        if path == vanished:
            path.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", remove_before_stat)

    assert _cache_size(tmp_path, 1024) == 8


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_cache_scan_handles_directory_scan_errors(tmp_path, monkeypatch, failure):
    def fail_walk(_path, *, followlinks, onerror=None):
        error = failure(f"private path: {tmp_path}")
        if onerror is not None:
            onerror(error)
        return iter(())

    monkeypatch.setattr(os, "walk", fail_walk)

    if failure is FileNotFoundError:
        assert _cache_size(tmp_path, 1024) == 0
    else:
        with pytest.raises(HttpError) as error:
            _cache_size(tmp_path, 1024)
        assert error.value.code == 500
        assert error.value.message == "lora_plugin_error"


def test_cache_scan_sanitizes_stat_permission_error(tmp_path, monkeypatch):
    entry = tmp_path / "private"
    entry.write_bytes(bytes(4))
    original = Path.lstat

    def deny_metadata(path, *args, **kwargs):
        if path == entry:
            raise PermissionError(f"private path: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_metadata)

    with pytest.raises(HttpError) as error:
        _cache_size(tmp_path, 1024)
    assert error.value.code == 500
    assert error.value.message == "lora_plugin_error"


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
