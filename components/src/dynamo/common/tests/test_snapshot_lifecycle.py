# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.lifecycle import SnapshotConfig, wake_restored_engine

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _PauseController:
    def __init__(self) -> None:
        self.paused = False
        self.resumed = False

    async def pause(self) -> None:
        self.paused = True

    async def resume(self) -> None:
        self.resumed = True

    def mark_resumed(self) -> None:
        pass


async def test_snapshot_lifecycle_returns_paused_after_restore_sentinel(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    controller = _PauseController()
    config = SnapshotConfig.from_env()
    assert config is not None

    lifecycle = asyncio.create_task(config.run_lifecycle(controller))
    try:
        for _ in range(100):
            if (tmp_path / READY_FOR_SNAPSHOT_FILE).exists():
                break
            await asyncio.sleep(0.01)

        assert controller.paused is True
        assert (tmp_path / READY_FOR_SNAPSHOT_FILE).exists()

        (tmp_path / RESTORE_COMPLETE_FILE).write_text("done", encoding="utf-8")

        assert await lifecycle is True
        assert controller.resumed is False
        assert not (tmp_path / READY_FOR_SNAPSHOT_FILE).exists()
        assert not (tmp_path / RESTORE_COMPLETE_FILE).exists()
    finally:
        if not lifecycle.done():
            lifecycle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle


async def test_snapshot_lifecycle_clears_capture_only_env_after_restore(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    controller = _PauseController()
    config = SnapshotConfig.from_env()
    assert config is not None

    lifecycle = asyncio.create_task(config.run_lifecycle(controller))
    try:
        for _ in range(100):
            if (tmp_path / READY_FOR_SNAPSHOT_FILE).exists():
                break
            await asyncio.sleep(0.01)

        (tmp_path / RESTORE_COMPLETE_FILE).write_text("done", encoding="utf-8")

        assert await lifecycle is True
        assert controller.resumed is False
        assert "HF_HUB_OFFLINE" not in os.environ
    finally:
        if not lifecycle.done():
            lifecycle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle


async def test_wake_restored_engine_resumes_without_lock(monkeypatch):
    monkeypatch.delenv("FAILOVER_LOCK_PATH", raising=False)
    controller = _PauseController()

    lock = await wake_restored_engine(controller)

    assert lock is None
    assert controller.resumed is True


def _patch_flock_lock(monkeypatch, fake_lock):
    flock_mod = pytest.importorskip("gpu_memory_service.failover_lock.flock")
    lock_cls = Mock(return_value=fake_lock)
    monkeypatch.setattr(flock_mod, "FlockFailoverLock", lock_cls)
    return lock_cls


async def test_wake_restored_engine_elects_then_resumes(monkeypatch):
    controller = _PauseController()
    runtime = SimpleNamespace(healthy=False)
    runtime.set_health_status = lambda ok: setattr(runtime, "healthy", ok)
    fake_lock = Mock()
    fake_lock.acquire = AsyncMock()
    monkeypatch.setenv("ENGINE_ID", "3")
    lock_cls = _patch_flock_lock(monkeypatch, fake_lock)

    lock = await wake_restored_engine(
        controller, runtime, lock_path="/tmp/failover.lock"
    )

    lock_cls.assert_called_once_with("/tmp/failover.lock")
    fake_lock.acquire.assert_awaited_once_with(engine_id="engine-3")
    assert lock is fake_lock
    assert runtime.healthy is True
    assert controller.resumed is True


async def test_wake_restored_engine_exits_if_wake_fails_after_lock(monkeypatch):
    class BoomController(_PauseController):
        async def resume(self) -> None:
            raise RuntimeError("wake failed")

    def _exit(code: int) -> None:
        raise SystemExit(code)

    runtime = SimpleNamespace(set_health_status=lambda _ok: None)
    fake_lock = Mock()
    fake_lock.acquire = AsyncMock()
    monkeypatch.setattr(os, "_exit", _exit)
    _patch_flock_lock(monkeypatch, fake_lock)

    with pytest.raises(SystemExit) as exc_info:
        await wake_restored_engine(
            BoomController(), runtime, lock_path="/tmp/failover.lock"
        )

    assert exc_info.value.code == 1
    fake_lock.acquire.assert_awaited_once()
