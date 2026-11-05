# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
    SNAPSHOT_RESTORE_CONTEXT_FILE,
)
from dynamo.common.snapshot.lifecycle import (
    SnapshotConfig,
    configure_snapshot_capture_env,
    elect_and_wake,
)
from dynamo.common.snapshot.restore_context import (
    refresh_snapshot_restore_config,
    write_snapshot_restore_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _PauseController:
    def __init__(self) -> None:
        self.paused = False
        self.resumed = False
        self.paused_with = None

    async def pause(self, level: int | None = None) -> None:
        self.paused = True
        self.paused_with = level

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
        assert (tmp_path / RESTORE_COMPLETE_FILE).read_text(encoding="utf-8") == "done"
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


async def test_elect_and_wake_resumes_without_lock(monkeypatch):
    monkeypatch.delenv("FAILOVER_LOCK_PATH", raising=False)
    controller = _PauseController()

    lock = await elect_and_wake(controller)

    assert lock is None
    assert controller.resumed is True


@pytest.mark.parametrize(
    "destination_policy", ["1", "0", "true", "false", "", None, "legacy"]
)
async def test_restore_applies_destination_offline_policy_before_runtime_parsing(
    monkeypatch, tmp_path, destination_policy
):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    if destination_policy is None or destination_policy == "legacy":
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", destination_policy)
    write_snapshot_restore_context()
    if destination_policy == "legacy":
        context_path = tmp_path / SNAPSHOT_RESTORE_CONTEXT_FILE
        context = json.loads(context_path.read_text(encoding="utf-8"))
        context["env"].pop("HF_HUB_OFFLINE", None)
        context_path.write_text(json.dumps(context), encoding="utf-8")

    for name in (
        "NCCL_IB_DISABLE",
        "NCCL_RAS_ENABLE",
        "TORCH_NCCL_ENABLE_MONITORING",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    configure_snapshot_capture_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    (tmp_path / RESTORE_COMPLETE_FILE).write_text("done", encoding="utf-8")
    controller = _PauseController()
    config = SnapshotConfig.from_env()
    assert config is not None
    assert await asyncio.wait_for(config.run_lifecycle(controller), timeout=1) is True

    expected_policy = None if destination_policy == "legacy" else destination_policy

    def parse_config():
        assert os.environ.get("HF_HUB_OFFLINE") == expected_policy
        return SimpleNamespace(discovery_backend="file")

    await refresh_snapshot_restore_config(SimpleNamespace(), parse_config)
    assert controller.resumed is False


def _patch_flock_lock(monkeypatch, fake_lock):
    flock_mod = pytest.importorskip("gpu_memory_service.failover_lock.flock")
    lock_cls = Mock(return_value=fake_lock)
    monkeypatch.setattr(flock_mod, "FlockFailoverLock", lock_cls)
    return lock_cls


async def test_elect_and_wake_elects_then_resumes(monkeypatch):
    controller = _PauseController()
    runtime = SimpleNamespace(healthy=False)
    runtime.set_health_status = lambda ok: setattr(runtime, "healthy", ok)
    fake_lock = Mock()
    fake_lock.acquire = AsyncMock()
    monkeypatch.setenv("ENGINE_ID", "3")
    lock_cls = _patch_flock_lock(monkeypatch, fake_lock)

    lock = await elect_and_wake(controller, runtime, lock_path="/tmp/failover.lock")

    lock_cls.assert_called_once_with("/tmp/failover.lock")
    fake_lock.acquire.assert_awaited_once_with(engine_id="engine-3")
    assert lock is fake_lock
    assert runtime.healthy is True
    assert controller.resumed is True


async def test_elect_and_wake_propagates_wake_failure_after_lock(monkeypatch):
    """A failed wake raises, matching the cold-start path. The process exits
    through normal shutdown, which releases the flock with its fd."""

    class BoomController(_PauseController):
        async def resume(self) -> None:
            raise RuntimeError("wake failed")

    runtime = SimpleNamespace(set_health_status=lambda _ok: None)
    fake_lock = Mock()
    fake_lock.acquire = AsyncMock()
    _patch_flock_lock(monkeypatch, fake_lock)

    with pytest.raises(RuntimeError, match="wake failed"):
        await elect_and_wake(BoomController(), runtime, lock_path="/tmp/failover.lock")

    fake_lock.acquire.assert_awaited_once()


async def test_elect_and_wake_reports_contended_acquire(monkeypatch):
    """A contended acquire is a real failover and drives the metric states."""
    controller = _PauseController()
    runtime = SimpleNamespace(set_health_status=lambda _ok: None)
    fake_lock = Mock()
    fake_lock.acquire = AsyncMock()
    fake_lock.was_contended = True
    metrics = Mock()
    _patch_flock_lock(monkeypatch, fake_lock)

    lock = await elect_and_wake(
        controller,
        runtime,
        lock_path="/tmp/failover.lock",
        failover_metrics=metrics,
    )

    # The caller owns the pause; the helper only elects and wakes.
    assert controller.paused is False
    assert controller.resumed is True
    assert lock.was_contended is True
    metrics.set_state.assert_any_call("standby")
    metrics.set_state.assert_any_call("waking")
    metrics.record_switch_attempt.assert_called_once()
