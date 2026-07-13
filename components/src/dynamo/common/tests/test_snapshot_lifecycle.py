# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.lifecycle import (
    SnapshotConfig,
    SnapshotLifecycleError,
    unified_snapshot_outcome,
)

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


async def test_snapshot_lifecycle_resumes_after_restore_sentinel(monkeypatch, tmp_path):
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
        assert controller.resumed is True
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
        assert controller.resumed is True
        assert "HF_HUB_OFFLINE" not in os.environ
    finally:
        if not lifecycle.done():
            lifecycle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle


async def test_snapshot_lifecycle_cancellation_cleans_sentinels(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    controller = _PauseController()
    config = SnapshotConfig.from_env()
    assert config is not None
    cancel = asyncio.Event()

    lifecycle = asyncio.create_task(
        config.run_lifecycle(controller, cancelled=cancel.wait)
    )
    for _ in range(100):
        if (tmp_path / READY_FOR_SNAPSHOT_FILE).exists():
            break
        await asyncio.sleep(0.01)

    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await lifecycle
    assert not (tmp_path / READY_FOR_SNAPSHOT_FILE).exists()


async def test_snapshot_lifecycle_annotates_resume_failure(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))

    class FailingResume(_PauseController):
        async def resume(self) -> None:
            raise RuntimeError("incompatible checkpoint image")

    config = SnapshotConfig.from_env()
    assert config is not None
    lifecycle = asyncio.create_task(config.run_lifecycle(FailingResume()))
    for _ in range(100):
        if (tmp_path / READY_FOR_SNAPSHOT_FILE).exists():
            break
        await asyncio.sleep(0.01)
    (tmp_path / RESTORE_COMPLETE_FILE).write_text("done", encoding="utf-8")

    with pytest.raises(
        SnapshotLifecycleError, match="incompatible checkpoint image"
    ) as exc:
        await lifecycle
    assert exc.value.phase == "restore_resume"


def _install_unified_outcome_types(monkeypatch):
    class Outcome:
        @staticmethod
        def exit_success():
            return ("exit_success", None)

        @staticmethod
        def prepared(config):
            return ("prepared", config)

    class Cancelled(Exception):
        pass

    class EngineShutdown(Exception):
        pass

    backend = ModuleType("dynamo.common.backend")
    backend.PreRuntimeOutcome = Outcome
    exceptions = ModuleType("dynamo.llm.exceptions")
    exceptions.Cancelled = Cancelled
    exceptions.EngineShutdown = EngineShutdown
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, exceptions.__name__, exceptions)
    return Cancelled, EngineShutdown


async def test_unified_snapshot_outcome_returns_capture_exit(monkeypatch):
    _install_unified_outcome_types(monkeypatch)

    class Controller:
        async def wait_for_restore(self, *, cancelled):
            assert callable(cancelled)
            return False

    async def cancelled():
        await asyncio.Future()

    outcome = await unified_snapshot_outcome(
        Controller(),
        argv=None,
        context=SimpleNamespace(cancelled=cancelled),
        backend_name="fake",
    )
    assert outcome == ("exit_success", None)


async def test_unified_snapshot_outcome_returns_replacement_config(monkeypatch):
    _install_unified_outcome_types(monkeypatch)
    replacement = object()
    monkeypatch.setattr(
        "dynamo.common.snapshot.restore_context.load_restored_runtime_config",
        lambda argv: replacement,
    )

    class Controller:
        async def wait_for_restore(self, *, cancelled):
            return True

    outcome = await unified_snapshot_outcome(
        Controller(),
        argv=["--namespace", "restored"],
        context=SimpleNamespace(cancelled=lambda: asyncio.Future()),
        backend_name="fake",
    )
    assert outcome == ("prepared", replacement)


async def test_unified_snapshot_outcome_preserves_restore_diagnostic(monkeypatch):
    _, EngineShutdown = _install_unified_outcome_types(monkeypatch)

    class Controller:
        async def wait_for_restore(self, *, cancelled):
            raise SnapshotLifecycleError(
                "restore_resume", "checkpoint engine version mismatch"
            )

    with pytest.raises(EngineShutdown, match="checkpoint engine version mismatch"):
        await unified_snapshot_outcome(
            Controller(),
            argv=None,
            context=SimpleNamespace(cancelled=lambda: asyncio.Future()),
            backend_name="fake",
        )


async def test_unified_snapshot_outcome_maps_cancellation(monkeypatch):
    Cancelled, _ = _install_unified_outcome_types(monkeypatch)

    class Controller:
        async def wait_for_restore(self, *, cancelled):
            raise asyncio.CancelledError

    with pytest.raises(Cancelled, match="fake snapshot lifecycle cancelled"):
        await unified_snapshot_outcome(
            Controller(),
            argv=None,
            context=SimpleNamespace(cancelled=lambda: asyncio.Future()),
            backend_name="fake",
        )
