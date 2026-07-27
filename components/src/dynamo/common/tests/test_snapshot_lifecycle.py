# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from dynamo.common.snapshot.constants import (
    GPU_UUIDS_FILE,
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.lifecycle import SnapshotConfig

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

    (tmp_path / GPU_UUIDS_FILE).write_text("recorded", encoding="utf-8")
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
        assert not (tmp_path / GPU_UUIDS_FILE).exists()
    finally:
        if not lifecycle.done():
            lifecycle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle


def test_record_visible_gpu_order(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    config = SnapshotConfig.from_env()
    assert config is not None
    uuids = [
        "GPU-aaaaaaaa-1111-2222-3333-444444444444",
        "GPU-bbbbbbbb-5555-6666-7777-888888888888",
    ]

    with patch(
        "dynamo.common.snapshot.lifecycle.subprocess.run",
        return_value=CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="\n".join(uuids) + "\n",
            stderr="",
        ),
    ) as run:
        config.record_visible_gpu_order()

    run.assert_called_once()
    assert (tmp_path / GPU_UUIDS_FILE).read_text(encoding="utf-8").splitlines() == uuids


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "GPU-aaaaaaaa-1111-2222-3333-444444444444\n"
        "GPU-aaaaaaaa-1111-2222-3333-444444444444\n",
        "not-a-gpu-uuid\n",
    ],
)
def test_record_visible_gpu_order_rejects_invalid_output(monkeypatch, tmp_path, stdout):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    config = SnapshotConfig.from_env()
    assert config is not None

    with patch(
        "dynamo.common.snapshot.lifecycle.subprocess.run",
        return_value=CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=stdout,
            stderr="",
        ),
    ):
        with pytest.raises(RuntimeError):
            config.record_visible_gpu_order()

    assert not (tmp_path / GPU_UUIDS_FILE).exists()


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
