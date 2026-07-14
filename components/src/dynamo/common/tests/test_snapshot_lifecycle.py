# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest

from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.lifecycle import (
    SnapshotConfig,
    configure_snapshot_capture_env,
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


@pytest.mark.parametrize(
    ("initial_p2p_disable", "expected_p2p_disable"),
    [(None, "0"), ("0", "0"), ("1", "1")],
)
def test_configure_snapshot_capture_env_preserves_explicit_p2p_disable(
    monkeypatch, initial_p2p_disable, expected_p2p_disable
):
    if initial_p2p_disable is None:
        monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    else:
        monkeypatch.setenv("NCCL_P2P_DISABLE", initial_p2p_disable)

    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    monkeypatch.setenv("NCCL_RAS_ENABLE", "1")
    monkeypatch.setenv("TORCH_NCCL_ENABLE_MONITORING", "1")
    monkeypatch.setenv("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    configure_snapshot_capture_env()

    assert os.environ["NCCL_P2P_DISABLE"] == expected_p2p_disable
    assert os.environ["NCCL_CUMEM_ENABLE"] == "0"
    assert os.environ["NCCL_NVLS_ENABLE"] == "0"
    assert os.environ["NCCL_IB_DISABLE"] == "1"
    assert os.environ["NCCL_RAS_ENABLE"] == "0"
    assert os.environ["TORCH_NCCL_ENABLE_MONITORING"] == "0"
    assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


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
