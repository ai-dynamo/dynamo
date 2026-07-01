# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

from dynamo.vllm.snapshot_backend import (
    BACKEND_NAME,
    DynamoSnapshotBackend,
    register_dynamo_snapshot_backend,
    select_dynamo_snapshot_backend,
)


def test_plugin_registration_is_idempotent():
    SleepModeBackendFactory._registry.pop(BACKEND_NAME, None)

    register_dynamo_snapshot_backend()
    register_dynamo_snapshot_backend()

    assert SleepModeBackendFactory.get_backend_class(BACKEND_NAME) is (
        DynamoSnapshotBackend
    )


def test_snapshot_mode_selects_backend(monkeypatch):
    config = SimpleNamespace(model_config=SimpleNamespace(sleep_mode_backend="cumem"))

    select_dynamo_snapshot_backend(config)
    assert config.model_config.sleep_mode_backend == "cumem"

    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", "/run/dynamo/snapshot")
    select_dynamo_snapshot_backend(config)
    assert config.model_config.sleep_mode_backend == BACKEND_NAME


def test_suspend_orders_checkpoint_before_allocator():
    backend = DynamoSnapshotBackend()
    backend._allocator = MagicMock()
    calls = []
    backend._allocator.suspend.side_effect = lambda level: calls.append(
        ("allocator_suspend", level)
    )

    with patch(
        "vllm.distributed.parallel_state.checkpoint_prepare_distributed_state",
        side_effect=lambda: calls.append(("checkpoint_prepare", None)),
    ):
        backend.suspend(2)

    assert calls == [("checkpoint_prepare", None), ("allocator_suspend", 2)]
    assert backend.state() == "SUSPENDED"


def test_resume_fully_wakes_before_checkpoint_restore():
    backend = DynamoSnapshotBackend()
    backend._state = "SUSPENDED"
    backend._allocator_suspended = True
    backend._allocator = MagicMock()
    calls = []
    backend._allocator.resume.side_effect = lambda tags: calls.append(
        ("allocator_resume", tags)
    )

    with patch(
        "vllm.distributed.parallel_state.checkpoint_restore_distributed_state",
        side_effect=lambda: calls.append(("checkpoint_restore", None)),
    ):
        backend.resume(["weights"])

    assert calls == [("allocator_resume", None), ("checkpoint_restore", None)]
    assert backend.state() == "RUNNING"


def test_restore_retry_does_not_wake_allocator_twice():
    backend = DynamoSnapshotBackend()
    backend._state = "SUSPENDED"
    backend._allocator_suspended = True
    backend._allocator = MagicMock()

    with patch(
        "vllm.distributed.parallel_state.checkpoint_restore_distributed_state",
        side_effect=[RuntimeError("restore failed"), None],
    ):
        with pytest.raises(RuntimeError, match="restore failed"):
            backend.resume()
        backend.resume()

    backend._allocator.resume.assert_called_once_with(None)
    assert backend.state() == "RUNNING"


def test_failed_suspend_rolls_back_before_rethrow():
    backend = DynamoSnapshotBackend()
    backend._allocator = MagicMock()
    calls = []
    backend._allocator.suspend.side_effect = RuntimeError("sleep failed")
    backend._allocator.resume.side_effect = lambda tags: calls.append(
        ("allocator_resume", tags)
    )

    with (
        patch(
            "vllm.distributed.parallel_state.checkpoint_prepare_distributed_state",
            side_effect=lambda: calls.append(("checkpoint_prepare", None)),
        ),
        patch(
            "vllm.distributed.parallel_state.checkpoint_restore_distributed_state",
            side_effect=lambda: calls.append(("checkpoint_restore", None)),
        ),
        pytest.raises(RuntimeError, match="sleep failed"),
    ):
        backend.suspend()

    assert calls == [
        ("checkpoint_prepare", None),
        ("allocator_resume", None),
        ("checkpoint_restore", None),
    ]
    assert backend.state() == "RUNNING"
    backend.resume()
    assert calls == [
        ("checkpoint_prepare", None),
        ("allocator_resume", None),
        ("checkpoint_restore", None),
    ]


def test_failed_suspend_retries_failed_allocator_rollback():
    backend = DynamoSnapshotBackend()
    backend._allocator = MagicMock()
    backend._allocator.suspend.side_effect = RuntimeError("sleep failed")
    backend._allocator.resume.side_effect = [
        RuntimeError("rollback failed"),
        None,
    ]

    with (
        patch("vllm.distributed.parallel_state.checkpoint_prepare_distributed_state"),
        patch(
            "vllm.distributed.parallel_state.checkpoint_restore_distributed_state"
        ) as restore,
        pytest.raises(RuntimeError, match="sleep failed"),
    ):
        backend.suspend()

    assert backend.state() == "SUSPENDED"
    backend.resume()

    assert backend._allocator.resume.call_count == 2
    restore.assert_called_once_with()
    assert backend.state() == "RUNNING"
