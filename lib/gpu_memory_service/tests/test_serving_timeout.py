# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from gpu_memory_service.common import serving_timeout

pytestmark = pytest.mark.pre_merge


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(serving_timeout, "_applied", False)
    monkeypatch.setattr(serving_timeout, "_warned", False)
    monkeypatch.setenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "1000")


def test_invalid_timeout_uses_default(monkeypatch):
    monkeypatch.setenv("DYN_GMS_SERVING_NCCL_TIMEOUT_S", "invalid")

    assert serving_timeout.serving_timeout_s() == 5.0


def test_disabled_timeout_is_noop(monkeypatch):
    monkeypatch.setenv("DYN_GMS_SERVING_NCCL_TIMEOUT_S", "0")

    assert serving_timeout.tighten_now() is False
    assert serving_timeout._applied is False


def test_uninitialized_distributed_is_noop(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    assert serving_timeout.apply_serving_collective_timeout(3.0) is False
    assert serving_timeout._applied is False


def test_applies_default_and_tracked_process_groups(monkeypatch):
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    group_a = object()
    group_b = object()
    calls: list[tuple[datetime.timedelta, object | None]] = []

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        c10d,
        "_set_pg_timeout",
        lambda timeout, group: calls.append((timeout, group)),
    )
    monkeypatch.setattr(
        c10d,
        "_world",
        SimpleNamespace(pg_map={group_a: object(), group_b: object()}),
    )

    assert serving_timeout.apply_serving_collective_timeout(2.5) is True
    assert calls == [
        (datetime.timedelta(seconds=2.5), None),
        (datetime.timedelta(seconds=2.5), group_a),
        (datetime.timedelta(seconds=2.5), group_b),
    ]
    assert serving_timeout._applied is True

    assert serving_timeout.apply_serving_collective_timeout(1.0) is False
    assert len(calls) == 3


def test_missing_private_timeout_api_is_noop(monkeypatch):
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.delattr(c10d, "_set_pg_timeout", raising=False)

    assert serving_timeout.apply_serving_collective_timeout(2.0) is False
    assert serving_timeout._applied is False
