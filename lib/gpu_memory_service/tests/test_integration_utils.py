# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python tests for shared GMS integration helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from gpu_memory_service.common.locks import RequestedLockType

_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "integrations" / "common" / "utils.py"
)
_UTILS_SPEC = importlib.util.spec_from_file_location(
    "gpu_memory_service_integrations_common_utils",
    _UTILS_PATH,
)
assert _UTILS_SPEC is not None and _UTILS_SPEC.loader is not None
_utils = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_utils)
finalize_gms_write = _utils.finalize_gms_write

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _FakeManager:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.mappings = {0: object(), 1: object()}

    def commit(self) -> None:
        self.calls.append("commit")

    def connect(self, lock_type: RequestedLockType) -> None:
        self.calls.append(("connect", lock_type))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")


def test_finalize_gms_write_reconnects_read_only() -> None:
    manager = _FakeManager()

    finalize_gms_write(manager)

    assert manager.calls == [
        "commit",
        ("connect", RequestedLockType.RO),
        "remap_all_vas",
    ]
