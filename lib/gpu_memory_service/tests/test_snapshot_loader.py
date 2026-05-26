# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

import pytest

try:
    from gpu_memory_service.cli.snapshot import loader
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_list_checkpoint_devices_discovers_device_directories(tmp_path):
    (tmp_path / "device-2").mkdir()
    (tmp_path / "device-0").mkdir()
    (tmp_path / "device-0-copy").mkdir()
    (tmp_path / "not-a-device").mkdir()
    (tmp_path / "device-1").write_text("not a directory", encoding="utf-8")

    assert loader._list_checkpoint_devices(str(tmp_path)) == [0, 2]


def test_list_checkpoint_devices_falls_back_to_cuda_discovery(tmp_path, monkeypatch):
    from gpu_memory_service.common import cuda_utils

    monkeypatch.setattr(cuda_utils, "list_devices", lambda: [7])

    assert loader._list_checkpoint_devices(str(tmp_path)) == [7]
