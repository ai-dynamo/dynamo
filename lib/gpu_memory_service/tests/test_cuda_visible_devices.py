# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-visible ordinal handling at GMS process boundaries."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _fake_nvml():
    calls = []
    module = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index))
        or f"index:{index}",
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid))
        or f"uuid:{uuid}",
        nvmlDeviceGetUUID=lambda handle: f"GPU-{handle}",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(free=7, total=11),
    )
    return module, calls


def test_socket_path_maps_cuda_visible_ordinal_to_nvml_device(monkeypatch, tmp_path):
    from gpu_memory_service.common import utils

    pynvml, calls = _fake_nvml()
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    utils.invalidate_uuid_cache()

    assert utils.get_socket_path(0).endswith("gms_GPU-index:3_weights.sock")
    assert utils.get_socket_path(1).endswith("gms_GPU-index:1_weights.sock")
    assert calls == [("index", 3), ("index", 1)]


def test_socket_uuid_cache_tracks_visibility_mapping(monkeypatch, tmp_path):
    from gpu_memory_service.common import utils

    pynvml, calls = _fake_nvml()
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    utils.invalidate_uuid_cache()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    assert "GPU-uuid:GPU-a" in utils.get_socket_path(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    assert "GPU-uuid:GPU-b" in utils.get_socket_path(0)
    assert calls == [("uuid", "GPU-a"), ("uuid", "GPU-b")]


def test_cuda_vmm_lists_visible_ordinals(monkeypatch):
    from gpu_memory_service.common.vmm import cuda_utils

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3")

    assert cuda_utils.CudaVMM().list_devices() == [0, 1]


def test_cuda_memory_info_uses_visible_mapping(monkeypatch):
    from gpu_memory_service.common.vmm import cuda_utils

    pynvml, calls = _fake_nvml()
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    assert cuda_utils.cuda_device_memory_info(0) == (7, 11)
    assert calls == [("index", 4)]
