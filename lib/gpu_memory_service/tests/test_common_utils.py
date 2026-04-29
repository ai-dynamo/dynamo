# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.common.utils import get_socket_path, invalidate_uuid_cache


@pytest.fixture(autouse=True)
def clear_uuid_cache():
    invalidate_uuid_cache()
    yield
    invalidate_uuid_cache()


def test_get_socket_path_uses_ordered_gpu_uuid_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_GPU_UUIDS", "GPU-0000,GPU-1111")

    assert get_socket_path(1) == str(tmp_path / "gms_GPU-1111_weights.sock")


def test_get_socket_path_uses_direct_gpu_uuid_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_GPU_UUIDS", "GPU-0000,GPU-1111")
    monkeypatch.setenv("GMS_GPU_UUID_1", "GPU-direct")

    assert get_socket_path(1, "metadata") == str(
        tmp_path / "gms_GPU-direct_metadata.sock"
    )


def test_get_socket_path_uses_visible_device_uuid_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")

    assert get_socket_path(0) == str(tmp_path / "gms_GPU-aaaa_weights.sock")
