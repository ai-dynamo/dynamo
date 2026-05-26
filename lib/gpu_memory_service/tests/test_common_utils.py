# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared GMS utilities."""

import os

import pytest

try:
    from gpu_memory_service.common import utils
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


@pytest.fixture(autouse=True)
def clear_uuid_cache():
    utils.invalidate_uuid_cache()
    yield
    utils.invalidate_uuid_cache()


def _socket_name(path: str) -> str:
    return os.path.basename(path)


def test_get_socket_path_prefers_per_device_uuid_env(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/tmp/gms-test")
    monkeypatch.setenv("GMS_GPU_UUID_0", "GPU-direct")
    monkeypatch.setenv("GMS_GPU_UUIDS", "GPU-ordered-0,GPU-ordered-1")

    assert _socket_name(utils.get_socket_path(0)) == "gms_GPU-direct_weights.sock"


def test_get_socket_path_uses_ordered_uuid_env(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/tmp/gms-test")
    monkeypatch.setenv("GMS_GPU_UUIDS", "GPU-ordered-0; GPU-ordered-1")

    assert _socket_name(utils.get_socket_path(1, "kv_cache")) == (
        "gms_GPU-ordered-1_kv_cache.sock"
    )


def test_get_socket_path_uses_visible_device_uuid_env(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/tmp/gms-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-visible-0,0,GPU-visible-1")

    assert _socket_name(utils.get_socket_path(1)) == "gms_GPU-visible-1_weights.sock"


def test_visible_device_uuid_split_ignores_numeric_ordinals():
    assert utils._split_visible_device_uuids("0,GPU-a,1,MIG-b") == ["GPU-a", "MIG-b"]
