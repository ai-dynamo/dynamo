# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gpu_memory_service.common import utils
from gpu_memory_service.core import device as device_identity

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.fixture(autouse=True)
def clear_device_uuid_cache():
    device_identity.invalidate_device_uuid_cache()
    yield
    device_identity.invalidate_device_uuid_cache()


def test_device_uuid_format_and_cache(monkeypatch):
    calls = []
    success = object()
    uuid_bytes = bytes(range(16))
    cuda = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=success),
        cuInit=lambda flags: (calls.append(("init", flags)) or success,),
        cuDeviceGet=lambda ordinal: (
            calls.append(("device", ordinal)) or success,
            42,
        ),
        cuDeviceGetUuid=lambda device: (
            calls.append(("uuid", device)) or success,
            SimpleNamespace(bytes=uuid_bytes),
        ),
    )
    monkeypatch.setattr(device_identity, "cuda", cuda)

    expected = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert device_identity.get_device_uuid(3) == expected
    assert device_identity.get_device_uuid(3) == expected
    assert calls == [("init", 0), ("device", 3), ("uuid", 42)]

    device_identity.invalidate_device_uuid_cache()
    assert device_identity.get_device_uuid(3) == expected
    assert calls == [
        ("init", 0),
        ("device", 3),
        ("uuid", 42),
        ("init", 0),
        ("device", 3),
        ("uuid", 42),
    ]


def test_device_uuid_driver_error_is_raised(monkeypatch):
    success = object()
    failure = object()
    cuda = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=success),
        cuInit=lambda flags: (success,),
        cuDeviceGet=lambda ordinal: (failure, 0),
        cuGetErrorString=lambda result: (success, b"invalid device ordinal"),
    )
    monkeypatch.setattr(device_identity, "cuda", cuda)

    with pytest.raises(
        RuntimeError,
        match="CUDA driver call cuDeviceGet failed: invalid device ordinal",
    ):
        device_identity.get_device_uuid(7)


def test_v0_socket_path_uses_shared_device_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        device_identity,
        "get_device_uuid",
        lambda device: f"GPU-visible-{device}",
    )
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))

    assert utils.get_socket_path(2, "kv_cache") == str(
        tmp_path / "gms_GPU-visible-2_kv_cache.sock"
    )


def test_v0_cache_invalidation_uses_shared_device_identity(monkeypatch):
    invalidated = []
    monkeypatch.setattr(
        device_identity,
        "invalidate_device_uuid_cache",
        lambda: invalidated.append(True),
    )

    utils.invalidate_uuid_cache()

    assert invalidated == [True]
