# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
pytest.importorskip("vllm.logger")

from gpu_memory_service.common.locks import RequestedLockType  # noqa: E402
from gpu_memory_service.v1.integrations.vllm import backend  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
]


def test_resume_refreshes_target_local_socket_paths_before_reconnect(
    monkeypatch,
) -> None:
    events = []

    def invalidate_device_uuid_cache() -> None:
        events.append("invalidate")

    def get_socket_path(device: int, domain: str) -> str:
        events.append(f"path:{device}:{domain}")
        return f"/target/GPU-target/{domain}.sock"

    monkeypatch.setattr(
        backend.device_identity,
        "invalidate_device_uuid_cache",
        invalidate_device_uuid_cache,
    )
    monkeypatch.setattr(backend.device_identity, "get_socket_path", get_socket_path)

    sleep_backend = backend.GMSV1SleepModeBackend.__new__(backend.GMSV1SleepModeBackend)
    sleep_backend._state = "SUSPENDED"
    sleep_backend._device = 2
    sleep_backend._weights = Mock()
    sleep_backend._kv_cache = Mock()
    sleep_backend._weights.refresh_socket_path.side_effect = lambda path: events.append(
        f"weights:refresh:{path}"
    )
    sleep_backend._kv_cache.refresh_socket_path.side_effect = (
        lambda path: events.append(f"kv_cache:refresh:{path}")
    )
    sleep_backend._kv_cache.connect.side_effect = lambda lock_type: events.append(
        f"kv_cache:connect:{lock_type.value}"
    )
    sleep_backend._weights.connect.side_effect = lambda lock_type: events.append(
        f"weights:connect:{lock_type.value}"
    )

    sleep_backend.resume()

    assert events == [
        "invalidate",
        "path:2:weights",
        "path:2:kv_cache",
        "weights:refresh:/target/GPU-target/weights.sock",
        "kv_cache:refresh:/target/GPU-target/kv_cache.sock",
        f"kv_cache:connect:{RequestedLockType.RW.value}",
        f"weights:connect:{RequestedLockType.RO.value}",
    ]
    sleep_backend._kv_cache.reallocate_all_handles.assert_called_once_with()
    sleep_backend._kv_cache.remap_all_vas.assert_called_once_with()
    sleep_backend._weights.remap_all_vas.assert_called_once_with()
    assert sleep_backend._state == "RUNNING"
