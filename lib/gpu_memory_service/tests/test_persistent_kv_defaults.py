# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default persistent-KV configuration helpers."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_persistent_kv_socket_defaults_to_existing_kv_cache_server(monkeypatch):
    import gpu_memory_service.common.utils as common_utils
    from gpu_memory_service.integrations.common import utils

    monkeypatch.delenv("GMS_VLLM_VMM_IPC_SOCKET", raising=False)
    monkeypatch.delenv("DYN_GMS_PERSISTENT_KV_SOCKET", raising=False)
    monkeypatch.setattr(
        common_utils,
        "get_socket_path",
        lambda device, tag: f"socket:{device}:{tag}",
    )

    assert utils.get_gms_persistent_kv_socket(7, "GMS_VLLM_VMM_IPC_SOCKET") == (
        "socket:7:kv_cache"
    )


def test_persistent_kv_socket_allows_explicit_override(monkeypatch):
    from gpu_memory_service.integrations.common import utils

    monkeypatch.setenv("GMS_VLLM_VMM_IPC_SOCKET", "/run/gms/explicit.sock")

    assert utils.get_gms_persistent_kv_socket(0, "GMS_VLLM_VMM_IPC_SOCKET") == (
        "/run/gms/explicit.sock"
    )


def test_persistent_kv_engine_id_prefers_explicit_env(monkeypatch):
    from gpu_memory_service.integrations.common import utils

    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "worker-a")

    assert (
        utils.get_gms_persistent_kv_engine_id("vllm", 0, "GMS_VLLM_VMM_IPC_ENGINE_ID")
        == "worker-a"
    )


def test_persistent_kv_engine_id_fallback_is_stable(monkeypatch):
    from gpu_memory_service.integrations.common import utils

    for name in (
        "GMS_VLLM_VMM_IPC_ENGINE_ID",
        "GMS_VLLM_ENGINE_ID",
        "DYN_GMS_ENGINE_ID",
        "DYN_ENGINE_ID",
        "DYN_NAMESPACE",
        "DYN_COMPONENT",
        "DYN_SYSTEM_NAME",
        "DYN_SYSTEM_PORT",
        "DYN_WORKER_ID",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DYN_NAMESPACE", "ns")
    monkeypatch.setenv("DYN_COMPONENT", "decode")
    monkeypatch.setenv("DYN_SYSTEM_PORT", "9091")
    monkeypatch.setenv("DYN_WORKER_ID", "3")

    assert (
        utils.get_gms_persistent_kv_engine_id("vllm", 1, "GMS_VLLM_VMM_IPC_ENGINE_ID")
        == "vllm|cuda=1|dyn_namespace=ns|dyn_component=decode|dyn_worker_id=3"
    )
