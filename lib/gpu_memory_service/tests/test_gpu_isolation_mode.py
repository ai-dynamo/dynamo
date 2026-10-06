# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DYN_GMS_GPU_ISOLATION selects coherent failover defaults; explicit settings win."""

import pytest
from gpu_memory_service.common import gpu_isolation
from gpu_memory_service.integrations.common import gpu_quiescence
from gpu_memory_service.server.gpu_quiescence import GPUQuiescenceManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]

_SETTINGS = (
    "DYN_GMS_GPU_ISOLATION",
    "DYN_GMS_GPU_QUIESCENCE_PROVIDER",
    "DYN_VLLM_GMS_GPU_QUIESCENCE_PROVIDER",
    "DYN_GMS_GPU_QUIESCENCE_COMMAND",
    "DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND",
    "DYN_GMS_GPU_CRASH_INTERLOCK",
    "DYN_VLLM_GMS_GPU_CRASH_INTERLOCK",
    "DYN_GMS_EXPERIMENTAL_PROCESS_LIFETIME_RECLAIM",
    "DYN_GMS_FAILOVER_RECLAIM_POLICY",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "DYN_GMS_MPS_PIPE_DIRECTORY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpu_quiescence, "_mps_client_possible", lambda: False)


@pytest.mark.parametrize(
    ("mode", "provider", "interlock", "policy"),
    [
        (None, "quarantine-only", False, "gpu-proof"),
        ("mps", "gms-mps", True, "gpu-proof"),
        ("process", "process-lifetime", False, "process-death-timeout"),
    ],
)
def test_mode_selects_coherent_defaults(monkeypatch, mode, provider, interlock, policy):
    if mode is not None:
        monkeypatch.setenv("DYN_GMS_GPU_ISOLATION", mode.upper())
    assert gpu_quiescence._provider_name("vllm") == provider
    assert gpu_quiescence.gpu_crash_interlock_enabled("vllm") is interlock
    assert gpu_isolation.default_reclaim_policy() == policy
    assert GPUQuiescenceManager.configured("vllm") is (provider == "gms-mps")


def test_process_mode_enables_process_lifetime_without_experimental_flag(
    monkeypatch,
):
    assert not gpu_quiescence.gpu_quiescence_provider_configured("vllm")
    monkeypatch.setenv("DYN_GMS_GPU_ISOLATION", "process")
    assert gpu_quiescence.gpu_quiescence_provider_configured("vllm")
    # Never with possible MPS clients: the server owns the context there.
    monkeypatch.setattr(gpu_quiescence, "_mps_client_possible", lambda: True)
    assert not gpu_quiescence.gpu_quiescence_provider_configured("vllm")


def test_explicit_settings_override_the_mode(monkeypatch):
    monkeypatch.setenv("DYN_GMS_GPU_ISOLATION", "process")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_VLLM_GMS_GPU_CRASH_INTERLOCK", "1")
    assert gpu_quiescence._provider_name("vllm") == "gms-mps"
    assert gpu_quiescence.gpu_crash_interlock_enabled("vllm")


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("DYN_GMS_GPU_ISOLATION", "none")
    with pytest.raises(ValueError, match="DYN_GMS_GPU_ISOLATION"):
        gpu_isolation.gpu_isolation_mode()
