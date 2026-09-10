# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest


def _disable_failover(monkeypatch) -> None:
    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "0")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "0")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "0")


def test_authoritative_startup_rejects_missing_live_hook(monkeypatch):
    from gpu_memory_service.integrations.vllm import (
        install_kv_leases,
        install_vmm_ipc_kv,
        startup,
    )

    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "1")
    monkeypatch.setattr(install_kv_leases, "lease_hooks_installed", lambda: True)
    monkeypatch.setattr(install_kv_leases, "engine_core_hook_installed", lambda: False)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "persistent_kv_hooks_installed", lambda: True
    )

    with pytest.raises(RuntimeError, match="EngineCore process hook.*API drift"):
        startup.verify_kv_failover_hooks()


def test_strict_install_is_idempotent_when_live_hooks_exist(monkeypatch):
    from gpu_memory_service.integrations.vllm import (
        install_kv_leases,
        install_vmm_ipc_kv,
        startup,
    )

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    calls = []
    monkeypatch.setattr(
        install_kv_leases, "install", lambda: calls.append("leases") or False
    )
    monkeypatch.setattr(
        install_kv_leases,
        "install_engine_core_hook",
        lambda: calls.append("core") or False,
    )
    monkeypatch.setattr(
        install_vmm_ipc_kv, "install", lambda: calls.append("vmm") or False
    )
    monkeypatch.setattr(install_kv_leases, "lease_hooks_installed", lambda: True)
    monkeypatch.setattr(install_kv_leases, "engine_core_hook_installed", lambda: True)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "persistent_kv_hooks_installed", lambda: True
    )

    startup.install_and_verify_kv_failover_hooks()
    startup.install_and_verify_kv_failover_hooks()
    assert calls == ["leases", "core", "vmm"] * 2


def test_weights_only_mode_tolerates_optional_hook_failure(monkeypatch):
    from gpu_memory_service.integrations.vllm import (
        install_kv_leases,
        install_vmm_ipc_kv,
        startup,
    )

    _disable_failover(monkeypatch)

    def unavailable():
        raise ImportError("optional vLLM KV API is unavailable")

    monkeypatch.setattr(install_kv_leases, "install", unavailable)
    monkeypatch.setattr(install_kv_leases, "install_engine_core_hook", unavailable)
    monkeypatch.setattr(install_vmm_ipc_kv, "install", unavailable)
    startup.install_and_verify_kv_failover_hooks()


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM unavailable")
def test_current_vllm_api_accepts_strict_install_in_fresh_process():
    env = {
        **os.environ,
        "GMS_VLLM_SHARED_KV": "1",
        "GMS_VLLM_KV_LEASES": "1",
        "GMS_VLLM_VMM_IPC_KV": "1",
        "GMS_KV_DIRECTORY_MODE": "authoritative",
    }
    script = (
        "from gpu_memory_service.integrations.vllm.startup import "
        "install_and_verify_kv_failover_hooks as install; install(); install()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
