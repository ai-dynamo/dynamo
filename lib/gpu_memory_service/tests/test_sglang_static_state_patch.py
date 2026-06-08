# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types

import gpu_memory_service.integrations.sglang as sglang_gms
import pytest
from gpu_memory_service.integrations.sglang import patches

pytestmark = pytest.mark.pre_merge


def test_patch_static_state_for_gms_targets_current_weight_updater(monkeypatch):
    module_name = "sglang.srt.managers.scheduler_components.weight_updater"
    module = types.ModuleType(module_name)
    module._export_static_state = lambda model: {"buffers": [("x", object())]}

    def _import_static_state(model, static_params):
        raise AssertionError("original import should be replaced")

    module._import_static_state = _import_static_state
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(patches, "_static_state_patched", False)

    patches.patch_static_state_for_gms()

    assert module._export_static_state(object()) == {"buffers": []}
    assert module._import_static_state(object(), {"buffers": [("x", object())]}) is None
    assert patches._static_state_patched is True


def _install_fake_scheduler(monkeypatch):
    module_name = "sglang.srt.managers.scheduler"
    module = types.ModuleType(module_name)

    class Scheduler:
        def __init__(self):
            self.calls = 0
            self.flushed = False
            self.completed = False

        def is_fully_idle(self):
            return True

        def flush_cache(self, empty_cache=True):
            assert empty_cache is False
            self.flushed = True

        def on_idle(self):
            self.calls += 1
            if not self.flushed:
                raise ValueError(
                    "pool memory leak detected! [full] total=256, "
                    "available=192, evictable=0, protected=0, "
                    "session_held=0, uncached=0"
                )
            self.completed = True
            return "ok"

    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(patches, "_idle_leak_recovery_patched", False)
    return Scheduler


def test_patch_idle_leak_recovery_flushes_only_in_gms_failover(monkeypatch):
    Scheduler = _install_fake_scheduler(monkeypatch)
    monkeypatch.setenv("GMS_SGLANG_SHARED_KV", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")

    patches.patch_idle_leak_recovery_for_gms()
    scheduler = Scheduler()

    assert scheduler.on_idle() == "ok"
    assert scheduler.flushed is True
    assert scheduler.completed is True
    assert scheduler.calls == 2


def test_patch_idle_leak_recovery_preserves_upstream_strictness(monkeypatch):
    Scheduler = _install_fake_scheduler(monkeypatch)
    monkeypatch.delenv("GMS_SGLANG_SHARED_KV", raising=False)
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)

    patches.patch_idle_leak_recovery_for_gms()
    scheduler = Scheduler()

    try:
        scheduler.on_idle()
    except ValueError as exc:
        assert "pool memory leak detected" in str(exc)
    else:
        raise AssertionError("expected upstream strict invariant to be preserved")
    assert scheduler.flushed is False
    assert scheduler.calls == 1


def test_configure_shared_failover_env_disables_sglang_tp_memory_check(monkeypatch):
    monkeypatch.setenv("GMS_SGLANG_SHARED_KV", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.delenv("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", raising=False)

    sglang_gms.configure_shared_failover_env()

    assert os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "0"


def test_configure_shared_failover_env_preserves_explicit_sglang_tp_memory_check(
    monkeypatch,
):
    monkeypatch.setenv("GMS_SGLANG_SHARED_KV", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "1")

    sglang_gms.configure_shared_failover_env()

    assert os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "1"
