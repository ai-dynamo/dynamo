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
