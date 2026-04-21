# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM GMS shadow mode configuration.

These tests only exercise pure-Python logic in
gpu_memory_service/integrations/trtllm/utils.py — no CUDA, no tensorrt_llm,
so they run anywhere.
"""

import pytest
from gpu_memory_service.integrations.trtllm.utils import (
    SHADOW_ALLREDUCE_STRATEGY,
    SHADOW_KV_CACHE_FRACTION,
    SHADOW_KV_CACHE_MAX_BYTES,
    configure_gms_lock_mode,
    force_nccl_allreduce_for_shadow,
    is_shadow_mode,
    is_shadow_standby,
    shrink_kv_cache_for_shadow,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("DYN_GMS_SHADOW_MODE", raising=False)
    monkeypatch.delenv("ENGINE_ID", raising=False)


def test_is_shadow_mode_default(clean_env):
    assert is_shadow_mode() is False


def test_is_shadow_mode_enabled(monkeypatch, clean_env):
    monkeypatch.setenv("DYN_GMS_SHADOW_MODE", "1")
    assert is_shadow_mode() is True


def test_engine_id_0_leaves_lock_mode_unset(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "0")
    extra = {}
    configure_gms_lock_mode(extra)
    assert "gms_read_only" not in extra


def test_engine_id_1_forces_read_only(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "1")
    extra = {}
    configure_gms_lock_mode(extra)
    assert extra["gms_read_only"] is True


def test_engine_id_missing_defaults_to_writer(clean_env):
    extra = {}
    configure_gms_lock_mode(extra)
    assert "gms_read_only" not in extra


def test_engine_id_0_with_explicit_read_only_raises(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "0")
    with pytest.raises(ValueError, match="primary writer"):
        configure_gms_lock_mode({"gms_read_only": True})


def test_engine_id_1_with_explicit_writer_raises(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "1")
    with pytest.raises(ValueError, match="requires gms_read_only=True"):
        configure_gms_lock_mode({"gms_read_only": False})


def test_engine_id_1_with_explicit_read_only_is_ok(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "1")
    extra = {"gms_read_only": True}
    configure_gms_lock_mode(extra)
    assert extra["gms_read_only"] is True


def test_configure_returns_same_dict(monkeypatch, clean_env):
    monkeypatch.setenv("ENGINE_ID", "2")
    extra = {}
    returned = configure_gms_lock_mode(extra)
    assert returned is extra


def test_is_shadow_standby_only_for_nonzero_shadow(monkeypatch, clean_env):
    # Shadow mode off: standby is always False
    monkeypatch.setenv("ENGINE_ID", "1")
    assert is_shadow_standby() is False

    monkeypatch.setenv("DYN_GMS_SHADOW_MODE", "1")
    # Engine 0 is the primary, not a standby
    monkeypatch.setenv("ENGINE_ID", "0")
    assert is_shadow_standby() is False

    # Engine 1+ in shadow mode is a standby
    monkeypatch.setenv("ENGINE_ID", "1")
    assert is_shadow_standby() is True

    monkeypatch.setenv("ENGINE_ID", "3")
    assert is_shadow_standby() is True


def test_shrink_kv_cache_clamps_fraction_and_bytes():
    class FakeCfg:
        free_gpu_memory_fraction = 0.9
        max_gpu_total_bytes = 0

    cfg = FakeCfg()
    shrink_kv_cache_for_shadow(cfg)
    assert cfg.free_gpu_memory_fraction == SHADOW_KV_CACHE_FRACTION
    assert cfg.max_gpu_total_bytes == SHADOW_KV_CACHE_MAX_BYTES


def test_force_nccl_allreduce_overrides_mnnvl():
    arg_map = {"allreduce_strategy": "MNNVL"}
    force_nccl_allreduce_for_shadow(arg_map)
    assert arg_map["allreduce_strategy"] == SHADOW_ALLREDUCE_STRATEGY


def test_force_nccl_allreduce_sets_when_unset():
    arg_map = {}
    force_nccl_allreduce_for_shadow(arg_map)
    assert arg_map["allreduce_strategy"] == SHADOW_ALLREDUCE_STRATEGY


def test_force_nccl_allreduce_is_idempotent():
    arg_map = {"allreduce_strategy": SHADOW_ALLREDUCE_STRATEGY}
    force_nccl_allreduce_for_shadow(arg_map)
    assert arg_map["allreduce_strategy"] == SHADOW_ALLREDUCE_STRATEGY
