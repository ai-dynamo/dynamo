# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM GMS shadow mode configuration.

These tests only exercise pure-Python logic in
gpu_memory_service/integrations/trtllm/utils.py — no CUDA, no tensorrt_llm,
so they run anywhere.
"""

import pytest
from gpu_memory_service.integrations.trtllm.utils import (
    configure_gms_lock_mode,
    is_shadow_mode,
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
