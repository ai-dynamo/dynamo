# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS-side MX integration utilities.

configure_mx_ports tests are pure Python — no GPU or modelexpress required.
get_mx_load_context tests require torch (model_loader.py imports it at
module level) and are skipped when torch is not available.
"""

import os
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.vllm.utils import configure_mx_ports

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class TestConfigureMxPorts:
    """Tests for configure_mx_ports() port offset logic."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("MX_METADATA_PORT", raising=False)
        monkeypatch.delenv("MX_WORKER_GRPC_PORT", raising=False)
        monkeypatch.delenv("MX_ENABLED", raising=False)
        monkeypatch.delenv("ENGINE_ID", raising=False)

    def test_noop_when_mx_disabled(self):
        configure_mx_ports(SimpleNamespace(tensor_parallel_size=1))
        assert "MX_METADATA_PORT" not in os.environ

    def test_engine1_tp2_offsets_by_tp_size(self, monkeypatch):
        monkeypatch.setenv("MX_ENABLED", "1")
        monkeypatch.setenv("ENGINE_ID", "1")
        configure_mx_ports(SimpleNamespace(tensor_parallel_size=2))
        assert os.environ["MX_METADATA_PORT"] == "5557"
        assert os.environ["MX_WORKER_GRPC_PORT"] == "6557"

    def test_respects_user_set_base_port(self, monkeypatch):
        monkeypatch.setenv("MX_ENABLED", "1")
        monkeypatch.setenv("ENGINE_ID", "1")
        monkeypatch.setenv("MX_METADATA_PORT", "7000")
        monkeypatch.setenv("MX_WORKER_GRPC_PORT", "8000")
        configure_mx_ports(SimpleNamespace(tensor_parallel_size=2))
        assert os.environ["MX_METADATA_PORT"] == "7002"
        assert os.environ["MX_WORKER_GRPC_PORT"] == "8002"


# torch is required to import model_loader.py (top-level `import torch`)
torch = pytest.importorskip("torch", reason="torch required for model_loader")
from gpu_memory_service.integrations.vllm.model_loader import (  # noqa: E402
    get_mx_load_context,
)


class TestGetMxLoadContext:
    """Tests for get_mx_load_context() guard logic."""

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        get_mx_load_context.__globals__["_mx_ctx"] = None
        monkeypatch.delenv("MX_ENABLED", raising=False)

    def test_returns_none_when_mx_disabled(self):
        result = get_mx_load_context(vllm_config=object(), model_config=object())
        assert result is None

    def test_returns_none_when_args_missing(self, monkeypatch):
        monkeypatch.setenv("MX_ENABLED", "1")
        assert get_mx_load_context() is None
        assert get_mx_load_context(vllm_config=object()) is None
        assert get_mx_load_context(model_config=object()) is None

    def test_returns_cached_singleton(self):
        sentinel = object()
        get_mx_load_context.__globals__["_mx_ctx"] = sentinel
        assert get_mx_load_context() is sentinel
