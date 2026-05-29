# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM-specific GMS patch helpers."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

try:
    from gpu_memory_service.integrations.vllm.patches import (
        patch_kv_cache_allocation_for_scratch,
    )
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_patch_kv_cache_allocation_for_scratch_uses_empty(monkeypatch):
    calls: list[tuple[str, int]] = []
    mempool_calls: list[tuple[str, str]] = []
    original_empty = torch.empty

    class FakeGPUModelRunner:
        def __init__(self) -> None:
            self.device = "cuda:0"
            self.runner_only_attn_layers = set()

        def _allocate_kv_cache_tensors(self, kv_cache_config):
            kv_cache_raw_tensors = {}
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                tensor = torch.zeros(
                    kv_cache_tensor.size,
                    dtype=torch.int8,
                    device=self.device,
                )
                for layer_name in kv_cache_tensor.shared_by:
                    kv_cache_raw_tensors[layer_name] = tensor
            return kv_cache_raw_tensors

    fake_gpu_model_runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    fake_gpu_model_runner.GPUModelRunner = FakeGPUModelRunner

    class FakeMixin:
        @staticmethod
        def allocate_uniform_kv_caches(*args, **kwargs):
            return torch.zeros(64, dtype=torch.int8, device=kwargs.get("device"))

    fake_mixin_module = types.ModuleType(
        "vllm.v1.worker.kv_connector_model_runner_mixin"
    )
    fake_mixin_module.KVConnectorModelRunnerMixin = FakeMixin

    fake_worker = types.ModuleType("vllm.v1.worker")
    fake_worker.gpu_model_runner = fake_gpu_model_runner
    fake_worker.kv_connector_model_runner_mixin = fake_mixin_module

    fake_v1 = types.ModuleType("vllm.v1")
    fake_v1.worker = fake_worker
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.v1 = fake_v1

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", fake_worker)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        fake_gpu_model_runner,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.kv_connector_model_runner_mixin",
        fake_mixin_module,
    )
    monkeypatch.setitem(
        patch_kv_cache_allocation_for_scratch.__globals__,
        "_kv_cache_alloc_patched",
        False,
    )

    fake_allocator = types.ModuleType("gpu_memory_service.client.torch.allocator")
    fake_manager = object()
    fake_allocator.get_gms_client_memory_manager = lambda tag: fake_manager
    fake_allocator.is_scratch = lambda manager: True

    @contextmanager
    def fake_gms_use_mem_pool(tag, device):
        mempool_calls.append((tag, str(device)))
        yield

    fake_allocator.gms_use_mem_pool = fake_gms_use_mem_pool
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.client.torch.allocator",
        fake_allocator,
    )

    def fake_empty(size, *, dtype, device):
        calls.append(("empty", int(size)))
        return original_empty(int(size), dtype=dtype)

    def fail_zeros(*args, **kwargs):
        raise AssertionError("scratch KV allocation must not zero-fill")

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "zeros", fail_zeros)

    patch_kv_cache_allocation_for_scratch()

    runner = FakeGPUModelRunner()
    kv_cache_config = types.SimpleNamespace(
        kv_cache_tensors=[
            types.SimpleNamespace(size=16, shared_by=["layer.0"]),
            types.SimpleNamespace(size=32, shared_by=["layer.1"]),
        ],
        kv_cache_groups=[
            types.SimpleNamespace(layer_names=["layer.0", "layer.1"]),
        ],
    )

    result = runner._allocate_kv_cache_tensors(kv_cache_config)

    assert set(result) == {"layer.0", "layer.1"}
    assert calls == [("empty", 16), ("empty", 32)]
    assert mempool_calls == [("kv_cache", "cuda:0"), ("kv_cache", "cuda:0")]
