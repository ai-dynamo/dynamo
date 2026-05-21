# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM-specific GMS patch helpers."""

from __future__ import annotations

import sys
import types

import pytest

try:
    from gpu_memory_service.integrations.vllm.patches import (
        fused_moe_cpu_routing_buffers_during_meta_init,
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


class _FakeTensor:
    def __init__(self, device_type: str) -> None:
        self.device = types.SimpleNamespace(type=device_type)


class _FakeTorch(types.ModuleType):
    int32 = object()
    _device_stack = ["cpu"]

    class device:
        def __init__(self, device_type: str) -> None:
            self.device_type = device_type

        def __enter__(self):
            _FakeTorch._device_stack.append(self.device_type)

        def __exit__(self, exc_type, exc, tb):
            _FakeTorch._device_stack.pop()

    def full(self, *args, **kwargs):
        return _FakeTensor(self._device_stack[-1])


@pytest.fixture
def fake_fused_moe_layer(monkeypatch):
    """Install minimal fake torch and vLLM FusedMoE layer modules."""
    fake_torch = _FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    package_names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ]
    packages = {name: types.ModuleType(name) for name in package_names}
    for package in packages.values():
        package.__path__ = []

    layer = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")

    def determine_expert_map():
        return fake_torch.full((4,), -1, dtype=fake_torch.int32)

    layer.determine_expert_map = determine_expert_map

    packages["vllm"].model_executor = packages["vllm.model_executor"]
    packages["vllm.model_executor"].layers = packages["vllm.model_executor.layers"]
    packages["vllm.model_executor.layers"].fused_moe = packages[
        "vllm.model_executor.layers.fused_moe"
    ]
    packages["vllm.model_executor.layers.fused_moe"].layer = layer

    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, layer.__name__, layer)

    return layer


def test_fused_moe_meta_init_patch_forces_expert_map_to_cpu(fake_fused_moe_layer):
    import torch

    with torch.device("meta"):
        before = fake_fused_moe_layer.determine_expert_map()
    assert before.device.type == "meta"

    with fused_moe_cpu_routing_buffers_during_meta_init():
        with torch.device("meta"):
            during = fake_fused_moe_layer.determine_expert_map()
    assert during.device.type == "cpu"

    with torch.device("meta"):
        after = fake_fused_moe_layer.determine_expert_map()
    assert after.device.type == "meta"


def test_fused_moe_meta_init_patch_restores_after_error(fake_fused_moe_layer):
    original = fake_fused_moe_layer.determine_expert_map

    with pytest.raises(RuntimeError, match="boom"):
        with fused_moe_cpu_routing_buffers_during_meta_init():
            assert fake_fused_moe_layer.determine_expert_map is not original
            raise RuntimeError("boom")

    assert fake_fused_moe_layer.determine_expert_map is original
