# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM-specific GMS patch helpers."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

try:
    from gpu_memory_service.integrations.vllm.patches import (
        fused_moe_cpu_routing_buffers_during_meta_init,
        meta_safe_module_to_during_meta_init,
        patch_moe_wna16_marlin_gemm_fake_impl,
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


def test_meta_safe_module_to_keeps_meta_tensors_on_meta():
    class _TinyVisionTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 3, device="meta"))
            self.register_buffer("scale", torch.empty(3, device="meta"))

    with pytest.raises(NotImplementedError, match="Cannot copy out of meta tensor"):
        _TinyVisionTower().to(device="cpu", dtype=torch.float16)

    with meta_safe_module_to_during_meta_init():
        tower = _TinyVisionTower().to(device="cpu", dtype=torch.float16)

    assert tower.weight.is_meta
    assert tower.scale.is_meta
    assert tower.weight.dtype is torch.float16
    assert tower.scale.dtype is torch.float16


def test_meta_safe_module_to_restores_after_error():
    original = torch.nn.Module.to

    with pytest.raises(RuntimeError, match="boom"):
        with meta_safe_module_to_during_meta_init():
            assert torch.nn.Module.to is not original
            raise RuntimeError("boom")

    assert torch.nn.Module.to is original


def test_patch_moe_wna16_marlin_gemm_fake_impl_accepts_vllm_021_signature(
    monkeypatch,
):
    import torch

    captured: dict[str, object] = {}
    stale = object()

    def fake_register_fake(qualname, *, allow_override=False):
        captured["qualname"] = qualname
        captured["allow_override"] = allow_override

        def decorator(fn):
            captured["fn"] = fn
            return fn

        return decorator

    fake_ops = types.SimpleNamespace(
        _moe_C=types.SimpleNamespace(moe_wna16_marlin_gemm=object())
    )

    fake_vllm_custom_ops = types.ModuleType("vllm._custom_ops")
    fake_vllm = types.ModuleType("vllm")
    fake_vllm._custom_ops = fake_vllm_custom_ops

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_vllm_custom_ops)
    monkeypatch.setattr(torch, "ops", fake_ops)
    monkeypatch.setattr(torch.library, "register_fake", fake_register_fake)

    fake_entry = types.SimpleNamespace(
        fake_impl=types.SimpleNamespace(kernels=[stale, stale])
    )
    fake_singleton_module = types.ModuleType("torch._library.simple_registry")
    fake_singleton_module.singleton = types.SimpleNamespace(
        find=lambda qualname: fake_entry
    )
    monkeypatch.setitem(
        sys.modules,
        "torch._library.simple_registry",
        fake_singleton_module,
    )

    monkeypatch.setitem(
        patch_moe_wna16_marlin_gemm_fake_impl.__globals__,
        "_moe_wna16_fake_patched",
        False,
    )

    patch_moe_wna16_marlin_gemm_fake_impl()

    assert captured["qualname"] == "_moe_C::moe_wna16_marlin_gemm"
    assert captured["allow_override"] is True
    assert len(fake_entry.fake_impl.kernels) == 1
    assert fake_entry.fake_impl.kernels[0] is not stale

    input_tensor = torch.empty((2, 4), dtype=torch.float16)
    output = torch.empty((2, 8), dtype=torch.float16)
    int_tensor = torch.empty((1,), dtype=torch.int32)
    topk = torch.empty((2, 1), dtype=torch.float32)

    result = captured["fn"](
        input_tensor,
        output,
        torch.empty((1, 1, 1), dtype=torch.int32),
        None,
        torch.empty((1, 1, 8), dtype=torch.float16),
        None,
        None,
        None,
        None,
        None,
        int_tensor,
        int_tensor,
        int_tensor,
        int_tensor,
        topk,
        16,
        1,
        False,
        0,
        2,
        8,
        4,
        True,
        False,
        True,
        False,
        -1,
        -1,
        -1,
    )

    assert result is output
