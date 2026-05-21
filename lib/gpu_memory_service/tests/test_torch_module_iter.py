# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for GMS torch module tensor helpers."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

try:
    from gpu_memory_service.client.torch.module import (
        _iter_module_tensors,
        materialize_module_from_gms,
        register_module_tensors,
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


class _CudaLikeTensor:
    is_cuda = True


class _FakeStorage:
    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes


class _FakeCudaTensor:
    is_cuda = True

    def __init__(self, *, numel: int, nbytes: int, data_ptr: int = 1) -> None:
        self._numel = numel
        self._storage = _FakeStorage(nbytes)
        self._data_ptr = data_ptr

    def numel(self) -> int:
        return self._numel

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def data_ptr(self) -> int:
        if self._numel == 0:
            raise AssertionError("zero-size tensors should not need a pointer")
        return self._data_ptr


class _ModuleWithReadOnlyTensorProperty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra = _CudaLikeTensor()

    @property
    def expert_map(self):
        return _CudaLikeTensor()


def test_iter_module_tensors_skips_read_only_tensor_properties(monkeypatch):
    """Read-only properties should not be registered as tensor attrs."""
    monkeypatch.setattr(
        torch, "is_tensor", lambda value: isinstance(value, _CudaLikeTensor)
    )

    tensors = list(_iter_module_tensors(_ModuleWithReadOnlyTensorProperty()))

    assert [(name, tensor_type) for name, _, tensor_type in tensors] == [
        ("extra", "tensor_attr")
    ]


class _NoMetadataManager:
    mappings = {}

    def metadata_list(self):
        return []

    def metadata_put(self, *args, **kwargs):
        raise AssertionError("zero-size tensors should not be registered")


def test_register_module_tensors_skips_zero_size_parameters():
    """Zero-size CUDA placeholders have no allocation to register."""
    module = torch.nn.Module()
    module._parameters["empty"] = _FakeCudaTensor(numel=0, nbytes=0)

    register_module_tensors(_NoMetadataManager(), module)


def test_register_module_tensors_keeps_strict_non_empty_parameters():
    """Non-empty parameters still must come from a GMS allocation."""
    module = torch.nn.Module()
    module._parameters["weight"] = _FakeCudaTensor(numel=1, nbytes=4)

    with pytest.raises(RuntimeError, match="not found in any GMS allocation"):
        register_module_tensors(_NoMetadataManager(), module)


def test_materialize_module_from_gms_recreates_zero_size_meta_tensors(monkeypatch):
    """Skipped zero-size meta tensors should become real tensors on read side."""
    original_empty_strided = torch.empty_strided

    def fake_empty_strided(shape, stride, *, dtype, device):
        return original_empty_strided(shape, stride, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch, "empty_strided", fake_empty_strided)

    module = torch.nn.Module()
    module.empty_param = torch.nn.Parameter(
        torch.empty((2, 0), device="meta", dtype=torch.float32),
        requires_grad=False,
    )
    module.register_buffer(
        "empty_buffer", torch.empty((0,), device="meta", dtype=torch.int32)
    )
    module.empty_attr = torch.empty((3, 0), device="meta", dtype=torch.float16)

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert not module.empty_param.is_meta
    assert not module.empty_buffer.is_meta
    assert not module.empty_attr.is_meta
    assert tuple(module.empty_param.shape) == (2, 0)
    assert tuple(module.empty_buffer.shape) == (0,)
    assert tuple(module.empty_attr.shape) == (3, 0)
    assert module.empty_param.dtype is torch.float32
    assert module.empty_buffer.dtype is torch.int32
    assert module.empty_attr.dtype is torch.float16


def test_materialize_module_from_gms_refreshes_cached_tensor_aliases(monkeypatch):
    """Objects that cached constructor-time meta params should see CUDA aliases."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="parameter",
            shape=(3,),
            stride=(1,),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.arange(3, dtype=torch.float32)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"gate.e_score_correction_bias": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.gate = torch.nn.Module()
    bias = torch.nn.Parameter(torch.empty(3, device="meta", dtype=torch.float32))
    module.gate.e_score_correction_bias = bias
    module.router = types.SimpleNamespace(e_score_correction_bias=bias)
    module.aliases = [bias]
    module.alias_dict = {"bias": bias}

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    materialized = module.gate.e_score_correction_bias
    assert not materialized.is_meta
    assert module.router.e_score_correction_bias is materialized
    assert module.aliases[0] is materialized
    assert module.alias_dict["bias"] is materialized
    torch.testing.assert_close(materialized, torch.arange(3, dtype=torch.float32))


def test_materialize_module_from_gms_preserves_shared_parameter_aliases(monkeypatch):
    """Duplicate metadata paths for the same Parameter should share one object."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="parameter",
            shape=(3,),
            stride=(1,),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def __init__(self) -> None:
            self.materialize_calls = 0

        def materialize(self, _manager, _device_index):
            self.materialize_calls += 1
            return torch.arange(3, dtype=torch.float32)

    spec = _FakeSpec()
    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {
            "gate.e_score_correction_bias": spec,
            "experts.e_score_correction_bias": spec,
            "experts.runner.gate.e_score_correction_bias": spec,
        },
    )

    module = torch.nn.Module()
    module.gate = torch.nn.Module()
    module.experts = torch.nn.Module()
    module.experts.runner = torch.nn.Module()
    module.experts.runner.gate = torch.nn.Module()
    bias = torch.nn.Parameter(torch.empty(3, device="meta", dtype=torch.float32))
    module.gate.e_score_correction_bias = bias
    module.experts.e_score_correction_bias = bias
    module.experts.runner.gate.e_score_correction_bias = bias

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    materialized = module.gate.e_score_correction_bias
    assert module.experts.e_score_correction_bias is materialized
    assert module.experts.runner.gate.e_score_correction_bias is materialized
    assert spec.materialize_calls == 1
    torch.testing.assert_close(materialized, torch.arange(3, dtype=torch.float32))
