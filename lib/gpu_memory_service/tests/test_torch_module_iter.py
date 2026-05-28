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


def test_register_module_tensors_skips_recomputable_non_persistent_buffer():
    """RoPE caches are constructor state, not GMS-published weights."""
    module = torch.nn.Module()
    module._buffers["cos_sin_cache"] = _FakeCudaTensor(numel=1, nbytes=4)
    module._non_persistent_buffers_set.add("cos_sin_cache")

    register_module_tensors(_NoMetadataManager(), module)


def test_register_module_tensors_skips_mla_post_load_attrs(monkeypatch):
    """MLA projection caches are rebuilt by vLLM post-load processing."""

    class _MLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_b_proj = object()
            self.kv_lora_rank = 1
            self.num_heads = 1
            self.W_UK_T = _CudaLikeTensor()

        def process_weights_after_loading(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        torch, "is_tensor", lambda value: isinstance(value, _CudaLikeTensor)
    )

    module = torch.nn.Module()
    module.mla = _MLA()

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


def test_materialize_module_from_gms_refreshes_deep_quant_aliases(monkeypatch):
    """Quant/kernel helper objects can cache tensors several levels deep."""

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
        lambda _manager: {"experts.w13_weight_scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    scale = torch.nn.Parameter(torch.empty(3, device="meta", dtype=torch.float32))
    module.experts.w13_weight_scale = scale

    # Deeper than the old alias-refresh depth of 6.
    cursor = module
    for idx in range(8):
        child = types.SimpleNamespace()
        setattr(cursor, f"quant_child_{idx}", child)
        cursor = child
    cursor.scale = scale

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    materialized = module.experts.w13_weight_scale
    assert cursor.scale is materialized
    torch.testing.assert_close(materialized, torch.arange(3, dtype=torch.float32))


def test_materialize_module_from_gms_uses_gms_layout_for_meta_params(monkeypatch):
    """Meta params should adopt writer-side post-processed layouts."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="parameter",
            shape=(6, 3),
            stride=(3, 1),
            dtype=torch.int32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.arange(18, dtype=torch.int32).reshape(6, 3)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    scale = torch.nn.Parameter(
        torch.empty((2, 3), device="meta", dtype=torch.float32),
        requires_grad=False,
    )
    scale.weight_loader = object()
    module.scale = scale
    module.cached_scale = scale

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert tuple(module.scale.shape) == (6, 3)
    assert module.scale.dtype is torch.int32
    assert module.scale.weight_loader is scale.weight_loader
    assert module.cached_scale is module.scale
    torch.testing.assert_close(
        module.scale,
        torch.arange(18, dtype=torch.int32).reshape(6, 3),
    )


def test_materialize_module_from_gms_rejects_non_meta_layout_mismatch(monkeypatch):
    """Real parameters must still match the writer metadata."""

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
        lambda _manager: {"scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.scale = torch.nn.Parameter(torch.empty(2, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="Shape/dtype mismatch for scale"):
        materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)


def test_materialize_module_from_gms_reports_mutable_tensor_clone_context(
    monkeypatch,
):
    """Mutable tensor clone failures should name the GMS entry."""

    class _BadTensor:
        def detach(self):
            return self

        def clone(self):
            raise RuntimeError("boom")

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="tensor_attr",
            shape=(2, 3),
            stride=(3, 1),
            dtype=torch.float16,
        )
        allocation_id = "alloc"
        offset_bytes = 128

        def materialize(self, _manager, _device_index):
            return _BadTensor()

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"layer.scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.layer = torch.nn.Module()

    with pytest.raises(
        RuntimeError,
        match=r"Failed to materialize mutable tensor_attr 'layer\.scale'",
    ) as exc_info:
        materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    message = str(exc_info.value)
    assert "shape=(2, 3)" in message
    assert "dtype=torch.float16" in message
    assert "offset_bytes=128" in message


def test_materialize_module_from_gms_ignores_stale_mla_attr_metadata(monkeypatch):
    """Stale metadata for derived MLA attrs should not be cloned."""

    class _BadTensor:
        def detach(self):
            raise RuntimeError("should not clone stale MLA metadata")

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="tensor_attr",
            shape=(2, 3, 4),
            stride=(12, 1, 3),
            dtype=torch.float16,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return _BadTensor()

    class _MLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_b_proj = object()
            self.kv_lora_rank = 1
            self.num_heads = 1
            self.W_UK_T = torch.empty((2, 3, 4), device="meta")

        def process_weights_after_loading(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"mla.W_UK_T": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.mla = _MLA()

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert module.mla.W_UK_T.is_meta


def test_materialize_module_from_gms_recomputes_non_persistent_meta_buffer(
    monkeypatch,
):
    """Stale GMS metadata for recomputable caches should not be cloned."""

    class _BadTensor:
        def detach(self):
            raise RuntimeError("should not clone stale cache metadata")

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="buffer",
            shape=(3, 2),
            stride=(2, 1),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return _BadTensor()

    class _Rope(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "cos_sin_cache",
                torch.empty((3, 2), device="meta", dtype=torch.float32),
                persistent=False,
            )

        def _compute_cos_sin_cache(self):
            return torch.arange(6, dtype=torch.float32).reshape(3, 2)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"rope.cos_sin_cache": _FakeSpec()},
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module._move_recomputed_buffer_to_device",
        lambda tensor, old_buffer, *, device_index: tensor.to(dtype=old_buffer.dtype),
    )

    module = torch.nn.Module()
    module.rope = _Rope()

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert not module.rope.cos_sin_cache.is_meta
    torch.testing.assert_close(
        module.rope.cos_sin_cache,
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )


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


def test_materialize_module_from_gms_recreates_runtime_meta_buffer_attrs(monkeypatch):
    """Runtime scratch attrs skipped from metadata should not stay meta."""
    original_empty_strided = torch.empty_strided

    def fake_empty_strided(shape, stride, *, dtype, device):
        return original_empty_strided(shape, stride, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch, "empty_strided", fake_empty_strided)

    module = torch.nn.Module()
    module.indexer = torch.nn.Module()
    module.indexer.indexer_op = torch.nn.Module()
    topk = torch.empty((4, 8), device="meta", dtype=torch.int32)
    module.indexer.topk_indices_buffer = topk
    module.indexer.indexer_op.topk_indices_buffer = topk
    module.helper = types.SimpleNamespace(runtime_workspace=topk)
    module.real_weight = torch.nn.Parameter(torch.empty((2, 2), device="meta"))

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    materialized = module.indexer.topk_indices_buffer
    assert not materialized.is_meta
    assert materialized.device.type == "cpu"
    assert tuple(materialized.shape) == (4, 8)
    assert materialized.dtype is torch.int32
    assert module.indexer.indexer_op.topk_indices_buffer is materialized
    assert module.helper.runtime_workspace is materialized
    # Non-runtime non-zero meta parameters are still not silently materialized.
    assert module.real_weight.is_meta
