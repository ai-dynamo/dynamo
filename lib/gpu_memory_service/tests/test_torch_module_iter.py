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
        snapshot_auxiliary_tensors,
    )
    from gpu_memory_service.client.torch.tensor import TensorMetadata
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

    def __init__(
        self,
        *,
        numel: int,
        nbytes: int,
        data_ptr: int = 1,
        shape: tuple[int, ...] = (),
        stride: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._numel = numel
        self._storage = _FakeStorage(nbytes)
        self._data_ptr = data_ptr
        self.shape = shape
        self.dtype = dtype
        self._stride = stride

    def numel(self) -> int:
        return self._numel

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def stride(self) -> tuple[int, ...]:
        return self._stride

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


class _DescriptorManager:
    def __init__(self) -> None:
        self.mappings = {
            0x1000: types.SimpleNamespace(allocation_id="descriptor", aligned_size=1)
        }
        self.puts = []
        self.create_mapping_calls = 0

    def create_mapping(self, *, size, tag):
        self.create_mapping_calls += 1
        assert size == 1
        assert tag == "weights"
        return 0x1000

    def metadata_put(self, *, key, allocation_id, offset_bytes, value):
        self.puts.append((key, allocation_id, offset_bytes, value))


def test_register_module_tensors_records_zero_size_parameter_descriptor():
    """Empty Parameters carry metadata but do not need a data pointer."""
    module = torch.nn.Module()
    module._parameters["empty"] = _FakeCudaTensor(
        numel=0,
        nbytes=0,
        shape=(2, 0),
        stride=(1, 1),
        dtype=torch.int32,
    )
    manager = _DescriptorManager()

    register_module_tensors(manager, module)

    assert len(manager.puts) == 1
    key, allocation_id, offset_bytes, value = manager.puts[0]
    assert (key, allocation_id, offset_bytes) == ("empty", "descriptor", 0)
    meta = TensorMetadata.from_bytes(value)
    assert meta.tensor_type == "empty_parameter"
    assert meta.shape == (2, 0)
    assert meta.stride == (1, 1)
    assert meta.dtype is torch.int32
    assert manager.create_mapping_calls == 1


def test_register_module_tensors_keeps_strict_non_empty_parameters():
    """Non-empty parameters still must come from a GMS allocation."""
    module = torch.nn.Module()
    module._parameters["weight"] = _FakeCudaTensor(numel=1, nbytes=4)

    with pytest.raises(RuntimeError, match="not found in any GMS allocation"):
        register_module_tensors(_NoMetadataManager(), module)


def test_register_module_tensors_skips_non_parameter_buffer():
    """Buffers are auxiliary runtime tensors, not shared weight mappings."""
    module = torch.nn.Module()
    module._buffers["cos_sin_cache"] = _FakeCudaTensor(numel=1, nbytes=4)
    module._non_persistent_buffers_set.add("cos_sin_cache")

    register_module_tensors(_NoMetadataManager(), module)


def test_register_module_tensors_skips_attention_scale_buffer():
    """Important buffer values are restored by the aux snapshot path."""
    module = torch.nn.Module()
    module.kv_cache_dtype = "auto"
    module.layer_name = "layers.0.self_attn"
    module._buffers["_k_scale"] = _FakeCudaTensor(numel=1, nbytes=4)

    register_module_tensors(_NoMetadataManager(), module)


def test_register_module_tensors_skips_tensor_attrs(monkeypatch):
    """Tensor attrs are auxiliary runtime tensors, not shared weights."""

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


def test_snapshot_auxiliary_tensors_records_all_non_parameters(monkeypatch):
    """Aux tensors are snapshotted structurally, without path-name filters."""

    class _FakeMapping:
        allocation_id = "source"
        aligned_size = 1 << 20

    class _Manager:
        device = 0
        mappings = {1000: _FakeMapping()}

        def __init__(self) -> None:
            self.puts: list[tuple[str, str]] = []

        def metadata_put(self, *, key, allocation_id, offset_bytes, value):
            self.puts.append((key, allocation_id))

    manager = _Manager()
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.empty(2, device="cuda" if torch.cuda.is_available() else "cpu")
    )
    module.register_buffer("scale", torch.empty(2, device=module.weight.device))
    module.cache = types.SimpleNamespace(
        banana=torch.empty(2, device=module.weight.device)
    )
    module.cached_scale = module.scale

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module._find_gms_mapping",
        lambda _manager, tensor: types.SimpleNamespace(
            mapping=types.SimpleNamespace(allocation_id=f"alloc-{id(tensor)}"),
            offset_bytes=0,
        ),
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module._move_aux_tensor_out_of_gms",
        lambda _manager, tensor, *, device_index: tensor.detach().clone(),
    )

    snapshots: list[torch.Tensor] = []

    def fake_snapshot(_manager, tensor, *, device_index):
        snapshots.append(tensor)
        return types.SimpleNamespace(
            allocation_id=f"snapshot-{len(snapshots)}",
            offset_bytes=0,
            tensor=tensor,
        )

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module._snapshot_aux_tensor_to_gms",
        fake_snapshot,
    )

    snapshotted = snapshot_auxiliary_tensors(manager, module, device_index=0)

    assert snapshotted == ["scale", "cache.banana", "cached_scale"]
    assert [key for key, _ in manager.puts] == snapshotted
    assert module.cached_scale is module.scale


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


def test_materialize_module_from_gms_uses_empty_parameter_descriptor(monkeypatch):
    """Writer-final empty params should override reader meta placeholders."""
    original_empty_strided = torch.empty_strided

    def fake_empty_strided(shape, stride, *, dtype, device):
        return original_empty_strided(shape, stride, dtype=dtype, device="cpu")

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="empty_parameter",
            shape=(4, 0),
            stride=(1, 1),
            dtype=torch.int32,
        )
        allocation_id = "descriptor"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            raise AssertionError("empty parameter descriptors have no payload")

    monkeypatch.setattr(torch, "empty_strided", fake_empty_strided)
    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"experts.w13_weight_g_idx": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    old = torch.nn.Parameter(
        torch.empty((4, 32), device="meta", dtype=torch.int32),
        requires_grad=False,
    )
    old.weight_loader = object()
    module.experts.w13_weight_g_idx = old
    module.experts.cached_g_idx = old

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    replacement = module.experts.w13_weight_g_idx
    assert not replacement.is_meta
    assert tuple(replacement.shape) == (4, 0)
    assert tuple(replacement.stride()) == (1, 1)
    assert replacement.dtype is torch.int32
    assert replacement.requires_grad is False
    assert replacement.weight_loader is old.weight_loader
    assert module.experts.cached_g_idx is replacement


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


def test_materialize_module_from_gms_registers_writer_added_parameter(monkeypatch):
    """Writer-only post-load parameters should remain registered parameters."""

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
        lambda _manager: {"experts.g1_scale_c": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.experts = torch.nn.Module()

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert "g1_scale_c" in module.experts._parameters
    assert isinstance(module.experts.g1_scale_c, torch.nn.Parameter)
    assert module.experts.g1_scale_c.requires_grad is False
    torch.testing.assert_close(module.experts.g1_scale_c, torch.arange(3.0))


def test_materialize_module_from_gms_drops_stale_packed_meta_parameter(monkeypatch):
    """Reader should not keep checkpoint-only packed params after import."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="parameter",
            shape=(2, 2),
            stride=(2, 1),
            dtype=torch.uint8,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.arange(4, dtype=torch.uint8).reshape(2, 2)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"experts.w13_weight": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    module.experts.w13_weight_packed = torch.nn.Parameter(
        torch.empty((2, 2), device="meta", dtype=torch.uint8),
        requires_grad=False,
    )
    module.experts.cached_weight = module.experts.w13_weight_packed

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert "w13_weight" in module.experts._parameters
    assert "w13_weight_packed" not in module.experts._parameters
    assert module.experts.cached_weight is module.experts.w13_weight
    torch.testing.assert_close(
        module.experts.w13_weight,
        torch.arange(4, dtype=torch.uint8).reshape(2, 2),
    )


def test_materialize_module_from_gms_drops_stale_renamed_scale_meta_parameter(
    monkeypatch,
):
    """Reader should drop scale placeholders renamed by writer post-load."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="parameter",
            shape=(),
            stride=(),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.ones((), dtype=torch.float32)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"linear.input_global_scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.linear = torch.nn.Module()
    module.linear.input_scale = torch.nn.Parameter(
        torch.empty((2,), device="meta", dtype=torch.float32),
        requires_grad=False,
    )

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert "input_global_scale" in module.linear._parameters
    assert "input_scale" not in module.linear._parameters
    torch.testing.assert_close(module.linear.input_global_scale, torch.ones(()))


def test_materialize_module_from_gms_drops_unpublished_temp_scale_parameter(
    monkeypatch,
):
    """Reader-only post-load temp params should not survive as meta tensors."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="aux_tensor",
            shape=(),
            stride=(),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.tensor(2.0, dtype=torch.float32)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"attn._q_scale": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.register_buffer(
        "_q_scale",
        torch.empty((), device="meta", dtype=torch.float32),
    )
    module.attn.q_scale = torch.nn.Parameter(
        torch.empty((), device="meta", dtype=torch.float32),
        requires_grad=False,
    )

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert "q_scale" not in module.attn._parameters
    assert not module.attn._q_scale.is_meta
    torch.testing.assert_close(module.attn._q_scale, torch.tensor(2.0))


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


def test_materialize_module_from_gms_clones_aux_tensor_metadata(monkeypatch):
    """Aux metadata is a copy source; runtime tensors get mutable copies."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="aux_tensor",
            shape=(3, 2),
            stride=(2, 1),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def __init__(self) -> None:
            self.materialize_calls = 0

        def materialize(self, _manager, _device_index):
            self.materialize_calls += 1
            return torch.arange(6, dtype=torch.float32).reshape(3, 2)

    spec = _FakeSpec()
    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {
            "rope.cos_sin_cache": spec,
            "rope.cached_alias": spec,
        },
    )

    module = torch.nn.Module()
    module.rope = torch.nn.Module()
    cache = torch.empty((3, 2), device="meta", dtype=torch.float32)
    module.rope.register_buffer("cos_sin_cache", cache, persistent=False)
    module.rope.cached_alias = cache

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert spec.materialize_calls == 1
    assert module.rope.cached_alias is module.rope.cos_sin_cache
    torch.testing.assert_close(
        module.rope.cos_sin_cache,
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )


def test_materialize_module_from_gms_installs_writer_added_aux_leaf(monkeypatch):
    """Aux attrs created by writer post-load are restored if the parent exists."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="aux_tensor",
            shape=(4,),
            stride=(1,),
            dtype=torch.float32,
        )
        allocation_id = "alloc"
        offset_bytes = 0

        def materialize(self, _manager, _device_index):
            return torch.arange(4, dtype=torch.float32)

    monkeypatch.setattr(
        "gpu_memory_service.client.torch.module.GMSTensorSpec.load_all",
        lambda _manager: {"experts.workspace": _FakeSpec()},
    )

    module = torch.nn.Module()
    module.experts = torch.nn.Module()

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert "workspace" in module.experts.__dict__
    torch.testing.assert_close(module.experts.workspace, torch.arange(4.0))


def test_materialize_module_from_gms_restores_aux_tensor_containers(monkeypatch):
    """Aux tensor paths inside helper containers are restored structurally."""

    class _FakeSpec:
        meta = types.SimpleNamespace(
            tensor_type="aux_tensor",
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
            "helper.tensor_list.0": spec,
            "helper.tensor_tuple.0": spec,
            "helper.tensor_dict.workspace": spec,
        },
    )

    module = torch.nn.Module()
    module.helper = types.SimpleNamespace(
        tensor_list=[torch.empty(3, device="meta", dtype=torch.float32)],
        tensor_tuple=(torch.empty(3, device="meta", dtype=torch.float32),),
        tensor_dict={"workspace": torch.empty(3, device="meta", dtype=torch.float32)},
    )

    materialize_module_from_gms(_NoMetadataManager(), module, device_index=0)

    assert spec.materialize_calls == 1
    materialized = module.helper.tensor_list[0]
    assert module.helper.tensor_tuple[0] is materialized
    assert module.helper.tensor_dict["workspace"] is materialized
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
