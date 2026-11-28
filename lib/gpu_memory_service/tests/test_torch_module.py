# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS torch module materialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import module as gms_module
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
)
from gpu_memory_service.client.torch.tensor import TensorMetadata

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@dataclass
class _FakeSpec:
    tensor: torch.Tensor
    meta: TensorMetadata
    allocation_id: str = "alloc-0"
    offset_bytes: int = 0

    def materialize(self, _manager: Any, _device_index: int) -> torch.Tensor:
        return self.tensor


def _spec_for(
    tensor: torch.Tensor,
    tensor_type: str = "parameter",
    *,
    allocation_id: str = "alloc-0",
    offset_bytes: int | None = None,
) -> _FakeSpec:
    return _FakeSpec(
        tensor=tensor,
        meta=TensorMetadata.from_tensor(tensor, tensor_type),
        allocation_id=allocation_id,
        offset_bytes=id(tensor) if offset_bytes is None else offset_bytes,
    )


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 4, bias=False)
        self.packed = torch.nn.Parameter(
            torch.empty(4, 8, dtype=torch.float32), requires_grad=True
        )


def test_materialize_replaces_meta_parameter_with_reshaped_and_repacked_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta placeholders predate post-load weight transforms (reshape/repack)."""
    reshaped = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    repacked = torch.arange(16, dtype=torch.int32).reshape(4, 4)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(
            lambda cls, _mgr: {
                "proj.weight": _spec_for(reshaped),
                "packed": _spec_for(repacked),
            }
        ),
    )

    with torch.device("meta"):
        model = _ToyModel()

    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert isinstance(model.proj.weight, torch.nn.Parameter)
    assert model.proj.weight.shape == (2, 2, 8)
    assert model.proj.weight.dtype == torch.float32
    assert model.proj.weight.requires_grad is True
    torch.testing.assert_close(model.proj.weight.data, reshaped, rtol=0, atol=0)

    assert isinstance(model.packed, torch.nn.Parameter)
    assert model.packed.shape == (4, 4)
    assert model.packed.dtype == torch.int32
    assert model.packed.requires_grad is False
    torch.testing.assert_close(model.packed.data, repacked, rtol=0, atol=0)


def test_materialize_rejects_shape_or_dtype_mismatch_on_concrete_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-meta parameters still fail fast on shape or dtype mismatch."""
    wrong_shape = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(lambda cls, _mgr: {"proj.weight": _spec_for(wrong_shape)}),
    )

    model = _ToyModel()
    with pytest.raises(RuntimeError, match="Shape/dtype mismatch for proj.weight"):
        materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]


class _SlottedHelper:
    __slots__ = ("bias", "__private_alias")

    def __init__(self, param: torch.nn.Parameter) -> None:
        self.bias = param
        self.__private_alias = param

    def get_private(self) -> torch.nn.Parameter:
        return self.__private_alias


class _DictHelper:
    def __init__(self, param: torch.nn.Parameter) -> None:
        self.bias = param
        self.bias_list = [param]
        self.bias_tuple = (param, "tag")
        self.bias_dict = {"primary": param}
        # Self-referential list verifies cycle protection during traversal.
        self.cyclic: list[Any] = [param]
        self.cyclic.append(self.cyclic)


class _AliasedRouterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Module()
        self.gate.e_score_correction_bias = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.bfloat16), requires_grad=False
        )
        self.dict_helper = _DictHelper(self.gate.e_score_correction_bias)
        self.slotted_helper = _SlottedHelper(self.gate.e_score_correction_bias)
        self.tied_peer = torch.nn.Module()
        self.tied_peer.bias_alias = self.gate.e_score_correction_bias
        self.optional_target = torch.nn.Module()
        self.optional_target.register_parameter("opt_bias", None)


def test_materialize_rebinds_aliased_meta_parameter_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replaced meta Parameters rebind stale aliases on helper objects and modules."""
    live_bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    opt_bias = torch.tensor([5.0, 6.0], dtype=torch.float32)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(
            lambda cls, _mgr: {
                "gate.e_score_correction_bias": _spec_for(live_bias),
                "optional_target.opt_bias": _spec_for(opt_bias),
            }
        ),
    )

    with torch.device("meta"):
        model = _AliasedRouterModel()

    stale_meta_param = model.gate.e_score_correction_bias
    assert stale_meta_param.is_meta
    assert model.dict_helper.bias is stale_meta_param

    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    materialized = model.gate.e_score_correction_bias
    assert materialized is not stale_meta_param
    assert not materialized.is_meta
    torch.testing.assert_close(materialized.data, live_bias, rtol=0, atol=0)

    assert model.dict_helper.bias is materialized
    assert model.dict_helper.bias_list[0] is materialized
    assert model.dict_helper.bias_tuple[0] is materialized
    assert model.dict_helper.bias_tuple[1] == "tag"
    assert model.dict_helper.bias_dict["primary"] is materialized
    assert model.dict_helper.cyclic[0] is materialized
    assert model.slotted_helper.bias is materialized
    assert model.slotted_helper.get_private() is materialized
    assert model.tied_peer.bias_alias is materialized

    assert isinstance(model.optional_target.opt_bias, torch.nn.Parameter)
    torch.testing.assert_close(
        model.optional_target.opt_bias.data, opt_bias, rtol=0, atol=0
    )


def test_materialize_registers_post_load_parameters_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameters/buffers created post-load register into _parameters/_buffers."""
    g1_scale_c = torch.tensor([0.25, 0.5], dtype=torch.float32)
    kv_scale = torch.tensor(0.125, dtype=torch.float32)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(
            lambda cls, _mgr: {
                "moe.g1_scale_c": _spec_for(g1_scale_c, tensor_type="parameter"),
                "attn._k_scale": _spec_for(kv_scale, tensor_type="buffer"),
            }
        ),
    )

    class _PostLoadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = torch.nn.Module()
            self.moe.g1_scale_c = None
            self.attn = torch.nn.Module()

    model = _PostLoadModel()
    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert "g1_scale_c" in model.moe._parameters
    assert "g1_scale_c" not in model.moe.__dict__
    assert isinstance(model.moe.g1_scale_c, torch.nn.Parameter)
    torch.testing.assert_close(model.moe.g1_scale_c.data, g1_scale_c, rtol=0, atol=0)
    # Re-registering via register_parameter must not raise KeyError.
    model.moe.register_parameter(
        "g1_scale_c", torch.nn.Parameter(g1_scale_c * 2, requires_grad=False)
    )

    assert "_k_scale" in model.attn._buffers
    assert "_k_scale" not in model.attn.__dict__
    torch.testing.assert_close(model.attn._k_scale, kv_scale, rtol=0, atol=0)


class _BiasPair(NamedTuple):
    bias: torch.nn.Parameter
    label: str


class _RaisingSlottedProxy:
    __slots__ = ("_bias",)

    def __init__(self, param: torch.nn.Parameter) -> None:
        self._bias = param

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"unexpected __getattr__({name!r})")


@dataclass(frozen=True, slots=True)
class _FrozenSlottedHolder:
    bias: torch.nn.Parameter


def test_materialize_handles_slotted_edge_cases_class_attrs_and_buffer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slotted proxies, frozen dataclasses, NamedTuples, class defaults, and buffer fallbacks."""
    live_bias = torch.tensor([2.0, 4.0], dtype=torch.float32)
    extra_param = torch.tensor([1.0, 3.0], dtype=torch.float32)
    class_default_param = torch.tensor([7.0], dtype=torch.float32)
    kv_scale = torch.tensor(0.25, dtype=torch.float32)

    class _ClassDefaultModule(torch.nn.Module):
        class_default = None

        def __init__(self) -> None:
            super().__init__()
            self.class_default = None
            self.meta_raw = torch.empty(2, device="meta")

    class _EdgeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Module()
            self.gate.bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.proxy = _RaisingSlottedProxy(self.gate.bias)
            self.frozen = _FrozenSlottedHolder(self.gate.bias)
            self.named_pair = _BiasPair(self.gate.bias, "ok")
            self.target = _ClassDefaultModule()
            self.kv = torch.nn.Module()
            self.kv.register_buffer("_k_scale", torch.empty((), dtype=torch.float32))

    specs_map = {
        "gate.bias": _spec_for(live_bias),
        "target.meta_raw": _spec_for(extra_param, tensor_type="parameter"),
        "target.class_default": _spec_for(class_default_param, tensor_type="parameter"),
        "kv._k_scale": _spec_for(kv_scale, tensor_type="parameter"),
    }
    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(lambda cls, _mgr: specs_map),
    )

    with torch.device("meta"):
        model = _EdgeModel()

    # Must not crash on _RaisingSlottedProxy.__getattr__, _FrozenSlottedHolder, or class_default.
    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert model.proxy._bias is model.gate.bias
    assert model.named_pair.bias is model.gate.bias
    assert model.named_pair.label == "ok"
    assert model.frozen.bias is model.gate.bias
    assert "meta_raw" in model.target._parameters
    torch.testing.assert_close(model.target.meta_raw.data, extra_param, rtol=0, atol=0)
    assert model.target.class_default.data_ptr() == class_default_param.data_ptr()
    # Parameter-over-buffer fallback clones into _buffers so GMS source stays unshared.
    assert model.kv._buffers["_k_scale"].data_ptr() != kv_scale.data_ptr()
    torch.testing.assert_close(model.kv._buffers["_k_scale"], kv_scale, rtol=0, atol=0)


def test_materialize_handles_slot_overriding_property_shallow_revisit_readonly_and_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subclass slots shadowing properties, shallow revisits, read-only props, and ties."""
    live_bias = torch.tensor([5.0, 6.0], dtype=torch.float32)
    live_ro = torch.tensor([9.0], dtype=torch.float32)
    live_tied = torch.tensor([3.0, 4.0], dtype=torch.float32)

    class _BaseWithProp:
        @property
        def bias(self) -> torch.nn.Parameter:
            raise AssertionError("base property should not be called")

    class _SlotOverrideHelper(_BaseWithProp):
        __slots__ = ("bias",)

        def __init__(self, param: torch.nn.Parameter) -> None:
            self.bias = param

    class _DeepNode:
        def __init__(self, target: Any) -> None:
            self.target = target

    class _ReadOnlyPropModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._stored = torch.empty(1, device="meta")

        @property
        def ro_attr(self) -> torch.Tensor:
            return self._stored

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Module()
            self.gate.bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            # Reach `shared_helper` at depth=6 first via `a_deep` (too deep to
            # traverse its nested `_SlotOverrideHelper`), then at depth=1 via `z_shallow`.
            leaf_helper = _SlotOverrideHelper(self.gate.bias)
            shared_helper = _DeepNode(leaf_helper)
            chain: Any = shared_helper
            self.z_shallow = _DeepNode(shared_helper)
            for _ in range(4):
                chain = _DeepNode(chain)
            self.a_deep = chain
            self.ro_mod = _ReadOnlyPropModule()
            self.left = torch.nn.Module()
            self.right = torch.nn.Module()
            tied_param = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.left.weight = tied_param
            self.right.weight = tied_param

    tied_spec = _spec_for(live_tied, allocation_id="alloc-tied", offset_bytes=128)
    specs_map = {
        "gate.bias": _spec_for(live_bias),
        "ro_mod.ro_attr": _spec_for(live_ro, tensor_type="tensor_attr"),
        "left.weight": tied_spec,
        "right.weight": tied_spec,
    }
    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(lambda cls, _mgr: specs_map),
    )

    with torch.device("meta"):
        model = _Model()

    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert model.z_shallow.target.target.bias is model.gate.bias
    assert model.ro_mod.ro_attr.is_meta
    assert model.left.weight is model.right.weight


@dataclass
class _IndexGroupHelper:
    logical_topk_indices: torch.Tensor


class _SparseAttnImplHelper:
    def __init__(self, buf: torch.Tensor) -> None:
        self.topk_indices_buffer = buf
        self.index_group = _IndexGroupHelper(logical_topk_indices=buf)


class _SharedScratchSubmodule(torch.nn.Module):
    def __init__(self, scratch: torch.Tensor, raw_map: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", scratch, persistent=False)
        self.topk_indices_buffer = scratch
        self._raw_map = raw_map
        self.impl = _SparseAttnImplHelper(scratch)

    @property
    def read_only_map(self) -> torch.Tensor:
        return self._raw_map


class _SharedScratchModel(torch.nn.Module):
    def __init__(self, scratch: torch.Tensor, raw_map: torch.Tensor) -> None:
        super().__init__()
        self.shared_attr = scratch
        self.layer0 = _SharedScratchSubmodule(scratch, raw_map)
        self.layer1 = _SharedScratchSubmodule(scratch, raw_map)


def test_materialize_deduplicates_shared_nonparam_clones_and_rebinds_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RO materialization clones shared GMS non-parameter regions once and rebinds helpers."""
    materialize_calls = {"scratch": 0, "map": 0}
    live_scratch = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    live_map = torch.tensor([0, 1], dtype=torch.int32)

    class _TrackingSpec:
        def __init__(
            self,
            tensor: torch.Tensor,
            tensor_type: str,
            alloc_id: str,
            offset: int,
            counter_key: str,
        ) -> None:
            self.tensor = tensor
            self.allocation_id = alloc_id
            self.offset_bytes = offset
            self.counter_key = counter_key
            self.meta = TensorMetadata.from_tensor(tensor, tensor_type)

        def materialize(self, _mgr: Any, _device_index: int) -> torch.Tensor:
            materialize_calls[self.counter_key] += 1
            return self.tensor

    specs = {
        "shared_attr": _TrackingSpec(
            live_scratch, "tensor_attr", "alloc-0", 0, "scratch"
        ),
        "layer0.cos_sin_cache": _TrackingSpec(
            live_scratch, "buffer", "alloc-0", 0, "scratch"
        ),
        "layer0.topk_indices_buffer": _TrackingSpec(
            live_scratch, "tensor_attr", "alloc-0", 0, "scratch"
        ),
        "layer1.cos_sin_cache": _TrackingSpec(
            live_scratch, "buffer", "alloc-0", 0, "scratch"
        ),
        "layer1.topk_indices_buffer": _TrackingSpec(
            live_scratch, "tensor_attr", "alloc-0", 0, "scratch"
        ),
        "layer0._raw_map": _TrackingSpec(live_map, "tensor_attr", "alloc-1", 64, "map"),
        "layer0.read_only_map": _TrackingSpec(
            live_map, "tensor_attr", "alloc-1", 64, "map"
        ),
        "layer1._raw_map": _TrackingSpec(live_map, "parameter", "alloc-1", 64, "map"),
    }
    monkeypatch.setattr(
        gms_module.GMSTensorSpec, "load_all", classmethod(lambda cls, _mgr: specs)
    )

    class _ParamOverBufModel(_SharedScratchModel):
        def __init__(self, scratch: torch.Tensor, raw_map: torch.Tensor) -> None:
            super().__init__(scratch, raw_map)
            del self.layer1._raw_map
            self.layer1.register_buffer("_raw_map", raw_map, persistent=False)

    with torch.device("meta"):
        model = _ParamOverBufModel(
            torch.empty((4, 4), dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        )

    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert materialize_calls["scratch"] == 1
    assert materialize_calls["map"] == 2
    mat = model.shared_attr
    assert mat.data_ptr() != live_scratch.data_ptr()
    assert model.layer0.cos_sin_cache is mat
    assert model.layer0.topk_indices_buffer is mat
    assert model.layer1.cos_sin_cache is mat
    assert model.layer1.topk_indices_buffer is mat
    assert model.layer0.impl.topk_indices_buffer is mat
    assert model.layer0.impl.index_group.logical_topk_indices is mat
    assert model.layer1.impl.index_group.logical_topk_indices is mat
    assert model.layer0.read_only_map is model.layer0._raw_map
    assert model.layer1._buffers["_raw_map"] is model.layer0._raw_map


def test_rebind_nonparameter_tensors_deduplicates_clones_and_rebinds_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RW rebinding clones shared pool allocations once, counts bytes once, and rebinds helpers."""
    scratch = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    raw_map = torch.tensor([2, 3], dtype=torch.int32)
    complex_base = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
    scratch_bytes = scratch.numel() * scratch.element_size()
    map_bytes = raw_map.numel() * raw_map.element_size()
    complex_bytes = complex_base.numel() * complex_base.element_size()

    model = _SharedScratchModel(scratch, raw_map)
    model.register_buffer("c_plain", complex_base, persistent=False)
    model.register_buffer("c_conj", complex_base.conj(), persistent=False)

    monkeypatch.setattr(gms_module, "_is_accelerator_tensor", lambda _t: True)
    fake_manager = SimpleNamespace(
        mappings={
            scratch.data_ptr(): SimpleNamespace(
                allocation_id="alloc-0",
                aligned_size=max(scratch_bytes, 64),
            ),
            raw_map.data_ptr(): SimpleNamespace(
                allocation_id="alloc-1",
                aligned_size=max(map_bytes, 64),
            ),
            complex_base.data_ptr(): SimpleNamespace(
                allocation_id="alloc-2",
                aligned_size=max(complex_bytes, 64),
            ),
        }
    )

    retained: list[torch.Tensor] = []
    rebound_bytes = rebind_nonparameter_tensors(
        fake_manager,
        model,
        retain_gms_tensors=retained,  # type: ignore[arg-type]
    )

    assert rebound_bytes == scratch_bytes + map_bytes + 2 * complex_bytes
    assert len(retained) == 4
    assert any(t is scratch for t in retained)

    clone = model.shared_attr
    assert clone is not scratch
    assert clone.data_ptr() != scratch.data_ptr()
    assert model.layer0.cos_sin_cache is clone
    assert model.layer0.topk_indices_buffer is clone
    assert model.layer1.cos_sin_cache is clone
    assert model.layer1.topk_indices_buffer is clone
    assert model.layer0.impl.topk_indices_buffer is clone
    assert model.layer0.impl.index_group.logical_topk_indices is clone
    assert model.layer1.impl.index_group.logical_topk_indices is clone
    assert model.layer0.read_only_map is model.layer0._raw_map
    assert model.layer0._raw_map is not raw_map
    torch.testing.assert_close(model._buffers["c_plain"], complex_base)
    torch.testing.assert_close(model._buffers["c_conj"], complex_base.conj())
    assert model._buffers["c_conj"] is not model._buffers["c_plain"]
