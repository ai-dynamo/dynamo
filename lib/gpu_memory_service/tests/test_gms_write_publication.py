# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for GMS storage-manifest publication and materialization."""

from __future__ import annotations

import sys
import weakref
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

import msgspec
import torch
from gpu_memory_service.client.torch import module as torch_module
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import STORAGE_MANIFEST_PREFIX, Slot
from gpu_memory_service.integrations.common import utils as common_utils
from gpu_memory_service.integrations.trtllm import model_loader as trtllm_model_loader
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _MetadataStore:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, int, bytes]] = {}


class _Manager:
    def __init__(
        self,
        store: _MetadataStore,
        allocations: dict[str, tuple[int, int]],
        *,
        mapped: bool,
    ) -> None:
        self.device = 0
        self.store = store
        self.allocations = allocations
        self.mappings = {}
        self.mapping_calls: list[str] = []
        self.freed: list[int] = []
        if mapped:
            for allocation_id, (va, nbytes) in allocations.items():
                self.mappings[va] = SimpleNamespace(
                    allocation_id=allocation_id,
                    aligned_size=nbytes,
                )

    def metadata_put(self, *, key, allocation_id, offset_bytes, value):
        self.store.entries[key] = (allocation_id, offset_bytes, value)
        return True

    def metadata_list(self, prefix=""):
        return sorted(key for key in self.store.entries if key.startswith(prefix))

    def metadata_get(self, key):
        return self.store.entries.get(key)

    def create_mapping(self, *, allocation_id):
        self.mapping_calls.append(allocation_id)
        for va, mapping in self.mappings.items():
            if mapping.allocation_id == allocation_id:
                return va
        va, nbytes = self.allocations[allocation_id]
        self.mappings[va] = SimpleNamespace(
            allocation_id=allocation_id,
            aligned_size=nbytes,
        )
        return va

    def free_va(self, va):
        self.freed.append(va)
        del self.mappings[va]


class _FakeAllocator:
    def __init__(self, events: list[str], *, fail_commit: bool = False):
        self.events = events
        self.fail_commit = fail_commit
        self.fail_close = False
        self.total_bytes = 100
        self.mappings = {
            1: SimpleNamespace(allocation_id="weight", aligned_size=60),
            2: SimpleNamespace(allocation_id="scratch", aligned_size=40),
        }

    def commit(self):
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def connect(self, lock_type):
        self.events.append("connect")

    def remap_all_vas(self):
        self.events.append("remap")

    def close(self, *, best_effort=False):
        self.events.append("close_best_effort" if best_effort else "close")
        if self.fail_close:
            raise RuntimeError("close failed")


@pytest.fixture
def cpu_storage_pointer(monkeypatch):
    def storage_from_pointer(data_ptr, size_bytes, _device_index):
        return torch._C._construct_storage_from_data_pointer(
            data_ptr,
            torch.device("cpu"),
            size_bytes,
        )

    monkeypatch.setattr(torch_module, "_storage_from_pointer", storage_from_pointer)


@pytest.fixture
def gms_write(monkeypatch):
    events = []
    allocator = _FakeAllocator(events)

    def register(_allocator, _model):
        events.append("register")
        return {"weight"}

    def prune(_allocator, *, referenced_allocation_ids):
        assert referenced_allocation_ids == {"weight"}
        events.append("prune")
        _allocator.mappings.pop(2)
        _allocator.total_bytes = 60

    monkeypatch.setattr(common_utils, "register_module_tensors", register)
    monkeypatch.setattr(common_utils, "prune_allocations", prune)
    monkeypatch.setattr(
        common_utils,
        "rebind_nonparameter_tensors",
        lambda _allocator, _model: events.append("rebind") or 12,
    )

    stats = common_utils.prepare_gms_write(allocator, object())
    return allocator, stats, events


@pytest.fixture(autouse=True)
def clear_pending_write(monkeypatch):
    monkeypatch.setattr(model_loader, "_pending_gms_client", None)
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)


def _writer_for_tensor(tensor: torch.Tensor):
    store = _MetadataStore()
    storage = tensor.untyped_storage()
    allocations = {"allocation": (storage.data_ptr(), storage.nbytes())}
    return store, _Manager(store, allocations, mapped=True)


def test_manifest_groups_exact_alias_slots_and_shared_storage_objects():
    backing = torch.arange(8, dtype=torch.float32)
    first = backing[1:7:2]
    second = torch.empty(0).set_(
        backing.untyped_storage(),
        2,
        (3,),
        (1,),
    )
    model = torch.nn.Module()
    model.first = first
    model.first_alias = first
    model.second = second
    store, writer = _writer_for_tensor(backing)

    assert register_module_tensors(writer, model) == {"allocation"}
    assert list(store.entries) == [f"{STORAGE_MANIFEST_PREFIX}0"]

    allocation_id, storage_base_offset, payload = next(iter(store.entries.values()))
    manifest = msgspec.msgpack.decode(payload)
    assert allocation_id == "allocation"
    assert storage_base_offset == 0
    assert set(manifest) == {"nbytes", "objects"}
    assert manifest["nbytes"] == backing.untyped_storage().nbytes()
    assert len(manifest["objects"]) == 2
    assert manifest["objects"][0] == {
        "dtype": "float32",
        "shape": [3],
        "stride": [2],
        "storage_offset_bytes": 4,
        "requires_grad": False,
        "slots": [
            {"path": "first", "kind": "attribute"},
            {"path": "first_alias", "kind": "attribute"},
        ],
    }
    assert manifest["objects"][1]["storage_offset_bytes"] == 8
    assert manifest["objects"][1]["slots"] == [{"path": "second", "kind": "attribute"}]
    assert "allocation_id" not in payload.decode("latin1")


def test_reader_replaces_split_and_shared_placeholder_topology(cpu_storage_pointer):
    backing = torch.arange(8, dtype=torch.float32)
    exact = backing[::2]
    distinct = torch.empty(0).set_(
        backing.untyped_storage(),
        1,
        (4,),
        (2,),
    )
    writer_model = torch.nn.Module()
    writer_model.exact_a = exact
    writer_model.exact_b = exact
    writer_model.distinct_a = backing
    writer_model.distinct_b = distinct
    store, writer = _writer_for_tensor(backing)
    register_module_tensors(writer, writer_model)

    shared_placeholder = torch.zeros(1)
    reader_model = torch.nn.Module()
    reader_model.exact_a = torch.zeros(2)
    reader_model.exact_b = torch.ones(3, dtype=torch.float64)
    reader_model.distinct_a = shared_placeholder
    reader_model.distinct_b = shared_placeholder
    names = ("exact_a", "exact_b", "distinct_a", "distinct_b")
    old_placeholders = tuple(reader_model.__dict__[name] for name in names)
    reader = _Manager(store, writer.allocations, mapped=False)

    materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader.mapping_calls == ["allocation"]
    assert reader_model.exact_a is reader_model.exact_b
    assert reader_model.distinct_a is not reader_model.distinct_b
    assert all(
        reader_model.__dict__[name] is not old
        for name, old in zip(names, old_placeholders, strict=True)
    )
    storage_ids = {
        reader_model.__dict__[name].untyped_storage()._cdata for name in names
    }
    assert len(storage_ids) == 1
    assert backing.untyped_storage()._cdata not in storage_ids
    torch.testing.assert_close(reader_model.exact_a, backing[::2])
    torch.testing.assert_close(reader_model.distinct_b, backing[1::2])
    reader_model.exact_a[1] = 99
    assert reader_model.distinct_a[2].item() == 99


def test_mixed_dtype_byte_offsets_and_one_mapping_per_allocation(
    cpu_storage_pointer,
):
    allocation = torch.tensor(
        [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
        dtype=torch.uint8,
    )
    first_storage = torch._C._construct_storage_from_data_pointer(
        allocation.data_ptr() + 4,
        torch.device("cpu"),
        8,
    )
    second_storage = torch._C._construct_storage_from_data_pointer(
        allocation.data_ptr() + 12,
        torch.device("cpu"),
        4,
    )
    octets = torch.empty(0, dtype=torch.uint8).set_(first_storage, 0, (8,), (1,))
    words = torch.empty(0, dtype=torch.int32).set_(first_storage, 0, (2,), (1,))
    tail = torch.empty(0, dtype=torch.int32).set_(second_storage, 0, (1,), (1,))
    writer_model = torch.nn.Module()
    writer_model.octets = octets
    writer_model.words = words
    writer_model.tail = tail
    store = _MetadataStore()
    allocations = {
        "allocation": (allocation.data_ptr(), allocation.untyped_storage().nbytes())
    }
    writer = _Manager(store, allocations, mapped=True)

    assert register_module_tensors(writer, writer_model) == {"allocation"}
    assert list(store.entries) == [
        f"{STORAGE_MANIFEST_PREFIX}0",
        f"{STORAGE_MANIFEST_PREFIX}1",
    ]
    assert [entry[1] for entry in store.entries.values()] == [4, 12]

    reader_model = torch.nn.Module()
    reader_model.octets = torch.empty(0)
    reader_model.words = torch.empty(0)
    reader_model.tail = torch.empty(0)
    reader = _Manager(store, allocations, mapped=True)
    materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader.mapping_calls == ["allocation"]
    assert reader_model.octets.dtype == torch.uint8
    assert reader_model.words.dtype == torch.int32
    assert (
        reader_model.octets.untyped_storage()._cdata
        == reader_model.words.untyped_storage()._cdata
        != reader_model.tail.untyped_storage()._cdata
    )
    assert reader_model.octets.untyped_storage()._cdata != first_storage._cdata
    assert reader_model.tail.untyped_storage()._cdata != second_storage._cdata
    torch.testing.assert_close(
        reader_model.words,
        torch.tensor([1, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        reader_model.tail,
        torch.tensor([3], dtype=torch.int32),
    )
    reader_model.words[0] = 257
    torch.testing.assert_close(
        reader_model.octets,
        torch.tensor([1, 1, 0, 0, 2, 0, 0, 0], dtype=torch.uint8),
    )


def test_nonpersistent_buffer_direct_list_and_tuple_bind_same_source_object(
    cpu_storage_pointer,
):
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.register_buffer("runtime", tensor, persistent=False)
    writer_model.direct = tensor
    writer_model.values = [tensor]
    writer_model.fixed = (tensor,)
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)

    reader_model = torch.nn.Module()
    reader_model.register_buffer("runtime", torch.zeros(1))
    reader_model.direct = torch.zeros(2)
    reader_model.values = [torch.zeros(3)]
    old_tuple = (torch.zeros(5),)
    reader_model.fixed = old_tuple
    reader = _Manager(store, writer.allocations, mapped=False)
    materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader_model.runtime is reader_model.direct
    assert reader_model.runtime is reader_model.values[0]
    assert reader_model.runtime is reader_model.fixed[0]
    assert reader_model.fixed is not old_tuple
    assert "runtime" in reader_model._non_persistent_buffers_set
    assert "runtime" not in reader_model.state_dict()


class _LoaderParameter(torch.nn.Parameter):
    __slots__ = ("slot_state",)

    def __new__(
        cls,
        data,
        *,
        weight_loader,
        tp_rank,
        requires_grad,
    ):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(
        self,
        data,
        *,
        weight_loader,
        tp_rank,
        requires_grad,
    ):
        self.weight_loader = weight_loader
        self.tp_rank = tp_rank
        self.slot_state = {"loader_state": True}


def test_parameter_subclass_state_comes_from_canonical_reader_template(
    cpu_storage_pointer,
):
    source = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    writer_model = torch.nn.Module()
    writer_model.register_parameter("left", source)
    writer_model.register_parameter("right", source)
    store, writer = _writer_for_tensor(source)
    register_module_tensors(writer, writer_model)

    def weight_loader(tensor):
        return tensor

    slot_state = {"loader_state": True}

    def template():
        parameter = _LoaderParameter(
            torch.empty(2, device="meta"),
            weight_loader=weight_loader,
            tp_rank=3,
            requires_grad=True,
        )
        parameter.slot_state = slot_state
        return parameter

    reader_model = torch.nn.Module()
    left = template()
    right = torch.nn.Parameter(torch.empty(2, device="meta"), requires_grad=False)
    right.uncomparable_state = torch.ones(2)
    reader_model.register_parameter("left", left)
    reader_model.register_parameter("right", right)
    reader = _Manager(store, writer.allocations, mapped=False)
    materialize_module_from_gms(reader, reader_model, device_index=0)

    parameter = reader_model.left
    assert parameter is reader_model.right
    assert parameter is not left and parameter is not right
    assert type(parameter) is _LoaderParameter
    assert parameter.weight_loader is weight_loader
    assert parameter.tp_rank == 3
    assert parameter.slot_state is slot_state
    assert parameter.requires_grad
    torch.testing.assert_close(parameter, source)


def test_reader_creates_writer_final_slots_absent_from_initial_model(
    cpu_storage_pointer,
):
    source = torch.nn.Parameter(
        torch.arange(4, dtype=torch.float32),
        requires_grad=False,
    )
    runtime = torch.arange(3, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.register_parameter("weight_scale", source)
    writer_model.register_buffer("runtime_scale", runtime, persistent=False)
    writer_model.final_scale = runtime
    store = _MetadataStore()
    allocations = {
        "parameter": (
            source.untyped_storage().data_ptr(),
            source.untyped_storage().nbytes(),
        ),
        "runtime": (
            runtime.untyped_storage().data_ptr(),
            runtime.untyped_storage().nbytes(),
        ),
    }
    writer = _Manager(store, allocations, mapped=True)
    register_module_tensors(writer, writer_model)

    reader_model = torch.nn.Module()
    reader_model.register_parameter("unrelated", torch.nn.Parameter(torch.empty(1)))
    reader = _Manager(store, writer.allocations, mapped=False)
    materialize_module_from_gms(reader, reader_model, device_index=0)

    assert type(reader_model.weight_scale) is torch.nn.Parameter
    assert not reader_model.weight_scale.requires_grad
    assert reader_model.runtime_scale is reader_model.final_scale
    assert "runtime_scale" in reader_model._non_persistent_buffers_set
    torch.testing.assert_close(reader_model.weight_scale, source)
    torch.testing.assert_close(reader_model.runtime_scale, runtime)


def test_reader_rejects_distinct_sources_for_tied_submodule_destination():
    first = torch.arange(2, dtype=torch.float32)
    second = torch.arange(2, dtype=torch.float32) + 10
    writer_model = torch.nn.Module()
    writer_model.a = torch.nn.Module()
    writer_model.b = torch.nn.Module()
    writer_model.a.value = first
    writer_model.b.value = second
    store = _MetadataStore()
    allocations = {
        "first": (
            first.untyped_storage().data_ptr(),
            first.untyped_storage().nbytes(),
        ),
        "second": (
            second.untyped_storage().data_ptr(),
            second.untyped_storage().nbytes(),
        ),
    }
    writer = _Manager(store, allocations, mapped=True)
    register_module_tensors(writer, writer_model)

    shared = torch.nn.Module()
    shared.value = torch.zeros(1)
    reader_model = torch.nn.Module()
    reader_model.a = shared
    reader_model.b = shared
    reader = _Manager(store, allocations, mapped=False)

    with pytest.raises(RuntimeError, match="same destination"):
        materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader.mapping_calls == []
    torch.testing.assert_close(shared.value, torch.zeros(1))


def test_reader_deduplicates_same_source_for_tied_submodule_destination(
    cpu_storage_pointer,
):
    source = torch.arange(2, dtype=torch.float32)
    writer_shared = torch.nn.Module()
    writer_shared.value = source
    writer_model = torch.nn.Module()
    writer_model.a = writer_shared
    writer_model.b = writer_shared
    store, writer = _writer_for_tensor(source)
    register_module_tensors(writer, writer_model)

    reader_shared = torch.nn.Module()
    reader_shared.value = torch.zeros(1)
    reader_model = torch.nn.Module()
    reader_model.a = reader_shared
    reader_model.b = reader_shared
    reader = _Manager(store, writer.allocations, mapped=False)

    materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader_model.a.value is reader_model.b.value
    torch.testing.assert_close(reader_model.a.value, source)


def test_writer_rejects_out_of_bounds_and_overlapping_storages():
    allocation = torch.arange(16, dtype=torch.uint8)
    first_storage = torch._C._construct_storage_from_data_pointer(
        allocation.data_ptr(),
        torch.device("cpu"),
        12,
    )
    second_storage = torch._C._construct_storage_from_data_pointer(
        allocation.data_ptr() + 4,
        torch.device("cpu"),
        12,
    )
    first = torch.empty(0, dtype=torch.uint8).set_(first_storage, 0, (12,), (1,))
    second = torch.empty(0, dtype=torch.uint8).set_(second_storage, 0, (12,), (1,))
    model = torch.nn.Module()
    model.first = first
    model.second = second
    store = _MetadataStore()
    manager = _Manager(
        store,
        {"allocation": (allocation.data_ptr(), allocation.untyped_storage().nbytes())},
        mapped=True,
    )

    with pytest.raises(RuntimeError, match="byte ranges overlap"):
        register_module_tensors(manager, model)
    assert store.entries == {}

    manager.mappings[allocation.data_ptr()].aligned_size = 8
    model.second = None
    with pytest.raises(RuntimeError, match="exceeds GMS allocation"):
        register_module_tensors(manager, model)
    assert store.entries == {}


def test_reader_bounds_failure_releases_only_new_mapping(
    cpu_storage_pointer,
    monkeypatch,
):
    existing = torch.arange(4, dtype=torch.float32)
    imported = torch.arange(4, dtype=torch.float32) + 10
    writer_model = torch.nn.Module()
    writer_model.existing = existing
    writer_model.imported = imported
    store = _MetadataStore()
    allocations = {
        "existing": (
            existing.untyped_storage().data_ptr(),
            existing.untyped_storage().nbytes(),
        ),
        "imported": (
            imported.untyped_storage().data_ptr(),
            imported.untyped_storage().nbytes(),
        ),
    }
    writer = _Manager(store, allocations, mapped=True)
    register_module_tensors(writer, writer_model)
    bad_key = next(
        key
        for key, (allocation_id, _, _) in store.entries.items()
        if allocation_id == "imported"
    )
    allocation_id, _, payload = store.entries[bad_key]
    store.entries[bad_key] = (
        allocation_id,
        imported.untyped_storage().nbytes(),
        payload,
    )
    reader_model = torch.nn.Module()
    reader_model.existing = torch.zeros(1)
    reader_model.imported = torch.zeros(1)
    reader = _Manager(store, allocations, mapped=False)
    existing_va, existing_nbytes = allocations["existing"]
    existing_mapping = SimpleNamespace(
        allocation_id="existing",
        aligned_size=existing_nbytes,
    )
    reader.mappings[existing_va] = existing_mapping
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    with pytest.raises(RuntimeError, match="exceeds allocation"):
        materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader.freed == [imported.untyped_storage().data_ptr()]
    assert reader.mappings == {existing_va: existing_mapping}


def test_reader_validates_storage_envelope_alignment_before_mapping():
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.runtime = tensor
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)
    key = f"{STORAGE_MANIFEST_PREFIX}0"
    allocation_id, _, payload = store.entries[key]
    store.entries[key] = (allocation_id, 2, payload)
    reader_model = torch.nn.Module()
    reader_model.runtime = torch.zeros(1)
    reader = _Manager(store, writer.allocations, mapped=False)

    with pytest.raises(RuntimeError, match="Unaligned storage envelope"):
        materialize_module_from_gms(reader, reader_model, device_index=0)

    assert reader.mapping_calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update({"unknown": True}),
        lambda manifest: manifest["objects"][0]["slots"][0].update({"kind": "invalid"}),
        lambda manifest: manifest["objects"][0]["slots"].append(
            manifest["objects"][0]["slots"][0].copy()
        ),
    ],
    ids=("unknown-field", "invalid-literal", "duplicate-slot"),
)
def test_reader_rejects_compact_malformed_manifest_cases(mutate):
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.runtime = tensor
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)
    key = f"{STORAGE_MANIFEST_PREFIX}0"
    allocation_id, offset, payload = store.entries[key]
    manifest = msgspec.msgpack.decode(payload)
    mutate(manifest)
    store.entries[key] = (
        allocation_id,
        offset,
        msgspec.msgpack.encode(manifest),
    )
    reader = _Manager(store, writer.allocations, mapped=False)

    with pytest.raises(RuntimeError, match="GMS storage manifest"):
        materialize_module_from_gms(reader, torch.nn.Module(), device_index=0)

    assert reader.mapping_calls == []


def test_reader_rejects_malformed_msgpack():
    store = _MetadataStore()
    store.entries[f"{STORAGE_MANIFEST_PREFIX}0"] = (
        "allocation",
        0,
        b"\xc1",
    )
    reader = _Manager(store, {}, mapped=False)

    with pytest.raises(RuntimeError, match="Invalid GMS storage manifest"):
        materialize_module_from_gms(reader, torch.nn.Module(), device_index=0)

    assert reader.mapping_calls == []


def test_reader_rejects_noncontiguous_ordinals_and_overlapping_manifests():
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.runtime = tensor
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)
    key = f"{STORAGE_MANIFEST_PREFIX}0"
    entry = store.entries.pop(key)
    store.entries[f"{STORAGE_MANIFEST_PREFIX}1"] = entry
    reader = _Manager(store, writer.allocations, mapped=False)

    with pytest.raises(RuntimeError, match="ordinals must be contiguous"):
        materialize_module_from_gms(reader, torch.nn.Module(), device_index=0)
    assert reader.mapping_calls == []

    store.entries[key] = store.entries.pop(f"{STORAGE_MANIFEST_PREFIX}1")
    allocation_id, _, payload = store.entries[key]
    second = msgspec.msgpack.decode(payload)
    second["objects"][0]["slots"][0]["path"] = "other"
    store.entries[f"{STORAGE_MANIFEST_PREFIX}1"] = (
        allocation_id,
        4,
        msgspec.msgpack.encode(second),
    )

    with pytest.raises(RuntimeError, match="manifests overlap"):
        materialize_module_from_gms(reader, torch.nn.Module(), device_index=0)
    assert reader.mapping_calls == []


def test_reader_keeps_mapping_when_clone_cleanup_cannot_synchronize(
    cpu_storage_pointer,
    monkeypatch,
):
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.runtime = tensor
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)
    reader_model = torch.nn.Module()
    reader_model.runtime = torch.zeros(1)
    reader = _Manager(store, writer.allocations, mapped=False)
    events: list[str] = []
    real_free = reader.free_va

    def fail_tensor(*_args, **_kwargs):
        raise RuntimeError("materialization failed")

    def fail_sync(_device):
        events.append("synchronize")
        raise RuntimeError("synchronize failed")

    def free(va):
        events.append("free")
        real_free(va)

    monkeypatch.setattr(torch_module, "_tensor_from_storage", fail_tensor)
    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    reader.free_va = free

    with pytest.raises(RuntimeError, match="synchronize failed"):
        materialize_module_from_gms(reader, reader_model, device_index=0)

    assert events == ["synchronize"]
    assert reader.freed == []
    assert list(reader.mappings) == [tensor.untyped_storage().data_ptr()]


def test_reader_ignores_metadata_outside_storage_prefix(
    cpu_storage_pointer,
):
    tensor = torch.arange(4, dtype=torch.float32)
    writer_model = torch.nn.Module()
    writer_model.runtime = tensor
    store, writer = _writer_for_tensor(tensor)
    register_module_tensors(writer, writer_model)
    store.entries["runtime"] = ("missing", 0, b"unrelated")
    reader_model = torch.nn.Module()
    reader_model.runtime = torch.zeros(1)
    reader = _Manager(store, writer.allocations, mapped=False)

    materialize_module_from_gms(reader, reader_model, device_index=0)

    torch.testing.assert_close(reader_model.runtime, tensor)


def test_reader_rejects_metadata_without_storage_manifests():
    store = _MetadataStore()
    store.entries["runtime"] = ("allocation", 0, b"unsupported")
    manager = _Manager(store, {}, mapped=False)

    with pytest.raises(RuntimeError, match="No GMS module storage manifests"):
        materialize_module_from_gms(manager, torch.nn.Module(), device_index=0)


def test_discovery_does_not_invoke_properties_or_walk_helper_graphs():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.property_reads = 0
            self.direct = torch.arange(4)
            self.helper = SimpleNamespace(hidden=torch.arange(4))

        @property
        def derived(self):
            self.property_reads += 1
            return self.direct

    model = Model()
    store, writer = _writer_for_tensor(model.direct)
    register_module_tensors(writer, model)
    manifest = msgspec.msgpack.decode(next(iter(store.entries.values()))[2])

    assert model.property_reads == 0
    assert manifest["objects"][0]["slots"] == [{"path": "direct", "kind": "attribute"}]


def test_unregistered_parameter_and_zero_byte_storage_are_rejected():
    parameter = torch.nn.Parameter(torch.ones(1))
    model = torch.nn.Module()
    model.__dict__["hidden"] = parameter
    _, writer = _writer_for_tensor(parameter)
    with pytest.raises(RuntimeError, match="Unregistered Parameter"):
        register_module_tensors(writer, model)

    empty = torch.empty(0)
    model = torch.nn.Module()
    model.empty = empty
    _, writer = _writer_for_tensor(empty)
    with pytest.raises(RuntimeError, match="zero-byte"):
        register_module_tensors(writer, model)


def test_publisher_rebind_clones_storage_once_and_retains_one_source_owner():
    backing = torch.arange(8, dtype=torch.float32)
    view = backing[1::2]
    assert view._base is backing
    model = torch.nn.Module()
    model.backing = backing
    model.backing_alias = backing
    model.view = view
    _, manager = _writer_for_tensor(backing)
    old_storage = backing.untyped_storage()._cdata
    rebound = rebind_nonparameter_tensors(manager, model)

    assert rebound == backing.untyped_storage().nbytes()
    assert model.backing is backing is model.backing_alias
    assert model.view is view
    assert model.backing.untyped_storage()._cdata != old_storage
    assert model.backing.untyped_storage()._cdata == model.view.untyped_storage()._cdata
    owners = model.__dict__["_gms_rebound_tensor_owners"].tensors
    assert len(owners) == 1
    assert owners[0].untyped_storage()._cdata == old_storage
    model.view.add_(10)
    torch.testing.assert_close(model.backing[1::2], model.view)


def test_swap_accepts_exclusively_owned_replacements_for_parameter_and_view():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    torch.autograd.graph.get_gradient_edge(parameter)
    with torch.no_grad():
        view = parameter[1::2]
    objects = [
        torch_module._DiscoveredObject(parameter, [Slot("weight", "parameter")]),
        torch_module._DiscoveredObject(view, [Slot("view", "attribute")]),
    ]
    target_storage = parameter.detach().clone().untyped_storage()
    replacements = {}
    for tensor_object in objects:
        tensor = tensor_object.tensor
        replacement = torch_module._tensor_from_storage(
            target_storage,
            list(tensor.shape),
            list(tensor.stride()),
            tensor.dtype,
            int(tensor.storage_offset()),
        )
        if isinstance(tensor, torch.nn.Parameter):
            replacement = torch_module._make_parameter(
                tensor,
                replacement,
                path="weight",
                requires_grad=True,
            )
        replacements[id(tensor)] = replacement
        del replacement

    torch_module._swap_discovered_objects(objects, replacements)

    assert replacements == {}
    assert parameter.untyped_storage()._cdata == view.untyped_storage()._cdata
    torch.testing.assert_close(parameter, torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(view, torch.tensor([1, 3, 5, 7], dtype=torch.float32))


def test_swap_preflight_failure_does_not_mutate_earlier_objects():
    first = torch.arange(4, dtype=torch.float32)
    second = torch.arange(4, dtype=torch.float32) + 10
    objects = [
        torch_module._DiscoveredObject(first, [Slot("first", "attribute")]),
        torch_module._DiscoveredObject(second, [Slot("second", "attribute")]),
    ]
    replacements = {
        id(first): torch.full_like(first, 20),
        id(second): torch.full_like(second, 30),
    }
    first_storage = first.untyped_storage()._cdata
    second_storage = second.untyped_storage()._cdata
    second_ref = weakref.ref(second)

    with pytest.raises(RuntimeError, match="second.*weakref"):
        torch_module._swap_discovered_objects(objects, replacements)

    assert second_ref() is second
    assert first.untyped_storage()._cdata == first_storage
    assert second.untyped_storage()._cdata == second_storage
    torch.testing.assert_close(first, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(second, torch.arange(4, dtype=torch.float32) + 10)


def test_rebound_tensor_owner_rejects_existing_none():
    tensor = torch.arange(4, dtype=torch.float32)
    model = torch.nn.Module()
    model.runtime = tensor
    model.__dict__["_gms_rebound_tensor_owners"] = None
    _, manager = _writer_for_tensor(tensor)

    with pytest.raises(RuntimeError, match="Reserved GMS attribute"):
        rebind_nonparameter_tensors(manager, model)


def test_prepare_and_publish_accounting(gms_write):
    allocator, stats, events = gms_write
    assert events == ["register", "prune"]
    assert stats.committed_bytes == 60
    assert stats.pruned_bytes == 40
    assert stats.pruned_count == 1

    model_loader._store_pending_gms_write(allocator, stats, 12)
    assert model_loader.get_imported_weights_bytes() == 60
    assert model_loader.get_model_memory_usage_offset_bytes() == 52
    assert model_loader.publish_pending_gms_write()
    assert events == ["register", "prune", "commit", "connect", "remap"]


def test_eager_finalize_rebinds_before_publication(gms_write):
    allocator, _, events = gms_write
    events.clear()
    allocator.total_bytes = 100
    allocator.mappings[2] = SimpleNamespace(
        allocation_id="scratch",
        aligned_size=40,
    )

    stats = common_utils.finalize_gms_write(allocator, object())

    assert stats.committed_bytes == 60
    assert events == [
        "register",
        "prune",
        "rebind",
        "commit",
        "connect",
        "remap",
    ]


def test_eager_finalize_releases_after_partial_rebind_without_publishing(
    gms_write, monkeypatch
):
    allocator, _, events = gms_write
    events.clear()
    allocator.fail_close = True
    allocator.total_bytes = 100
    allocator.mappings[2] = SimpleNamespace(
        allocation_id="scratch",
        aligned_size=40,
    )

    model = SimpleNamespace(partially_rebound=False)

    def fail_rebind(_allocator, rebound_model):
        events.append("rebind")
        rebound_model.partially_rebound = True
        raise RuntimeError("rebind failed")

    monkeypatch.setattr(common_utils, "rebind_nonparameter_tensors", fail_rebind)

    with pytest.raises(RuntimeError, match="rebind failed"):
        common_utils.finalize_gms_write(allocator, model)

    assert model.partially_rebound
    assert events == ["register", "prune", "rebind", "close_best_effort"]


def test_eager_finalize_commit_failure_releases_and_preserves_error(gms_write):
    allocator, _, events = gms_write
    events.clear()
    allocator.fail_commit = True
    allocator.fail_close = True
    allocator.total_bytes = 100
    allocator.mappings[2] = SimpleNamespace(
        allocation_id="scratch",
        aligned_size=40,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        common_utils.finalize_gms_write(allocator, object())

    assert events == [
        "register",
        "prune",
        "rebind",
        "commit",
        "close_best_effort",
    ]


@pytest.mark.parametrize("failure_stage", ("load", "move", "empty_cache"))
def test_trt_rw_pre_finalization_failure_closes_once_and_preserves_error(
    monkeypatch, failure_stage
):
    events = []

    def fail_at(stage):
        events.append(stage)
        if stage == failure_stage:
            raise RuntimeError(f"{stage} failed")

    def close(*, best_effort):
        assert best_effort
        events.append("close")
        raise RuntimeError("close failed")

    def original_load(*_args):
        fail_at("load")
        return object(), object()

    monkeypatch.setattr(
        trtllm_model_loader,
        "gms_use_mem_pool",
        lambda *_args: nullcontext(),
    )
    monkeypatch.setattr(
        trtllm_model_loader,
        "_move_untracked_params",
        lambda *_args: fail_at("move"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: fail_at("empty_cache"))
    client = SimpleNamespace(close=close)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        trtllm_model_loader._load_rw(
            object(),
            None,
            None,
            client,
            0,
            original_load,
        )

    assert events.count("close") == 1
    assert events[-1] == "close"


def test_trt_rw_finalization_failure_is_not_closed_twice(monkeypatch):
    events = []

    def close(*, best_effort):
        assert best_effort
        events.append("close")
        raise RuntimeError("close failed")

    client = SimpleNamespace(close=close)

    def fail_finalize(finalize_client, _model):
        events.append("finalize")
        try:
            finalize_client.close(best_effort=True)
        except RuntimeError:
            pass
        raise RuntimeError("finalize failed")

    monkeypatch.setattr(
        trtllm_model_loader,
        "gms_use_mem_pool",
        lambda *_args: nullcontext(),
    )
    monkeypatch.setattr(trtllm_model_loader, "_move_untracked_params", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(trtllm_model_loader, "finalize_gms_write", fail_finalize)

    with pytest.raises(RuntimeError, match="finalize failed"):
        trtllm_model_loader._load_rw(
            object(),
            None,
            None,
            client,
            0,
            lambda *_args: (object(), object()),
        )

    assert events == ["finalize", "close"]


def test_publication_failure_uses_best_effort_close_and_preserves_error(gms_write):
    allocator, stats, events = gms_write
    allocator.fail_commit = True
    model_loader._store_pending_gms_write(allocator, stats, 12)

    with pytest.raises(RuntimeError, match="commit failed"):
        model_loader.publish_pending_gms_write()

    assert events == ["register", "prune", "commit", "close_best_effort"]
    assert not model_loader.has_pending_gms_write()


def test_abort_pending_write_uses_best_effort_close(gms_write):
    allocator, stats, events = gms_write
    model_loader._store_pending_gms_write(allocator, stats, 12)

    assert model_loader.abort_pending_gms_write()
    assert events == ["register", "prune", "close_best_effort"]
    assert not model_loader.has_pending_gms_write()


def test_write_load_failure_uses_best_effort_close_and_preserves_error(monkeypatch):
    events = []
    client = SimpleNamespace(
        close=lambda *, best_effort: events.append(
            "close_best_effort" if best_effort else "close"
        )
    )
    monkeypatch.setattr(
        model_loader,
        "_load_write_mode_impl",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        model_loader._load_write_mode(client, None, None, None, torch.device("cpu"))

    assert events == ["close_best_effort"]
    assert not model_loader.has_pending_gms_write()


def test_read_mode_runs_meta_post_load_once_before_materialization(monkeypatch):
    events = []
    model = SimpleNamespace(eval=lambda: model)
    manager = SimpleNamespace(close=lambda: events.append("close"), total_bytes=16)
    loader_utils = ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.initialize_model = lambda **_kwargs: (
        events.append("initialize") or model
    )
    loader_utils.process_weights_after_loading = lambda *_args: events.append(
        "post-load"
    )
    torch_utils = ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(
        model_loader,
        "setup_meta_tensor_workaround",
        lambda: events.append("setup"),
    )
    monkeypatch.setattr(
        model_loader,
        "materialize_module_from_gms",
        lambda *_args, **_kwargs: events.append("materialize"),
    )
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    model_config = SimpleNamespace(dtype=torch.float32)
    assert model_loader._load_read_mode(manager, object(), model_config, 0) is model
    assert events == ["setup", "initialize", "post-load", "materialize"]


def test_meta_post_load_failure_prevents_materialization(monkeypatch):
    events = []

    def fail_close():
        events.append("close")
        raise RuntimeError("close failed")

    manager = SimpleNamespace(close=fail_close)
    model = SimpleNamespace(eval=lambda: model)
    loader_utils = ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.initialize_model = lambda **_kwargs: (
        events.append("initialize") or model
    )

    def fail_post_load(*_args):
        events.append("post-load")
        raise RuntimeError("post-load failed")

    loader_utils.process_weights_after_loading = fail_post_load
    torch_utils = ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(
        model_loader,
        "materialize_module_from_gms",
        lambda *_args, **_kwargs: events.append("materialize"),
    )

    with pytest.raises(RuntimeError, match="post-load failed"):
        model_loader._load_read_mode(
            manager,
            object(),
            SimpleNamespace(dtype=torch.float32),
            0,
        )

    assert events == ["initialize", "post-load", "close"]
