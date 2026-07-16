# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish and materialize PyTorch module storage through GMS."""

from __future__ import annotations

import copyreg
import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import msgspec
import torch
from gpu_memory_service.client.torch.tensor import (
    STORAGE_MANIFEST_PREFIX,
    Slot,
    StorageManifest,
    TensorObject,
    _dtype_from_name,
    _storage_from_pointer,
    _tensor_from_storage,
    _validate_layout,
)

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

_REBOUND_TENSOR_OWNERS_ATTR = "_gms_rebound_tensor_owners"


class _ReboundTensorOwners:
    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []


@dataclass
class _DiscoveredObject:
    tensor: torch.Tensor
    slots: list[Slot]


@dataclass
class _DiscoveredStorage:
    storage: torch.UntypedStorage
    objects: list[_DiscoveredObject]
    allocation_id: str | None = None
    storage_base_offset: int | None = None

    @property
    def has_parameter(self) -> bool:
        return any(
            slot.kind == "parameter"
            for tensor_object in self.objects
            for slot in tensor_object.slots
        )


@dataclass(frozen=True)
class _StoredManifest:
    allocation_id: str
    storage_base_offset: int
    manifest: StorageManifest


@dataclass(frozen=True)
class _SlotTarget:
    slot: Slot
    module: torch.nn.Module
    attr: str
    existing: torch.Tensor | None
    index: int | None = None

    @property
    def destination(self) -> tuple[int, str, int | None]:
        if self.index is not None:
            container = self.module.__dict__[self.attr]
            if isinstance(container, list):
                return id(container), "", self.index
            return id(self.module.__dict__), self.attr, self.index
        return id(self.module), self.attr, None

    def install(self, tensor: torch.Tensor) -> None:
        if self.slot.kind == "parameter":
            self.module._parameters[self.attr] = tensor
        elif self.slot.kind in ("persistent_buffer", "nonpersistent_buffer"):
            self.module._buffers[self.attr] = tensor
            if self.slot.kind == "persistent_buffer":
                self.module._non_persistent_buffers_set.discard(self.attr)
            else:
                self.module._non_persistent_buffers_set.add(self.attr)
        elif self.index is None:
            self.module.__dict__[self.attr] = tensor
        else:
            container = self.module.__dict__[self.attr]
            if isinstance(container, list):
                container[self.index] = tensor
            else:
                values = list(container)
                values[self.index] = tensor
                self.module.__dict__[self.attr] = tuple(values)


@dataclass(frozen=True)
class _MaterializedObject:
    tensor: torch.Tensor
    targets: tuple[_SlotTarget, ...]


def _iter_module_slots(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, Slot]]:
    for module_path, module in model.named_modules(remove_duplicate=False):
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            path = f"{module_path}.{name}" if module_path else name
            yield parameter, Slot(path, "parameter")

        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            kind = (
                "nonpersistent_buffer"
                if name in module._non_persistent_buffers_set
                else "persistent_buffer"
            )
            path = f"{module_path}.{name}" if module_path else name
            yield buffer, Slot(path, kind)

        registered = (
            set(module._parameters) | set(module._buffers) | set(module._modules)
        )
        for name, value in module.__dict__.items():
            if name in registered or name.startswith("__"):
                continue
            if torch.is_tensor(value):
                path = f"{module_path}.{name}" if module_path else name
                yield value, Slot(path, "attribute")
            elif type(value) in (list, tuple):
                for index, element in enumerate(value):
                    if torch.is_tensor(element):
                        name_with_index = f"{name}.{index}"
                        path = (
                            f"{module_path}.{name_with_index}"
                            if module_path
                            else name_with_index
                        )
                        yield element, Slot(path, "attribute")


def _discover_module_storage(model: torch.nn.Module) -> list[_DiscoveredStorage]:
    objects: dict[int, _DiscoveredObject] = {}
    for tensor, slot in _iter_module_slots(model):
        tensor_object = objects.get(id(tensor))
        if tensor_object is None:
            tensor_object = _DiscoveredObject(tensor, [])
            objects[id(tensor)] = tensor_object
        if slot not in tensor_object.slots:
            tensor_object.slots.append(slot)

    storages: dict[int, _DiscoveredStorage] = {}
    for tensor_object in objects.values():
        tensor = tensor_object.tensor
        has_parameter_slot = any(
            slot.kind == "parameter" for slot in tensor_object.slots
        )
        if isinstance(tensor, torch.nn.Parameter) and not has_parameter_slot:
            raise RuntimeError(
                f"Unregistered Parameter at {tensor_object.slots[0].path!r} "
                "has no parameter slot"
            )
        if has_parameter_slot and not isinstance(tensor, torch.nn.Parameter):
            raise RuntimeError(
                f"GMS parameter slot {tensor_object.slots[0].path!r} "
                "does not contain a Parameter"
            )
        if not has_parameter_slot and type(tensor) is not torch.Tensor:
            raise RuntimeError(
                "GMS does not support non-parameter Tensor subclass at "
                f"{tensor_object.slots[0].path!r}: {type(tensor).__name__}"
            )
        if tensor.is_conj() or tensor.is_neg():
            raise RuntimeError(
                "GMS does not support lazy conjugate/negative tensor "
                f"{tensor_object.slots[0].path!r}"
            )
        if has_parameter_slot:
            if not tensor.is_leaf or tensor.grad_fn is not None:
                raise RuntimeError(
                    f"GMS only supports leaf Parameters at "
                    f"{tensor_object.slots[0].path!r}"
                )
        elif tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError(
                "GMS inference storage does not support autograd tensor "
                f"{tensor_object.slots[0].path!r}"
            )

        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            raise RuntimeError(
                "GMS module manifests do not support zero-byte StorageImpls: "
                f"{tensor_object.slots[0].path!r}"
            )
        storage_offset_bytes = int(tensor.storage_offset()) * tensor.dtype.itemsize
        if (storage.data_ptr() + storage_offset_bytes) % tensor.dtype.itemsize:
            raise RuntimeError(
                f"GMS tensor data is unaligned at {tensor_object.slots[0].path!r}"
            )
        _validate_layout(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            int(tensor.storage_offset()),
            int(storage.nbytes()),
        )
        storage_id = int(storage._cdata)
        discovered_storage = storages.get(storage_id)
        if discovered_storage is None:
            discovered_storage = _DiscoveredStorage(storage, [])
            storages[storage_id] = discovered_storage
        discovered_storage.objects.append(tensor_object)
    return list(storages.values())


def _locate_storage(
    storages: list[_DiscoveredStorage],
    mappings: dict[int, object],
    *,
    require_parameters: bool,
) -> list[_DiscoveredStorage]:
    located: list[_DiscoveredStorage] = []
    for ordinal, discovered_storage in enumerate(storages):
        storage_ptr = int(discovered_storage.storage.data_ptr())
        storage_nbytes = int(discovered_storage.storage.nbytes())
        storage_end = storage_ptr + storage_nbytes
        containing: list[tuple[int, object]] = []
        for va, mapping in mappings.items():
            mapping_end = int(va) + int(mapping.aligned_size)
            overlaps = max(int(va), storage_ptr) < min(mapping_end, storage_end)
            if overlaps and not (int(va) <= storage_ptr and storage_end <= mapping_end):
                raise RuntimeError(
                    f"Storage {ordinal} exceeds GMS allocation "
                    f"{mapping.allocation_id!r}"
                )
            if int(va) <= storage_ptr and storage_end <= mapping_end:
                containing.append((int(va), mapping))

        if len(containing) > 1:
            raise RuntimeError(f"Storage {ordinal} belongs to multiple GMS allocations")
        if not containing:
            if require_parameters and discovered_storage.has_parameter:
                raise RuntimeError(
                    f"Parameter {discovered_storage.objects[0].slots[0].path!r} "
                    "is not contained in a GMS allocation"
                )
            logger.debug(
                "[GMS] Skipping storage for %r outside GMS allocations",
                discovered_storage.objects[0].slots[0].path,
            )
            continue

        va, mapping = containing[0]
        discovered_storage.allocation_id = str(mapping.allocation_id)
        discovered_storage.storage_base_offset = storage_ptr - va
        located.append(discovered_storage)

    intervals: dict[str, list[tuple[int, int]]] = {}
    for discovered_storage in located:
        start = discovered_storage.storage_base_offset
        if start is None or discovered_storage.allocation_id is None:
            raise RuntimeError("Located GMS storage is missing its allocation envelope")
        intervals.setdefault(discovered_storage.allocation_id, []).append(
            (start, start + discovered_storage.storage.nbytes())
        )
    for allocation_id, allocation_intervals in intervals.items():
        ordered = sorted(allocation_intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] > current[0]:
                raise RuntimeError(
                    "Distinct StorageImpl byte ranges overlap in allocation "
                    f"{allocation_id!r}: {previous} and {current}"
                )
    return located


def _manifest_for_storage(discovered_storage: _DiscoveredStorage) -> StorageManifest:
    return StorageManifest(
        nbytes=int(discovered_storage.storage.nbytes()),
        objects=tuple(
            TensorObject(
                dtype=str(tensor_object.tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor_object.tensor.shape),
                stride=tuple(tensor_object.tensor.stride()),
                storage_offset_bytes=(
                    int(tensor_object.tensor.storage_offset())
                    * tensor_object.tensor.dtype.itemsize
                ),
                requires_grad=bool(tensor_object.tensor.requires_grad),
                slots=tuple(tensor_object.slots),
            )
            for tensor_object in discovered_storage.objects
        ),
    )


def _validate_manifest(
    manifest: StorageManifest,
    *,
    key: str,
    storage_base_offset: int | None = None,
) -> None:
    if manifest.nbytes <= 0:
        raise RuntimeError(f"GMS storage manifest {key!r} has nonpositive size")
    if not manifest.objects:
        raise RuntimeError(f"GMS storage manifest {key!r} has no tensor objects")
    if storage_base_offset is not None and storage_base_offset < 0:
        raise RuntimeError(f"GMS storage manifest {key!r} has negative base offset")
    paths: set[str] = set()
    for tensor_object in manifest.objects:
        try:
            dtype = _dtype_from_name(tensor_object.dtype)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid dtype in GMS storage manifest {key!r}: {exc}"
            ) from exc
        if tensor_object.storage_offset_bytes % dtype.itemsize:
            raise RuntimeError(
                f"Unaligned tensor byte offset in GMS storage manifest {key!r}"
            )
        if storage_base_offset is not None and storage_base_offset % dtype.itemsize:
            raise RuntimeError(
                f"Unaligned storage envelope in GMS storage manifest {key!r}"
            )
        try:
            _validate_layout(
                tensor_object.shape,
                tensor_object.stride,
                dtype,
                tensor_object.storage_offset_bytes // dtype.itemsize,
                manifest.nbytes,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid tensor layout in GMS storage manifest {key!r}: {exc}"
            ) from exc
        if not tensor_object.slots:
            raise RuntimeError(
                f"Tensor object in GMS storage manifest {key!r} has no slots"
            )
        has_parameter = any(slot.kind == "parameter" for slot in tensor_object.slots)
        if tensor_object.requires_grad and not has_parameter:
            raise RuntimeError(
                f"Non-parameter tensor object in GMS storage manifest {key!r} "
                "requires gradients"
            )
        for slot in tensor_object.slots:
            if not slot.path or slot.path in paths:
                raise RuntimeError(
                    f"Duplicate or empty slot path in GMS storage manifest {key!r}: "
                    f"{slot.path!r}"
                )
            paths.add(slot.path)


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> set[str]:
    """Publish one typed storage manifest for each GMS-backed StorageImpl."""
    storages = _locate_storage(
        _discover_module_storage(model),
        gms_client_memory_manager.mappings,
        require_parameters=True,
    )
    entries: list[tuple[str, str, int, bytes]] = []
    referenced_allocation_ids: set[str] = set()
    for ordinal, discovered_storage in enumerate(storages):
        if (
            discovered_storage.allocation_id is None
            or discovered_storage.storage_base_offset is None
        ):
            raise RuntimeError("Located GMS storage is missing its allocation envelope")
        key = f"{STORAGE_MANIFEST_PREFIX}{ordinal}"
        manifest = _manifest_for_storage(discovered_storage)
        _validate_manifest(
            manifest,
            key=key,
            storage_base_offset=discovered_storage.storage_base_offset,
        )
        encoded = msgspec.msgpack.encode(manifest)
        entries.append(
            (
                key,
                discovered_storage.allocation_id,
                discovered_storage.storage_base_offset,
                encoded,
            )
        )
        referenced_allocation_ids.add(discovered_storage.allocation_id)

    for key, allocation_id, storage_base_offset, encoded in entries:
        if not gms_client_memory_manager.metadata_put(
            key=key,
            allocation_id=allocation_id,
            offset_bytes=storage_base_offset,
            value=encoded,
        ):
            raise RuntimeError(f"Failed to publish GMS storage manifest {key!r}")
    return referenced_allocation_ids


def _load_storage_manifests(
    manager: "GMSClientMemoryManager",
) -> list[_StoredManifest]:
    loaded: list[tuple[int, _StoredManifest]] = []
    for key in manager.metadata_list(STORAGE_MANIFEST_PREFIX):
        suffix = key.removeprefix(STORAGE_MANIFEST_PREFIX)
        if not suffix.isascii() or not suffix.isdecimal() or str(int(suffix)) != suffix:
            raise RuntimeError(f"Invalid GMS storage manifest key {key!r}")
        got = manager.metadata_get(key)
        if got is None:
            raise RuntimeError(f"GMS storage manifest disappeared: {key!r}")
        allocation_id, storage_base_offset, value = got
        if type(storage_base_offset) is not int:
            raise RuntimeError(
                f"GMS storage manifest {key!r} has a non-integer base offset"
            )
        try:
            manifest = msgspec.msgpack.decode(value, type=StorageManifest)
        except msgspec.DecodeError as exc:
            raise RuntimeError(f"Invalid GMS storage manifest {key!r}: {exc}") from exc
        _validate_manifest(
            manifest,
            key=key,
            storage_base_offset=storage_base_offset,
        )
        loaded.append(
            (
                int(suffix),
                _StoredManifest(
                    str(allocation_id),
                    storage_base_offset,
                    manifest,
                ),
            )
        )
    loaded.sort(key=lambda item: item[0])
    if not loaded:
        raise RuntimeError("No GMS module storage manifests found")
    if [ordinal for ordinal, _ in loaded] != list(range(len(loaded))):
        raise RuntimeError("GMS storage manifest ordinals must be contiguous")

    intervals: dict[str, list[tuple[int, int]]] = {}
    paths: set[str] = set()
    for _, stored in loaded:
        start = stored.storage_base_offset
        intervals.setdefault(stored.allocation_id, []).append(
            (start, start + stored.manifest.nbytes)
        )
        for tensor_object in stored.manifest.objects:
            for slot in tensor_object.slots:
                if slot.path in paths:
                    raise RuntimeError(f"Duplicate GMS destination slot {slot.path!r}")
                paths.add(slot.path)
    for allocation_id, allocation_intervals in intervals.items():
        ordered = sorted(allocation_intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] > current[0]:
                raise RuntimeError(
                    "GMS storage manifests overlap in allocation "
                    f"{allocation_id!r}: {previous} and {current}"
                )
    return [stored for _, stored in loaded]


def _resolve_module(model: torch.nn.Module, path: list[str]) -> torch.nn.Module:
    module = model
    for name in path:
        if name not in module._modules or module._modules[name] is None:
            raise RuntimeError(f"Unsupported GMS destination module path {name!r}")
        module = module._modules[name]
    return module


def _has_data_descriptor(module: torch.nn.Module, attr: str) -> bool:
    descriptor = next(
        (cls.__dict__[attr] for cls in type(module).__mro__ if attr in cls.__dict__),
        None,
    )
    return hasattr(descriptor, "__set__")


def _class_defines_attribute(module: torch.nn.Module, attr: str) -> bool:
    return any(attr in cls.__dict__ for cls in type(module).__mro__)


def _resolve_slot_target(
    model: torch.nn.Module,
    slot: Slot,
) -> _SlotTarget:
    parts = slot.path.split(".")
    if any(not part for part in parts):
        raise RuntimeError(f"Invalid GMS destination slot path {slot.path!r}")

    if slot.kind == "parameter":
        module = _resolve_module(model, parts[:-1])
        attr = parts[-1]
        if (
            attr in module._buffers
            or attr in module._modules
            or attr in module.__dict__
            or _class_defines_attribute(module, attr)
        ):
            raise RuntimeError(
                f"GMS parameter destination {slot.path!r} collides with "
                "another module slot"
            )
        existing = module._parameters.get(attr)
        if existing is not None and not isinstance(existing, torch.nn.Parameter):
            raise RuntimeError(
                f"GMS parameter destination {slot.path!r} is not a Parameter"
            )
        return _SlotTarget(slot, module, attr, existing)

    if slot.kind in ("persistent_buffer", "nonpersistent_buffer"):
        module = _resolve_module(model, parts[:-1])
        attr = parts[-1]
        if (
            attr in module._parameters
            or attr in module._modules
            or attr in module.__dict__
            or _class_defines_attribute(module, attr)
        ):
            raise RuntimeError(
                f"GMS buffer destination {slot.path!r} collides with "
                "another module slot"
            )
        existing = module._buffers.get(attr)
        if existing is not None and not torch.is_tensor(existing):
            raise RuntimeError(f"GMS buffer destination {slot.path!r} is not a Tensor")
        return _SlotTarget(slot, module, attr, existing)

    if len(parts) >= 2 and parts[-1].isdecimal():
        module = _resolve_module(model, parts[:-2])
        attr = parts[-2]
        container = module.__dict__.get(attr)
        index = int(parts[-1])
        if type(container) not in (list, tuple) or index >= len(container):
            raise RuntimeError(f"GMS sequence destination {slot.path!r} is unsupported")
        existing = container[index] if torch.is_tensor(container[index]) else None
        return _SlotTarget(slot, module, attr, existing, index)

    module = _resolve_module(model, parts[:-1])
    attr = parts[-1]
    if attr in module._parameters or attr in module._buffers or attr in module._modules:
        raise RuntimeError(
            f"GMS attribute destination {slot.path!r} collides with another module slot"
        )
    existing = module.__dict__.get(attr)
    if attr in module.__dict__ and not torch.is_tensor(existing):
        raise RuntimeError(
            f"GMS attribute destination {slot.path!r} is not a direct tensor"
        )
    if attr not in module.__dict__ and _has_data_descriptor(module, attr):
        raise RuntimeError(
            f"GMS attribute destination {slot.path!r} is not representable"
        )
    return _SlotTarget(slot, module, attr, existing)


def _parameter_template(
    targets: tuple[_SlotTarget, ...],
) -> torch.nn.Parameter | None:
    return next(
        (
            target.existing
            for target in targets
            if target.slot.kind == "parameter"
            and isinstance(target.existing, torch.nn.Parameter)
        ),
        None,
    )


def _parameter_slot_names(parameter_type: type[torch.nn.Parameter]) -> tuple[str, ...]:
    names: list[str] = []
    for cls in parameter_type.__mro__:
        declared = cls.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{cls.__name__.lstrip('_')}{name}"
            if name not in names:
                names.append(name)
    return tuple(names)


def _make_parameter(
    template: torch.nn.Parameter | None,
    tensor: torch.Tensor,
    *,
    path: str,
    requires_grad: bool,
) -> torch.nn.Parameter:
    if template is None:
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)
    try:
        parameter = torch.Tensor._make_subclass(type(template), tensor, requires_grad)
        parameter.__dict__ = template.__dict__.copy()
        for name in _parameter_slot_names(type(template)):
            if hasattr(template, name):
                setattr(parameter, name, getattr(template, name))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot materialize GMS parameter {path!r} as "
            f"{type(template).__name__}: {exc}"
        ) from exc
    return parameter


def _clone_storage(
    storage: torch.UntypedStorage,
) -> tuple[torch.UntypedStorage, torch.Tensor]:
    source_owner = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
        storage,
        0,
        (storage.nbytes(),),
        (1,),
    )
    return source_owner.clone().untyped_storage(), source_owner


def _free_imported_mappings(
    manager: "GMSClientMemoryManager",
    imported: list[int],
) -> None:
    for va in reversed(imported):
        try:
            manager.free_va(va)
        except BaseException:
            logger.exception("[GMS] Failed to release imported mapping at %#x", va)


def materialize_module_from_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
) -> None:
    """Replace supported module slots with objects reconstructed from GMS."""
    if int(gms_client_memory_manager.device) != device_index:
        raise RuntimeError(
            "GMS manager device does not match materialization device: "
            f"{gms_client_memory_manager.device} vs {device_index}"
        )
    stored_manifests = _load_storage_manifests(gms_client_memory_manager)
    targets: list[list[tuple[TensorObject, tuple[_SlotTarget, ...]]]] = []
    destinations: dict[tuple[int, str, int | None], tuple[int, int]] = {}
    for manifest_index, stored in enumerate(stored_manifests):
        manifest_targets: list[tuple[TensorObject, tuple[_SlotTarget, ...]]] = []
        for object_index, tensor_object in enumerate(stored.manifest.objects):
            object_id = (manifest_index, object_index)
            unique_targets: dict[tuple[int, str, int | None], _SlotTarget] = {}
            for slot in tensor_object.slots:
                target = _resolve_slot_target(model, slot)
                previous_target = unique_targets.get(target.destination)
                if previous_target is not None:
                    if previous_target.slot.kind != target.slot.kind:
                        raise RuntimeError(
                            "GMS source slots resolve to incompatible destination "
                            f"{slot.path!r}"
                        )
                    continue
                previous_object = destinations.setdefault(target.destination, object_id)
                if previous_object != object_id:
                    raise RuntimeError(
                        "Distinct GMS tensor objects resolve to the same destination "
                        f"{slot.path!r}"
                    )
                unique_targets[target.destination] = target
            object_targets = tuple(unique_targets.values())
            manifest_targets.append((tensor_object, object_targets))
        targets.append(manifest_targets)

    allocation_ids = list(
        dict.fromkeys(stored.allocation_id for stored in stored_manifests)
    )
    existing_vas = set(gms_client_memory_manager.mappings)
    mapped: dict[str, int] = {}
    imported: list[int] = []
    clone_started = False
    installation_started = False
    try:
        for allocation_id in allocation_ids:
            va = gms_client_memory_manager.create_mapping(allocation_id=allocation_id)
            mapped[allocation_id] = va
            if va not in existing_vas:
                imported.append(va)
            mapping = gms_client_memory_manager.mappings.get(va)
            if mapping is None or str(mapping.allocation_id) != allocation_id:
                raise RuntimeError(
                    f"GMS mapping for allocation {allocation_id!r} is inconsistent"
                )

        materialized: list[_MaterializedObject] = []
        for stored, manifest_targets in zip(stored_manifests, targets, strict=True):
            mapping = gms_client_memory_manager.mappings[mapped[stored.allocation_id]]
            if (
                stored.storage_base_offset < 0
                or stored.storage_base_offset + stored.manifest.nbytes
                > int(mapping.aligned_size)
            ):
                raise RuntimeError(
                    f"GMS storage manifest exceeds allocation {stored.allocation_id!r}"
                )
            source_storage = _storage_from_pointer(
                mapped[stored.allocation_id] + stored.storage_base_offset,
                stored.manifest.nbytes,
                device_index,
            )
            has_parameter = any(
                slot.kind == "parameter"
                for tensor_object in stored.manifest.objects
                for slot in tensor_object.slots
            )
            if has_parameter:
                target_storage = source_storage
            else:
                clone_started = True
                target_storage, _ = _clone_storage(source_storage)

            for tensor_object, object_targets in manifest_targets:
                dtype = _dtype_from_name(tensor_object.dtype)
                tensor = _tensor_from_storage(
                    target_storage,
                    list(tensor_object.shape),
                    list(tensor_object.stride),
                    dtype,
                    tensor_object.storage_offset_bytes // dtype.itemsize,
                )
                parameter_targets = tuple(
                    target
                    for target in object_targets
                    if target.slot.kind == "parameter"
                )
                if parameter_targets:
                    tensor = _make_parameter(
                        _parameter_template(object_targets),
                        tensor,
                        path=parameter_targets[0].slot.path,
                        requires_grad=tensor_object.requires_grad,
                    )
                materialized.append(_MaterializedObject(tensor, object_targets))

        installation_started = True
        for tensor_object in materialized:
            for target in tensor_object.targets:
                target.install(tensor_object.tensor)
    except BaseException:
        if not installation_started:
            if clone_started and imported:
                torch.cuda.synchronize(device_index)
            _free_imported_mappings(gms_client_memory_manager, imported)
        raise

    meta_tensors = [name for name, value in model.named_parameters() if value.is_meta]
    meta_tensors += [name for name, value in model.named_buffers() if value.is_meta]
    if meta_tensors:
        logger.warning(
            "[GMS] %d meta tensors not in storage manifests: %s",
            len(meta_tensors),
            meta_tensors[:10],
        )


def _swap_tensor_contents(
    existing: torch.Tensor,
    replacement: torch.Tensor,
    *,
    path: str,
) -> None:
    if not hasattr(torch.utils, "swap_tensors"):
        raise RuntimeError("GMS publisher rebinding requires torch.utils.swap_tensors")
    try:
        torch.utils.swap_tensors(existing, replacement)
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot rebind GMS tensor {path!r}: {exc}") from exc


def _swap_discovered_objects(
    objects: list[_DiscoveredObject],
    replacements: dict[int, torch.Tensor],
) -> None:
    if not hasattr(torch.utils, "swap_tensors"):
        raise RuntimeError("GMS publisher rebinding requires torch.utils.swap_tensors")

    object_ids = {id(tensor_object.tensor) for tensor_object in objects}
    group_base_uses: dict[int, int] = {}
    for tensor_object in objects:
        base = tensor_object.tensor._base
        if base is not None and id(base) in object_ids:
            group_base_uses[id(base)] = group_base_uses.get(id(base), 0) + 1

    def base_depth(tensor_object: _DiscoveredObject) -> int:
        depth = 0
        base = tensor_object.tensor._base
        while base is not None and id(base) in object_ids:
            depth += 1
            base = base._base
        return depth

    ordered = sorted(objects, key=base_depth, reverse=True)
    for tensor_object in ordered:
        existing = tensor_object.tensor
        replacement = replacements[id(existing)]
        path = tensor_object.slots[0].path
        for tensor, name in ((existing, "t1"), (replacement, "t2")):
            if weakref.getweakrefs(tensor):
                raise RuntimeError(
                    f"Cannot rebind GMS tensor {path!r}: "
                    f"{name} has weakref associated with it"
                )

        existing_slots = set(copyreg._slotnames(existing.__class__))
        replacement_slots = set(copyreg._slotnames(replacement.__class__))
        if existing_slots != replacement_slots:
            raise RuntimeError(
                f"Cannot rebind GMS tensor {path!r}: "
                "replacement has different Python slots"
            )

    accumulate_grad_checks: list[tuple[torch.Tensor, int, str]] = []
    for tensor_object in ordered:
        existing = tensor_object.tensor
        replacement = replacements[id(existing)]
        path = tensor_object.slots[0].path
        for tensor, name, released_uses in (
            (existing, "t1", group_base_uses.get(id(existing), 0)),
            (replacement, "t2", 0),
        ):
            use_count = tensor._use_count() - released_uses
            ownership_error = (
                f"Cannot rebind GMS tensor {path!r}: expected use_count of "
                f"{name} to be 1 or 2 with an AccumulateGrad node but got "
                f"{use_count}"
            )
            if use_count > 1:
                if use_count != 2 or not tensor.is_leaf:
                    raise RuntimeError(ownership_error)
                accumulate_grad_checks.append((tensor, released_uses, ownership_error))

    for tensor, released_uses, ownership_error in accumulate_grad_checks:
        torch.autograd.graph.get_gradient_edge(tensor)
        if tensor._use_count() - released_uses != 2:
            raise RuntimeError(ownership_error)

    for tensor_object in ordered:
        replacement = replacements.pop(id(tensor_object.tensor))
        _swap_tensor_contents(
            tensor_object.tensor,
            replacement,
            path=tensor_object.slots[0].path,
        )
        del replacement


def rebind_nonparameter_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> int:
    """Clone each non-parameter-only GMS storage once into writable CUDA memory."""
    storages = _locate_storage(
        _discover_module_storage(model),
        gms_client_memory_manager.mappings,
        require_parameters=False,
    )
    candidates = [
        discovered_storage
        for discovered_storage in storages
        if not discovered_storage.has_parameter
    ]
    if not candidates:
        return 0

    if _REBOUND_TENSOR_OWNERS_ATTR not in model.__dict__:
        namespace_collision = any(
            _REBOUND_TENSOR_OWNERS_ATTR in namespace
            for namespace in (
                model._parameters,
                model._buffers,
                model._modules,
            )
        )
        class_collision = any(
            _REBOUND_TENSOR_OWNERS_ATTR in cls.__dict__ for cls in type(model).__mro__
        )
        if namespace_collision or class_collision:
            raise RuntimeError(
                f"Reserved GMS attribute {_REBOUND_TENSOR_OWNERS_ATTR!r} "
                "collides with an existing model attribute"
            )
        owners = _ReboundTensorOwners()
        model.__dict__[_REBOUND_TENSOR_OWNERS_ATTR] = owners
    else:
        owners = model.__dict__[_REBOUND_TENSOR_OWNERS_ATTR]
    if not isinstance(owners, _ReboundTensorOwners):
        raise RuntimeError(
            f"Reserved GMS attribute {_REBOUND_TENSOR_OWNERS_ATTR!r} "
            "collides with an existing model attribute"
        )

    replacements: dict[int, torch.Tensor] = {}
    source_owners: list[torch.Tensor] = []
    objects: list[_DiscoveredObject] = []
    for discovered_storage in candidates:
        target_storage, source_owner = _clone_storage(discovered_storage.storage)
        source_owners.append(source_owner)
        for tensor_object in discovered_storage.objects:
            tensor = tensor_object.tensor
            replacements[id(tensor)] = _tensor_from_storage(
                target_storage,
                list(tensor.shape),
                list(tensor.stride()),
                tensor.dtype,
                int(tensor.storage_offset()),
            )
            objects.append(tensor_object)

    _swap_discovered_objects(objects, replacements)
    owners.tensors.extend(source_owners)
    return sum(discovered_storage.storage.nbytes() for discovered_storage in candidates)
