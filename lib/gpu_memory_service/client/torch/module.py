# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish and materialize PyTorch module storage through GMS.

Write:
  module -> discover storages -> match GMS mappings -> build manifests -> metadata
Read:
  metadata -> validate/resolve -> GMS map -> parameter-containing storage
                              `-> clone ---> all other storage
Writer rebind:
  discover/match -> skip parameter-containing storage
                 `-> clone/swap non-parameter-only storage -> retain source owners
"""

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
    ModuleTensorBinding,
    ModuleTensorKind,
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

_DISPLACED_SOURCE_TENSORS_ATTR = "_gms_displaced_source_tensors"
_DISPLACED_SOURCE_TENSORS_SENTINEL = object()


@dataclass
class _DiscoveredTensor:
    """One Python tensor object plus every module binding that aliases it."""

    tensor: torch.Tensor
    bindings: list[ModuleTensorBinding]


@dataclass
class _DiscoveredStorage:
    storage: torch.UntypedStorage
    objects: list[_DiscoveredTensor]

    @property
    def has_parameter(self) -> bool:
        return any(
            binding.kind is ModuleTensorKind.PARAMETER
            for tensor_object in self.objects
            for binding in tensor_object.bindings
        )


@dataclass(frozen=True)
class _StoredManifest:
    allocation_id: str
    storage_base_offset: int
    manifest: StorageManifest


@dataclass(frozen=True)
class _ResolvedTensorDestination:
    binding: ModuleTensorBinding
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
        if self.binding.kind is ModuleTensorKind.PARAMETER:
            self.module._parameters[self.attr] = tensor
        elif (
            self.binding.kind is ModuleTensorKind.PERSISTENT_BUFFER
            or self.binding.kind is ModuleTensorKind.NONPERSISTENT_BUFFER
        ):
            self.module._buffers[self.attr] = tensor
            if self.binding.kind is ModuleTensorKind.PERSISTENT_BUFFER:
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


def _get_displaced_source_tensors(
    module: torch.nn.Module,
) -> list[torch.Tensor] | None:
    namespace_collision = any(
        _DISPLACED_SOURCE_TENSORS_ATTR in namespace
        for namespace in (module._parameters, module._buffers, module._modules)
    )
    class_collision = any(
        _DISPLACED_SOURCE_TENSORS_ATTR in cls.__dict__ for cls in type(module).__mro__
    )
    if not namespace_collision and not class_collision:
        if _DISPLACED_SOURCE_TENSORS_ATTR not in module.__dict__:
            return None
        owner_state = module.__dict__[_DISPLACED_SOURCE_TENSORS_ATTR]
        if (
            type(owner_state) is tuple
            and len(owner_state) == 2
            and owner_state[0] is _DISPLACED_SOURCE_TENSORS_SENTINEL
            and type(owner_state[1]) is list
            and all(torch.is_tensor(tensor) for tensor in owner_state[1])
        ):
            return owner_state[1]
    raise RuntimeError(
        f"Reserved GMS attribute {_DISPLACED_SOURCE_TENSORS_ATTR!r} "
        "collides with an existing model attribute"
    )


def _iter_module_tensor_bindings(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, ModuleTensorBinding]]:
    """Yield each supported tensor and its direct module binding."""
    for module_path, module in model.named_modules(remove_duplicate=False):
        _get_displaced_source_tensors(module)
        prefix = f"{module_path}." if module_path else ""
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            yield (
                parameter,
                ModuleTensorBinding(f"{prefix}{name}", ModuleTensorKind.PARAMETER),
            )

        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            kind = (
                ModuleTensorKind.NONPERSISTENT_BUFFER
                if name in module._non_persistent_buffers_set
                else ModuleTensorKind.PERSISTENT_BUFFER
            )
            yield buffer, ModuleTensorBinding(f"{prefix}{name}", kind)

        registered = (
            set(module._parameters) | set(module._buffers) | set(module._modules)
        )
        for name, value in module.__dict__.items():
            if (
                name in registered
                or name == _DISPLACED_SOURCE_TENSORS_ATTR
                or name.startswith("__")
            ):
                continue
            if torch.is_tensor(value):
                yield (
                    value,
                    ModuleTensorBinding(f"{prefix}{name}", ModuleTensorKind.ATTRIBUTE),
                )
            elif type(value) in (list, tuple):
                for index, element in enumerate(value):
                    if torch.is_tensor(element):
                        yield (
                            element,
                            ModuleTensorBinding(
                                f"{prefix}{name}.{index}",
                                ModuleTensorKind.ATTRIBUTE,
                            ),
                        )


def _discover_module_storages(model: torch.nn.Module) -> list[_DiscoveredStorage]:
    """Group supported module tensors by Python identity and StorageImpl."""
    objects: dict[int, _DiscoveredTensor] = {}
    for tensor, binding in _iter_module_tensor_bindings(model):
        tensor_object = objects.get(id(tensor))
        if tensor_object is None:
            tensor_object = _DiscoveredTensor(tensor, [])
            objects[id(tensor)] = tensor_object
        if binding not in tensor_object.bindings:
            tensor_object.bindings.append(binding)

    storages: dict[int, _DiscoveredStorage] = {}
    for tensor_object in objects.values():
        tensor = tensor_object.tensor
        has_parameter_binding = any(
            binding.kind is ModuleTensorKind.PARAMETER
            for binding in tensor_object.bindings
        )
        if isinstance(tensor, torch.nn.Parameter) and not has_parameter_binding:
            raise RuntimeError(
                f"Unregistered Parameter at {tensor_object.bindings[0].path!r} "
                "has no parameter binding"
            )
        if has_parameter_binding and not isinstance(tensor, torch.nn.Parameter):
            raise RuntimeError(
                f"GMS parameter binding {tensor_object.bindings[0].path!r} "
                "does not contain a Parameter"
            )
        if not has_parameter_binding and type(tensor) is not torch.Tensor:
            raise RuntimeError(
                "GMS does not support non-parameter Tensor subclass at "
                f"{tensor_object.bindings[0].path!r}: {type(tensor).__name__}"
            )
        if tensor.is_conj() or tensor.is_neg():
            raise RuntimeError(
                "GMS does not support lazy conjugate/negative tensor "
                f"{tensor_object.bindings[0].path!r}"
            )
        if has_parameter_binding:
            if not tensor.is_leaf or tensor.grad_fn is not None:
                raise RuntimeError(
                    f"GMS only supports leaf Parameters at "
                    f"{tensor_object.bindings[0].path!r}"
                )
        elif tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError(
                "GMS inference storage does not support autograd tensor "
                f"{tensor_object.bindings[0].path!r}"
            )

        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            raise RuntimeError(
                "GMS module manifests do not support zero-byte StorageImpls: "
                f"{tensor_object.bindings[0].path!r}"
            )
        storage_offset_bytes = int(tensor.storage_offset()) * tensor.dtype.itemsize
        if (storage.data_ptr() + storage_offset_bytes) % tensor.dtype.itemsize:
            raise RuntimeError(
                f"GMS tensor data is unaligned at {tensor_object.bindings[0].path!r}"
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


def _match_storages_to_gms_mappings(
    storages: list[_DiscoveredStorage],
    mappings: dict[int, object],
    *,
    require_parameters: bool,
) -> list[tuple[_DiscoveredStorage, str, int]]:
    """Match complete StorageImpl envelopes to their containing GMS mappings."""
    located: list[tuple[_DiscoveredStorage, str, int]] = []
    for ordinal, discovered_storage in enumerate(storages):
        storage_ptr = int(discovered_storage.storage.data_ptr())
        storage_nbytes = int(discovered_storage.storage.nbytes())
        storage_end = storage_ptr + storage_nbytes
        containing: tuple[int, object] | None = None
        for va, mapping in mappings.items():
            mapping_start = int(va)
            mapping_end = mapping_start + int(mapping.aligned_size)
            contains = mapping_start <= storage_ptr and storage_end <= mapping_end
            if (
                max(mapping_start, storage_ptr) < min(mapping_end, storage_end)
                and not contains
            ):
                raise RuntimeError(
                    f"Storage {ordinal} exceeds GMS allocation "
                    f"{mapping.allocation_id!r}"
                )
            if contains:
                if containing is not None:
                    raise RuntimeError(
                        f"Storage {ordinal} belongs to multiple GMS allocations"
                    )
                containing = (mapping_start, mapping)

        if containing is None:
            if require_parameters and discovered_storage.has_parameter:
                raise RuntimeError(
                    f"Parameter {discovered_storage.objects[0].bindings[0].path!r} "
                    "is not contained in a GMS allocation"
                )
            logger.debug(
                "[GMS] Skipping storage for %r outside GMS allocations",
                discovered_storage.objects[0].bindings[0].path,
            )
            continue

        va, mapping = containing
        located.append(
            (discovered_storage, str(mapping.allocation_id), storage_ptr - va)
        )

    intervals: dict[str, list[tuple[int, int]]] = {}
    for discovered_storage, allocation_id, start in located:
        intervals.setdefault(allocation_id, []).append(
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


def _build_storage_manifest(
    discovered_storage: _DiscoveredStorage,
) -> StorageManifest:
    """Build one manifest for a discovered StorageImpl and all of its tensors."""
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
                bindings=tuple(tensor_object.bindings),
            )
            for tensor_object in discovered_storage.objects
        ),
    )


def _validate_storage_manifest(
    manifest: StorageManifest,
    *,
    key: str,
    storage_base_offset: int | None = None,
) -> None:
    """Validate manifest layouts, bindings, and optional allocation envelope."""
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
        if not tensor_object.bindings:
            raise RuntimeError(
                f"Tensor object in GMS storage manifest {key!r} has no bindings"
            )
        has_parameter = any(
            binding.kind is ModuleTensorKind.PARAMETER
            for binding in tensor_object.bindings
        )
        if tensor_object.requires_grad and not has_parameter:
            raise RuntimeError(
                f"Non-parameter tensor object in GMS storage manifest {key!r} "
                "requires gradients"
            )
        for binding in tensor_object.bindings:
            if not binding.path or binding.path in paths:
                raise RuntimeError(
                    f"Duplicate or empty binding path in GMS storage manifest {key!r}: "
                    f"{binding.path!r}"
                )
            paths.add(binding.path)


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> set[str]:
    """Publish one typed storage manifest for each GMS-backed StorageImpl."""
    storages = _match_storages_to_gms_mappings(
        _discover_module_storages(model),
        gms_client_memory_manager.mappings,
        require_parameters=True,
    )
    entries: list[tuple[str, str, int, bytes]] = []
    referenced_allocation_ids: set[str] = set()
    for ordinal, (
        discovered_storage,
        allocation_id,
        storage_base_offset,
    ) in enumerate(storages):
        key = f"{STORAGE_MANIFEST_PREFIX}{ordinal}"
        manifest = _build_storage_manifest(discovered_storage)
        _validate_storage_manifest(
            manifest,
            key=key,
            storage_base_offset=storage_base_offset,
        )
        encoded = msgspec.msgpack.encode(manifest)
        entries.append((key, allocation_id, storage_base_offset, encoded))
        referenced_allocation_ids.add(allocation_id)

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
    """Decode, validate, and order every published storage manifest."""
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
        _validate_storage_manifest(
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
            for binding in tensor_object.bindings:
                if binding.path in paths:
                    raise RuntimeError(
                        f"Duplicate GMS destination binding {binding.path!r}"
                    )
                paths.add(binding.path)
    for allocation_id, allocation_intervals in intervals.items():
        ordered = sorted(allocation_intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] > current[0]:
                raise RuntimeError(
                    "GMS storage manifests overlap in allocation "
                    f"{allocation_id!r}: {previous} and {current}"
                )
    return [stored for _, stored in loaded]


def _resolve_submodule_path(
    model: torch.nn.Module,
    path: list[str],
) -> torch.nn.Module:
    module = model
    for name in path:
        if name not in module._modules or module._modules[name] is None:
            raise RuntimeError(f"Unsupported GMS destination module path {name!r}")
        module = module._modules[name]
    return module


_MISSING = object()


def _static_module_attribute(module: torch.nn.Module, attr: str) -> object:
    return next(
        (cls.__dict__[attr] for cls in type(module).__mro__ if attr in cls.__dict__),
        _MISSING,
    )


def _resolve_tensor_destination(
    model: torch.nn.Module,
    binding: ModuleTensorBinding,
) -> _ResolvedTensorDestination:
    """Resolve one source binding to a validated reader-model destination."""
    parts = binding.path.split(".")
    if any(not part for part in parts):
        raise RuntimeError(f"Invalid GMS destination binding path {binding.path!r}")

    if binding.kind is ModuleTensorKind.PARAMETER:
        module = _resolve_submodule_path(model, parts[:-1])
        attr = parts[-1]
        if (
            attr in module._buffers
            or attr in module._modules
            or attr in module.__dict__
            or _static_module_attribute(module, attr) is not _MISSING
        ):
            raise RuntimeError(
                f"GMS parameter destination {binding.path!r} collides with "
                "another module binding"
            )
        existing = module._parameters.get(attr)
        if existing is not None and not isinstance(existing, torch.nn.Parameter):
            raise RuntimeError(
                f"GMS parameter destination {binding.path!r} is not a Parameter"
            )
        return _ResolvedTensorDestination(binding, module, attr, existing)

    if (
        binding.kind is ModuleTensorKind.PERSISTENT_BUFFER
        or binding.kind is ModuleTensorKind.NONPERSISTENT_BUFFER
    ):
        module = _resolve_submodule_path(model, parts[:-1])
        attr = parts[-1]
        if (
            attr in module._parameters
            or attr in module._modules
            or attr in module.__dict__
            or _static_module_attribute(module, attr) is not _MISSING
        ):
            raise RuntimeError(
                f"GMS buffer destination {binding.path!r} collides with "
                "another module binding"
            )
        existing = module._buffers.get(attr)
        if existing is not None and not torch.is_tensor(existing):
            raise RuntimeError(
                f"GMS buffer destination {binding.path!r} is not a Tensor"
            )
        return _ResolvedTensorDestination(binding, module, attr, existing)

    if len(parts) >= 2 and parts[-1].isdecimal():
        module = _resolve_submodule_path(model, parts[:-2])
        attr = parts[-2]
        container = module.__dict__.get(attr)
        index = int(parts[-1])
        if type(container) not in (list, tuple) or index >= len(container):
            raise RuntimeError(
                f"GMS sequence destination {binding.path!r} is unsupported"
            )
        existing = container[index] if torch.is_tensor(container[index]) else None
        return _ResolvedTensorDestination(binding, module, attr, existing, index)

    module = _resolve_submodule_path(model, parts[:-1])
    attr = parts[-1]
    if attr in module._parameters or attr in module._buffers or attr in module._modules:
        raise RuntimeError(
            f"GMS attribute destination {binding.path!r} collides with "
            "another module binding"
        )
    existing = module.__dict__.get(attr)
    if attr in module.__dict__ and not torch.is_tensor(existing):
        raise RuntimeError(
            f"GMS attribute destination {binding.path!r} is not a direct tensor"
        )
    if attr not in module.__dict__ and hasattr(
        _static_module_attribute(module, attr), "__set__"
    ):
        raise RuntimeError(
            f"GMS attribute destination {binding.path!r} is not representable"
        )
    return _ResolvedTensorDestination(binding, module, attr, existing)


def _make_parameter_from_template(
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
        for name in dict.fromkeys(copyreg._slotnames(type(template))):
            if hasattr(template, name):
                setattr(parameter, name, getattr(template, name))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot materialize GMS parameter {path!r} as "
            f"{type(template).__name__}: {exc}"
        ) from exc
    return parameter


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
    """Map parameter-containing storages read-only and clone all other storages."""
    if int(gms_client_memory_manager.device) != device_index:
        raise RuntimeError(
            "GMS manager device does not match materialization device: "
            f"{gms_client_memory_manager.device} vs {device_index}"
        )
    stored_manifests = _load_storage_manifests(gms_client_memory_manager)
    resolved: list[
        list[tuple[TensorObject, tuple[_ResolvedTensorDestination, ...]]]
    ] = []
    destinations: dict[tuple[int, str, int | None], tuple[int, int]] = {}
    for manifest_index, stored in enumerate(stored_manifests):
        manifest_destinations: list[
            tuple[TensorObject, tuple[_ResolvedTensorDestination, ...]]
        ] = []
        for object_index, tensor_object in enumerate(stored.manifest.objects):
            object_id = (manifest_index, object_index)
            unique_destinations: dict[
                tuple[int, str, int | None], _ResolvedTensorDestination
            ] = {}
            for binding in tensor_object.bindings:
                destination = _resolve_tensor_destination(model, binding)
                previous_destination = unique_destinations.get(destination.destination)
                if previous_destination is not None:
                    if previous_destination.binding.kind is not binding.kind:
                        raise RuntimeError(
                            "GMS source bindings resolve to incompatible destination "
                            f"{binding.path!r}"
                        )
                    continue
                previous_object = destinations.setdefault(
                    destination.destination, object_id
                )
                if previous_object != object_id:
                    raise RuntimeError(
                        "Distinct GMS tensor objects resolve to the same destination "
                        f"{binding.path!r}"
                    )
                unique_destinations[destination.destination] = destination
            object_destinations = tuple(unique_destinations.values())
            manifest_destinations.append((tensor_object, object_destinations))
        resolved.append(manifest_destinations)

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

        materialized: list[
            tuple[torch.Tensor, tuple[_ResolvedTensorDestination, ...]]
        ] = []
        for stored, manifest_destinations in zip(
            stored_manifests, resolved, strict=True
        ):
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
                binding.kind is ModuleTensorKind.PARAMETER
                for tensor_object in stored.manifest.objects
                for binding in tensor_object.bindings
            )
            if has_parameter:
                target_storage = source_storage
            else:
                clone_started = True
                target_storage = source_storage.clone()

            for tensor_object, object_destinations in manifest_destinations:
                dtype = _dtype_from_name(tensor_object.dtype)
                tensor = _tensor_from_storage(
                    target_storage,
                    list(tensor_object.shape),
                    list(tensor_object.stride),
                    dtype,
                    tensor_object.storage_offset_bytes // dtype.itemsize,
                )
                parameter_destinations = tuple(
                    destination
                    for destination in object_destinations
                    if destination.binding.kind is ModuleTensorKind.PARAMETER
                )
                if parameter_destinations:
                    template = next(
                        (
                            destination.existing
                            for destination in object_destinations
                            if destination.binding.kind is ModuleTensorKind.PARAMETER
                            and isinstance(destination.existing, torch.nn.Parameter)
                        ),
                        None,
                    )
                    tensor = _make_parameter_from_template(
                        template,
                        tensor,
                        path=parameter_destinations[0].binding.path,
                        requires_grad=tensor_object.requires_grad,
                    )
                materialized.append((tensor, object_destinations))

        installation_started = True
        for tensor, object_destinations in materialized:
            for destination in object_destinations:
                destination.install(tensor)
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


def _swap_discovered_tensors(
    objects: list[_DiscoveredTensor],
    replacements: dict[int, torch.Tensor],
) -> None:
    """Preflight and swap a group of identity-preserving tensor replacements."""
    if not hasattr(torch.utils, "swap_tensors"):
        raise RuntimeError("GMS publisher rebinding requires torch.utils.swap_tensors")

    object_ids = {id(tensor_object.tensor) for tensor_object in objects}
    group_base_uses: dict[int, int] = {}
    for tensor_object in objects:
        base = tensor_object.tensor._base
        if base is not None and id(base) in object_ids:
            group_base_uses[id(base)] = group_base_uses.get(id(base), 0) + 1

    def base_depth(tensor_object: _DiscoveredTensor) -> int:
        depth = 0
        base = tensor_object.tensor._base
        while base is not None and id(base) in object_ids:
            depth += 1
            base = base._base
        return depth

    ordered = sorted(objects, key=base_depth, reverse=True)
    accumulate_grad_checks: list[tuple[torch.Tensor, int, str]] = []
    for tensor_object in ordered:
        existing = tensor_object.tensor
        replacement = replacements[id(existing)]
        path = tensor_object.bindings[0].path
        for tensor, name, released_uses in (
            (existing, "t1", group_base_uses.get(id(existing), 0)),
            (replacement, "t2", 0),
        ):
            if weakref.getweakrefs(tensor):
                raise RuntimeError(
                    f"Cannot rebind GMS tensor {path!r}: "
                    f"{name} has weakref associated with it"
                )
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

        existing_slots = set(copyreg._slotnames(existing.__class__))
        replacement_slots = set(copyreg._slotnames(replacement.__class__))
        if existing_slots != replacement_slots:
            raise RuntimeError(
                f"Cannot rebind GMS tensor {path!r}: "
                "replacement has different Python slots"
            )

    for tensor, released_uses, ownership_error in accumulate_grad_checks:
        torch.autograd.graph.get_gradient_edge(tensor)
        if tensor._use_count() - released_uses != 2:
            raise RuntimeError(ownership_error)

    for tensor_object in ordered:
        replacement = replacements.pop(id(tensor_object.tensor))
        path = tensor_object.bindings[0].path
        try:
            torch.utils.swap_tensors(tensor_object.tensor, replacement)
        except RuntimeError as exc:
            raise RuntimeError(f"Cannot rebind GMS tensor {path!r}: {exc}") from exc
        del replacement


def rebind_nonparameter_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> int:
    """Clone each non-parameter-only GMS storage once."""
    displaced_source_tensors = _get_displaced_source_tensors(model)
    storages = _match_storages_to_gms_mappings(
        _discover_module_storages(model),
        gms_client_memory_manager.mappings,
        require_parameters=False,
    )
    candidates = [
        discovered_storage
        for discovered_storage, _, _ in storages
        if not discovered_storage.has_parameter
    ]
    if not candidates:
        return 0

    replacements: dict[int, torch.Tensor] = {}
    source_owners: list[torch.Tensor] = []
    objects: list[_DiscoveredTensor] = []
    for discovered_storage in candidates:
        source_storage = discovered_storage.storage
        source_owner = torch.empty(
            0, dtype=torch.uint8, device=source_storage.device
        ).set_(
            source_storage,
            0,
            (source_storage.nbytes(),),
            (1,),
        )
        target_storage = source_storage.clone()
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

    _swap_discovered_tensors(objects, replacements)
    if displaced_source_tensors is None:
        model.__dict__[_DISPLACED_SOURCE_TENSORS_ATTR] = (
            _DISPLACED_SOURCE_TENSORS_SENTINEL,
            source_owners,
        )
    else:
        displaced_source_tensors.extend(source_owners)
    return sum(discovered_storage.storage.nbytes() for discovered_storage in candidates)
