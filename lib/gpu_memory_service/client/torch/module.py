# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module tensor operations for GPU Memory Service.

This module provides module-level tensor operations:
- Module tensor iteration
- Tensor registration (write path)
- Tensor materialization (read path)
"""

from __future__ import annotations

import copyreg
import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from gpu_memory_service.client.torch.tensor import (
    GMSTensorSpec,
    TensorMetadata,
    _storage_from_pointer,
    _tensor_from_storage,
    _validate_layout,
)

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

_REBOUND_TENSOR_OWNERS_ATTR = "_gms_rebound_tensor_owners"


class _ReboundTensorOwners:
    """Model-scoped owners for GMS storage displaced by tensor swaps."""

    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []


def _get_or_create_rebound_tensor_owners(
    model: torch.nn.Module,
) -> _ReboundTensorOwners:
    owners = model.__dict__.get(_REBOUND_TENSOR_OWNERS_ATTR)
    if isinstance(owners, _ReboundTensorOwners):
        return owners
    namespace_collision = any(
        _REBOUND_TENSOR_OWNERS_ATTR in model.__dict__[namespace]
        for namespace in ("_parameters", "_buffers", "_modules")
    )
    class_collision = any(
        _REBOUND_TENSOR_OWNERS_ATTR in cls.__dict__ for cls in type(model).__mro__
    )
    if (
        _REBOUND_TENSOR_OWNERS_ATTR in model.__dict__
        or namespace_collision
        or class_collision
    ):
        raise RuntimeError(
            f"Reserved GMS attribute {_REBOUND_TENSOR_OWNERS_ATTR!r} "
            "collides with an existing model attribute"
        )
    owners = _ReboundTensorOwners()
    model.__dict__[_REBOUND_TENSOR_OWNERS_ATTR] = owners
    return owners


# =============================================================================
# Module Tensor Iteration
# =============================================================================


def _iter_module_tensors(
    module: torch.nn.Module,
    prefix: str = "",
) -> Iterator[Tuple[str, torch.Tensor, str]]:
    """Iterate over all CUDA tensors in a module tree.

    Yields (qualified_name, tensor, tensor_type) for:
    - Parameters (tensor_type="parameter")
    - Buffers (tensor_type="buffer")
    - Other tensor attributes like _k_scale (tensor_type="tensor_attr")

    Args:
        module: The nn.Module to iterate.
        prefix: Prefix for qualified names (used in recursion).

    Yields:
        (name, tensor, tensor_type) tuples for each CUDA tensor.
    """
    # Parameters
    for name, param in module._parameters.items():
        if param is not None and param.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, param, "parameter")

    # Buffers
    for name, buf in module._buffers.items():
        if buf is not None and buf.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, buf, "buffer")

    # Other tensor attributes explicitly stored on the module instance.
    # Do not inspect class descriptors: properties may return derived tensor
    # aliases that are not writable materialization targets.
    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
    )
    for attr_name, attr_val in module.__dict__.items():
        if attr_name in skip or attr_name.startswith("__"):
            continue

        if torch.is_tensor(attr_val) and attr_val.is_cuda:
            qualified = f"{prefix}{attr_name}" if prefix else attr_name
            yield (qualified, attr_val, "tensor_attr")
        elif isinstance(attr_val, (list, tuple)) and attr_val:
            if all(torch.is_tensor(x) and x.is_cuda for x in attr_val):
                for i, x in enumerate(attr_val):
                    qualified = (
                        f"{prefix}{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                    )
                    yield (qualified, x, "tensor_attr")

    # Recurse into submodules
    for name, submodule in module._modules.items():
        if submodule is not None:
            subprefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from _iter_module_tensors(submodule, subprefix)


def _resolve_module_attr(
    root: torch.nn.Module, qualified_name: str
) -> Tuple[torch.nn.Module, str]:
    """Resolve a dotted name to (parent_module, leaf_attr).

    Handles ModuleList/Sequential (numeric indices) and ModuleDict (key access).
    """
    parts = qualified_name.split(".")
    mod = root
    for p in parts[:-1]:
        if isinstance(mod, torch.nn.Module) and p in mod._modules:
            mod = mod._modules[p]
        elif isinstance(mod, torch.nn.Module) and p in mod.__dict__:
            value = mod.__dict__[p]
            if not isinstance(value, (list, tuple)):
                raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
            mod = value
        elif isinstance(mod, (list, tuple)) and p.isdigit():
            mod = mod[int(p)]
        else:
            raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
    return mod, parts[-1]


def _resolve_existing_tensor(
    model: torch.nn.Module,
    name: str,
    mod: object,
    attr: str,
    tensor_type: str,
) -> torch.Tensor | None:
    """Return the Tensor object at a supported direct module location."""
    if attr.isdigit() and isinstance(mod, (list, tuple)):
        owner, container_attr = _resolve_module_attr(model, name.rsplit(".", 1)[0])
        if isinstance(getattr(type(owner), container_attr, None), property):
            return None
        value = mod[int(attr)]
        return value if torch.is_tensor(value) else None

    if tensor_type == "buffer" and attr in mod._buffers:
        value = mod._buffers[attr]
        return value if torch.is_tensor(value) else None

    if isinstance(getattr(type(mod), attr, None), property):
        return None
    value = getattr(mod, attr, None)
    return value if torch.is_tensor(value) else None


def _swap_tensor_contents(
    existing: torch.Tensor,
    replacement: torch.Tensor,
    *,
    name: str,
) -> None:
    """Swap TensorImpls while preserving ``existing`` Python identity."""
    if torch.torch_version.TorchVersion(torch.__version__) < (2, 4) or not hasattr(
        torch.utils, "swap_tensors"
    ):
        raise RuntimeError(
            "GMS identity-preserving tensor materialization requires "
            "torch.utils.swap_tensors with TensorImpl use-count safety "
            "(PyTorch 2.4+)"
        )
    try:
        torch.utils.swap_tensors(existing, replacement)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot preserve Python identity for GMS tensor {name!r}: "
            f"torch.utils.swap_tensors failed: {exc}"
        ) from exc


def _copy_parameter_state(
    parameter: torch.nn.Parameter,
    replacement: torch.nn.Parameter,
    *,
    name: str,
) -> None:
    """Give an exact-class replacement the original Parameter's Python state."""
    try:
        replacement.__dict__ = parameter.__dict__
        for slot in copyreg._slotnames(type(parameter)) or ():
            if hasattr(parameter, slot):
                setattr(replacement, slot, getattr(parameter, slot))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot preserve Python state for GMS parameter {name!r} "
            f"of type {type(parameter).__name__}: {exc}"
        ) from exc


def _make_parameter_replacement(
    parameter: torch.nn.Parameter,
    tensor: torch.Tensor,
    *,
    name: str,
) -> torch.nn.Parameter:
    """Wrap ``tensor`` without invoking a heterogeneous Parameter constructor."""
    try:
        replacement = torch.Tensor._make_subclass(
            type(parameter), tensor, parameter.requires_grad
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot materialize GMS parameter {name!r} as "
            f"{type(parameter).__name__}: {exc}"
        ) from exc
    _copy_parameter_state(parameter, replacement, name=name)
    return replacement


def _validate_parameter_swap(
    parameter: torch.nn.Parameter,
    *,
    name: str,
) -> None:
    """Fail before reader mutation if an exact-class Parameter cannot be swapped."""
    if weakref.getweakrefs(parameter):
        raise RuntimeError(
            f"Cannot preserve Python identity for GMS parameter {name!r}: "
            "torch.utils.swap_tensors does not support weak references"
        )
    use_count = parameter._use_count()
    if use_count > 1:
        raise RuntimeError(
            f"Cannot preserve Python identity for GMS parameter {name!r}: "
            f"TensorImpl use count is {use_count}, expected 1"
        )
    probe = torch.empty(0, dtype=parameter.dtype, device="meta")
    _make_parameter_replacement(parameter, probe, name=name)


def _tensor_source(spec: GMSTensorSpec):
    return (
        spec.allocation_id,
        spec.offset_bytes,
        spec.meta.shape,
        spec.meta.stride,
        spec.meta.dtype,
    )


def _registered_parameters(
    model: torch.nn.Module,
) -> dict[int, torch.nn.Parameter]:
    """Index every registered Parameter by exact Python identity."""
    return {
        id(parameter): parameter
        for module in model.modules()
        for parameter in module._parameters.values()
        if parameter is not None
    }


def _validate_plain_nonparameter(tensor: torch.Tensor, *, name: str) -> None:
    if type(tensor) is not torch.Tensor:
        raise RuntimeError(
            f"GMS does not support non-parameter Tensor subclass at {name!r}: "
            f"{type(tensor).__name__}"
        )


def _validate_observed_storage(
    observed: dict[int, tuple[str, torch.Tensor]],
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Reject observed views or distinct objects sharing storage."""
    if tensor._base is not None:
        raise RuntimeError(f"GMS cannot materialize tensor view {name!r}")

    storage_ptr = int(tensor.untyped_storage().data_ptr())
    if storage_ptr == 0:
        return
    previous = observed.get(storage_ptr)
    if previous is not None and previous[1] is not tensor:
        raise RuntimeError(
            "GMS cannot materialize distinct tensors that share storage: "
            f"{previous[0]!r} and {name!r}"
        )
    observed[storage_ptr] = (name, tensor)


@dataclass
class _TensorDescriptor:
    name: str
    tensor: torch.Tensor
    meta: TensorMetadata
    buffer_persistent: bool
    object_group_id: int


@dataclass
class _StorageComponent:
    storage_group_id: int
    storage_nbytes: int
    members: list[_TensorDescriptor]
    allocation_id: str | None = None
    storage_base_offset: int | None = None


def _storage_impl_identity(tensor: torch.Tensor) -> int:
    """Return the private identity used only to compare StorageImpl topology."""
    return int(tensor.untyped_storage()._cdata)


def _validate_tensor_contract(
    tensor: torch.Tensor,
    *,
    name: str,
    is_parameter: bool,
) -> None:
    if tensor.is_conj() or tensor.is_neg():
        raise RuntimeError(
            f"GMS does not support lazy conjugate/negative tensor {name!r}"
        )
    element_size = tensor.element_size()
    if tensor.untyped_storage().data_ptr() % element_size:
        raise RuntimeError(f"GMS tensor storage is unaligned at {name!r}")
    _validate_layout(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().nbytes()),
    )
    if not is_parameter:
        _validate_plain_nonparameter(tensor, name=name)
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError(
                f"GMS inference storage does not support autograd tensor {name!r}"
            )
    elif not tensor.is_leaf or tensor.grad_fn is not None:
        raise RuntimeError(f"GMS only supports leaf Parameters at {name!r}")


def _discover_storage_components(
    model: torch.nn.Module,
) -> list[_StorageComponent]:
    entries = list(_iter_module_tensors(model))
    # PyTorch exposes no public StorageImpl identity. Keep this private value
    # isolated to grouping wrappers that must share one reconstructed storage.
    components_by_storage: dict[int, _StorageComponent] = {}
    object_groups: dict[int, int] = {}

    for name, tensor, tensor_type in entries:
        mod, attr = _resolve_module_attr(model, name)
        if _resolve_existing_tensor(model, name, mod, attr, tensor_type) is not tensor:
            continue
        _validate_tensor_contract(
            tensor,
            name=name,
            is_parameter=isinstance(tensor, torch.nn.Parameter),
        )
        object_group_id = object_groups.setdefault(id(tensor), len(object_groups))

        storage = tensor.untyped_storage()
        storage_id = _storage_impl_identity(tensor)
        component = components_by_storage.get(storage_id)
        if component is None:
            component = _StorageComponent(
                storage_group_id=len(components_by_storage),
                storage_nbytes=int(storage.nbytes()),
                members=[],
            )
            components_by_storage[storage_id] = component
        component.members.append(
            _TensorDescriptor(
                name=name,
                tensor=tensor,
                meta=TensorMetadata.from_tensor(tensor, tensor_type),
                buffer_persistent=(
                    tensor_type == "buffer"
                    and attr not in mod._non_persistent_buffers_set
                ),
                object_group_id=object_group_id,
            )
        )
    return list(components_by_storage.values())


def _locate_components(
    components: list[_StorageComponent],
    mappings: dict[int, object],
    *,
    require_parameters: bool,
) -> list[_StorageComponent]:
    located: list[_StorageComponent] = []
    for component in components:
        storage_ptr = int(component.members[0].tensor.untyped_storage().data_ptr())
        for va, mapping in mappings.items():
            end = storage_ptr + component.storage_nbytes
            mapping_end = va + mapping.aligned_size
            if max(va, storage_ptr) < min(mapping_end, end) and not (
                va <= storage_ptr and end <= mapping_end
            ):
                raise RuntimeError(
                    f"Storage component {component.storage_group_id} exceeds "
                    f"GMS allocation {mapping.allocation_id!r}"
                )
            if va <= storage_ptr and end <= mapping_end:
                component.allocation_id = str(mapping.allocation_id)
                component.storage_base_offset = storage_ptr - va
                located.append(component)
                break
        else:
            if require_parameters and any(
                isinstance(member.tensor, torch.nn.Parameter)
                for member in component.members
            ):
                raise RuntimeError(
                    f"Tensor {component.members[0].name!r} not found in any "
                    "GMS allocation"
                )
            logger.debug(
                "[GMS] Skipping storage component for %r outside GMS allocations",
                component.members[0].name,
            )
    _validate_component_intervals(located)
    return located


def _validate_component_intervals(components: list[_StorageComponent]) -> None:
    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for component in components:
        start = int(component.storage_base_offset)
        grouped.setdefault(str(component.allocation_id), []).append(
            (start, start + component.storage_nbytes, component.storage_group_id)
        )
    for allocation_id, intervals in grouped.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] > current[0]:
                raise RuntimeError(
                    "Distinct StorageImpl byte ranges overlap in allocation "
                    f"{allocation_id!r}: storage groups "
                    f"{previous[2]} and {current[2]}"
                )


def _unique_members(
    components: list[_StorageComponent],
) -> list[_TensorDescriptor]:
    return list(
        {
            member.object_group_id: member
            for component in components
            for member in component.members
        }.values()
    )


def _swap_order(members: list[_TensorDescriptor]) -> list[_TensorDescriptor]:
    tensors = {id(member.tensor) for member in members}

    def depth(member: _TensorDescriptor) -> int:
        result = 0
        base = member.tensor._base
        while base is not None and id(base) in tensors:
            result += 1
            base = base._base
        return result

    return sorted(members, key=depth, reverse=True)


def _has_parameter_accumulator_owner(
    tensor: torch.Tensor,
    expected_use_count: int,
) -> bool:
    """Disambiguate one AccumulateGrad owner from an unknown TensorImpl owner."""
    if not (
        isinstance(tensor, torch.nn.Parameter)
        and tensor.requires_grad
        and tensor.is_leaf
        and tensor._use_count() == expected_use_count + 1
    ):
        return False
    gradient_edge = torch.autograd.graph.get_gradient_edge(tensor)
    return (
        gradient_edge.node is not None and tensor._use_count() == expected_use_count + 1
    )


def _validate_swap_ownership(members: list[_TensorDescriptor]) -> None:
    tensors = {id(member.tensor): member.tensor for member in members}
    child_counts: dict[int, int] = {}
    for member in members:
        base = member.tensor._base
        if base is not None and tensors.get(id(base)) is base:
            child_counts[id(base)] = child_counts.get(id(base), 0) + 1

    # TensorImpl use counts are private, but public swap_tensors enforces them.
    # Comparing here lets the complete operation fail before any mutation.
    for member in members:
        tensor = member.tensor
        if weakref.getweakrefs(tensor):
            raise RuntimeError(
                f"Cannot preserve Python identity for GMS tensor {member.name!r}: "
                "torch.utils.swap_tensors does not support weak references"
            )
        expected = 1 + child_counts.get(id(tensor), 0)
        use_count = tensor._use_count()
        parameter_accumulator = _has_parameter_accumulator_owner(tensor, expected)
        if use_count != expected and not parameter_accumulator:
            raise RuntimeError(
                f"Cannot safely swap GMS tensor {member.name!r}: TensorImpl use "
                f"count is {use_count}, expected {expected} from discovered tensors"
            )


def _replacement_for(
    member: _TensorDescriptor,
    storage: torch.UntypedStorage,
) -> torch.Tensor:
    tensor = _tensor_from_storage(
        storage,
        list(member.meta.shape),
        list(member.meta.stride),
        member.meta.dtype,
        int(member.tensor.storage_offset())
        if member.meta.storage is None
        else int(member.meta.storage["storage_offset"]),
    )
    if isinstance(member.tensor, torch.nn.Parameter):
        return _make_parameter_replacement(member.tensor, tensor, name=member.name)
    return tensor


def _apply_replacements(
    members: list[_TensorDescriptor],
    replacements: dict[int, torch.Tensor],
) -> None:
    """Swap child-first; failures are terminal and the caller discards the model."""
    for member in _swap_order(members):
        replacement = replacements.pop(member.object_group_id)
        _swap_tensor_contents(member.tensor, replacement, name=member.name)
        del replacement


# =============================================================================
# Public API - Registration and Materialization
# =============================================================================


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> set[str]:
    """Register all model tensors into the GMS metadata store.

    Args:
        gms_client_memory_manager: GMS client memory manager in write mode.
        model: PyTorch model to register.

    Returns:
        Allocation IDs referenced by registered tensors.
    """
    components = _locate_components(
        _discover_storage_components(model),
        gms_client_memory_manager.mappings,
        require_parameters=True,
    )
    _validate_swap_ownership(_unique_members(components))
    referenced_allocation_ids: set[str] = set()
    for component in components:
        assert component.allocation_id is not None
        assert component.storage_base_offset is not None
        referenced_allocation_ids.add(component.allocation_id)
        for member in component.members:
            element_size = torch.empty((), dtype=member.meta.dtype).element_size()
            offset_bytes = (
                component.storage_base_offset
                + member.tensor.storage_offset() * element_size
            )
            meta = TensorMetadata.from_tensor(
                member.tensor,
                member.meta.tensor_type,
                storage={
                    "schema_version": 1,
                    "storage_group_id": component.storage_group_id,
                    "object_group_id": member.object_group_id,
                    "storage_base_offset": component.storage_base_offset,
                    "storage_nbytes": component.storage_nbytes,
                    "storage_offset": int(member.tensor.storage_offset()),
                    "buffer_persistent": member.buffer_persistent,
                },
            )
            if not gms_client_memory_manager.metadata_put(
                key=member.name,
                allocation_id=component.allocation_id,
                offset_bytes=offset_bytes,
                value=meta.to_bytes(),
            ):
                raise RuntimeError(f"Failed to register GMS tensor {member.name!r}")
    return referenced_allocation_ids


def _resolve_slot(
    model: torch.nn.Module,
    name: str,
    meta: TensorMetadata,
) -> torch.Tensor:
    mod, attr = _resolve_module_attr(model, name)
    if meta.tensor_type == "parameter":
        if not isinstance(mod, torch.nn.Module) or attr not in mod._parameters:
            raise RuntimeError(
                f"GMS tensor {name!r} is a Parameter but the reader slot is not"
            )
        tensor = mod._parameters[attr]
        if not isinstance(tensor, torch.nn.Parameter):
            raise RuntimeError(f"Reader Parameter slot {name!r} is empty or invalid")
    elif meta.tensor_type == "buffer":
        if not isinstance(mod, torch.nn.Module) or attr not in mod._buffers:
            raise RuntimeError(
                f"GMS tensor {name!r} is a buffer but the reader slot is not"
            )
        tensor = mod._buffers[attr]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Reader buffer slot {name!r} is empty or invalid")
        persistent = attr not in mod._non_persistent_buffers_set
        if meta.storage is not None and persistent != meta.storage["buffer_persistent"]:
            raise RuntimeError(f"Reader buffer persistence differs at {name!r}")
    else:
        tensor = _resolve_existing_tensor(model, name, mod, attr, meta.tensor_type)
        if tensor is None:
            raise RuntimeError(
                f"GMS tensor attribute {name!r} requires a preexisting direct "
                "Tensor object"
            )
        if isinstance(mod, torch.nn.Module) and (
            attr in mod._parameters or attr in mod._buffers
        ):
            raise RuntimeError(f"Reader tensor attribute slot differs at {name!r}")
    return tensor


def _components_from_specs(
    specs: dict[str, GMSTensorSpec],
    model: torch.nn.Module,
) -> list[_StorageComponent]:
    if any(not spec.meta.is_storage_component for spec in specs.values()):
        raise RuntimeError("Cannot mix legacy and storage-component GMS metadata")
    components: dict[tuple[str, int], _StorageComponent] = {}
    objects: dict[int, torch.Tensor] = {}
    reader_objects: dict[int, int] = {}
    object_sources: dict[int, tuple[str, int]] = {}
    object_layouts: dict[int, tuple[object, ...]] = {}
    component_storages: dict[tuple[str, int], int] = {}
    storage_components: dict[int, tuple[str, int]] = {}

    for name, spec in specs.items():
        meta = spec.meta
        tensor = _resolve_slot(model, name, meta)
        if tuple(tensor.shape) != meta.shape or tensor.dtype != meta.dtype:
            raise RuntimeError(
                f"Shape/dtype mismatch for {name}: "
                f"existing={tuple(tensor.shape)}/{tensor.dtype}, "
                f"gms={meta.shape}/{meta.dtype}"
            )
        _validate_tensor_contract(
            tensor,
            name=name,
            is_parameter=isinstance(tensor, torch.nn.Parameter),
        )

        assert meta.storage is not None
        if meta.tensor_type != "buffer" and meta.storage["buffer_persistent"]:
            raise RuntimeError(f"Invalid buffer persistence at GMS tensor {name!r}")
        storage_group_id = int(meta.storage["storage_group_id"])
        object_group_id = int(meta.storage["object_group_id"])
        storage_base_offset = int(meta.storage["storage_base_offset"])
        storage_offset = int(meta.storage["storage_offset"])
        storage_nbytes = int(meta.storage["storage_nbytes"])
        element_size = torch.empty((), dtype=meta.dtype).element_size()
        expected_offset = storage_base_offset + storage_offset * element_size
        if expected_offset % element_size:
            raise RuntimeError(f"Unaligned GMS tensor metadata at {name!r}")
        if spec.offset_bytes != expected_offset:
            raise RuntimeError(
                f"GMS legacy offset disagrees with tensor layout at {name!r}"
            )
        try:
            _validate_layout(
                meta.shape,
                meta.stride,
                meta.dtype,
                storage_offset,
                storage_nbytes,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid GMS tensor metadata at {name!r}: {exc}"
            ) from exc

        previous_object = objects.setdefault(object_group_id, tensor)
        previous_reader = reader_objects.setdefault(id(tensor), object_group_id)
        layout = (meta.shape, meta.stride, meta.dtype, storage_offset)
        previous_layout = object_layouts.setdefault(object_group_id, layout)
        if previous_object is not tensor:
            raise RuntimeError(
                f"Reader exact-alias topology differs for GMS tensor {name!r}"
            )
        if previous_reader != object_group_id or previous_layout != layout:
            raise RuntimeError(
                f"Reader tensor aliases incompatible GMS entries at {name!r}"
            )

        component_key = (spec.allocation_id, storage_group_id)
        previous_source = object_sources.setdefault(object_group_id, component_key)
        if previous_source != component_key:
            raise RuntimeError(
                f"Reader tensor aliases incompatible GMS entries at {name!r}"
            )
        storage_identity = _storage_impl_identity(tensor)
        previous_storage = component_storages.setdefault(
            component_key, storage_identity
        )
        previous_component = storage_components.setdefault(
            storage_identity, component_key
        )
        if previous_storage != storage_identity or previous_component != component_key:
            raise RuntimeError(
                f"Reader StorageImpl topology differs at GMS tensor {name!r}"
            )
        component = components.get(component_key)
        if component is None:
            component = _StorageComponent(
                storage_group_id=storage_group_id,
                storage_nbytes=storage_nbytes,
                members=[],
                allocation_id=spec.allocation_id,
                storage_base_offset=storage_base_offset,
            )
            components[component_key] = component
        elif (
            component.storage_nbytes != storage_nbytes
            or component.storage_base_offset != storage_base_offset
        ):
            raise RuntimeError(f"Inconsistent GMS storage component at tensor {name!r}")
        component.members.append(
            _TensorDescriptor(
                name=name,
                tensor=tensor,
                meta=meta,
                buffer_persistent=bool(meta.storage["buffer_persistent"]),
                object_group_id=object_group_id,
            )
        )
    result = list(components.values())
    _validate_component_intervals(result)
    return result


def _mapped_allocations(
    manager: "GMSClientMemoryManager",
    components: list[_StorageComponent],
) -> tuple[dict[str, int], list[int]]:
    existing = {
        str(mapping.allocation_id): int(va) for va, mapping in manager.mappings.items()
    }
    allocation_ids = {component.allocation_id for component in components}
    mapped: dict[str, int] = {}
    imported: list[int] = []
    try:
        for allocation_id in allocation_ids:
            assert allocation_id is not None
            if allocation_id in existing:
                va = existing[allocation_id]
            else:
                va = manager.create_mapping(allocation_id=allocation_id)
                imported.append(va)
            mapped[allocation_id] = va
            allocation_nbytes = int(manager.mappings[va].aligned_size)
            for component in components:
                if component.allocation_id != allocation_id:
                    continue
                assert component.storage_base_offset is not None
                if (
                    component.storage_base_offset < 0
                    or component.storage_base_offset + component.storage_nbytes
                    > allocation_nbytes
                ):
                    raise RuntimeError(
                        f"GMS storage component {component.storage_group_id} "
                        f"exceeds allocation {allocation_id!r}"
                    )
    except BaseException:
        for va in reversed(imported):
            manager.free_va(va)
        raise
    return mapped, imported


def _materialize_component_specs(
    manager: "GMSClientMemoryManager",
    specs: dict[str, GMSTensorSpec],
    model: torch.nn.Module,
    device_index: int,
) -> None:
    components = _components_from_specs(specs, model)
    members = _unique_members(components)
    _validate_swap_ownership(members)
    mapped, imported = _mapped_allocations(manager, components)
    replacements: dict[int, torch.Tensor] = {}
    try:
        for component in components:
            assert component.allocation_id is not None
            assert component.storage_base_offset is not None
            source_storage = _storage_from_pointer(
                mapped[component.allocation_id] + component.storage_base_offset,
                component.storage_nbytes,
                device_index,
            )
            if any(
                isinstance(member.tensor, torch.nn.Parameter)
                for member in component.members
            ):
                target_storage = source_storage
            else:
                target_storage = (
                    torch.empty(0, dtype=torch.uint8, device=source_storage.device)
                    .set_(source_storage, 0, (source_storage.nbytes(),), (1,))
                    .clone()
                    .untyped_storage()
                )
            for member in _unique_members([component]):
                replacements[member.object_group_id] = _replacement_for(
                    member, target_storage
                )
        _apply_replacements(members, replacements)
    except BaseException:
        for va in reversed(imported):
            manager.free_va(va)
        raise


def materialize_module_from_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
) -> None:
    """Materialize model tensors from GMS.

    A failure after tensor mutation begins is terminal; discard the model.

    Args:
        gms_client_memory_manager: GMS client memory manager in read mode.
        model: Model to populate with tensors.
        device_index: CUDA device index.
    """
    specs = GMSTensorSpec.load_all(gms_client_memory_manager)
    if any(spec.meta.is_storage_component for spec in specs.values()):
        _materialize_component_specs(
            gms_client_memory_manager, specs, model, device_index
        )
        return

    parameters = _registered_parameters(model)
    aliases = {}
    resolved = []
    observed_storage = {}
    validated_parameters = set()
    target_device = torch.device("cuda", device_index)
    for name, spec in specs.items():
        mod, attr = _resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type
        existing = _resolve_slot(model, name, spec.meta)
        parameter = parameters.get(id(existing))
        if existing.shape != spec.meta.shape or existing.dtype != spec.meta.dtype:
            raise RuntimeError(
                f"Shape/dtype mismatch for {name}: "
                f"existing={tuple(existing.shape)}/{existing.dtype}, "
                f"gms={spec.meta.shape}/{spec.meta.dtype}"
            )
        source = _tensor_source(spec)
        previous = aliases.get(id(existing))
        if previous is not None and previous[0] is existing:
            if previous[1] != source:
                raise RuntimeError(
                    f"Reader tensor aliases incompatible GMS entries at {name!r}"
                )
        else:
            aliases[id(existing)] = (existing, source)
            if parameter is None:
                _validate_plain_nonparameter(existing, name=name)
            _validate_observed_storage(observed_storage, name=name, tensor=existing)
        if (
            parameter is not None
            and id(parameter) not in validated_parameters
            and (parameter.is_meta or parameter.device != target_device)
        ):
            _validate_parameter_swap(parameter, name=name)
            validated_parameters.add(id(parameter))
        resolved.append((name, spec, existing, parameter))

    materialized = set()
    for name, spec, existing, parameter in resolved:
        if id(existing) in materialized:
            continue
        tensor_type = spec.meta.tensor_type
        if tensor_type in ("tensor_attr", "buffer") and parameter is None:
            tensor = spec.materialize(gms_client_memory_manager, device_index)
            private = tensor.detach().clone()
            _swap_tensor_contents(existing, private, name=name)
            materialized.add(id(existing))
            continue

        tensor = spec.materialize(gms_client_memory_manager, device_index)
        if parameter is not None:
            if parameter.is_meta or parameter.device != tensor.device:
                replacement = _make_parameter_replacement(parameter, tensor, name=name)
                _swap_tensor_contents(parameter, replacement, name=name)
            else:
                parameter.data = tensor
            materialized.add(id(parameter))
            continue
    # Check for meta tensors and warn
    meta_tensors = [n for n, p in model.named_parameters() if p.is_meta]
    meta_tensors += [n for n, b in model.named_buffers() if b.is_meta]
    if meta_tensors:
        logger.warning(
            "[GMS] %d meta tensors not in metadata: %s",
            len(meta_tensors),
            meta_tensors[:10],
        )


def rebind_nonparameter_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    retain_gms_tensors: list[torch.Tensor] | None = None,
) -> int:
    """Re-bind GMS-resident non-parameter tensors to private clones.

    The publisher builds the whole model inside the GMS memory pool, so
    buffers and tensor attributes (fp8 KV scales, quantization ranges, ...)
    land in the same committed allocations as the weights, which are
    remapped read-only after publish. Unlike parameters, these tensors can
    be written after load (for example ``init_fp8_kv_scales`` on wake),
    which faults on the read-only mapping. Cloning them into ordinary CUDA
    memory gives the publisher the same binding semantics importers get
    from ``materialize_module_from_gms``: parameters stay on the shared
    read-only mapping, everything else is private and writable. The GMS
    copies stay registered so importers can still materialize from them.

    Must run before CUDA graph capture: the clones live at new addresses.

    Discoverable exact aliases keep their original Python Tensor object.
    Distinct discovered tensors sharing one StorageImpl are rebuilt together
    over one private storage copy. Discovery remains limited to registered
    fields and direct module instance attributes (including tensor lists/tuples).

    Returns the number of bytes rebound, i.e. how much memory is duplicated
    between the read-only GMS copies and the private clones.

    One detached storage owner per copied component retains the original GMS
    allocation. If ``retain_gms_tensors`` is provided, each owner is also
    appended for compatibility with deferred publication.

    A failure after tensor mutation begins is terminal; discard the model.
    """
    components = _locate_components(
        _discover_storage_components(model),
        gms_client_memory_manager.mappings,
        require_parameters=False,
    )
    candidates = [
        component
        for component in components
        if not any(
            isinstance(member.tensor, torch.nn.Parameter)
            for member in component.members
        )
    ]
    if not candidates:
        return 0
    members = _unique_members(candidates)
    _validate_swap_ownership(members)
    replacements: dict[int, torch.Tensor] = {}
    source_owners: list[torch.Tensor] = []
    for component in candidates:
        source_storage = component.members[0].tensor.untyped_storage()
        source_owner = torch.empty(
            0, dtype=torch.uint8, device=source_storage.device
        ).set_(
            source_storage,
            0,
            (component.storage_nbytes,),
            (1,),
        )
        target_storage = source_owner.clone().untyped_storage()
        source_owners.append(source_owner)
        for member in _unique_members([component]):
            replacements[member.object_group_id] = _replacement_for(
                member, target_storage
            )

    owners = _get_or_create_rebound_tensor_owners(model)
    owner_count = len(owners.tensors)
    retained_count = len(retain_gms_tensors) if retain_gms_tensors is not None else 0
    try:
        _apply_replacements(members, replacements)
    except BaseException:
        del owners.tensors[owner_count:]
        if retain_gms_tensors is not None:
            del retain_gms_tensors[retained_count:]
        if owner_count == 0 and not owners.tensors:
            model.__dict__.pop(_REBOUND_TENSOR_OWNERS_ATTR, None)
        raise

    owners.tensors.extend(source_owners)
    if retain_gms_tensors is not None:
        retain_gms_tensors.extend(source_owners)
    return sum(component.storage_nbytes for component in candidates)
