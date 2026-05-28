# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module tensor operations for GPU Memory Service.

This module provides module-level tensor operations:
- Module tensor iteration
- Tensor registration (write path)
- Tensor materialization (read path)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from types import FunctionType, MethodType, ModuleType
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Tuple, cast

import torch
from gpu_memory_service.client.torch.tensor import (
    GMSTensorSpec,
    TensorMetadata,
    _tensor_from_pointer,
)

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)
_EMPTY_PARAMETER_TENSOR_TYPE = "empty_parameter"


# =============================================================================
# Module Tensor Iteration
# =============================================================================


def _is_zero_size_tensor(tensor: torch.Tensor) -> bool:
    """True when a tensor has no payload to register with GMS.

    vLLM uses zero-size CUDA parameters as kernel placeholders for optional
    metadata such as non-actorder g_idx/sort-index tensors. They have no data to
    share through GMS, and their pointers may not correspond to a real
    allocation.
    """
    if tensor.numel() == 0:
        return True
    try:
        return int(tensor.untyped_storage().nbytes()) == 0
    except Exception:
        return False


def _empty_like_on_device(tensor: torch.Tensor, *, device_index: int) -> torch.Tensor:
    return torch.empty_strided(
        tuple(tensor.shape),
        tuple(int(s) for s in tensor.stride()),
        dtype=tensor.dtype,
        device=torch.device("cuda", device_index),
    )


def _storage_nbytes_for_shape_stride(tensor: torch.Tensor) -> int:
    element_size = torch.tensor([], dtype=tensor.dtype).element_size()
    shape = tuple(tensor.shape)
    stride = tuple(int(s) for s in tensor.stride())
    if shape and stride:
        max_offset = sum(
            s * (d - 1) for s, d in zip(stride, shape, strict=True) if d > 0
        )
        return (max_offset + 1) * element_size
    return element_size


class _GMSMappingMatch(NamedTuple):
    va: int
    mapping: Any
    offset_bytes: int


class _TensorAliasKey:
    """Identity key for tensor aliases captured before materialization."""

    __slots__ = ("tensor", "_id", "shape", "stride", "dtype")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self._id = id(tensor)
        self.shape = tuple(tensor.shape)
        self.stride = tuple(int(s) for s in tensor.stride())
        self.dtype = tensor.dtype

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _TensorAliasKey)
            and self.tensor is other.tensor
            and self.shape == other.shape
            and self.stride == other.stride
            and self.dtype == other.dtype
        )


_TensorAliasMap = dict[_TensorAliasKey, torch.Tensor]


class _AuxTensorSnapshot(NamedTuple):
    allocation_id: str
    offset_bytes: int
    tensor: torch.Tensor


def _get_tensor_alias(
    alias_map: _TensorAliasMap, tensor: torch.Tensor
) -> torch.Tensor | None:
    return alias_map.get(_TensorAliasKey(tensor))


def _set_tensor_alias(
    alias_map: _TensorAliasMap, source: torch.Tensor, replacement: torch.Tensor
) -> None:
    alias_map[_TensorAliasKey(source)] = replacement


def _tuple_like(value: tuple, items: list[Any]) -> tuple:
    if hasattr(value, "_fields"):
        return type(value)(*items)
    return tuple(items)


def _find_gms_mapping(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
) -> _GMSMappingMatch | None:
    ptr = int(tensor.data_ptr())
    for va, mapping in gms_client_memory_manager.mappings.items():
        if va <= ptr < va + mapping.aligned_size:
            return _GMSMappingMatch(va, mapping, ptr - va)
    return None


def _put_tensor_metadata(
    gms_client_memory_manager: "GMSClientMemoryManager",
    *,
    key: str,
    allocation_id: str,
    offset_bytes: int,
    tensor: torch.Tensor,
    tensor_type: str,
) -> None:
    meta = TensorMetadata.from_tensor(tensor, tensor_type)
    gms_client_memory_manager.metadata_put(
        key=key,
        allocation_id=allocation_id,
        offset_bytes=offset_bytes,
        value=meta.to_bytes(),
    )


def _descriptor_allocation_id(
    gms_client_memory_manager: "GMSClientMemoryManager",
) -> str:
    """Return a tiny backing allocation used only to satisfy metadata linkage."""
    allocation_id = getattr(
        gms_client_memory_manager,
        "_tensor_descriptor_allocation_id",
        None,
    )
    if allocation_id is not None:
        return str(allocation_id)

    va = gms_client_memory_manager.create_mapping(size=1, tag="weights")
    allocation_id = gms_client_memory_manager.mappings[va].allocation_id
    setattr(
        gms_client_memory_manager,
        "_tensor_descriptor_allocation_id",
        allocation_id,
    )
    return str(allocation_id)


def _runtime_cuda_copy(
    tensor: torch.Tensor,
    *,
    device_index: int,
) -> torch.Tensor:
    device = getattr(tensor, "device", torch.device("cuda", device_index))
    if getattr(device, "type", None) != "cuda":
        device = torch.device("cuda", device_index)
    replacement = torch.empty_strided(
        tuple(tensor.shape),
        tuple(int(s) for s in tensor.stride()),
        dtype=tensor.dtype,
        device=device,
    )
    replacement.copy_(tensor.detach())
    if replacement.is_cuda:
        torch.cuda.synchronize(replacement.device)
    _copy_tensor_attrs(tensor, replacement)
    return replacement


def _move_aux_tensor_out_of_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
    *,
    device_index: int,
) -> torch.Tensor:
    if _find_gms_mapping(gms_client_memory_manager, tensor) is None:
        return tensor

    replacement = _runtime_cuda_copy(tensor, device_index=device_index)
    if _find_gms_mapping(gms_client_memory_manager, replacement) is not None:
        raise RuntimeError(
            "Auxiliary tensor copies must be allocated outside the GMS mempool"
        )
    return replacement


def _keep_aux_snapshot_alive(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
) -> None:
    refs = getattr(gms_client_memory_manager, "_aux_tensor_refs", None)
    if refs is None:
        refs = []
        setattr(gms_client_memory_manager, "_aux_tensor_refs", refs)
    refs.append(tensor)


def _snapshot_aux_tensor_to_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
    *,
    device_index: int,
) -> _AuxTensorSnapshot:
    va = gms_client_memory_manager.create_mapping(
        size=_storage_nbytes_for_shape_stride(tensor),
        tag="weights",
    )
    mapping = gms_client_memory_manager.mappings[va]
    snapshot = _tensor_from_pointer(
        va,
        list(tensor.shape),
        [int(s) for s in tensor.stride()],
        tensor.dtype,
        device_index,
    )
    snapshot.copy_(tensor.detach())
    if snapshot.is_cuda:
        torch.cuda.synchronize(snapshot.device)

    _keep_aux_snapshot_alive(gms_client_memory_manager, snapshot)
    return _AuxTensorSnapshot(
        allocation_id=mapping.allocation_id,
        offset_bytes=0,
        tensor=snapshot,
    )


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

    # Only inspect instance attrs. `dir(module)` also includes class
    # properties, including read-only aliases over registered buffers.
    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
        | {"_parameters", "_buffers", "_modules"}
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
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif hasattr(mod, "__getitem__"):
            try:
                mod = mod[int(p)] if p.isdigit() else mod[p]
            except Exception:
                raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
        else:
            raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
    return mod, parts[-1]


def _child_for_path(owner: Any, part: str) -> Any:
    if isinstance(owner, (list, tuple)):
        return owner[int(part)]
    if isinstance(owner, Mapping):
        if part in owner:
            return owner[part]
        if part.isdigit() and int(part) in owner:
            return owner[int(part)]
        raise AttributeError(f"Cannot resolve {part!r} in mapping")
    if hasattr(owner, part):
        return getattr(owner, part)
    if hasattr(owner, "__getitem__"):
        try:
            return owner[int(part)] if part.isdigit() else owner[part]
        except Exception:
            pass
    raise AttributeError(f"Cannot resolve {part!r}")


def _assign_child_for_path(owner: Any, part: str, value: Any) -> Any:
    if isinstance(owner, list):
        owner[int(part)] = value
        return owner
    if isinstance(owner, tuple):
        items = list(owner)
        items[int(part)] = value
        return _tuple_like(owner, items)
    if isinstance(owner, MutableMapping):
        key: Any = part
        if part not in owner and part.isdigit() and int(part) in owner:
            key = int(part)
        owner[key] = value
        return owner
    if isinstance(owner, torch.nn.Module) and part in owner._buffers:
        owner._buffers[part] = value
        return owner
    setattr(owner, part, value)
    return owner


def _path_value(root: Any, qualified_name: str) -> Any:
    value = root
    for part in qualified_name.split("."):
        value = _child_for_path(value, part)
    return value


def _path_parent(root: Any, qualified_name: str) -> tuple[Any, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _child_for_path(parent, part)
    return parent, parts[-1]


def _set_path_value(root: Any, qualified_name: str, value: Any) -> None:
    parts = qualified_name.split(".")

    def replace(owner: Any, index: int) -> Any:
        part = parts[index]
        if index == len(parts) - 1:
            return _assign_child_for_path(owner, part, value)

        child = _child_for_path(owner, part)
        new_child = replace(child, index + 1)
        if new_child is child:
            return owner
        return _assign_child_for_path(owner, part, new_child)

    replace(root, 0)


def _is_read_only_property_path(root: Any, qualified_name: str) -> bool:
    parts = qualified_name.split(".")
    owner = root
    try:
        for part in parts[:-1]:
            owner = _child_for_path(owner, part)
    except AttributeError:
        return False

    descriptor = getattr(type(owner), parts[-1], None)
    return isinstance(descriptor, property) and descriptor.fset is None


def _materialize_zero_size_meta_tensors(
    module: torch.nn.Module,
    *,
    device_index: int,
    alias_map: _TensorAliasMap,
    prefix: str = "",
) -> list[str]:
    """Replace skipped zero-size meta tensors with equivalent CUDA tensors."""
    materialized: list[str] = []

    for name, param in module._parameters.items():
        if param is None or not param.is_meta or not _is_zero_size_tensor(param):
            continue
        qualified = f"{prefix}{name}" if prefix else name
        tensor = _empty_like_on_device(param, device_index=device_index)
        replacement = torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
        _set_tensor_alias(alias_map, param, replacement)
        module._parameters[name] = replacement
        materialized.append(qualified)

    for name, buf in module._buffers.items():
        if buf is None or not buf.is_meta or not _is_zero_size_tensor(buf):
            continue
        qualified = f"{prefix}{name}" if prefix else name
        replacement = _empty_like_on_device(buf, device_index=device_index)
        _set_tensor_alias(alias_map, buf, replacement)
        module._buffers[name] = replacement
        materialized.append(qualified)

    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
        | {"_parameters", "_buffers", "_modules"}
    )
    for attr_name, attr_val in list(module.__dict__.items()):
        if attr_name in skip or attr_name.startswith("__"):
            continue

        qualified = f"{prefix}{attr_name}" if prefix else attr_name
        if (
            torch.is_tensor(attr_val)
            and attr_val.is_meta
            and _is_zero_size_tensor(attr_val)
        ):
            replacement = _empty_like_on_device(attr_val, device_index=device_index)
            _set_tensor_alias(alias_map, attr_val, replacement)
            try:
                setattr(module, attr_name, replacement)
            except AttributeError:
                logger.debug(
                    "[GMS] Skipping read-only zero-size meta tensor attr %r",
                    qualified,
                )
                continue
            materialized.append(qualified)
        elif isinstance(attr_val, list):
            updated = list(attr_val)
            changed = False
            for i, value in enumerate(updated):
                if (
                    torch.is_tensor(value)
                    and value.is_meta
                    and _is_zero_size_tensor(value)
                ):
                    replacement = _empty_like_on_device(
                        value, device_index=device_index
                    )
                    _set_tensor_alias(alias_map, value, replacement)
                    updated[i] = replacement
                    materialized.append(f"{qualified}.{i}")
                    changed = True
            if changed:
                setattr(module, attr_name, updated)
        elif isinstance(attr_val, tuple):
            updated_tuple = []
            changed = False
            for i, value in enumerate(attr_val):
                if (
                    torch.is_tensor(value)
                    and value.is_meta
                    and _is_zero_size_tensor(value)
                ):
                    replacement = _empty_like_on_device(
                        value, device_index=device_index
                    )
                    _set_tensor_alias(alias_map, value, replacement)
                    updated_tuple.append(replacement)
                    materialized.append(f"{qualified}.{i}")
                    changed = True
                else:
                    updated_tuple.append(value)
            if changed:
                setattr(module, attr_name, tuple(updated_tuple))

    for name, submodule in module._modules.items():
        if submodule is not None:
            subprefix = f"{prefix}{name}." if prefix else f"{name}."
            materialized.extend(
                _materialize_zero_size_meta_tensors(
                    submodule,
                    device_index=device_index,
                    alias_map=alias_map,
                    prefix=subprefix,
                )
            )

    return materialized


def _copy_tensor_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    attrs = getattr(src, "__dict__", None)
    if attrs:
        dst.__dict__.update(attrs)


def _metadata_shape_stride_dtype(
    meta: TensorMetadata,
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    return (
        tuple(int(d) for d in meta.shape),
        tuple(int(s) for s in meta.stride),
        meta.dtype,
    )


def _tensor_matches_metadata(tensor: torch.Tensor, meta: TensorMetadata) -> bool:
    shape, stride, dtype = _metadata_shape_stride_dtype(meta)
    return (
        tuple(tensor.shape) == shape
        and tuple(int(s) for s in tensor.stride()) == stride
        and tensor.dtype == dtype
    )


def _empty_tensor_from_metadata(
    meta: TensorMetadata,
    *,
    device_index: int,
) -> torch.Tensor:
    shape, stride, dtype = _metadata_shape_stride_dtype(meta)
    return torch.empty_strided(
        shape,
        stride,
        dtype=dtype,
        device=torch.device("cuda", device_index),
    )


def _clone_mutable_gms_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    tensor_type: str,
    spec: GMSTensorSpec,
) -> torch.Tensor:
    try:
        cloned = tensor.detach().clone()
        if cloned.is_cuda:
            torch.cuda.synchronize(cloned.device)
        return cloned
    except Exception as e:
        raise RuntimeError(
            f"Failed to materialize mutable {tensor_type} {name!r} from GMS: "
            f"shape={tuple(spec.meta.shape)}, dtype={spec.meta.dtype}, "
            f"stride={tuple(spec.meta.stride)}, "
            f"allocation_id={spec.allocation_id!r}, "
            f"offset_bytes={int(spec.offset_bytes)}"
        ) from e


def _replace_aliases_in_value(
    value: Any,
    alias_map: _TensorAliasMap,
    *,
    visited: set[int],
    depth: int,
) -> tuple[Any, int]:
    if torch.is_tensor(value):
        replacement = _get_tensor_alias(alias_map, value)
        if replacement is not None:
            return replacement, 1
        return value, 0

    if depth <= 0 or value is None:
        return value, 0

    if isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            FunctionType,
            MethodType,
        ),
    ) or isinstance(value, type):
        return value, 0

    value_id = id(value)
    if value_id in visited:
        return value, 0
    visited.add(value_id)

    if isinstance(value, list):
        changed = 0
        for idx, item in enumerate(value):
            new_item, item_changed = _replace_aliases_in_value(
                item, alias_map, visited=visited, depth=depth - 1
            )
            if item_changed:
                value[idx] = new_item
                changed += item_changed
        return value, changed

    if isinstance(value, tuple):
        changed = 0
        items = []
        for item in value:
            new_item, item_changed = _replace_aliases_in_value(
                item, alias_map, visited=visited, depth=depth - 1
            )
            items.append(new_item)
            changed += item_changed
        if not changed:
            return value, 0
        return _tuple_like(value, items), changed

    if isinstance(value, MutableMapping):
        changed = 0
        for key, item in list(value.items()):
            new_item, item_changed = _replace_aliases_in_value(
                item, alias_map, visited=visited, depth=depth - 1
            )
            if item_changed:
                value[key] = new_item
                changed += item_changed
        return value, changed

    if isinstance(value, (Mapping, Sequence, torch.nn.Module, ModuleType)):
        return value, 0

    try:
        attrs = vars(value)
    except TypeError:
        attrs = {}

    changed = 0
    for attr_name, attr_val in list(attrs.items()):
        if attr_name.startswith("__"):
            continue
        new_val, attr_changed = _replace_aliases_in_value(
            attr_val, alias_map, visited=visited, depth=depth - 1
        )
        if not attr_changed:
            continue
        try:
            setattr(value, attr_name, new_val)
        except (AttributeError, TypeError):
            continue
        changed += attr_changed

    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot_name in slots:
        if slot_name in ("__dict__", "__weakref__") or slot_name.startswith("__"):
            continue
        try:
            slot_val = getattr(value, slot_name)
        except AttributeError:
            continue
        new_val, slot_changed = _replace_aliases_in_value(
            slot_val, alias_map, visited=visited, depth=depth - 1
        )
        if not slot_changed:
            continue
        try:
            setattr(value, slot_name, new_val)
        except (AttributeError, TypeError):
            continue
        changed += slot_changed
    return value, changed


def _refresh_tensor_aliases(
    model: torch.nn.Module,
    alias_map: _TensorAliasMap,
    *,
    max_depth: int = 12,
) -> int:
    """Refresh references that cached tensors before materialization."""
    if not alias_map:
        return 0

    refreshed = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            replacement = _get_tensor_alias(alias_map, param)
            if replacement is None:
                continue
            if not isinstance(replacement, torch.nn.Parameter):
                replacement = torch.nn.Parameter(
                    replacement,
                    requires_grad=(
                        param.requires_grad
                        if isinstance(param, torch.nn.Parameter)
                        else False
                    ),
                )
            module._parameters[name] = replacement
            refreshed += 1

        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            replacement = _get_tensor_alias(alias_map, buf)
            if replacement is None:
                continue
            module._buffers[name] = replacement
            refreshed += 1

        skip = {
            "_parameters",
            "_buffers",
            "_modules",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
        }
        for attr_name, attr_val in list(module.__dict__.items()):
            if attr_name in skip or attr_name.startswith("__"):
                continue
            new_val, changed = _replace_aliases_in_value(
                attr_val, alias_map, visited=set(), depth=max_depth
            )
            if not changed:
                continue
            # Write through __dict__ so cached tensor attrs do not become newly
            # registered Parameters/Buffers on nn.Module instances.
            module.__dict__[attr_name] = new_val
            refreshed += changed

    return refreshed


def snapshot_auxiliary_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
    max_depth: int = 12,
) -> list[str]:
    """Snapshot non-parameter tensors as mutable runtime state.

    GMS committed weight mappings are read-only after publish. vLLM keeps
    runtime caches, scales, routing tables, and scratch buffers in buffers or
    helper attrs; those tensors may be read or mutated by kernels, so they must
    not remain live aliases of committed weight memory.
    """
    snapshotted: list[str] = []
    alias_map: _TensorAliasMap = {}
    snapshot_by_source: dict[_TensorAliasKey, _AuxTensorSnapshot] = {}
    snapshot_by_location: dict[
        tuple[str, int, tuple[int, ...], tuple[int, ...], torch.dtype],
        _AuxTensorSnapshot,
    ] = {}
    registered_parameter_ids = {
        id(param)
        for module in model.modules()
        for param in module._parameters.values()
        if param is not None
    }

    def set_value(owner: Any, key: str | int, value: Any) -> None:
        if isinstance(owner, list):
            owner[int(key)] = value
        elif isinstance(owner, MutableMapping):
            owner[key] = value
        elif isinstance(owner, torch.nn.Module):
            owner.__dict__[str(key)] = value
        else:
            setattr(owner, str(key), value)

    def snapshot_tensor(tensor: torch.Tensor, path: str) -> torch.Tensor | None:
        if id(tensor) in registered_parameter_ids:
            return None
        if not tensor.is_cuda:
            return None
        if _is_zero_size_tensor(tensor):
            return None

        key = _TensorAliasKey(tensor)
        snapshot = snapshot_by_source.get(key)
        if snapshot is None:
            runtime_tensor = _move_aux_tensor_out_of_gms(
                gms_client_memory_manager,
                tensor,
                device_index=device_index,
            )
            if runtime_tensor is not tensor:
                _set_tensor_alias(alias_map, tensor, runtime_tensor)

            found = _find_gms_mapping(gms_client_memory_manager, tensor)
            if found is not None:
                location = (
                    found.mapping.allocation_id,
                    found.offset_bytes,
                    tuple(tensor.shape),
                    tuple(int(s) for s in tensor.stride()),
                    tensor.dtype,
                )
                snapshot = snapshot_by_location.get(location)
            if snapshot is None:
                snapshot = _snapshot_aux_tensor_to_gms(
                    gms_client_memory_manager,
                    runtime_tensor,
                    device_index=device_index,
                )
                found_snapshot = _find_gms_mapping(
                    gms_client_memory_manager, snapshot.tensor
                )
                if found_snapshot is not None:
                    snapshot_by_location[
                        (
                            found_snapshot.mapping.allocation_id,
                            found_snapshot.offset_bytes,
                            tuple(snapshot.tensor.shape),
                            tuple(int(s) for s in snapshot.tensor.stride()),
                            snapshot.tensor.dtype,
                        )
                    ] = snapshot

            snapshot_by_source[key] = snapshot
            tensor_for_meta = runtime_tensor
        else:
            tensor_for_meta = _get_tensor_alias(alias_map, tensor)
            if tensor_for_meta is None:
                tensor_for_meta = tensor

        _put_tensor_metadata(
            gms_client_memory_manager,
            key=path,
            allocation_id=snapshot.allocation_id,
            offset_bytes=snapshot.offset_bytes,
            tensor=tensor_for_meta,
            tensor_type="aux_tensor",
        )
        snapshotted.append(path)
        return tensor_for_meta

    def visit_value(
        owner: Any,
        key: str | int,
        value: Any,
        *,
        path: str,
        visited: set[int],
        depth: int,
    ) -> None:
        if torch.is_tensor(value):
            replacement = snapshot_tensor(cast(torch.Tensor, value), path)
            if replacement is not None and replacement is not value:
                set_value(owner, key, replacement)
            return

        if depth <= 0 or value is None:
            return

        if isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                bool,
                torch.dtype,
                torch.device,
                FunctionType,
                MethodType,
                ModuleType,
            ),
        ) or isinstance(value, type):
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, list):
            for idx, item in enumerate(value):
                visit_value(
                    value,
                    idx,
                    item,
                    path=f"{path}.{idx}",
                    visited=visited,
                    depth=depth - 1,
                )
            return

        if isinstance(value, tuple):
            updated = list(value)
            changed = False
            for idx, item in enumerate(value):
                before = updated[idx]
                visit_value(
                    updated,
                    idx,
                    item,
                    path=f"{path}.{idx}",
                    visited=visited,
                    depth=depth - 1,
                )
                changed = changed or updated[idx] is not before
            if changed:
                set_value(owner, key, _tuple_like(value, updated))
            return

        if isinstance(value, MutableMapping):
            for item_key, item in list(value.items()):
                visit_value(
                    value,
                    item_key,
                    item,
                    path=f"{path}.{item_key}",
                    visited=visited,
                    depth=depth - 1,
                )
            return

        if isinstance(value, (Mapping, Sequence, torch.nn.Module, ModuleType)):
            return

        try:
            attrs = vars(value)
        except TypeError:
            attrs = {}

        for attr_name, attr_val in list(attrs.items()):
            if attr_name.startswith("__"):
                continue
            visit_value(
                value,
                attr_name,
                attr_val,
                path=f"{path}.{attr_name}",
                visited=visited,
                depth=depth - 1,
            )

        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot_name in slots:
            if slot_name in ("__dict__", "__weakref__") or slot_name.startswith("__"):
                continue
            try:
                slot_val = getattr(value, slot_name)
            except AttributeError:
                continue
            visit_value(
                value,
                slot_name,
                slot_val,
                path=f"{path}.{slot_name}",
                visited=visited,
                depth=depth - 1,
            )

    for module_prefix, module in (
        ("", model),
        *[(name, mod) for name, mod in model.named_modules() if name],
    ):
        for name, buf in module._buffers.items():
            if buf is None:
                continue
            qualified = f"{module_prefix}.{name}" if module_prefix else name
            replacement = snapshot_tensor(buf, qualified)
            if replacement is not None and replacement is not buf:
                module._buffers[name] = replacement

        skip = (
            set(module._parameters.keys())
            | set(module._buffers.keys())
            | set(module._modules.keys())
            | {"_parameters", "_buffers", "_modules"}
        )
        for attr_name, attr_val in list(module.__dict__.items()):
            if attr_name in skip or attr_name.startswith("__"):
                continue
            qualified = f"{module_prefix}.{attr_name}" if module_prefix else attr_name
            visit_value(
                module,
                attr_name,
                attr_val,
                path=qualified,
                visited=set(),
                depth=max_depth,
            )

    refreshed = _refresh_tensor_aliases(model, alias_map)
    if refreshed:
        logger.debug("[GMS] Refreshed %d auxiliary tensor aliases", refreshed)

    return snapshotted


def _materialize_skipped_meta_tensors(
    model: torch.nn.Module,
    *,
    device_index: int,
    alias_map: _TensorAliasMap,
) -> None:
    """Materialize descriptor-only tensors omitted from GMS metadata."""
    zero_size_meta_tensors = _materialize_zero_size_meta_tensors(
        model, device_index=device_index, alias_map=alias_map
    )
    if zero_size_meta_tensors:
        logger.debug(
            "[GMS] Materialized %d zero-size meta tensors not in metadata: %s",
            len(zero_size_meta_tensors),
            zero_size_meta_tensors[:10],
        )


def _materialize_empty_parameter(
    model: torch.nn.Module,
    name: str,
    spec: GMSTensorSpec,
    *,
    device_index: int,
    alias_map: _TensorAliasMap,
) -> None:
    try:
        mod, attr = _resolve_module_attr(model, name)
    except AttributeError:
        parent, leaf = _path_parent(model, name)
        if not isinstance(parent, torch.nn.Module):
            raise
        mod = parent
        attr = leaf

    old = None
    requires_grad = False
    if isinstance(mod, torch.nn.Module) and attr in mod._parameters:
        old = mod._parameters[attr]
        if isinstance(old, torch.nn.Parameter):
            requires_grad = old.requires_grad
    else:
        old = getattr(mod, attr, None)
        if isinstance(old, torch.nn.Parameter):
            requires_grad = old.requires_grad

    if torch.is_tensor(old):
        existing = _get_tensor_alias(alias_map, old)
        if existing is not None:
            if not _tensor_matches_metadata(existing, spec.meta):
                raise RuntimeError(
                    f"Shape/dtype mismatch for empty parameter alias {name}: "
                    f"existing={tuple(existing.shape)}/{existing.dtype}, "
                    f"gms={tuple(spec.meta.shape)}/{spec.meta.dtype}"
                )
            if isinstance(mod, torch.nn.Module):
                mod._parameters[attr] = cast(torch.nn.Parameter, existing)
            else:
                setattr(mod, attr, existing)
            return

    tensor = _empty_tensor_from_metadata(spec.meta, device_index=device_index)
    replacement = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    if torch.is_tensor(old):
        _copy_tensor_attrs(old, replacement)
        _set_tensor_alias(alias_map, old, replacement)

    if isinstance(mod, torch.nn.Module):
        mod._parameters[attr] = replacement
    else:
        setattr(mod, attr, replacement)


def _iter_local_materialized_tensors(
    module: torch.nn.Module,
) -> Iterator[torch.Tensor]:
    for tensor in module._parameters.values():
        if torch.is_tensor(tensor) and not tensor.is_meta:
            yield tensor
    for tensor in module._buffers.values():
        if torch.is_tensor(tensor) and not tensor.is_meta:
            yield tensor
    skip = set(module._parameters) | set(module._buffers) | set(module._modules)
    for name, value in module.__dict__.items():
        if name in skip or name.startswith("__"):
            continue
        if torch.is_tensor(value) and not value.is_meta:
            yield value


def _unique_matching_local_tensor(
    module: torch.nn.Module,
    tensor: torch.Tensor,
) -> torch.Tensor | None:
    matches: list[torch.Tensor] = []
    seen: set[int] = set()
    for candidate in _iter_local_materialized_tensors(module):
        if (
            tuple(candidate.shape) != tuple(tensor.shape)
            or tuple(int(s) for s in candidate.stride())
            != tuple(int(s) for s in tensor.stride())
            or candidate.dtype != tensor.dtype
        ):
            continue
        key = id(candidate)
        if key in seen:
            continue
        seen.add(key)
        matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    return None


def _drop_unpublished_meta_parameters(
    module: torch.nn.Module,
    *,
    published_parameter_names: set[str],
    alias_map: _TensorAliasMap,
    prefix: str = "",
) -> list[str]:
    """Drop reader-only meta params absent from the writer-final manifest."""
    dropped: list[str] = []

    for name, param in list(module._parameters.items()):
        if param is None or not param.is_meta:
            continue
        qualified = f"{prefix}{name}" if prefix else name
        if qualified in published_parameter_names:
            continue
        existing = _get_tensor_alias(alias_map, param)
        if existing is not None:
            if not isinstance(existing, torch.nn.Parameter):
                existing = torch.nn.Parameter(
                    existing,
                    requires_grad=(
                        param.requires_grad
                        if isinstance(param, torch.nn.Parameter)
                        else False
                    ),
                )
                _set_tensor_alias(alias_map, param, existing)
            module._parameters[name] = existing
            continue
        replacement = _unique_matching_local_tensor(module, param)
        if replacement is not None:
            _set_tensor_alias(alias_map, param, replacement)
        del module._parameters[name]
        dropped.append(qualified)

    for name, submodule in module._modules.items():
        if submodule is not None:
            subprefix = f"{prefix}{name}." if prefix else f"{name}."
            dropped.extend(
                _drop_unpublished_meta_parameters(
                    submodule,
                    published_parameter_names=published_parameter_names,
                    alias_map=alias_map,
                    prefix=subprefix,
                )
            )

    return dropped


# =============================================================================
# Public API - Registration and Materialization
# =============================================================================


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> set[str]:
    """Register non-empty Parameters as shared GMS weight state.

    Args:
        gms_client_memory_manager: GMS client memory manager in write mode.
        model: PyTorch model to register.

    Returns:
        Allocation IDs referenced by registered tensors.
    """
    referenced_allocation_ids: set[str] = set()
    for name, tensor, tensor_type in _iter_module_tensors(model):
        if tensor_type != "parameter":
            logger.debug("[GMS] Skipping auxiliary %s %r", tensor_type, name)
            continue

        if _is_zero_size_tensor(tensor):
            allocation_id = _descriptor_allocation_id(gms_client_memory_manager)
            _put_tensor_metadata(
                gms_client_memory_manager,
                key=name,
                allocation_id=allocation_id,
                offset_bytes=0,
                tensor=tensor,
                tensor_type=_EMPTY_PARAMETER_TENSOR_TYPE,
            )
            logger.debug(
                "[GMS] Registered descriptor-only zero-size parameter %r",
                name,
            )
            referenced_allocation_ids.add(allocation_id)
            continue

        found = _find_gms_mapping(gms_client_memory_manager, tensor)
        if found is not None:
            _, mapping, offset = found
            _put_tensor_metadata(
                gms_client_memory_manager,
                key=name,
                allocation_id=mapping.allocation_id,
                offset_bytes=offset,
                tensor=tensor,
                tensor_type="parameter",
            )
            referenced_allocation_ids.add(mapping.allocation_id)
        else:
            raise RuntimeError(f"Tensor {name!r} not found in any GMS allocation")
    return referenced_allocation_ids


def materialize_module_from_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
) -> None:
    """Materialize model tensors from GMS.

    Args:
        gms_client_memory_manager: GMS client memory manager in read mode.
        model: Model to populate with tensors.
        device_index: CUDA device index.
    """
    specs = GMSTensorSpec.load_all(gms_client_memory_manager)
    alias_map: _TensorAliasMap = {}
    materialized_by_location: dict[
        tuple[str, int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor
    ] = {}
    mutable_by_location: dict[
        tuple[str, int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor
    ] = {}

    for name, spec in specs.items():
        tensor_type = spec.meta.tensor_type
        if tensor_type == _EMPTY_PARAMETER_TENSOR_TYPE:
            _materialize_empty_parameter(
                model,
                name,
                spec,
                device_index=device_index,
                alias_map=alias_map,
            )
            continue

        try:
            mod, attr = _resolve_module_attr(model, name)
        except AttributeError:
            if tensor_type == "aux_tensor":
                mod = None
                attr = ""
            else:
                raise

        location = (
            spec.allocation_id,
            int(spec.offset_bytes),
            tuple(spec.meta.shape),
            tuple(spec.meta.stride),
            spec.meta.dtype,
        )
        tensor = materialized_by_location.get(location)
        if tensor is None:
            tensor = spec.materialize(gms_client_memory_manager, device_index)
            materialized_by_location[location] = tensor

        if tensor_type == "aux_tensor":
            try:
                old = _path_value(model, name)
            except AttributeError:
                try:
                    _path_parent(model, name)
                except AttributeError:
                    logger.debug("[GMS] Skipping aux tensor for missing path %r", name)
                    continue
                old = None
            replacement = mutable_by_location.get(location)
            if replacement is None:
                replacement = _clone_mutable_gms_tensor(
                    tensor, name=name, tensor_type=tensor_type, spec=spec
                )
                mutable_by_location[location] = replacement
            if torch.is_tensor(old):
                _set_tensor_alias(alias_map, old, replacement)
            try:
                _set_path_value(model, name, replacement)
            except AttributeError:
                if _is_read_only_property_path(model, name):
                    logger.debug("[GMS] Skipping read-only aux property %r", name)
                    continue
                raise
            continue

        # Legacy metadata for non-parameters is cloned into mutable runtime memory.
        if tensor_type in ("tensor_attr", "buffer"):
            try:
                old = _path_value(model, name)
            except AttributeError:
                _path_parent(model, name)
                old = None
            replacement = mutable_by_location.get(location)
            if replacement is None:
                replacement = _clone_mutable_gms_tensor(
                    tensor, name=name, tensor_type=tensor_type, spec=spec
                )
                mutable_by_location[location] = replacement
            if torch.is_tensor(old):
                _set_tensor_alias(alias_map, old, replacement)
            try:
                _set_path_value(model, name, replacement)
            except AttributeError:
                if tensor_type == "tensor_attr" and _is_read_only_property_path(
                    model, name
                ):
                    logger.debug("[GMS] Skipping read-only property %r", name)
                    continue
                raise
            continue

        # Parameters: in-place update or replace meta tensors
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            param = mod._parameters[attr]
            if param is not None:
                existing = _get_tensor_alias(alias_map, param)
                if existing is not None:
                    if existing.shape != tensor.shape or existing.dtype != tensor.dtype:
                        raise RuntimeError(
                            f"Shape/dtype mismatch for shared alias {name}: "
                            f"existing={tuple(existing.shape)}/{existing.dtype}, "
                            f"gms={tuple(tensor.shape)}/{tensor.dtype}"
                        )
                    mod._parameters[attr] = cast(torch.nn.Parameter, existing)
                    continue
                if not param.is_meta and (
                    param.shape != tensor.shape or param.dtype != tensor.dtype
                ):
                    raise RuntimeError(
                        f"Shape/dtype mismatch for {name}: "
                        f"param={tuple(param.shape)}/{param.dtype}, "
                        f"gms={tuple(tensor.shape)}/{tensor.dtype}"
                    )
                if param.is_meta or param.device != tensor.device:
                    replacement = torch.nn.Parameter(
                        tensor, requires_grad=param.requires_grad
                    )
                    _copy_tensor_attrs(param, replacement)
                    _set_tensor_alias(alias_map, param, replacement)
                    mod._parameters[attr] = replacement
                else:
                    param.data = tensor
                    _set_tensor_alias(alias_map, param, param)
                continue

        # Writer post-load hooks can add parameters that are absent from the
        # RO meta model because those hooks are skipped there.
        old = getattr(mod, attr, None)
        if torch.is_tensor(old):
            _set_tensor_alias(alias_map, old, tensor)
        if tensor_type == "parameter" and isinstance(mod, torch.nn.Module):
            replacement = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(mod, attr, replacement)
            _set_tensor_alias(alias_map, tensor, replacement)
        else:
            setattr(mod, attr, tensor)

    _materialize_skipped_meta_tensors(
        model, device_index=device_index, alias_map=alias_map
    )

    published_parameter_names = {
        name
        for name, spec in specs.items()
        if spec.meta.tensor_type in {"parameter", _EMPTY_PARAMETER_TENSOR_TYPE}
    }
    dropped = _drop_unpublished_meta_parameters(
        model,
        published_parameter_names=published_parameter_names,
        alias_map=alias_map,
    )
    if dropped:
        logger.debug(
            "[GMS] Dropped %d unpublished meta parameters: %s",
            len(dropped),
            dropped[:10],
        )

    refreshed = _refresh_tensor_aliases(model, alias_map)
    if refreshed:
        logger.debug("[GMS] Refreshed %d cached tensor aliases", refreshed)

    # Check for meta tensors and warn
    meta_tensors = [n for n, p in model.named_parameters() if p.is_meta]
    meta_tensors += [n for n, b in model.named_buffers() if b.is_meta]
    if meta_tensors:
        logger.warning(
            "[GMS] %d meta tensors not in metadata: %s",
            len(meta_tensors),
            meta_tensors[:10],
        )
