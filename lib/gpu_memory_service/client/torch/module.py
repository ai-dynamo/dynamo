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
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


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


def _get_tensor_alias(
    alias_map: _TensorAliasMap, tensor: torch.Tensor
) -> torch.Tensor | None:
    return alias_map.get(_TensorAliasKey(tensor))


def _set_tensor_alias(
    alias_map: _TensorAliasMap, source: torch.Tensor, replacement: torch.Tensor
) -> None:
    alias_map[_TensorAliasKey(source)] = replacement


def _find_gms_mapping(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
) -> _GMSMappingMatch | None:
    ptr = int(tensor.data_ptr())
    for va, mapping in gms_client_memory_manager.mappings.items():
        if va <= ptr < va + mapping.aligned_size:
            return _GMSMappingMatch(va, mapping, ptr - va)
    return None


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
        try:
            return type(value)(*items), changed
        except TypeError:
            return tuple(items), changed

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
        return value, 0

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
    return value, changed


def _refresh_tensor_aliases(
    model: torch.nn.Module,
    alias_map: _TensorAliasMap,
    *,
    max_depth: int = 6,
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


def move_tensor_attrs_out_of_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
    name_substrings: tuple[str, ...] = ("workspace",),
    max_depth: int = 6,
) -> list[str]:
    """Move mutable runtime tensor attrs out of the GMS weight layout."""
    moved: list[str] = []
    alias_map: _TensorAliasMap = {}
    target_device = torch.device("cuda", device_index)

    def should_move(name: str) -> bool:
        lowered = name.lower()
        return any(substring in lowered for substring in name_substrings)

    def move_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, bool] | None:
        if _is_zero_size_tensor(tensor):
            return None

        replacement = _get_tensor_alias(alias_map, tensor)
        if replacement is not None:
            return replacement, False

        found = _find_gms_mapping(gms_client_memory_manager, tensor)
        if found is None:
            return None

        replacement = torch.empty_strided(
            tuple(tensor.shape),
            tuple(int(s) for s in tensor.stride()),
            dtype=tensor.dtype,
            device=target_device,
        )
        if _find_gms_mapping(gms_client_memory_manager, replacement) is not None:
            raise RuntimeError(
                "move_tensor_attrs_out_of_gms must be called outside the GMS "
                "weights mempool"
            )
        replacement.copy_(tensor.detach())
        _copy_tensor_attrs(tensor, replacement)
        _set_tensor_alias(alias_map, tensor, replacement)
        return replacement, True

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
            if value.is_cuda and should_move(path):
                moved_tensor = move_tensor(cast(torch.Tensor, value))
                if moved_tensor is not None:
                    replacement, is_new = moved_tensor
                    set_value(owner, key, replacement)
                    if is_new:
                        moved.append(path)
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
                try:
                    replacement = type(value)(*updated)
                except TypeError:
                    replacement = tuple(updated)
                set_value(owner, key, replacement)
            return

        if isinstance(value, MutableMapping):
            for item_key, item in list(value.items()):
                item_path = f"{path}.{item_key}"
                visit_value(
                    value,
                    item_key,
                    item,
                    path=item_path,
                    visited=visited,
                    depth=depth - 1,
                )
            return

        if isinstance(value, (Mapping, Sequence, torch.nn.Module)):
            return

        try:
            attrs = vars(value)
        except TypeError:
            return

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

    def set_value(owner: Any, key: str | int, value: Any) -> None:
        if isinstance(owner, list):
            owner[int(key)] = value
        elif isinstance(owner, MutableMapping):
            owner[key] = value
        elif isinstance(owner, torch.nn.Module):
            owner.__dict__[str(key)] = value
        else:
            setattr(owner, str(key), value)

    def visit_module(module: torch.nn.Module, prefix: str = "") -> None:
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
            visit_value(
                module,
                attr_name,
                attr_val,
                path=qualified,
                visited=set(),
                depth=max_depth,
            )

        for name, submodule in module._modules.items():
            if submodule is not None:
                subprefix = f"{prefix}{name}." if prefix else f"{name}."
                visit_module(submodule, subprefix)

    visit_module(model)

    refreshed = _refresh_tensor_aliases(model, alias_map)
    if refreshed:
        logger.debug("[GMS] Refreshed %d tensor aliases after moving attrs", refreshed)

    leftovers = _find_named_tensors_in_gms(
        gms_client_memory_manager,
        model,
        name_substrings=name_substrings,
        max_depth=max_depth,
    )
    if leftovers:
        raise RuntimeError(
            "Mutable runtime tensor attrs still live in GMS: "
            + ", ".join(leftovers[:10])
        )

    return moved


def _find_named_tensors_in_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    name_substrings: tuple[str, ...],
    max_depth: int,
) -> list[str]:
    matches: list[str] = []

    def should_check(path: str) -> bool:
        lowered = path.lower()
        return any(substring in lowered for substring in name_substrings)

    def visit_value(value: Any, *, path: str, visited: set[int], depth: int) -> None:
        if torch.is_tensor(value):
            if (
                value.is_cuda
                and should_check(path)
                and not _is_zero_size_tensor(value)
                and _find_gms_mapping(gms_client_memory_manager, value) is not None
            ):
                matches.append(path)
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

        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                visit_value(
                    item,
                    path=f"{path}.{idx}",
                    visited=visited,
                    depth=depth - 1,
                )
            return

        if isinstance(value, Mapping):
            for item_key, item in value.items():
                visit_value(
                    item,
                    path=f"{path}.{item_key}",
                    visited=visited,
                    depth=depth - 1,
                )
            return

        if isinstance(value, (Sequence, torch.nn.Module)):
            return

        try:
            attrs = vars(value)
        except TypeError:
            return

        for attr_name, attr_val in attrs.items():
            if attr_name.startswith("__"):
                continue
            visit_value(
                attr_val,
                path=f"{path}.{attr_name}",
                visited=visited,
                depth=depth - 1,
            )

    for module_prefix, module in (
        ("", model),
        *[(name, mod) for name, mod in model.named_modules() if name],
    ):
        skip = (
            set(module._parameters.keys())
            | set(module._buffers.keys())
            | set(module._modules.keys())
            | {"_parameters", "_buffers", "_modules"}
        )
        for attr_name, attr_val in module.__dict__.items():
            if attr_name in skip or attr_name.startswith("__"):
                continue
            qualified = f"{module_prefix}.{attr_name}" if module_prefix else attr_name
            visit_value(attr_val, path=qualified, visited=set(), depth=max_depth)

    return matches


# =============================================================================
# Public API - Registration and Materialization
# =============================================================================


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> None:
    """Register all model tensors into the GMS metadata store.

    Args:
        gms_client_memory_manager: GMS client memory manager in write mode.
        model: PyTorch model to register.
    """
    for name, tensor, tensor_type in _iter_module_tensors(model):
        if _is_zero_size_tensor(tensor):
            logger.debug(
                "[GMS] Skipping zero-size %s %r - no allocation to register",
                tensor_type,
                name,
            )
            continue

        found = _find_gms_mapping(gms_client_memory_manager, tensor)
        if found is not None:
            _, mapping, offset = found
            meta = TensorMetadata.from_tensor(tensor, tensor_type)
            gms_client_memory_manager.metadata_put(
                key=name,
                allocation_id=mapping.allocation_id,
                offset_bytes=offset,
                value=meta.to_bytes(),
            )
        else:
            # No mapping matched - tensor pointer not in any GMS allocation
            if tensor_type == "parameter":
                # Parameters are model weights - must be in GMS allocations
                raise RuntimeError(f"Tensor {name!r} not found in any GMS allocation")
            # Buffers and tensor_attrs may be dynamically allocated (e.g., KV cache)
            logger.debug(
                "[GMS] Skipping %s %r - not in GMS allocations", tensor_type, name
            )


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

    for name, spec in specs.items():
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
        mod, attr = _resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type

        # Tensor attrs and buffers: clone since they may be mutated
        if tensor_type in ("tensor_attr", "buffer"):
            if (
                tensor_type == "buffer"
                and hasattr(mod, "_buffers")
                and attr in mod._buffers
            ):
                old = mod._buffers[attr]
                replacement = tensor.detach().clone()
                if torch.is_tensor(old):
                    _set_tensor_alias(alias_map, old, replacement)
                mod._buffers[attr] = replacement
            else:
                replacement = tensor.detach().clone()
                old = getattr(mod, attr, None)
                if torch.is_tensor(old):
                    _set_tensor_alias(alias_map, old, replacement)
                try:
                    setattr(mod, attr, replacement)
                except AttributeError:
                    descriptor = getattr(type(mod), attr, None)
                    if (
                        tensor_type == "tensor_attr"
                        and isinstance(descriptor, property)
                        and descriptor.fset is None
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
                if param.shape != tensor.shape or param.dtype != tensor.dtype:
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

        # Fallback: set as attribute
        old = getattr(mod, attr, None)
        if torch.is_tensor(old):
            _set_tensor_alias(alias_map, old, tensor)
        setattr(mod, attr, tensor)

    zero_size_meta_tensors = _materialize_zero_size_meta_tensors(
        model, device_index=device_index, alias_map=alias_map
    )
    if zero_size_meta_tensors:
        logger.debug(
            "[GMS] Materialized %d zero-size meta tensors not in metadata: %s",
            len(zero_size_meta_tensors),
            zero_size_meta_tensors[:10],
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
