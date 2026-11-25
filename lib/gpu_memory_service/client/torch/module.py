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
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    MethodType,
    ModuleType,
)
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

_MISSING = object()

_SKIP_ALIAS_TYPES = (
    torch.nn.Module,
    torch.Tensor,
    type,
    ModuleType,
    FunctionType,
    MethodType,
    BuiltinFunctionType,
    BuiltinMethodType,
    logging.Logger,
    logging.Handler,
    str,
    bytes,
    bytearray,
    memoryview,
    int,
    float,
    bool,
)


# =============================================================================
# Module Tensor Iteration
# =============================================================================


def _is_accelerator_tensor(t: "torch.Tensor") -> bool:
    """True for CUDA or XPU device tensors (the backends GMS supports)."""
    return t.is_cuda or getattr(t, "is_xpu", False)


def _iter_module_tensors(
    module: torch.nn.Module,
    prefix: str = "",
) -> Iterator[Tuple[str, torch.Tensor, str]]:
    """Iterate over all accelerator (CUDA/XPU) tensors in a module tree.

    Yields (qualified_name, tensor, tensor_type) for:
    - Parameters (tensor_type="parameter")
    - Buffers (tensor_type="buffer")
    - Other tensor attributes like _k_scale (tensor_type="tensor_attr")

    Args:
        module: The nn.Module to iterate.
        prefix: Prefix for qualified names (used in recursion).

    Yields:
        (name, tensor, tensor_type) tuples for each accelerator tensor.
    """
    # Parameters
    for name, param in module._parameters.items():
        if param is not None and _is_accelerator_tensor(param):
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, param, "parameter")

    # Buffers
    for name, buf in module._buffers.items():
        if buf is not None and _is_accelerator_tensor(buf):
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, buf, "buffer")

    # Other tensor attributes (not params/buffers/submodules)
    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
    )
    for attr_name in dir(module):
        if attr_name in skip or attr_name.startswith("__"):
            continue
        try:
            attr_val = getattr(module, attr_name, None)
        except Exception:
            continue

        if torch.is_tensor(attr_val) and _is_accelerator_tensor(attr_val):
            qualified = f"{prefix}{attr_name}" if prefix else attr_name
            yield (qualified, attr_val, "tensor_attr")
        elif isinstance(attr_val, (list, tuple)) and attr_val:
            if all(torch.is_tensor(x) and _is_accelerator_tensor(x) for x in attr_val):
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
    referenced_allocation_ids: set[str] = set()
    for name, tensor, tensor_type in _iter_module_tensors(model):
        ptr = int(tensor.data_ptr())

        # Find allocation containing this tensor
        for va, mapping in gms_client_memory_manager.mappings.items():
            if va <= ptr < va + mapping.aligned_size:
                offset = ptr - va
                meta = TensorMetadata.from_tensor(tensor, tensor_type)
                gms_client_memory_manager.metadata_put(
                    key=name,
                    allocation_id=mapping.allocation_id,
                    offset_bytes=offset,
                    value=meta.to_bytes(),
                )
                referenced_allocation_ids.add(mapping.allocation_id)
                break
        else:
            # No mapping matched - tensor pointer not in any GMS allocation
            if tensor_type == "parameter":
                # Parameters are model weights - must be in GMS allocations
                raise RuntimeError(f"Tensor {name!r} not found in any GMS allocation")
            # Buffers and tensor_attrs may be dynamically allocated (e.g., KV cache)
            logger.debug(
                "[GMS] Skipping %s %r - not in GMS allocations", tensor_type, name
            )
    return referenced_allocation_ids


def _get_instance_dict(obj: object) -> dict[str, object] | None:
    """Return ``obj``'s own ``__dict__`` without invoking ``__getattr__``."""
    try:
        d = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return None
    return d if isinstance(d, dict) else None


_MISSING = object()


def _effective_descriptor(cls: type, attr: str) -> object:
    """Return the first descriptor for ``attr`` in ``cls.__mro__`` or ``_MISSING``."""
    for base in cls.__mro__:
        if attr in base.__dict__:
            return base.__dict__[attr]
    return _MISSING


def _is_property(cls: type, attr: str) -> bool:
    """True if ``attr`` resolves to a ``property`` descriptor on ``cls``."""
    return isinstance(_effective_descriptor(cls, attr), property)


def _is_readonly_property(mod: object, attr: str) -> bool:
    """True if ``attr`` resolves to a read-only ``property`` on ``mod``."""
    desc = _effective_descriptor(type(mod), attr)
    return isinstance(desc, property) and desc.fset is None


def _iter_slot_names(cls: type) -> Iterator[str]:
    """Yield non-property attribute names declared in ``__slots__`` on ``cls``."""
    for base in cls.__mro__:
        slots = base.__dict__.get("__slots__")
        if not slots:
            continue
        names = (slots,) if isinstance(slots, str) else slots
        for name in names:
            if name in ("__dict__", "__weakref__"):
                continue
            resolved = name
            if name.startswith("__") and not name.endswith("__"):
                stripped = base.__name__.lstrip("_")
                if stripped:
                    resolved = f"_{stripped}{name}"
            if _is_property(cls, resolved):
                continue
            yield resolved


def _rebind_replaced_parameters(
    root: torch.nn.Module,
    replaced_params: dict[int, tuple[torch.nn.Parameter, torch.nn.Parameter]],
    *,
    max_depth: int = 6,
) -> int:
    """Rebind aliased references to replaced ``Parameter`` objects.

    Custom routers, MoE runners, or weight-tied modules may retain direct
    Python references (attributes, slots, or containers) to meta ``Parameter``
    instances created during model construction. When materialization replaces
    those parameters with new ``torch.nn.Parameter`` objects, update stale
    aliases to point to the materialized parameters.
    """
    # Maps id(obj) -> shallowest depth seen so a shallower path revisits it.
    visited: dict[int, int] = {}
    stack: list[tuple[object, int]] = [(mod, 0) for mod in root.modules()]
    rebound = 0

    def _maybe_enqueue(val: object, depth: int) -> None:
        next_depth = depth + 1
        if next_depth > max_depth or isinstance(val, _SKIP_ALIAS_TYPES):
            return
        prev_depth = visited.get(id(val))
        if prev_depth is not None and next_depth >= prev_depth:
            return
        has_dict = _get_instance_dict(val) is not None
        has_slots = any(True for _ in _iter_slot_names(type(val)))
        if has_dict or has_slots:
            stack.append((val, next_depth))

    def _rebind_value(val: object, depth: int) -> tuple[object, int]:
        pair = replaced_params.get(id(val))
        if pair is not None and val is pair[0]:
            return pair[1], 1
        if depth > max_depth:
            return val, 0

        if isinstance(val, list):
            vid = id(val)
            prev_depth = visited.get(vid)
            if prev_depth is not None and depth >= prev_depth:
                return val, 0
            visited[vid] = depth
            count = 0
            for idx, item in enumerate(val):
                updated, n = _rebind_value(item, depth + 1)
                if n and updated is not item:
                    val[idx] = updated
                count += n
            return val, count

        if isinstance(val, tuple) and (type(val) is tuple or hasattr(val, "_fields")):
            items = []
            changed = False
            count = 0
            for item in val:
                updated, n = _rebind_value(item, depth + 1)
                if n and updated is not item:
                    changed = True
                items.append(updated)
                count += n
            if not changed:
                return val, count
            rebuilt = type(val)(*items) if hasattr(val, "_fields") else tuple(items)
            return rebuilt, count

        if isinstance(val, dict):
            vid = id(val)
            prev_depth = visited.get(vid)
            if prev_depth is not None and depth >= prev_depth:
                return val, 0
            visited[vid] = depth
            count = 0
            for k, item in list(val.items()):
                updated, n = _rebind_value(item, depth + 1)
                if n and updated is not item:
                    val[k] = updated
                count += n
            return val, count

        _maybe_enqueue(val, depth)
        return val, 0

    while stack:
        obj, depth = stack.pop()
        oid = id(obj)
        prev_depth = visited.get(oid)
        if prev_depth is not None and depth >= prev_depth:
            continue
        visited[oid] = depth

        if isinstance(obj, torch.nn.Module):
            for pname, p in list(obj._parameters.items()):
                if p is None:
                    continue
                pair = replaced_params.get(id(p))
                if pair is not None and p is pair[0]:
                    obj._parameters[pname] = pair[1]
                    rebound += 1

        d = _get_instance_dict(obj)
        if d is not None:
            is_module = isinstance(obj, torch.nn.Module)
            for k, v in list(d.items()):
                if is_module and k in ("_parameters", "_buffers", "_modules"):
                    continue
                updated, n = _rebind_value(v, depth)
                if n and updated is not v:
                    d[k] = updated
                rebound += n

        for slot_name in _iter_slot_names(type(obj)):
            try:
                v = object.__getattribute__(obj, slot_name)
            except AttributeError:
                continue
            updated, n = _rebind_value(v, depth)
            if not n:
                continue
            if updated is v:
                rebound += n
                continue
            try:
                object.__setattr__(obj, slot_name, updated)
                rebound += n
            except (AttributeError, TypeError):
                pass

    return rebound


def _can_register_missing_tensor(
    mod: object,
    attr: str,
    *,
    other_registry: str,
) -> bool:
    """True if a post-load parameter or buffer can be registered on ``mod``."""
    if not isinstance(mod, torch.nn.Module):
        return False
    if getattr(type(mod), attr, _MISSING) is not _MISSING:
        return False
    if attr in getattr(mod, other_registry) or attr in mod._modules:
        return False
    d = _get_instance_dict(mod)
    if not hasattr(mod, attr):
        return True
    if d is not None and attr in d:
        existing = d[attr]
        return existing is None or (torch.is_tensor(existing) and existing.is_meta)
    return False


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
    # Maps id(old_param) -> (old_param, new_param); keeping old_param alive
    # here prevents id() reuse until alias rebinding finishes below.
    replaced_params: dict[int, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    materialized_params: dict[
        tuple[int, str, int, tuple[int, ...], tuple[int, ...], torch.dtype],
        torch.nn.Parameter,
    ] = {}

    for name, spec in specs.items():
        tensor = spec.materialize(gms_client_memory_manager, device_index)
        mod, attr = _resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type

        # Tensor attrs and buffers: clone since they may be mutated
        if tensor_type in ("tensor_attr", "buffer"):
            cloned = tensor.detach().clone()
            is_existing_buffer = (
                tensor_type == "buffer"
                and hasattr(mod, "_buffers")
                and attr in mod._buffers
            )
            if is_existing_buffer:
                mod._buffers[attr] = cloned
            elif tensor_type == "buffer" and _can_register_missing_tensor(
                mod, attr, other_registry="_parameters"
            ):
                mod.__dict__.pop(attr, None)
                mod.register_buffer(attr, cloned)
            elif _is_readonly_property(mod, attr):
                logger.debug(
                    "[GMS] Skipping read-only property %s on %s",
                    name,
                    type(mod).__name__,
                )
            else:
                setattr(mod, attr, cloned)
            continue

        # Parameters: replace meta placeholders (which predate post-load
        # weight transforms like reshaping or quantized repacking), register
        # post-load parameters, or update live parameters in place.
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            param = mod._parameters[attr]
            if param is None:
                mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
                continue
            if param.shape != tensor.shape or param.dtype != tensor.dtype:
                if not param.is_meta:
                    raise RuntimeError(
                        f"Shape/dtype mismatch for {name}: "
                        f"param={tuple(param.shape)}/{param.dtype}, "
                        f"gms={tuple(tensor.shape)}/{tensor.dtype}"
                    )
                logger.debug(
                    "[GMS] Replacing meta parameter %s with post-processed "
                    "tensor: meta=%s/%s, gms=%s/%s",
                    name,
                    tuple(param.shape),
                    param.dtype,
                    tuple(tensor.shape),
                    tensor.dtype,
                )
            if param.is_meta or param.device != tensor.device:
                param_key = (
                    id(param),
                    spec.allocation_id,
                    spec.offset_bytes,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                )
                new_param = materialized_params.get(param_key)
                if new_param is None:
                    requires_grad = param.requires_grad and (
                        not param.is_meta
                        or tensor.is_floating_point()
                        or tensor.is_complex()
                    )
                    new_param = torch.nn.Parameter(tensor, requires_grad=requires_grad)
                    materialized_params[param_key] = new_param
                    replaced_params.setdefault(id(param), (param, new_param))
                mod._parameters[attr] = new_param
            else:
                param.data = tensor
            continue

        if tensor_type == "parameter" and _can_register_missing_tensor(
            mod, attr, other_registry="_buffers"
        ):
            mod.__dict__.pop(attr, None)
            mod.register_parameter(
                attr, torch.nn.Parameter(tensor, requires_grad=False)
            )
            continue

        if (
            tensor_type == "parameter"
            and isinstance(mod, torch.nn.Module)
            and attr in mod._buffers
        ):
            # Parameter overwrote a buffer on the writer; clone so the reader's
            # buffer registry never points directly at read-only GMS memory.
            mod._buffers[attr] = tensor.detach().clone()
            continue

        # Fallback: set as attribute
        if _is_readonly_property(mod, attr):
            logger.debug(
                "[GMS] Skipping read-only property %s on %s",
                name,
                type(mod).__name__,
            )
        else:
            setattr(mod, attr, tensor)

    if replaced_params:
        rebound = _rebind_replaced_parameters(model, replaced_params)
        if rebound:
            logger.debug("[GMS] Rebound %d aliased Parameter reference(s)", rebound)

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

    Returns the number of bytes rebound, i.e. how much memory is duplicated
    between the read-only GMS copies and the private clones.

    If ``retain_gms_tensors`` is provided, the original GMS-backed tensors
    are appended to it. A deferred writer that rebinds before commit must
    keep those references alive so the underlying GMS pool allocations are
    not freed before the layout is published.
    """
    mappings = gms_client_memory_manager.mappings
    rebound_bytes = 0
    for name, tensor, tensor_type in list(_iter_module_tensors(model)):
        if tensor_type == "parameter":
            continue
        ptr = int(tensor.data_ptr())
        if not any(
            va <= ptr < va + mapping.aligned_size for va, mapping in mappings.items()
        ):
            # Allocated outside the GMS pool; already private.
            continue

        mod, attr = _resolve_module_attr(model, name)
        if (
            tensor_type == "buffer"
            and hasattr(mod, "_buffers")
            and attr in mod._buffers
        ):
            mod._buffers[attr] = tensor.detach().clone()
        elif attr.isdigit() and not isinstance(mod, torch.nn.Module):
            # Element of a tensor list/tuple attribute.
            if isinstance(mod, list):
                mod[int(attr)] = tensor.detach().clone()
            elif isinstance(mod, tuple):
                # Tuples are immutable: rebuild the tuple on its owner.
                container_name, _ = name.rsplit(".", 1)
                owner, container_attr = _resolve_module_attr(model, container_name)
                if isinstance(getattr(type(owner), container_attr, None), property):
                    # Read-only derived attribute; the underlying tensors
                    # are iterated (and rebound) separately.
                    logger.debug("[GMS] Skipping property attribute %r", name)
                    continue
                elements = list(mod)
                elements[int(attr)] = tensor.detach().clone()
                setattr(owner, container_attr, tuple(elements))
            else:
                logger.debug("[GMS] Cannot rebind container element %r", name)
                continue
        else:
            if isinstance(getattr(type(mod), attr, None), property):
                # Read-only derived attribute; the underlying tensor is
                # iterated (and rebound) separately.
                logger.debug("[GMS] Skipping property attribute %r", name)
                continue
            setattr(mod, attr, tensor.detach().clone())
        if retain_gms_tensors is not None:
            retain_gms_tensors.append(tensor)
        rebound_bytes += tensor.numel() * tensor.element_size()

    return rebound_bytes
