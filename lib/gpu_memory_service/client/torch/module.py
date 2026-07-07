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
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


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

    for name, spec in specs.items():
        tensor = spec.materialize(gms_client_memory_manager, device_index)
        mod, attr = _resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type

        # Tensor attrs and buffers: clone since they may be mutated
        if tensor_type in ("tensor_attr", "buffer"):
            if (
                tensor_type == "buffer"
                and hasattr(mod, "_buffers")
                and attr in mod._buffers
            ):
                mod._buffers[attr] = tensor.detach().clone()
            else:
                setattr(mod, attr, tensor.detach().clone())
            continue

        # Parameters: in-place update or replace meta tensors
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            param = mod._parameters[attr]
            if param is not None:
                if param.shape != tensor.shape or param.dtype != tensor.dtype:
                    raise RuntimeError(
                        f"Shape/dtype mismatch for {name}: "
                        f"param={tuple(param.shape)}/{param.dtype}, "
                        f"gms={tuple(tensor.shape)}/{tensor.dtype}"
                    )
                if param.is_meta or param.device != tensor.device:
                    mod._parameters[attr] = torch.nn.Parameter(
                        tensor, requires_grad=param.requires_grad
                    )
                else:
                    param.data = tensor
                continue

        # Fallback: set as attribute
        setattr(mod, attr, tensor)

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
        rebound_bytes += tensor.numel() * tensor.element_size()

    return rebound_bytes
