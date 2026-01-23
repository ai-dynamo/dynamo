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
) -> None:
    """Register all model tensors into the GMS metadata store.

    Args:
        gms_client_memory_manager: GMS client memory manager in write mode.
        model: PyTorch model to register.
    """
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
