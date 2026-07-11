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
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata

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


def _tensor_source(
    spec: GMSTensorSpec,
) -> tuple[str, int, tuple[int, ...], tuple[int, ...], torch.dtype,]:
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
    parameters = _registered_parameters(model)
    aliases: dict[
        int,
        tuple[
            torch.Tensor,
            tuple[
                str,
                int,
                tuple[int, ...],
                tuple[int, ...],
                torch.dtype,
            ],
        ],
    ] = {}
    resolved: list[
        tuple[
            str,
            GMSTensorSpec,
            object,
            str,
            torch.Tensor | None,
            torch.nn.Parameter | None,
        ]
    ] = []
    observed_storage: dict[int, tuple[str, torch.Tensor]] = {}
    validated_parameters: set[int] = set()
    target_device = torch.device("cuda", device_index)

    # Resolve and validate every observed target before materializing or
    # mutating any model tensor. Exact aliases are supported; aliases hidden
    # inside arbitrary helper objects are intentionally not discoverable.
    for name, spec in specs.items():
        mod, attr = _resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type
        existing = _resolve_existing_tensor(model, name, mod, attr, tensor_type)
        parameter = parameters.get(id(existing)) if existing is not None else None

        if existing is not None:
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
                is_nonparameter = parameter is None
                if is_nonparameter:
                    _validate_plain_nonparameter(existing, name=name)
                _validate_observed_storage(
                    observed_storage,
                    name=name,
                    tensor=existing,
                )
            if (
                parameter is not None
                and id(parameter) not in validated_parameters
                and (parameter.is_meta or parameter.device != target_device)
            ):
                _validate_parameter_swap(parameter, name=name)
                validated_parameters.add(id(parameter))
        resolved.append((name, spec, mod, attr, existing, parameter))

    materialized: set[int] = set()
    for name, spec, mod, attr, existing, parameter in resolved:
        if existing is not None and id(existing) in materialized:
            continue
        tensor_type = spec.meta.tensor_type

        # Tensor attrs and buffers need private writable storage. Swap the
        # TensorImpl so every exact Python alias of an existing tensor survives.
        if tensor_type in ("tensor_attr", "buffer") and parameter is None:
            if existing is not None:
                tensor = spec.materialize(gms_client_memory_manager, device_index)
                private = tensor.detach().clone()
                _swap_tensor_contents(existing, private, name=name)
                materialized.add(id(existing))
            else:
                tensor = spec.materialize(gms_client_memory_manager, device_index)
                private = tensor.detach().clone()
                if tensor_type == "buffer" and attr in mod._buffers:
                    mod._buffers[attr] = private
                elif attr.isdigit() or isinstance(
                    getattr(type(mod), attr, None), property
                ):
                    raise RuntimeError(
                        f"Cannot materialize GMS tensor {name!r} without "
                        "an existing Tensor object"
                    )
                else:
                    setattr(mod, attr, private)
            continue

        tensor = spec.materialize(gms_client_memory_manager, device_index)

        # A registered Parameter always stays on the dedicated parameter path,
        # even when this metadata entry names an ordinary instance alias.
        if parameter is not None:
            if parameter.is_meta or parameter.device != tensor.device:
                replacement = _make_parameter_replacement(
                    parameter,
                    tensor,
                    name=name,
                )
                _swap_tensor_contents(parameter, replacement, name=name)
            else:
                parameter.data = tensor
            materialized.add(id(parameter))
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
    Discovery is limited to registered fields and direct module instance
    attributes (including tensor lists/tuples); aliases hidden in arbitrary
    helper objects are not inspected. Observed distinct tensor objects that
    share storage (for example, views) are unsupported.

    Returns the number of bytes rebound, i.e. how much memory is duplicated
    between the read-only GMS copies and the private clones.

    Displaced GMS TensorImpls remain owned by a private model holder for the
    model's lifetime. If ``retain_gms_tensors`` is provided, each displaced
    owner is also appended for compatibility with deferred publication.
    """
    mappings = gms_client_memory_manager.mappings
    rebound_bytes = 0
    module_tensors = list(_iter_module_tensors(model))
    parameters = {
        id(tensor): tensor
        for _, tensor, tensor_type in module_tensors
        if tensor_type == "parameter"
    }
    seen: dict[int, torch.Tensor] = {}
    candidates: list[tuple[str, torch.Tensor]] = []
    storage_owners: dict[int, tuple[str, torch.Tensor]] = {}
    for name, tensor, tensor_type in module_tensors:
        mod, attr = _resolve_module_attr(model, name)
        if _resolve_existing_tensor(model, name, mod, attr, tensor_type) is not tensor:
            continue
        if seen.get(id(tensor)) is tensor:
            continue
        seen[id(tensor)] = tensor
        ptr = int(tensor.data_ptr())
        if not any(
            va <= ptr < va + mapping.aligned_size for va, mapping in mappings.items()
        ):
            continue

        is_candidate = not (
            tensor_type == "parameter" or parameters.get(id(tensor)) is tensor
        )
        if tensor._base is not None:
            raise RuntimeError(f"GMS cannot rebind tensor view {name!r}")
        if is_candidate:
            _validate_plain_nonparameter(tensor, name=name)

        storage_ptr = int(tensor.untyped_storage().data_ptr())
        if storage_ptr == 0:
            if is_candidate:
                candidates.append((name, tensor))
            continue
        previous = storage_owners.get(storage_ptr)
        if previous is not None and previous[1] is not tensor:
            raise RuntimeError(
                "GMS cannot rebind distinct tensors that share storage: "
                f"{previous[0]!r} and {name!r}"
            )
        storage_owners[storage_ptr] = (name, tensor)
        if is_candidate:
            candidates.append((name, tensor))

    # Drop discovery references before swap_tensors checks TensorImpl use counts.
    module_tensors.clear()
    tensor = None
    mod = None
    prepared = [(name, tensor, tensor.detach().clone()) for name, tensor in candidates]
    if not prepared:
        return 0
    owners = _get_or_create_rebound_tensor_owners(model)
    owner_count = len(owners.tensors)
    retained_count = len(retain_gms_tensors) if retain_gms_tensors is not None else 0
    completed: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    try:
        for name, tensor, private in prepared:
            _swap_tensor_contents(tensor, private, name=name)
            completed.append((name, tensor, private))
    except BaseException as swap_error:
        rollback_error: BaseException | None = None
        for name, tensor, private in reversed(completed):
            try:
                _swap_tensor_contents(tensor, private, name=name)
            except BaseException as exc:
                rollback_error = rollback_error or exc
        del owners.tensors[owner_count:]
        if retain_gms_tensors is not None:
            del retain_gms_tensors[retained_count:]
        if owner_count == 0 and not owners.tensors:
            model.__dict__.pop(_REBOUND_TENSOR_OWNERS_ATTR, None)
        if rollback_error is not None:
            raise RuntimeError(
                "GMS tensor rebind failed and rollback was incomplete"
            ) from rollback_error
        raise swap_error

    displaced = [private for _, _, private in completed]
    owners.tensors.extend(displaced)
    if retain_gms_tensors is not None:
        retain_gms_tensors.extend(displaced)
    rebound_bytes = sum(
        tensor.numel() * tensor.element_size() for _, tensor, _ in completed
    )

    return rebound_bytes
