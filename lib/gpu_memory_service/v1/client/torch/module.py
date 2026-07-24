# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize live module storages before the GMS model-load pool is destroyed."""

from __future__ import annotations

import copyreg
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ..memory_manager import LocalMapping


@dataclass
class _DiscoveredTensor:
    tensor: "torch.Tensor"
    names: list[str]
    is_parameter: bool


@dataclass
class _DiscoveredStorage:
    storage: "torch.UntypedStorage"
    objects: list[_DiscoveredTensor]

    @property
    def has_parameter(self) -> bool:
        return any(tensor_object.is_parameter for tensor_object in self.objects)


def _iter_module_tensors(model: object):
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("GMS V1 model loader did not return a torch.nn.Module")
    for module_path, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_path}." if module_path else ""
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            yield parameter, f"{prefix}{name}", True
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            yield buffer, f"{prefix}{name}", False

        registered = (
            set(module._parameters) | set(module._buffers) | set(module._modules)
        )
        for name, value in module.__dict__.items():
            if name in registered or name.startswith("__"):
                continue
            if torch.is_tensor(value):
                yield value, f"{prefix}{name}", isinstance(value, torch.nn.Parameter)
            elif type(value) in (list, tuple):
                for index, element in enumerate(value):
                    if torch.is_tensor(element):
                        yield (
                            element,
                            f"{prefix}{name}.{index}",
                            isinstance(element, torch.nn.Parameter),
                        )


def _discover_module_storages(model: object) -> list[_DiscoveredStorage]:
    objects: dict[int, _DiscoveredTensor] = {}
    for tensor, name, is_parameter in _iter_module_tensors(model):
        tensor_object = objects.get(id(tensor))
        if tensor_object is None:
            tensor_object = _DiscoveredTensor(tensor, [], False)
            objects[id(tensor)] = tensor_object
        if name not in tensor_object.names:
            tensor_object.names.append(name)
        tensor_object.is_parameter |= is_parameter

    storages: dict[int, _DiscoveredStorage] = {}
    for tensor_object in objects.values():
        storage = tensor_object.tensor.untyped_storage()
        storage_id = int(storage._cdata)
        discovered = storages.get(storage_id)
        if discovered is None:
            discovered = _DiscoveredStorage(storage, [])
            storages[storage_id] = discovered
        discovered.objects.append(tensor_object)
    return list(storages.values())


def _containing_mapping(
    discovered: _DiscoveredStorage,
    mappings: tuple["LocalMapping", ...],
) -> "LocalMapping | None":
    storage_start = int(discovered.storage.data_ptr())
    storage_end = storage_start + int(discovered.storage.nbytes())
    containing = None
    for mapping in mappings:
        overlaps = max(mapping.base, storage_start) < min(mapping.end, storage_end)
        contains = mapping.base <= storage_start and storage_end <= mapping.end
        if overlaps and not contains:
            raise RuntimeError(
                f"Storage {discovered.objects[0].names[0]!r} crosses a GMS mapping"
            )
        if contains:
            if containing is not None:
                raise RuntimeError(
                    f"Storage {discovered.objects[0].names[0]!r} belongs to "
                    "multiple GMS mappings"
                )
            containing = mapping
    return containing


def _tensor_from_storage(
    template: "torch.Tensor",
    storage: "torch.UntypedStorage",
) -> "torch.Tensor":
    import torch

    if type(template) is not torch.Tensor:
        raise RuntimeError(
            f"GMS V1 cannot normalize tensor subclass {type(template).__name__}"
        )
    if template.is_conj() or template.is_neg() or template.grad_fn is not None:
        raise RuntimeError("GMS V1 cannot normalize a lazy or autograd tensor")
    replacement = torch.empty(0, dtype=template.dtype, device=template.device).set_(
        storage,
        int(template.storage_offset()),
        tuple(template.shape),
        tuple(template.stride()),
    )
    replacement.requires_grad_(template.requires_grad)
    return replacement


def _swap_tensors(
    objects: list[_DiscoveredTensor],
    replacements: dict[int, "torch.Tensor"],
) -> None:
    import torch

    object_ids = {id(tensor_object.tensor) for tensor_object in objects}
    base_uses: dict[int, int] = {}
    for tensor_object in objects:
        base = tensor_object.tensor._base
        if base is not None and id(base) in object_ids:
            base_uses[id(base)] = base_uses.get(id(base), 0) + 1

    def base_depth(tensor_object: _DiscoveredTensor) -> int:
        depth = 0
        base = tensor_object.tensor._base
        while base is not None and id(base) in object_ids:
            depth += 1
            base = base._base
        return depth

    ordered = sorted(objects, key=base_depth, reverse=True)
    accumulate_grad_checks: list[tuple["torch.Tensor", int, str]] = []
    for tensor_object in ordered:
        existing = tensor_object.tensor
        replacement = replacements[id(existing)]
        name = tensor_object.names[0]
        for tensor, label, released_uses in (
            (existing, "existing", base_uses.get(id(existing), 0)),
            (replacement, "replacement", 0),
        ):
            if weakref.getweakrefs(tensor):
                raise RuntimeError(
                    f"Cannot normalize GMS V1 tensor {name!r} with a weak reference"
                )
            use_count = tensor._use_count() - released_uses
            ownership_error = (
                f"Cannot normalize GMS V1 tensor {name!r}: unexpected {label} "
                f"use count {use_count}"
            )
            if use_count > 1:
                if use_count != 2 or not tensor.is_leaf:
                    raise RuntimeError(ownership_error)
                accumulate_grad_checks.append((tensor, released_uses, ownership_error))
        if set(copyreg._slotnames(type(existing))) != set(
            copyreg._slotnames(type(replacement))
        ):
            raise RuntimeError(
                f"Cannot normalize GMS V1 tensor {name!r} with different slots"
            )

    for tensor, released_uses, ownership_error in accumulate_grad_checks:
        torch.autograd.graph.get_gradient_edge(tensor)
        if tensor._use_count() - released_uses != 2:
            raise RuntimeError(ownership_error)

    for tensor_object in ordered:
        replacement = replacements.pop(id(tensor_object.tensor))
        try:
            torch.utils.swap_tensors(tensor_object.tensor, replacement)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot normalize GMS V1 tensor {tensor_object.names[0]!r}: {exc}"
            ) from exc


def normalize_model_storages(
    model: object,
    mappings: tuple["LocalMapping", ...],
) -> None:
    """Clone each complete non-Parameter-only GMS storage once."""
    candidates = [
        discovered
        for discovered in _discover_module_storages(model)
        if _containing_mapping(discovered, mappings) is not None
        and not discovered.has_parameter
    ]
    replacements = {}
    objects = []
    for discovered in candidates:
        target_storage = discovered.storage.clone()
        for tensor_object in discovered.objects:
            replacements[id(tensor_object.tensor)] = _tensor_from_storage(
                tensor_object.tensor, target_storage
            )
            objects.append(tensor_object)
    _swap_tensors(objects, replacements)


def validate_model_storage_ownership(
    model: object,
    mappings: tuple["LocalMapping", ...],
) -> None:
    """Validate that surviving GMS mappings back only Parameter storages."""
    discovered = _discover_module_storages(model)
    parameter_storage_ids = {
        int(storage.storage._cdata)
        for storage in discovered
        if storage.has_parameter
        and storage.storage.device.type == "cuda"
        and storage.storage.nbytes()
    }
    parameter_mappings = set()
    for storage in discovered:
        if storage.storage.device.type != "cuda":
            continue
        if storage.has_parameter and not storage.storage.nbytes():
            continue
        mapping = _containing_mapping(storage, mappings)
        storage_id = int(storage.storage._cdata)
        if storage.has_parameter:
            if mapping is None:
                raise RuntimeError(
                    f"CUDA Parameter storage {storage.objects[0].names[0]!r} "
                    "is not contained in a surviving GMS mapping"
                )
            parameter_mappings.add(mapping.base)
        elif (
            storage.storage.nbytes()
            and mapping is not None
            and storage_id not in parameter_storage_ids
        ):
            raise RuntimeError(
                f"Non-Parameter storage {storage.objects[0].names[0]!r} "
                "still overlaps a surviving GMS mapping"
            )
    missing = {mapping.base for mapping in mappings} - parameter_mappings
    if missing:
        raise RuntimeError(
            "Surviving GMS mapping has no Parameter storage: "
            + ", ".join(f"0x{base:x}" for base in sorted(missing))
        )
