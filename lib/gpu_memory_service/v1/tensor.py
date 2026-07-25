# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize captured tensors for the Snapshot profile."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpu_memory_service.core.client.torch.tensor import isolate_tensors, tensor_span

if TYPE_CHECKING:
    import torch
    from gpu_memory_service.core.client.memory_manager import LocalMapping


@dataclass
class _DiscoveredTensor:
    tensor: "torch.Tensor"
    is_parameter: bool


@dataclass
class _DiscoveredStorage:
    storage: "torch.UntypedStorage"
    objects: list[_DiscoveredTensor]


def _iter_live_tensors(model: object):
    import torch

    yield from model.parameters()
    for value in gc.get_objects():
        if (
            isinstance(type(value), torch._C._TensorMeta)
            and value.layout is torch.strided
        ):
            yield value


def _discover_live_storages(model: object) -> list[_DiscoveredStorage]:
    import torch

    objects: dict[int, _DiscoveredTensor] = {}
    for tensor in _iter_live_tensors(model):
        tensor_id = int(tensor._cdata)
        tensor_object = objects.get(tensor_id)
        if tensor_object is None:
            objects[tensor_id] = _DiscoveredTensor(
                tensor,
                isinstance(tensor, torch.nn.Parameter),
            )
        elif isinstance(tensor, torch.nn.Parameter):
            tensor_object.is_parameter = True

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
    for mapping in mappings:
        if mapping.base <= storage_start and storage_end <= mapping.end:
            return mapping
    return None


def normalize_captured_tensors(
    model: object,
    mappings: tuple["LocalMapping", ...],
) -> tuple[int, int]:
    """Rebind every captured non-Parameter TensorImpl to cloned storage."""
    gc.collect()
    retained_parameter_spans: list[tuple[int, int]] = []
    copied_out_bytes = 0
    for discovered in _discover_live_storages(model):
        if _containing_mapping(discovered, mappings) is None:
            continue
        parameters = [
            tensor_object
            for tensor_object in discovered.objects
            if tensor_object.is_parameter
        ]
        storage_start = int(discovered.storage.data_ptr())
        retained_parameter_spans.extend(
            (storage_start + start, storage_start + end)
            for tensor_object in parameters
            if tensor_object.tensor.numel()
            for start, end in (tensor_span(tensor_object.tensor),)
        )
        non_parameters = [
            tensor_object
            for tensor_object in discovered.objects
            if not tensor_object.is_parameter
        ]
        copied_out_bytes += isolate_tensors(
            [tensor_object.tensor for tensor_object in non_parameters]
        )

    retained_gms_parameter_span_bytes = 0
    retained_end = 0
    for start, end in sorted(retained_parameter_spans):
        retained_gms_parameter_span_bytes += max(end - max(start, retained_end), 0)
        retained_end = max(retained_end, end)
    return retained_gms_parameter_span_bytes, copied_out_bytes
