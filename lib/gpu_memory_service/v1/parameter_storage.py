# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move non-Parameter tensors out of GMS storage before weight publication.

Dynamo Snapshot invokes this copy only after a whole engine has been put to
sleep by GMS. Restore preserves the same Python/Torch process state, TensorImpls,
post-partition StorageImpl graph, layouts, allocation IDs, and CUDA VA
reservations. Model construction, loading, and this copy do not run again after
restore. The rank-local GMS sidecar survives separately and retains committed
weight backing. Copies made with Torch's default allocator are ordinary
process-owned Snapshot state. KV physical backing and contents are not retained;
fresh backing is mapped at preserved VAs.

One GMS source storage may contain Parameters and other tensor views::

    before
    GMS storage: [0-------------63]
                  |Parameter|
                       |view A------|
                            |view B|
                                          |view C|

    after
    GMS RO:      [0-------------63]
                  |Parameter|
    default #1:       [view A------]
                          [view B]       (overlap and relative offsets preserved)
    default #2:                            [view C]

Mixed Parameter/non-Parameter aliasing is deliberately severed. Each overlapping
connected component of non-empty bounding storage byte ranges gets one compact
copy; disjoint ranges get separate storage. A copied component may rebase its
absolute storage offset, but relative aliases and offsets within it are preserved.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gpu_memory_service.v1.mapping import LocalMapping


def tensor_storage_byte_bounds(tensor: "torch.Tensor") -> tuple[int, int]:
    """Return the bounding storage-relative byte range touched by a tensor."""
    element_size = int(tensor.element_size())
    start = int(tensor.storage_offset())
    end = start
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        start += min(extent, 0)
        end += max(extent, 0)
    return start * element_size, (end + 1) * element_size


def _rebind(
    tensors: list["torch.Tensor"],
    storage: "torch.UntypedStorage",
    source_start: int,
) -> None:
    with torch.inference_mode():
        for tensor in tensors:
            tensor.set_(
                storage,
                (
                    int(tensor.storage_offset())
                    - source_start // int(tensor.element_size())
                ),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )


def _clone_storage_spans_and_rebind_tensors(
    tensors: Iterable["torch.Tensor"],
) -> int:
    """Copy overlapping storage spans while preserving TensorImpls and aliases."""
    by_storage: dict[int, tuple[torch.UntypedStorage, dict[int, torch.Tensor]]] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        _, objects = by_storage.setdefault(int(storage._cdata), (storage, {}))
        objects[int(tensor._cdata)] = tensor

    copied_bytes = 0
    for storage, objects_by_id in by_storage.values():
        objects = list(objects_by_id.values())
        zero_elements = [tensor for tensor in objects if not tensor.numel()]
        if zero_elements:
            target = torch.empty(
                0,
                dtype=torch.uint8,
                device=storage.device,
            ).untyped_storage()
            _rebind(zero_elements, target, 0)

        groups: list[tuple[int, int, list[torch.Tensor]]] = []
        spans = [
            (*tensor_storage_byte_bounds(tensor), tensor)
            for tensor in objects
            if tensor.numel()
        ]
        for start, end, tensor in sorted(spans, key=lambda item: (item[0], item[1])):
            if groups and start < groups[-1][1]:
                group_start, group_end, group_tensors = groups[-1]
                groups[-1] = (
                    group_start,
                    max(group_end, end),
                    [*group_tensors, tensor],
                )
            else:
                groups.append((start, end, [tensor]))

        for start, end, group_tensors in groups:
            alignment = math.lcm(
                *(int(tensor.element_size()) for tensor in group_tensors)
            )
            source_start = start // alignment * alignment
            source = torch.empty(
                0,
                dtype=torch.uint8,
                device=storage.device,
            ).set_(
                storage,
                source_start,
                (end - source_start,),
                (1,),
            )
            target_storage = source.clone().untyped_storage()
            copied_bytes += int(target_storage.nbytes())
            _rebind(group_tensors, target_storage, source_start)
    return copied_bytes


def _iter_live_tensors(model: object):
    yield from model.parameters()
    for value in gc.get_objects():
        if (
            isinstance(type(value), torch._C._TensorMeta)
            and value.layout is torch.strided
        ):
            yield value


def _discover_live_tensors(model: object) -> list[tuple["torch.Tensor", bool]]:
    objects: dict[int, tuple[torch.Tensor, bool]] = {}
    for tensor in _iter_live_tensors(model):
        tensor_id = int(tensor._cdata)
        tensor_object = objects.get(tensor_id)
        if tensor_object is None:
            objects[tensor_id] = (
                tensor,
                isinstance(tensor, torch.nn.Parameter),
            )
        elif isinstance(tensor, torch.nn.Parameter):
            objects[tensor_id] = (tensor_object[0], True)
    return list(objects.values())


def _containing_mapping(
    tensor: "torch.Tensor",
    mappings: tuple["LocalMapping", ...],
) -> "LocalMapping | None":
    storage = tensor.untyped_storage()
    storage_start = int(storage.data_ptr())
    storage_end = storage_start + int(storage.nbytes())
    for mapping in mappings:
        if (
            mapping.base <= storage_start
            and storage_end <= mapping.base + mapping.aligned_size
        ):
            return mapping
    return None


def copy_non_parameter_tensors_to_default_allocator(
    model: object,
    mappings: tuple["LocalMapping", ...],
) -> tuple[int, int]:
    """Copy live non-Parameters out of GMS and return span/copy byte counts."""
    gc.collect()
    retained_parameter_spans: list[tuple[int, int]] = []
    non_parameters: list[torch.Tensor] = []
    for tensor, is_parameter in _discover_live_tensors(model):
        if _containing_mapping(tensor, mappings) is None:
            continue
        if not is_parameter:
            non_parameters.append(tensor)
            continue
        if tensor.numel():
            storage_start = int(tensor.untyped_storage().data_ptr())
            start, end = tensor_storage_byte_bounds(tensor)
            retained_parameter_spans.append(
                (storage_start + start, storage_start + end)
            )

    parameter_span_bytes = 0
    retained_end = 0
    for start, end in sorted(retained_parameter_spans):
        parameter_span_bytes += max(end - max(start, retained_end), 0)
        retained_end = max(retained_end, end)
    copied_out_bytes = _clone_storage_spans_and_rebind_tensors(non_parameters)
    return parameter_span_bytes, copied_out_bytes
