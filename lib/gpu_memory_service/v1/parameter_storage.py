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
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.core.client.torch.storage_rebinding import (
    clone_storage_spans_and_rebind_tensors,
    tensor_storage_byte_bounds,
)

if TYPE_CHECKING:
    from gpu_memory_service.core.client.memory_manager import LocalMapping


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
        if mapping.base <= storage_start and storage_end <= mapping.end:
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
    copied_out_bytes = clone_storage_spans_and_rebind_tensors(non_parameters)
    return parameter_span_bytes, copied_out_bytes
