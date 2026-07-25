# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alias-preserving isolation of tensors from shared allocator storage."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _tensor_span(tensor: "torch.Tensor") -> tuple[int, int]:
    element_size = int(tensor.element_size())
    start = int(tensor.storage_offset())
    end = start
    for size, stride in zip(tensor.shape, tensor.stride()):
        extent = (int(size) - 1) * int(stride)
        start += min(extent, 0)
        end += max(extent, 0)
    return start * element_size, (end + 1) * element_size


def _rebind(
    tensors: list["torch.Tensor"],
    storage: "torch.UntypedStorage",
    source_start: int,
) -> None:
    import torch

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


def isolate_tensors(
    tensors: Iterable["torch.Tensor"],
    retain_sources: list["torch.Tensor"] | None = None,
) -> int:
    """Copy tensor spans to private storage while preserving objects and aliases."""
    import torch

    by_storage: dict[int, tuple[torch.UntypedStorage, dict[int, torch.Tensor]]] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        _, objects = by_storage.setdefault(int(storage._cdata), (storage, {}))
        objects[int(tensor._cdata)] = tensor

    copied_bytes = 0
    for storage, objects_by_id in by_storage.values():
        objects = list(objects_by_id.values())
        if retain_sources is not None:
            retain_sources.append(
                torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
                    storage,
                    0,
                    (int(storage.nbytes()),),
                    (1,),
                )
            )

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
            (*_tensor_span(tensor), tensor) for tensor in objects if tensor.numel()
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
