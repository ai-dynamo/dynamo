# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor storage utilities and the GMS module storage manifest."""

from __future__ import annotations

from enum import Enum

import msgspec
import torch

STORAGE_MANIFEST_PREFIX = "torch.module.storage/"


class ModuleTensorKind(str, Enum):
    """Classify how a tensor is bound to a module."""

    PARAMETER = "parameter"
    PERSISTENT_BUFFER = "persistent_buffer"
    NONPERSISTENT_BUFFER = "nonpersistent_buffer"
    ATTRIBUTE = "attribute"


class ModuleTensorBinding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Name one module binding for a tensor object."""

    path: str
    kind: ModuleTensorKind


class TensorObject(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Describe one tensor identity within a storage manifest."""

    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset_bytes: int
    requires_grad: bool
    bindings: tuple[ModuleTensorBinding, ...]


class StorageManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Describe one shared StorageImpl and its tensor objects."""

    nbytes: int
    objects: tuple[TensorObject, ...]


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype) or str(dtype) != f"torch.{name}":
        raise ValueError(f"Unknown or noncanonical dtype: {name!r}")
    return dtype


def _storage_from_pointer(
    data_ptr: int,
    size_bytes: int,
    device_index: int,
) -> torch.UntypedStorage:
    """Create non-owning CUDA storage for an existing mapped byte range."""
    if data_ptr < 0 or size_bytes < 0:
        raise ValueError("Storage pointer and size must be nonnegative")
    return torch._C._construct_storage_from_data_pointer(
        data_ptr,
        torch.device("cuda", device_index),
        size_bytes,
    )


def _tensor_from_storage(
    storage: torch.UntypedStorage,
    shape: list[int],
    stride: list[int],
    dtype: torch.dtype,
    storage_offset: int = 0,
) -> torch.Tensor:
    """Create an independent TensorImpl wrapper over ``storage``."""
    _validate_layout(shape, stride, dtype, storage_offset, storage.nbytes())
    return torch.empty(0, dtype=dtype, device=storage.device).set_(
        storage,
        storage_offset,
        shape,
        stride,
    )


def _tensor_from_pointer(
    data_ptr: int,
    shape: list[int],
    stride: list[int],
    dtype: torch.dtype,
    device_index: int,
) -> torch.Tensor:
    """Create a non-owning CUDA tensor from a raw pointer."""
    storage_size_bytes = _layout_end_bytes(shape, stride, dtype, 0)
    storage = _storage_from_pointer(data_ptr, storage_size_bytes, device_index)
    return _tensor_from_storage(storage, shape, stride, dtype)


def _layout_end_bytes(
    shape: list[int] | tuple[int, ...],
    stride: list[int] | tuple[int, ...],
    dtype: torch.dtype,
    storage_offset: int,
) -> int:
    """Return the exclusive byte end needed by a nonnegative strided layout."""
    if len(shape) != len(stride):
        raise ValueError(
            f"Shape and stride length mismatch: {len(shape)} vs {len(stride)}"
        )
    if any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("Tensor shape must contain nonnegative integers")
    if any(type(step) is not int or step < 0 for step in stride):
        raise ValueError("Tensor stride must contain nonnegative integers")
    if type(storage_offset) is not int or storage_offset < 0:
        raise ValueError("Tensor storage offset must be a nonnegative integer")
    if any(dim == 0 for dim in shape):
        return storage_offset * dtype.itemsize
    last_element = storage_offset + sum(
        step * (dim - 1) for dim, step in zip(shape, stride, strict=True)
    )
    return (last_element + 1) * dtype.itemsize


def _validate_layout(
    shape: list[int] | tuple[int, ...],
    stride: list[int] | tuple[int, ...],
    dtype: torch.dtype,
    storage_offset: int,
    storage_nbytes: int,
) -> None:
    if type(storage_nbytes) is not int or storage_nbytes < 0:
        raise ValueError("Storage size must be a nonnegative integer")
    if _layout_end_bytes(shape, stride, dtype, storage_offset) > storage_nbytes:
        raise ValueError("Tensor layout exceeds its storage bounds")
