# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor utilities for GPU Memory Service.

This module provides low-level tensor functionality:
- Tensor creation from CUDA pointers
- Tensor metadata serialization/deserialization
- GMS tensor spec for metadata store entries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager


# =============================================================================
# Tensor Creation from CUDA Pointer
# =============================================================================


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
    shape: List[int],
    stride: List[int],
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
    shape: List[int],
    stride: List[int],
    dtype: torch.dtype,
    device_index: int,
) -> torch.Tensor:
    """Create a torch.Tensor from a raw CUDA pointer without copying data.

    Uses PyTorch's internal APIs to create a tensor that aliases existing
    GPU memory. The tensor does NOT own the memory - the caller must ensure
    the memory remains valid for the tensor's lifetime.

    Args:
        data_ptr: CUDA device pointer (virtual address) to the tensor data.
        shape: Tensor dimensions.
        stride: Tensor strides (in elements, not bytes).
        dtype: Tensor data type.
        device_index: CUDA device index where the memory resides.

    Returns:
        A tensor aliasing the specified GPU memory.
    """
    storage_size_bytes = _layout_end_bytes(shape, stride, dtype, 0)
    storage = _storage_from_pointer(data_ptr, storage_size_bytes, device_index)
    return _tensor_from_storage(storage, shape, stride, dtype)


# =============================================================================
# Tensor Metadata - serialization format for metadata store
# =============================================================================


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string (e.g., 'torch.float16') to torch.dtype."""
    s = str(dtype_str)
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    dt = getattr(torch, s, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Unknown dtype: {dtype_str!r}")
    return dt


_TENSOR_TYPES = frozenset(("parameter", "buffer", "tensor_attr"))
_STORAGE_FIELDS = frozenset(
    (
        "schema_version",
        "storage_group_id",
        "object_group_id",
        "storage_base_offset",
        "storage_nbytes",
        "storage_offset",
        "buffer_persistent",
    )
)
_BASE_FIELDS = frozenset(("shape", "dtype", "stride", "tensor_type"))


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_int_list(value: object, field: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(_strict_int(item, field) for item in value)


def _layout_end_bytes(
    shape: List[int] | Tuple[int, ...],
    stride: List[int] | Tuple[int, ...],
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
    element_size = torch.empty((), dtype=dtype).element_size()
    if any(dim == 0 for dim in shape):
        return storage_offset * element_size
    last_element = storage_offset + sum(
        step * (dim - 1) for dim, step in zip(shape, stride, strict=True)
    )
    return (last_element + 1) * element_size


def _validate_layout(
    shape: List[int] | Tuple[int, ...],
    stride: List[int] | Tuple[int, ...],
    dtype: torch.dtype,
    storage_offset: int,
    storage_nbytes: int,
) -> None:
    if type(storage_nbytes) is not int or storage_nbytes < 0:
        raise ValueError("Storage size must be a nonnegative integer")
    if _layout_end_bytes(shape, stride, dtype, storage_offset) > storage_nbytes:
        raise ValueError("Tensor layout exceeds its storage bounds")


@dataclass(frozen=True)
class TensorMetadata:
    """Metadata for a tensor stored in the GMS metadata store."""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    stride: Tuple[int, ...]
    tensor_type: str = "parameter"  # "parameter", "buffer", or "tensor_attr"
    storage: dict[str, object] | None = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        tensor_type: str = "parameter",
        *,
        storage: dict[str, object] | None = None,
    ) -> "TensorMetadata":
        """Create TensorMetadata from an existing tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(int(s) for s in tensor.stride()),
            tensor_type=tensor_type,
            storage=storage,
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "TensorMetadata":
        """Parse metadata from JSON bytes."""
        obj = json.loads(value.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Tensor metadata must be a JSON object")
        is_storage_component = "storage" in obj
        if is_storage_component and set(obj) != _BASE_FIELDS | {"storage"}:
            raise ValueError("Storage-component metadata has unknown or missing fields")
        shape = _strict_int_list(obj["shape"], "shape")
        dtype = _parse_dtype(obj["dtype"])

        if is_storage_component:
            stride = _strict_int_list(obj["stride"], "stride")
        elif "stride" in obj and obj["stride"] is not None:
            stride = _strict_int_list(obj["stride"], "stride")
        else:
            # Legacy format: compute contiguous stride
            stride = []
            acc = 1
            for d in reversed(shape):
                stride.append(acc)
                acc *= d
            stride = tuple(reversed(stride)) if stride else ()

        if len(shape) != len(stride):
            raise ValueError("Tensor shape and stride lengths differ")
        tensor_type = obj.get("tensor_type", "parameter")
        if tensor_type not in _TENSOR_TYPES:
            raise ValueError(f"Unknown tensor type: {tensor_type!r}")

        if "storage" not in obj:
            return cls(
                shape=shape,
                dtype=dtype,
                stride=stride,
                tensor_type=tensor_type,
            )
        storage = obj["storage"]
        if not isinstance(storage, dict) or set(storage) != _STORAGE_FIELDS:
            raise ValueError("Storage-component metadata fields must be exact")
        if _strict_int(storage["schema_version"], "schema_version") != 1:
            raise ValueError("Unsupported storage-component schema version")
        if type(storage["buffer_persistent"]) is not bool:
            raise ValueError("buffer_persistent must be a boolean")
        normalized = {
            field: _strict_int(storage[field], field)
            for field in _STORAGE_FIELDS
            if field not in ("schema_version", "buffer_persistent")
        }
        normalized["schema_version"] = 1
        normalized["buffer_persistent"] = storage["buffer_persistent"]
        return cls(shape, dtype, stride, tensor_type, normalized)

    def to_bytes(self) -> bytes:
        """Serialize to JSON bytes for metadata store."""
        obj = {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "stride": list(self.stride),
            "tensor_type": self.tensor_type,
        }
        if self.storage is not None:
            obj["storage"] = self.storage
        return json.dumps(obj, sort_keys=True).encode("utf-8")

    @property
    def is_storage_component(self) -> bool:
        return self.storage is not None


# =============================================================================
# GMS Tensor Spec - metadata entry from store
# =============================================================================


@dataclass(frozen=True)
class GMSTensorSpec:
    """A tensor entry from the GMS metadata store."""

    key: str
    name: str
    allocation_id: str
    offset_bytes: int
    meta: TensorMetadata

    @classmethod
    def load_all(
        cls, gms_client_memory_manager: "GMSClientMemoryManager"
    ) -> Dict[str, "GMSTensorSpec"]:
        """Load all metadata entries.

        Returns:
            Mapping of tensor name -> GMSTensorSpec.
        """
        specs: Dict[str, GMSTensorSpec] = {}

        for key in gms_client_memory_manager.metadata_list():
            got = gms_client_memory_manager.metadata_get(key)
            if got is None:
                raise RuntimeError(f"Metadata key disappeared: {key}")

            allocation_id, offset_bytes, value = got

            if key in specs:
                raise RuntimeError(f"Duplicate tensor name: {key}")

            specs[key] = cls(
                key=key,
                name=key,
                allocation_id=str(allocation_id),
                offset_bytes=int(offset_bytes),
                meta=TensorMetadata.from_bytes(value),
            )

        return specs

    def materialize(
        self,
        gms_client_memory_manager: "GMSClientMemoryManager",
        device_index: int,
    ) -> torch.Tensor:
        """Create a tensor aliasing mapped CUDA memory."""
        base_va = gms_client_memory_manager.create_mapping(
            allocation_id=self.allocation_id
        )
        mapping = gms_client_memory_manager.mappings[base_va]
        allocation_nbytes = int(mapping.aligned_size)
        tensor_nbytes = _layout_end_bytes(
            self.meta.shape,
            self.meta.stride,
            self.meta.dtype,
            0,
        )
        if (
            self.offset_bytes < 0
            or self.offset_bytes + tensor_nbytes > allocation_nbytes
        ):
            raise ValueError(f"Tensor {self.name!r} exceeds allocation bounds")
        ptr = int(base_va) + int(self.offset_bytes)

        return _tensor_from_pointer(
            ptr,
            list(self.meta.shape),
            list(self.meta.stride),
            self.meta.dtype,
            device_index,
        )
