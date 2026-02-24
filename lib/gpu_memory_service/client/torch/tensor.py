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
    device = torch.device("cuda", device_index)

    # Calculate storage size in bytes based on stride (handles non-contiguous tensors)
    # For non-contiguous tensors, the memory footprint is larger than numel * element_size
    element_size = torch.tensor([], dtype=dtype).element_size()

    if shape and stride:
        if len(shape) != len(stride):
            raise ValueError(
                f"Shape and stride length mismatch: {len(shape)} vs {len(stride)}"
            )
        # Maximum offset = sum of stride[i] * (shape[i] - 1) for all dimensions
        max_offset = sum(
            s * (d - 1) for s, d in zip(stride, shape, strict=True) if d > 0
        )
        required_elements = max_offset + 1
    else:
        # Scalar tensor or empty tensor
        required_elements = 1

    storage_size_bytes = required_elements * element_size

    # Create storage from raw pointer (does not take ownership)
    storage = torch._C._construct_storage_from_data_pointer(
        data_ptr, device, storage_size_bytes
    )

    # Create tensor from storage with metadata
    metadata = {
        "size": torch.Size(shape),
        "stride": stride,
        "storage_offset": 0,
        "dtype": dtype,
    }

    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)


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


@dataclass(frozen=True)
class TensorMetadata:
    """Metadata for a tensor stored in the GMS metadata store."""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    stride: Tuple[int, ...]
    tensor_type: str = "parameter"  # "parameter", "buffer", or "tensor_attr"

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, tensor_type: str = "parameter"
    ) -> "TensorMetadata":
        """Create TensorMetadata from an existing tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(int(s) for s in tensor.stride()),
            tensor_type=tensor_type,
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "TensorMetadata":
        """Parse metadata from JSON bytes."""
        obj = json.loads(value.decode("utf-8"))
        shape = tuple(int(x) for x in obj["shape"])
        dtype = _parse_dtype(obj["dtype"])

        if "stride" in obj and obj["stride"] is not None:
            stride = tuple(int(x) for x in obj["stride"])
        else:
            # Legacy format: compute contiguous stride
            stride = []
            acc = 1
            for d in reversed(shape):
                stride.append(acc)
                acc *= d
            stride = tuple(reversed(stride)) if stride else ()

        return cls(
            shape=shape,
            dtype=dtype,
            stride=stride,
            tensor_type=obj.get("tensor_type", "parameter"),
        )

    def to_bytes(self) -> bytes:
        """Serialize to JSON bytes for metadata store."""
        return json.dumps(
            {
                "shape": list(self.shape),
                "dtype": str(self.dtype),
                "stride": list(self.stride),
                "tensor_type": self.tensor_type,
            },
            sort_keys=True,
        ).encode("utf-8")


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
        base_va = gms_client_memory_manager.import_allocation(self.allocation_id)
        ptr = int(base_va) + int(self.offset_bytes)

        return _tensor_from_pointer(
            ptr,
            list(self.meta.shape),
            list(self.meta.stride),
            self.meta.dtype,
            device_index,
        )
