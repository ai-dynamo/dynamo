# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple, overload

import numpy as np
import torch

if TYPE_CHECKING:
    from dynamo import nixl_connect


_TORCH_DTYPES = {
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int64": torch.int64,
    "float64": torch.float64,
}


@overload
async def read_decoded_media_via_nixl(
    connector: nixl_connect.Connector,
    decoded_meta: Dict[str, Any],
    return_metadata: Literal[False] = False,
    *,
    trim_alpha: bool = True,
) -> np.ndarray:
    ...


@overload
async def read_decoded_media_via_nixl(
    connector: nixl_connect.Connector,
    decoded_meta: Dict[str, Any],
    return_metadata: Literal[True],
    *,
    trim_alpha: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any] | None]:
    ...


async def read_decoded_media_via_nixl(
    connector: nixl_connect.Connector,
    decoded_meta: Dict[str, Any],
    return_metadata: bool = False,
    *,
    trim_alpha: bool = True,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any] | None]:
    """
    Read pre-decoded media data via NIXL RDMA transfer, into a CPU numpy array.

    Args:
        connector: Initialized NIXL connector for RDMA operations.
        decoded_meta: Metadata dict from the frontent, containing nixl_metadata, shape, dtype, nixl_descriptor, and metadata.

    Returns:
        np.ndarray containing the transferred media data.
        Dict[str, Any] containing the media metadata.
    """
    try:
        from dynamo import nixl_connect
        from dynamo.nixl_connect import (
            OperationKind,
            RdmaMetadata,
            SerializedDescriptor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "NIXL is required to read decoded media via frontend decoding; "
            "install dynamo.nixl_connect to enable RDMA media transfers."
        ) from exc

    rdma_metadata = decoded_meta["nixl_metadata"]
    descriptor = decoded_meta["nixl_descriptor"]
    remote_device = (
        "cpu"
        if descriptor.get("mem_type", "dram").lower() == "dram"
        else f"cuda:{descriptor.get('device_id', 0)}"
    )

    rdma_metadata = RdmaMetadata(
        descriptors=[
            SerializedDescriptor(
                device=remote_device,
                ptr=descriptor["addr"],
                size=descriptor["size"],
            )
        ],
        nixl_metadata=rdma_metadata,
        notification_key=str(uuid.uuid4()),
        operation_kind=int(OperationKind.READ),
    )

    # Create empty tensor to receive RDMA data
    shape = decoded_meta["shape"]
    dtype_str = decoded_meta.get("dtype", "uint8").lower()
    try:
        dtype = _TORCH_DTYPES[dtype_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported media tensor dtype: {dtype_str!r}") from exc
    tensor = torch.empty(shape, dtype=dtype)
    local_descriptor = nixl_connect.Descriptor(tensor)

    read_op = await connector.begin_read(rdma_metadata, local_descriptor)
    await read_op.wait_for_completion()

    array = tensor.numpy()  # zero-copy
    if trim_alpha:
        array = array[..., :3]  # ignore alpha in decoded RGB(A) media
    if return_metadata:
        return array, decoded_meta.get("metadata")
    else:
        return array
