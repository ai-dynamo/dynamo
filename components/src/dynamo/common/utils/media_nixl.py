# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import uuid
from typing import Any, Dict, Tuple

import numpy as np
import torch

from dynamo import nixl_connect
from dynamo.nixl_connect import OperationKind, RdmaMetadata, SerializedDescriptor

logger = logging.getLogger(__name__)


async def read_decoded_media_via_nixl(
    connector: nixl_connect.Connector,
    decoded_meta: Dict[str, Any],
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read pre-decoded media data via NIXL RDMA transfer, into a CPU numpy array.

    Args:
        connector: Initialized NIXL connector for RDMA operations.
        decoded_meta: Metadata dict from the frontent, containing nixl_metadata, shape, dtype, nixl_descriptor, and metadata.

    Returns:
        np.ndarray containing the transferred media data.
        Dict[str, Any] containing the media metadata.
    """
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
    alloc_start = time.perf_counter()
    tensor = torch.empty(shape, dtype=getattr(torch, dtype_str))
    alloc_end = time.perf_counter()
    local_descriptor = nixl_connect.Descriptor(tensor)

    read_start = time.perf_counter()
    read_op = await connector.begin_read(rdma_metadata, local_descriptor)
    await read_op.wait_for_completion()
    read_end = time.perf_counter()

    logger.debug(
        f"Loaded media via NIXL RDMA: shape={shape}, "
        f"read_time={read_end - read_start:.4f}s, "
        f"alloc_time={alloc_end - alloc_start:.6f}s"
    )

    array = tensor.numpy()  # zero-copy
    array = array[..., :3]  # ignore alpha
    if return_metadata:
        return array, decoded_meta.get("metadata")
    else:
        return array
