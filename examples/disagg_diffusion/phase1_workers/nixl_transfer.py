# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL RDMA tensor transfer for disaggregated diffusion stages.

Provides GPU-direct tensor transfer between stage workers. Only small
metadata (shapes, dtypes, NIXL descriptor ~1.5 KB) travels over the ZMQ
control plane; actual tensor data (embeddings, latents) transfers
GPU→GPU via NIXL RDMA.

Usage inside PipelineStage.forward() (synchronous context)::

    sender = NixlTensorSender()
    meta = sender.send({"latents": tensor})   # registers & returns metadata
    # ... pass meta via ZMQ ...

    receiver = NixlTensorReceiver()
    tensors = receiver.recv(meta, device="cuda")  # RDMA pull
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

try:
    import dynamo.nixl_connect as nixl_connect
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False
    logger.info("NIXL not available — falling back to ZMQ tensor transfer")


class _PersistentConnector:
    """Lazily-initialized NIXL Connector singleton per process."""

    _instance: Optional[nixl_connect.Connector] = None if NIXL_AVAILABLE else None

    @classmethod
    async def get(cls) -> nixl_connect.Connector:
        if cls._instance is None:
            cls._instance = nixl_connect.Connector()
            await cls._instance.initialize()
        return cls._instance


class NixlTensorSender:
    """Register GPU tensors as NIXL-readable. Returns metadata for the receiver.

    The readable is kept alive via a background task so the sender process
    can return immediately after yielding metadata.
    """

    def __init__(self):
        self._pending: list = []

    def send(self, tensors: Dict[str, torch.Tensor]) -> dict:
        """Register tensors and return metadata dict (synchronous wrapper)."""
        return asyncio.run(self._async_send(tensors))

    async def _async_send(self, tensors: Dict[str, torch.Tensor]) -> dict:
        # Clean completed tasks
        self._pending = [t for t in self._pending if not t.done()]

        connector = await _PersistentConnector.get()

        # Flatten all tensors into a single contiguous buffer
        flat = torch.cat([t.contiguous().view(-1) for t in tensors.values()])
        descriptor = nixl_connect.Descriptor(flat)
        readable = await connector.create_readable(descriptor)
        raw_meta = readable.metadata()

        meta = {
            "tensor_keys": list(tensors.keys()),
            "shapes": {k: list(t.shape) for k, t in tensors.items()},
            "dtypes": {k: str(t.dtype).removeprefix("torch.") for k, t in tensors.items()},
            "nixl_metadata": raw_meta.model_dump() if hasattr(raw_meta, "model_dump") else raw_meta,
        }

        # Keep readable alive until the receiver has pulled the data
        async def _keep_alive():
            try:
                await readable.wait_for_completion()
            except Exception as e:
                logger.warning("NIXL readable wait failed: %s", e)

        task = asyncio.ensure_future(_keep_alive())
        self._pending.append(task)
        return meta


class NixlTensorReceiver:
    """Pull tensors from a remote sender via NIXL RDMA."""

    def recv(self, meta: dict, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Pull tensors described by metadata. Returns {name: tensor}."""
        return asyncio.run(self._async_recv(meta, device))

    async def _async_recv(self, meta: dict, device: str) -> Dict[str, torch.Tensor]:
        connector = await _PersistentConnector.get()

        # Calculate total size and per-tensor specs
        specs = []
        total_bytes = 0
        for key in meta["tensor_keys"]:
            shape = meta["shapes"][key]
            dtype = getattr(torch, meta["dtypes"][key])
            numel = 1
            for s in shape:
                numel *= s
            size = numel * dtype.itemsize
            specs.append((key, shape, dtype, size))
            total_bytes += size

        # Allocate receive buffer directly on target device (GPU-direct)
        flat = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        descriptor = nixl_connect.Descriptor(flat)

        rdma_meta = nixl_connect.RdmaMetadata.model_validate(meta["nixl_metadata"])
        read_op = await connector.begin_read(rdma_meta, descriptor)
        await read_op.wait_for_completion()

        # Slice the flat buffer into individual tensors
        result = {}
        offset = 0
        for key, shape, dtype, size in specs:
            result[key] = flat[offset:offset + size].view(dtype=dtype).reshape(shape)
            offset += size

        return result
