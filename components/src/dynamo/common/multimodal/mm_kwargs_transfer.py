# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL-based transfer of pre-processed mm_kwargs tensors from frontend to backend.

When the frontend runs vLLM's HF processor, it produces mm_kwargs (a dict of
named tensors like pixel_values, image_grid_thw, etc.). Rather than having the
backend re-run the expensive HF processor, we transfer these tensors via NIXL
and let the backend construct a pre-rendered MultiModalInput that skips the
processor entirely.

This module provides:
- MmKwargsSender: registers mm_kwargs tensors with NIXL on the frontend side
- MmKwargsReceiver: pulls tensors via NIXL READ on the backend side
- MmKwargsTransferMetadata: the wire protocol between the two
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, Awaitable

import torch
from pydantic import BaseModel

from dynamo.common.utils import nvtx_utils as _nvtx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------


class TensorTransferSpec(BaseModel):
    """Metadata for a single tensor within mm_kwargs."""

    field_name: str  # e.g. "pixel_values", "image_grid_thw"
    shape: list[int]
    dtype_str: str
    serialized_request: Any  # RdmaMetadata from nixl_connect


class MmKwargsTransferMetadata(BaseModel):
    """Metadata for transferring all mm_kwargs tensors via NIXL.

    Sent from frontend to backend alongside the routing request.
    """

    modality: str  # e.g. "image"
    tensor_specs: list[TensorTransferSpec]
    mm_hashes: list[str]  # frontend-computed hashes for consistency


# ---------------------------------------------------------------------------
# Sender (frontend side)
# ---------------------------------------------------------------------------


class MmKwargsSender:
    """Registers mm_kwargs tensors with NIXL for remote READ access.

    Usage::

        sender = MmKwargsSender()
        metadata, completion = await sender.prepare(mm_features, "image")
        # ... send metadata to backend ...
        await completion  # wait for backend to finish reading
    """

    def __init__(self) -> None:
        # Lazy import to avoid hard dependency when NIXL is not available.
        try:
            from dynamo import nixl_connect

            self._connector = nixl_connect.Connector()
            self._nixl_connect = nixl_connect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("nixl_connect not available; MmKwargsSender disabled")

    async def prepare(
        self,
        mm_features: list[Any],  # list[MultiModalFeatureSpec]
        modality: str = "image",
    ) -> tuple[MmKwargsTransferMetadata | None, list[Awaitable[None]]]:
        """Register mm_kwargs tensors from mm_features with NIXL.

        Args:
            mm_features: MultiModalFeatureSpec list from EngineCoreRequest.
            modality: The modality to extract (default "image").

        Returns:
            (transfer_metadata, completion_futures) or (None, []) if no
            tensors to transfer.
        """
        if not self._available:
            logger.info("[NIXL-Sender] NIXL not available, skipping")
            return None, []
        if not mm_features:
            logger.info("[NIXL-Sender] No mm_features to send")
            return None, []
        logger.debug(
            "[NIXL-Sender] Preparing %d mm_features for NIXL transfer", len(mm_features)
        )

        tensor_specs: list[TensorTransferSpec] = []
        completions: list[Awaitable[None]] = []
        mm_hashes: list[str] = []

        rng = _nvtx.start_range("mm_nixl:sender_prepare", color="magenta")
        for i, feat in enumerate(mm_features):
            if feat.mm_hash:
                mm_hashes.append(feat.mm_hash)

            if feat.data is None:
                logger.debug("[NIXL-Sender] feature[%d]: data is None, skipping", i)
                continue

            # Pickle the kwargs item — preserves MultiModalFieldElem + field objects
            pickled_item = pickle.dumps(feat.data)
            logger.debug(
                "[NIXL-Sender] feature[%d]: pickled kwargs_item (%d bytes)",
                i,
                len(pickled_item),
            )

            # Register pickled bytes as a NIXL transfer
            pickled_tensor = torch.frombuffer(
                bytearray(pickled_item), dtype=torch.uint8
            )
            descriptor = self._nixl_connect.Descriptor(pickled_tensor)
            readable_op = await self._connector.create_readable(descriptor)

            spec = TensorTransferSpec(
                field_name="__pickled_kwargs_item__",
                shape=[len(pickled_item)],
                dtype_str="uint8",
                serialized_request=readable_op.metadata().model_dump(),
            )
            tensor_specs.append(spec)
            completions.append(readable_op.wait_for_completion())

        _nvtx.end_range(rng)
        if not tensor_specs:
            return None, []

        metadata = MmKwargsTransferMetadata(
            modality=modality,
            tensor_specs=tensor_specs,
            mm_hashes=mm_hashes,
        )
        return metadata, completions


# ---------------------------------------------------------------------------
# Receiver (backend side)
# ---------------------------------------------------------------------------


class MmKwargsReceiver:
    """Pulls mm_kwargs tensors from the frontend via NIXL READ.

    Usage::

        receiver = MmKwargsReceiver()
        mm_kwargs = await receiver.receive(transfer_metadata)
        # mm_kwargs is a dict like {"pixel_values": tensor, ...}
    """

    def __init__(self) -> None:
        try:
            from dynamo import nixl_connect

            self._connector = nixl_connect.Connector()
            self._nixl_connect = nixl_connect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("nixl_connect not available; MmKwargsReceiver disabled")

    async def receive(self, metadata: MmKwargsTransferMetadata) -> dict[str, Any]:
        """Pull all data described in metadata via NIXL READ.

        Returns:
            Dict mapping field_name to received data (tensor or bytes).
        """
        if not self._available:
            raise RuntimeError("NIXL not available for mm_kwargs reception")

        rng = _nvtx.start_range("mm_nixl:receiver_read", color="magenta")
        results: dict[str, Any] = {}
        read_tasks = []

        for spec in metadata.tensor_specs:
            dtype = getattr(torch, spec.dtype_str, torch.float32)
            local_tensor = torch.empty(spec.shape, dtype=dtype)
            descriptor = self._nixl_connect.Descriptor(local_tensor)

            rdma_metadata = self._nixl_connect.RdmaMetadata.model_validate(
                spec.serialized_request
            )

            async def _do_read(
                rm=rdma_metadata, desc=descriptor, name=spec.field_name, t=local_tensor
            ):
                read_op = await self._connector.begin_read(rm, desc)
                await read_op.wait_for_completion()
                # For pickled items, convert tensor bytes back to bytes.
                # Use a list to support multiple items (e.g. 3 images).
                if name == "__pickled_kwargs_item__":
                    results.setdefault(name, []).append(bytes(t.numpy().tobytes()))
                else:
                    results[name] = t

            read_tasks.append(_do_read())

        await asyncio.gather(*read_tasks)
        _nvtx.end_range(rng)
        return results
