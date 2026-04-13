# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL-based transfer of pre-processed mm_kwargs tensors from frontend to backend.

When the frontend runs vLLM's HF processor, it produces mm_kwargs (a dict of
named tensors like pixel_values, image_grid_thw, etc.). Rather than having the
backend re-run the expensive HF processor, we transfer these tensors via NIXL
and let the backend construct a pre-rendered MultiModalInput that skips the
processor entirely.

Tensors are registered directly with NIXL (zero-copy) and the backend reads
them via RDMA.  Only lightweight field metadata (~200 bytes) is serialized —
no pickle of the full MultiModalKwargsItem.

This module provides:
- MmKwargsSender: registers mm_kwargs tensors with NIXL on the frontend side
- MmKwargsReceiver: pulls tensors via NIXL READ on the backend side
- MmKwargsTransferMetadata: the wire protocol between the two
"""

from __future__ import annotations

import asyncio
import logging
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
    # Field metadata for reconstructing MultiModalFieldElem on the backend.
    field_type: str = ""  # e.g. "batched", "flat", "shared"
    field_keep_on_cpu: bool = False
    field_batch_size: int = 1  # for SharedField
    field_slices: list[
        list[int]
    ] | None = None  # for FlatField: [[start,stop,step],...]
    field_dim: int = 0  # for FlatField


class MmKwargsTransferMetadata(BaseModel):
    """Metadata for transferring all mm_kwargs tensors via NIXL.

    Sent from frontend to backend alongside the routing request.
    """

    modality: str  # e.g. "image"
    tensor_specs: list[TensorTransferSpec]
    mm_hashes: list[str]  # frontend-computed hashes for consistency


# ---------------------------------------------------------------------------
# Field metadata helpers
# ---------------------------------------------------------------------------


def _serialize_field(field: Any) -> dict[str, Any]:
    """Extract serializable metadata from a BaseMultiModalField."""
    cls_name = type(field).__name__
    result: dict[str, Any] = {
        "field_keep_on_cpu": getattr(field, "keep_on_cpu", False),
    }
    if cls_name == "MultiModalBatchedField":
        result["field_type"] = "batched"
    elif cls_name == "MultiModalSharedField":
        result["field_type"] = "shared"
        result["field_batch_size"] = getattr(field, "batch_size", 1)
    elif cls_name == "MultiModalFlatField":
        result["field_type"] = "flat"
        result["field_dim"] = getattr(field, "dim", 0)
        slices = getattr(field, "slices", None)
        if slices is not None:
            serialized_slices = []
            for s in slices:
                if isinstance(s, slice):
                    serialized_slices.append(
                        [
                            s.start if s.start is not None else 0,
                            s.stop if s.stop is not None else -1,
                            s.step if s.step is not None else 1,
                        ]
                    )
            result["field_slices"] = serialized_slices if serialized_slices else None
    else:
        # Unknown field type — fall back to batched
        logger.warning("Unknown field type %s, treating as batched", cls_name)
        result["field_type"] = "batched"
    return result


def _deserialize_field(spec: TensorTransferSpec) -> Any:
    """Reconstruct a BaseMultiModalField from transfer metadata."""
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFlatField,
        MultiModalSharedField,
    )

    if spec.field_type == "shared":
        return MultiModalSharedField(
            batch_size=spec.field_batch_size,
            keep_on_cpu=spec.field_keep_on_cpu,
        )
    elif spec.field_type == "flat":
        slices: list[slice] = []
        if spec.field_slices:
            slices = [
                slice(s[0], s[1] if s[1] != -1 else None, s[2])
                for s in spec.field_slices
            ]
        return MultiModalFlatField(
            slices=slices,
            dim=spec.field_dim,
            keep_on_cpu=spec.field_keep_on_cpu,
        )
    else:  # "batched" or unknown
        return MultiModalBatchedField(
            keep_on_cpu=spec.field_keep_on_cpu,
        )


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

        Tensors are registered directly (zero-copy, no pickle).  Only
        lightweight field metadata is serialized alongside the NIXL
        descriptors.

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
            "[NIXL-Sender] Preparing %d mm_features for NIXL transfer",
            len(mm_features),
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

            # feat.data is a MultiModalKwargsItem (UserDict[str, MultiModalFieldElem])
            # Register each tensor directly — no pickle.
            kwargs_item = feat.data
            for key, elem in kwargs_item.items():
                if elem.data is None:
                    continue
                tensor = elem.data
                if not isinstance(tensor, torch.Tensor):
                    # NestedTensors can be lists — skip non-tensor entries
                    logger.debug(
                        "[NIXL-Sender] feature[%d] key=%s: skipping non-tensor data",
                        i,
                        key,
                    )
                    continue

                # Make contiguous for NIXL registration
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                descriptor = self._nixl_connect.Descriptor(tensor)
                readable_op = await self._connector.create_readable(descriptor)

                field_meta = _serialize_field(elem.field)
                spec = TensorTransferSpec(
                    field_name=f"{i}:{key}",  # "0:pixel_values", "0:image_grid_thw"
                    shape=list(tensor.shape),
                    dtype_str=str(tensor.dtype).replace("torch.", ""),
                    serialized_request=readable_op.metadata().model_dump(),
                    **field_meta,
                )
                tensor_specs.append(spec)
                completions.append(readable_op.wait_for_completion())

            logger.debug(
                "[NIXL-Sender] feature[%d]: registered %d tensor(s) directly",
                i,
                sum(1 for s in tensor_specs if s.field_name.startswith(f"{i}:")),
            )

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
        kwargs_items = await receiver.receive(transfer_metadata)
        # kwargs_items is a list of MultiModalKwargsItem
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

    async def receive(self, metadata: MmKwargsTransferMetadata) -> list[Any]:
        """Pull tensors via NIXL READ and reconstruct MultiModalKwargsItems.

        Returns:
            List of MultiModalKwargsItem, one per mm_feature.
        """
        if not self._available:
            raise RuntimeError("NIXL not available for mm_kwargs reception")

        from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem

        rng = _nvtx.start_range("mm_nixl:receiver_read", color="magenta")

        # Group specs by feature index (parsed from "0:pixel_values" → 0)
        specs_by_feature: dict[int, list[tuple[str, TensorTransferSpec]]] = {}
        for spec in metadata.tensor_specs:
            parts = spec.field_name.split(":", 1)
            feat_idx = int(parts[0])
            key = parts[1]
            specs_by_feature.setdefault(feat_idx, []).append((key, spec))

        # Read all tensors in parallel
        read_results: dict[str, torch.Tensor] = {}
        read_tasks = []

        for spec in metadata.tensor_specs:
            dtype = getattr(torch, spec.dtype_str, torch.float32)
            local_tensor = torch.empty(spec.shape, dtype=dtype)
            descriptor = self._nixl_connect.Descriptor(local_tensor)

            rdma_metadata = self._nixl_connect.RdmaMetadata.model_validate(
                spec.serialized_request
            )

            async def _do_read(
                rm=rdma_metadata,
                desc=descriptor,
                name=spec.field_name,
                t=local_tensor,
            ):
                read_op = await self._connector.begin_read(rm, desc)
                await read_op.wait_for_completion()
                read_results[name] = t

            read_tasks.append(_do_read())

        await asyncio.gather(*read_tasks)

        # Reconstruct MultiModalKwargsItems from received tensors + metadata
        kwargs_items: list[MultiModalKwargsItem] = []
        for feat_idx in sorted(specs_by_feature.keys()):
            item_dict: dict[str, MultiModalFieldElem] = {}
            for key, spec in specs_by_feature[feat_idx]:
                full_name = f"{feat_idx}:{key}"
                tensor = read_results.get(full_name)
                if tensor is None:
                    logger.warning("[NIXL-Receiver] Missing tensor for %s", full_name)
                    continue
                field = _deserialize_field(spec)
                item_dict[key] = MultiModalFieldElem(data=tensor, field=field)
            kwargs_items.append(MultiModalKwargsItem(item_dict))

        _nvtx.end_range(rng)
        return kwargs_items
