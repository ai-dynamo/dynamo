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
import multiprocessing.shared_memory as shm
import os
import pickle
from queue import Queue
from typing import Any, Awaitable

import torch
from pydantic import BaseModel

from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.runtime import run_async

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
# Receiver (backend side) — pre-registered descriptor pool
# ---------------------------------------------------------------------------

# Default max pickled kwargs size: 8MB (covers most single-image models).
_DEFAULT_MAX_ITEM_BYTES = 8 * 1024 * 1024
# Number of pre-warmed descriptors (concurrent images in flight).
_DEFAULT_POOL_SIZE = 16


class MmKwargsReceiver:
    """Pulls mm_kwargs tensors from the frontend via NIXL READ.

    Uses pre-registered descriptor pooling to avoid per-request NIXL
    registration overhead (~20-30ms → ~1ms per transfer).

    Usage::

        receiver = MmKwargsReceiver()
        mm_kwargs = await receiver.receive(transfer_metadata)
        # mm_kwargs is a dict like {"__pickled_kwargs_item__": [bytes, ...]}
    """

    def __init__(
        self,
        max_item_bytes: int = _DEFAULT_MAX_ITEM_BYTES,
        pool_size: int = _DEFAULT_POOL_SIZE,
    ) -> None:
        try:
            from dynamo import nixl_connect

            self._connector = nixl_connect.Connector()
            self._nixl_connect = nixl_connect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("nixl_connect not available; MmKwargsReceiver disabled")
            return

        self._max_item_bytes = max_item_bytes
        self._pool: Queue[nixl_connect.Descriptor] = Queue()
        self._inuse: dict[int, tuple[nixl_connect.Descriptor, bool]] = {}
        self._tensor_id_counter = 0

        # Pre-allocate and pre-register descriptors.
        connection = run_async(self._connector._create_connection)
        for _ in range(pool_size):
            buf = torch.zeros(max_item_bytes, dtype=torch.uint8)
            desc = nixl_connect.Descriptor(buf)
            desc.register_with_connector(connection)
            self._pool.put(desc)
        logger.info(
            "MmKwargsReceiver: pre-registered %d descriptors (%d bytes each)",
            pool_size,
            max_item_bytes,
        )

    def _acquire_descriptor(
        self, size_bytes: int
    ) -> tuple[Any, torch.Tensor, bool, int | None]:
        """Get a descriptor from the pool or create a dynamic one.

        Returns (descriptor, tensor_view, is_dynamic, original_size).
        """
        if not self._pool.empty():
            desc = self._pool.get()
            if size_bytes <= self._max_item_bytes:
                original_size = desc._data_size
                desc._data_size = size_bytes
                assert desc._data_ref is not None
                tensor_view = desc._data_ref[:size_bytes]
                return desc, tensor_view, False, original_size
            else:
                # Too large for pre-allocated buffer — put it back, create dynamic
                self._pool.put(desc)

        # Dynamic fallback
        buf = torch.empty(size_bytes, dtype=torch.uint8)
        desc = self._nixl_connect.Descriptor(buf)
        return desc, buf, True, None

    def _release_descriptor(
        self,
        desc: Any,
        is_dynamic: bool,
        original_size: int | None,
    ) -> None:
        """Return a descriptor to the pool or discard dynamic ones."""
        if is_dynamic:
            return  # GC will clean up
        if original_size is not None:
            desc._data_size = original_size
        self._pool.put(desc)

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
        # Track acquired descriptors for release after reads complete.
        acquired: list[tuple[Any, bool, int | None]] = []

        for spec in metadata.tensor_specs:
            size_bytes = 1
            for s in spec.shape:
                size_bytes *= s
            # uint8 → 1 byte per element
            desc, tensor_view, is_dynamic, orig_size = self._acquire_descriptor(
                size_bytes
            )
            acquired.append((desc, is_dynamic, orig_size))

            rdma_metadata = self._nixl_connect.RdmaMetadata.model_validate(
                spec.serialized_request
            )

            async def _do_read(
                rm=rdma_metadata,
                d=desc,
                name=spec.field_name,
                t=tensor_view,
                sz=size_bytes,
            ):
                read_op = await self._connector.begin_read(rm, d)
                await read_op.wait_for_completion()
                if name == "__pickled_kwargs_item__":
                    results.setdefault(name, []).append(bytes(t[:sz].numpy().tobytes()))
                else:
                    results[name] = t[:sz]

            read_tasks.append(_do_read())

        await asyncio.gather(*read_tasks)

        # Release all descriptors back to pool.
        for desc, is_dynamic, orig_size in acquired:
            self._release_descriptor(desc, is_dynamic, orig_size)

        _nvtx.end_range(rng)
        return results


# ---------------------------------------------------------------------------
# Shared Memory transfer (same-node, ~1.5ms for 3.7MB)
# ---------------------------------------------------------------------------


class MmKwargsShmSender:
    """Transfers pickled mm_kwargs via shared memory (same-node only).

    ~30x faster than NIXL for CPU→CPU same-machine transfers.

    Usage::

        sender = MmKwargsShmSender()
        shm_meta, cleanup = sender.prepare(mm_features, "image")
        # ... send shm_meta to backend via extra_args ...
        # ... after backend confirms receipt:
        cleanup()
    """

    def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        """Pickle mm_kwargs into shared memory.

        Returns:
            (shm_metadata_dict, shm_handles) or (None, []) if nothing to send.
            Caller must keep shm_handles alive until backend reads, then call
            handle.close() and handle.unlink() for each.
        """
        if not mm_features:
            return None, []

        rng = _nvtx.start_range("mm_shm:sender_prepare", color="cyan")
        items: list[dict[str, Any]] = []
        handles: list[shm.SharedMemory] = []
        mm_hashes: list[str] = []

        for i, feat in enumerate(mm_features):
            if feat.mm_hash:
                mm_hashes.append(feat.mm_hash)
            if feat.data is None:
                continue

            pickled = pickle.dumps(feat.data)
            import uuid

            name = f"mm_kwargs_{os.getpid()}_{uuid.uuid4().hex[:12]}_{i}"
            sm = shm.SharedMemory(name=name, create=True, size=len(pickled))
            sm.buf[: len(pickled)] = pickled
            handles.append(sm)
            items.append({"name": name, "size": len(pickled)})
            logger.debug(
                "[SHM-Sender] feature[%d]: wrote %d bytes to shm %s",
                i,
                len(pickled),
                name,
            )

        _nvtx.end_range(rng)
        if not items:
            return None, []

        meta = {
            "modality": modality,
            "items": items,
            "mm_hashes": mm_hashes,
        }
        return meta, handles

    @staticmethod
    def cleanup(handles: list[Any]) -> None:
        """Release shared memory after backend has read."""
        for sm in handles:
            try:
                sm.close()
                sm.unlink()
            except Exception:
                pass


class MmKwargsShmReceiver:
    """Reads pickled mm_kwargs from shared memory.

    Usage::

        receiver = MmKwargsShmReceiver()
        result = receiver.receive(shm_metadata)
        # result is {"__pickled_kwargs_item__": [bytes, bytes, ...]}
    """

    def receive(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Read from shared memory and return pickled bytes.

        Returns:
            Dict with "__pickled_kwargs_item__" key mapping to list of bytes.
        """
        rng = _nvtx.start_range("mm_shm:receiver_read", color="cyan")
        results: dict[str, Any] = {}

        for item in meta.get("items", []):
            name = item["name"]
            size = item["size"]
            try:
                sm = shm.SharedMemory(name=name, create=False)
                data = bytes(sm.buf[:size])
                sm.close()
                results.setdefault("__pickled_kwargs_item__", []).append(data)
                logger.debug("[SHM-Receiver] read %d bytes from shm %s", size, name)
            except Exception:
                logger.warning(
                    "[SHM-Receiver] failed to read shm %s",
                    name,
                    exc_info=True,
                )

        _nvtx.end_range(rng)
        return results
