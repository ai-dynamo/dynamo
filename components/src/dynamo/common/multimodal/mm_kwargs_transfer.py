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
- MmKwargsSender: abstract base class for frontend-side transfer
- MmKwargsNixlSender: NIXL RDMA implementation (cross-node)
- MmKwargsShmSender: shared memory implementation (same-node, ~2ms)
- MmKwargsNixlReceiver: pulls tensors via NIXL READ on the backend side
- MmKwargsShmReceiver: reads pickled mm_kwargs from shared memory
- MmKwargsTransferMetadata: the wire protocol between the two
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing.shared_memory as shm
import os
import pickle
import uuid
from abc import ABC, abstractmethod
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


class MmKwargsShmItem(BaseModel):
    """Metadata for a single shared-memory segment carrying one pickled item."""

    name: str
    size: int


class MmKwargsShmTransferMetadata(BaseModel):
    """Metadata for transferring mm_kwargs via shared memory (same-node).

    Sent from frontend to backend alongside the routing request. Mirrors
    ``MmKwargsTransferMetadata`` for the SHM transport.
    """

    modality: str  # e.g. "image"
    items: list[MmKwargsShmItem]
    mm_hashes: list[str]


# ---------------------------------------------------------------------------
# Sender (frontend side) — abstract base + concrete implementations
# ---------------------------------------------------------------------------


class MmKwargsSender(ABC):
    """Abstract base for frontend-side mm_kwargs transfer.

    Subclasses implement different transport mechanisms (NIXL RDMA, shared
    memory). The caller in vllm_processor treats them uniformly:

        sender = MmKwargsShmSender()  # or MmKwargsNixlSender()
        extra_args, cleanup_items = await sender.prepare(mm_features, "image")
        if extra_args:
            dynamo_preproc["extra_args"].update(extra_args)
        # ... after streaming:
        await sender.cleanup(cleanup_items)

    The base class owns the per-request NVTX annotation via class-level
    ``_nvtx_label`` / ``_nvtx_color`` attributes that subclasses override.
    ``prepare`` is a concrete template method that wraps ``_prepare`` with
    ``_nvtx.annotate`` so the callsite in ``vllm_processor`` doesn't need
    to know which transport is in use.
    """

    # Subclasses override these to tag their own NVTX range.
    _nvtx_label: str = "mm_frontend:sender_prepare"
    _nvtx_color: str = "cyan"

    # Subclass hook label used for the pickle step.
    _pickle_nvtx_label: str = "mm_frontend:pickle_dumps"

    def _is_available(self) -> bool:
        """Return True if this transport is usable. Default True; NIXL overrides."""
        return True

    async def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        """Serialize and register mm_kwargs for transfer to the backend.

        Walks mm_features, collects per-feature mm_hashes, skips
        ``data is None`` features, pickles each remaining ``feat.data``,
        and delegates the transport-specific hand-off to
        :meth:`_encode_item`. Once all items are encoded, calls
        :meth:`_assemble_extra_args` to build the dict returned to the
        caller.

        Returns:
            (extra_args_update, cleanup_items)
            extra_args_update: dict to merge into dynamo_preproc["extra_args"],
                or None if nothing was transferred.
            cleanup_items: opaque list passed back to cleanup() after the
                response stream completes.
        """
        with _nvtx.annotate(self._nvtx_label, color=self._nvtx_color):
            if not self._is_available() or not mm_features:
                return None, []

            encoded_items: list[Any] = []
            cleanup_items: list[Any] = []
            mm_hashes: list[str] = []

            for i, feat in enumerate(mm_features):
                if feat.mm_hash:
                    mm_hashes.append(feat.mm_hash)
                if feat.data is None:
                    continue

                with _nvtx.annotate(self._pickle_nvtx_label, color=self._nvtx_color):
                    pickled = pickle.dumps(feat.data)

                encoded, cleanup = await self._encode_item(i, pickled)
                encoded_items.append(encoded)
                if cleanup is not None:
                    cleanup_items.append(cleanup)

            if not encoded_items:
                return None, []

            return (
                self._assemble_extra_args(modality, encoded_items, mm_hashes),
                cleanup_items,
            )

    @abstractmethod
    async def _encode_item(self, idx: int, pickled: bytes) -> tuple[Any, Any | None]:
        """Subclass hook: hand off pickled bytes to the transport.

        Returns ``(encoded_item, cleanup_item)`` — ``encoded_item`` is whatever
        the subclass wants to collect (e.g. a ``TensorTransferSpec`` or
        ``MmKwargsShmItem``); ``cleanup_item`` is an opaque handle the caller
        passes to :meth:`cleanup` (e.g. an NIXL completion future or an
        ``shm.SharedMemory`` handle), or ``None`` if no cleanup is needed.
        """

    @abstractmethod
    def _assemble_extra_args(
        self,
        modality: str,
        encoded_items: list[Any],
        mm_hashes: list[str],
    ) -> dict[str, Any]:
        """Subclass hook: build the ``extra_args`` update dict from encoded items."""

    @abstractmethod
    async def cleanup(self, items: list[Any]) -> None:
        """Release transfer resources after the backend has consumed them."""


class MmKwargsNixlSender(MmKwargsSender):
    """Registers mm_kwargs tensors with NIXL for remote READ access.

    Usage::

        sender = MmKwargsNixlSender()
        extra_args, cleanup_items = await sender.prepare(mm_features, "image")
        # ... send extra_args to backend ...
        await sender.cleanup(cleanup_items)  # wait for backend to finish reading
    """

    _nvtx_label = "mm_frontend:nixl_sender_prepare"
    _nvtx_color = "magenta"
    _pickle_nvtx_label = "mm_nixl:pickle_dumps"

    def __init__(self) -> None:
        # Lazy import to avoid hard dependency when NIXL is not available.
        try:
            from dynamo import nixl_connect

            self._connector = nixl_connect.Connector()
            self._nixl_connect = nixl_connect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("nixl_connect not available; MmKwargsNixlSender disabled")

    def _is_available(self) -> bool:
        return self._available

    async def _encode_item(
        self, idx: int, pickled: bytes
    ) -> tuple[TensorTransferSpec, Awaitable[None]]:
        """Register pickled bytes with NIXL and return the spec + completion."""
        with _nvtx.annotate("mm_nixl:register_descriptor", color="magenta"):
            pickled_tensor = torch.frombuffer(bytearray(pickled), dtype=torch.uint8)
            descriptor = self._nixl_connect.Descriptor(pickled_tensor)
            readable_op = await self._connector.create_readable(descriptor)
        logger.debug(
            "[NIXL-Sender] feature[%d]: registered %d bytes", idx, len(pickled)
        )
        spec = TensorTransferSpec(
            field_name="__pickled_kwargs_item__",
            shape=[len(pickled)],
            dtype_str="uint8",
            serialized_request=readable_op.metadata().model_dump(),
        )
        return spec, readable_op.wait_for_completion()

    def _assemble_extra_args(
        self,
        modality: str,
        encoded_items: list[Any],
        mm_hashes: list[str],
    ) -> dict[str, Any]:
        metadata = MmKwargsTransferMetadata(
            modality=modality,
            tensor_specs=encoded_items,
            mm_hashes=mm_hashes,
        )
        logger.debug(
            "[NIXL-Sender] registered %d tensor(s) for transfer",
            len(metadata.tensor_specs),
        )
        return {"mm_kwargs_nixl": metadata.model_dump()}

    async def cleanup(self, items: list[Any]) -> None:
        """Await NIXL completion futures so memory regions can be deregistered."""
        if not items:
            return
        try:
            await asyncio.gather(*items)
            logger.debug("[NIXL-Sender] all transfers completed")
        except Exception:
            logger.warning("[NIXL-Sender] transfer completion failed", exc_info=True)


# ---------------------------------------------------------------------------
# Receiver (backend side) — abstract base + concrete implementations
# ---------------------------------------------------------------------------


class MmKwargsReceiver(ABC):
    """Abstract base for backend-side mm_kwargs reception.

    Subclasses implement different transport mechanisms (NIXL RDMA, shared
    memory). The caller in handlers treats them uniformly::

        receiver: MmKwargsReceiver = self._select_mm_kwargs_receiver(extra_args)
        results = await receiver.receive(metadata)

    The base class owns the per-request NVTX annotation via class-level
    ``_nvtx_label`` / ``_nvtx_color`` attributes that subclasses override.
    ``receive`` is a concrete template method that wraps ``_receive`` with
    ``_nvtx.annotate`` so callers don't need to know which transport is in
    use.
    """

    # Subclasses override these to tag their own NVTX range.
    _nvtx_label: str = "mm_backend:receiver_read"
    _nvtx_color: str = "cyan"

    async def receive(self, metadata: BaseModel) -> dict[str, Any]:
        """Fetch pickled mm_kwargs items using this transport.

        Returns a dict keyed by field name (typically a single key
        ``"__pickled_kwargs_item__"`` → ``list[bytes]``).
        """
        with _nvtx.annotate(self._nvtx_label, color=self._nvtx_color):
            return await self._receive(metadata)

    @abstractmethod
    async def _receive(self, metadata: BaseModel) -> dict[str, Any]:
        """Subclass hook: do the actual transport-specific read."""


# Default max pickled kwargs size: 8MB (covers most single-image models).
_DEFAULT_MAX_ITEM_BYTES = 8 * 1024 * 1024
# Number of pre-warmed descriptors (concurrent images in flight).
_DEFAULT_POOL_SIZE = 16


class MmKwargsNixlReceiver(MmKwargsReceiver):
    """Pulls mm_kwargs tensors from the frontend via NIXL READ.

    Uses pre-registered descriptor pooling to avoid per-request NIXL
    registration overhead (~20-30ms → ~1ms per transfer).

    Usage::

        receiver = MmKwargsNixlReceiver()
        mm_kwargs = await receiver.receive(transfer_metadata)
        # mm_kwargs is a dict like {"__pickled_kwargs_item__": [bytes, ...]}
    """

    _nvtx_label = "mm_backend:nixl_receiver_read"
    _nvtx_color = "magenta"

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
            logger.warning("nixl_connect not available; MmKwargsNixlReceiver disabled")
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
            "MmKwargsNixlReceiver: pre-registered %d descriptors (%d bytes each)",
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
                if desc._data_ref is None:
                    raise RuntimeError("Pre-allocated descriptor has no data reference")
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

    async def _receive(self, metadata: BaseModel) -> dict[str, Any]:
        """Pull all data described in metadata via NIXL READ.

        Returns:
            Dict mapping field_name to received data (tensor or bytes).
        """
        assert isinstance(metadata, MmKwargsTransferMetadata)
        if not self._available:
            raise RuntimeError("NIXL not available for mm_kwargs reception")

        # Pre-allocate result slots to preserve spec order regardless of
        # completion order from asyncio.gather.
        read_tasks = []
        # Track acquired descriptors for release after reads complete.
        acquired: list[tuple[Any, bool, int | None]] = []
        # Store (spec_index, field_name, tensor_view, size_bytes) per task.
        task_meta: list[tuple[int, str, Any, int]] = []

        for idx, spec in enumerate(metadata.tensor_specs):
            size_bytes = 1
            for s in spec.shape:
                size_bytes *= s
            # uint8 → 1 byte per element
            desc, tensor_view, is_dynamic, orig_size = self._acquire_descriptor(
                size_bytes
            )
            acquired.append((desc, is_dynamic, orig_size))
            task_meta.append((idx, spec.field_name, tensor_view, size_bytes))

            rdma_metadata = self._nixl_connect.RdmaMetadata.model_validate(
                spec.serialized_request
            )

            async def _do_read(rm=rdma_metadata, d=desc):
                read_op = await self._connector.begin_read(rm, d)
                await read_op.wait_for_completion()

            read_tasks.append(_do_read())

        await asyncio.gather(*read_tasks)

        # Collect results in spec order (not completion order).
        results: dict[str, Any] = {}
        for idx, name, tensor_view, sz in task_meta:
            if name == "__pickled_kwargs_item__":
                results.setdefault(name, []).append(
                    bytes(tensor_view[:sz].numpy().tobytes())
                )
            else:
                results[name] = tensor_view[:sz]

        # Release all descriptors back to pool.
        for desc, is_dynamic, orig_size in acquired:
            self._release_descriptor(desc, is_dynamic, orig_size)

        return results


# ---------------------------------------------------------------------------
# Shared Memory transfer (same-node, ~1.5ms for 3.7MB)
# ---------------------------------------------------------------------------


class MmKwargsShmSender(MmKwargsSender):
    """Transfers pickled mm_kwargs via shared memory (same-node only).

    ~30x faster than NIXL for CPU→CPU same-machine transfers.

    Usage::

        sender = MmKwargsShmSender()
        extra_args, cleanup_items = await sender.prepare(mm_features, "image")
        # ... send extra_args to backend via extra_args ...
        await sender.cleanup(cleanup_items)
    """

    _nvtx_label = "mm_frontend:shm_sender_prepare"
    _nvtx_color = "cyan"
    _pickle_nvtx_label = "mm_shm:pickle_dumps"

    async def _encode_item(
        self, idx: int, pickled: bytes
    ) -> tuple[MmKwargsShmItem, shm.SharedMemory]:
        """Write pickled bytes to a fresh shared-memory segment."""
        name = f"mm_kwargs_{os.getpid()}_{uuid.uuid4().hex[:12]}_{idx}"
        with _nvtx.annotate("mm_shm:create_and_write", color="cyan"):
            sm = shm.SharedMemory(name=name, create=True, size=len(pickled))
            sm.buf[: len(pickled)] = pickled
        logger.debug(
            "[SHM-Sender] feature[%d]: wrote %d bytes to shm %s",
            idx,
            len(pickled),
            name,
        )
        return MmKwargsShmItem(name=name, size=len(pickled)), sm

    def _assemble_extra_args(
        self,
        modality: str,
        encoded_items: list[Any],
        mm_hashes: list[str],
    ) -> dict[str, Any]:
        metadata = MmKwargsShmTransferMetadata(
            modality=modality, items=encoded_items, mm_hashes=mm_hashes
        )
        logger.debug(
            "[SHM-Sender] wrote %d item(s) to shared memory", len(encoded_items)
        )
        return {"mm_kwargs_shm": metadata.model_dump()}

    async def cleanup(self, items: list[Any]) -> None:
        """Release shared memory handles after the backend has read."""
        for sm in items:
            try:
                sm.close()
                sm.unlink()
            except FileNotFoundError:
                pass  # Already unlinked (e.g., by resource_tracker)
            except Exception:
                logger.warning(
                    "Failed to clean up shared memory handle",
                    exc_info=True,
                )


class MmKwargsShmReceiver(MmKwargsReceiver):
    """Reads pickled mm_kwargs from shared memory.

    Usage::

        receiver = MmKwargsShmReceiver()
        result = await receiver.receive(shm_metadata)
        # result is {"__pickled_kwargs_item__": [bytes, bytes, ...]}
    """

    _nvtx_label = "mm_backend:shm_receiver_read"
    _nvtx_color = "cyan"

    async def _receive(self, metadata: BaseModel) -> dict[str, Any]:
        """Read from shared memory and return pickled bytes.

        Returns:
            Dict with "__pickled_kwargs_item__" key mapping to list of bytes.
        """
        assert isinstance(metadata, MmKwargsShmTransferMetadata)
        results: dict[str, Any] = {}

        for item in metadata.items:
            with _nvtx.annotate("mm_shm:open_and_read", color="cyan"):
                sm = shm.SharedMemory(name=item.name, create=False)
                data = bytes(sm.buf[: item.size])
                sm.close()
            results.setdefault("__pickled_kwargs_item__", []).append(data)
            logger.debug(
                "[SHM-Receiver] read %d bytes from shm %s", item.size, item.name
            )

        return results
