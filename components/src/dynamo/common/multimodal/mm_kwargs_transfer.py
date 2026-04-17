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
- MmKwargsReceiver: pulls tensors via NIXL READ on the backend side
- MmKwargsShmReceiver: reads pickled mm_kwargs from shared memory
- MmKwargsTransferMetadata: the wire protocol between the two
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing.shared_memory as shm
import os
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any

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
    """

    @abstractmethod
    async def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        """Serialize and register mm_kwargs for transfer to the backend.

        Returns:
            (extra_args_update, cleanup_items)
            extra_args_update: dict to merge into dynamo_preproc["extra_args"],
                or None if nothing was transferred.
            cleanup_items: opaque list passed back to cleanup() after the
                response stream completes.
        """

    @abstractmethod
    async def cleanup(self, items: list[Any]) -> None:
        """Release transfer resources after the backend has consumed them."""


_DEFAULT_NIXL_SENDER_POOL_SIZE = 8
_DEFAULT_NIXL_SENDER_MAX_ITEM_BYTES = 8 * 1024 * 1024  # 8 MB


class MmKwargsNixlSender(MmKwargsSender):
    """Registers mm_kwargs tensors with NIXL for remote READ access.

    Uses a pool of pre-registered send buffers so that NIXL registration
    (~34ms per-request without pooling) is paid once at startup per slot.
    Per-request cost is just a memcpy into the pre-registered buffer + metadata
    re-serialization (~0-1ms).

    Usage::

        sender = MmKwargsNixlSender()
        extra_args, cleanup_items = await sender.prepare(mm_features, "image")
        # ... send extra_args to backend ...
        await sender.cleanup(cleanup_items)  # wait for backend to finish reading
    """

    def __init__(
        self,
        pool_size: int = _DEFAULT_NIXL_SENDER_POOL_SIZE,
        max_item_bytes: int = _DEFAULT_NIXL_SENDER_MAX_ITEM_BYTES,
    ) -> None:
        try:
            from dynamo import nixl_connect

            self._connector = nixl_connect.Connector()
            self._nixl_connect = nixl_connect
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("nixl_connect not available; MmKwargsNixlSender disabled")
            return

        self._max_item_bytes = max_item_bytes
        self._pool: Queue = Queue()

        # Pre-warm the pool: pay NIXL registration once per slot at startup.
        connection = run_async(self._connector._create_connection)
        t0 = time.perf_counter()
        for _ in range(pool_size):
            slot = self._make_slot(connection, max_item_bytes)
            self._pool.put(slot)
        logger.info(
            "MmKwargsNixlSender: pre-registered %d sender slots (%d bytes each) in %.0fms",
            pool_size,
            max_item_bytes,
            (time.perf_counter() - t0) * 1e3,
        )

    def _make_slot(self, connection: Any, max_bytes: int) -> Any:
        """Allocate and pre-register one sender buffer slot."""
        nixl_connect = self._nixl_connect
        ReadableOperation = nixl_connect.ReadableOperation

        # Subclass suppresses _release() so the descriptor stays registered
        # across multiple requests.  The pool manages the descriptor lifetime.
        class _NonReleasingOp(ReadableOperation):
            def _release(self) -> None:
                pass

        buf = torch.zeros(max_bytes, dtype=torch.uint8)
        desc = nixl_connect.Descriptor(buf)
        desc.register_with_connector(connection)  # one-time cost (~34ms)

        # Creating ReadableOperation calls register_with_connector — no-op since
        # desc is already registered.
        op = _NonReleasingOp(connection, desc)
        return (buf, desc, op, max_bytes)  # tuple: (buffer, descriptor, op, capacity)

    def _slot_load(self, slot: Any, pickled: bytes) -> None:
        """Copy pickled bytes into slot and reset op state for a new request."""
        import numpy as np

        buf, desc, op, _cap = slot
        size = len(pickled)
        # Zero-copy view of pickled bytes, then bulk copy into pre-registered buf.
        buf.numpy()[:size] = np.frombuffer(pickled, dtype=np.uint8)

        # Update size so serialized metadata reflects the actual payload.
        desc._data_size = size
        desc._serialized = None  # clear cached SerializedDescriptor
        op._serialized_request = None  # force RdmaMetadata re-serialization
        # Re-arm the status so wait_for_completion polls for the new notification.
        op._status = self._nixl_connect.OperationStatus.INITIALIZED

    def _slot_restore(self, slot: Any) -> None:
        """Restore full-capacity size on descriptor before returning to pool."""
        _buf, desc, _op, cap = slot
        desc._data_size = cap

    async def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        """Copy mm_kwargs into pre-registered buffers and expose via NIXL READ.

        Returns:
            (extra_args_update, cleanup_items) or (None, []).
            cleanup_items is a list of (slot_or_None, completion_coroutine).
        """
        if not self._available:
            return None, []
        if not mm_features:
            return None, []

        tensor_specs: list[TensorTransferSpec] = []
        cleanup_items: list[Any] = []
        mm_hashes: list[str] = []

        t_prepare_start = time.perf_counter()
        rng = _nvtx.start_range("mm_nixl:sender_prepare", color="magenta")
        total_bytes = 0

        for i, feat in enumerate(mm_features):
            if feat.mm_hash:
                mm_hashes.append(feat.mm_hash)
            if feat.data is None:
                continue

            t0 = time.perf_counter()
            with _nvtx.annotate("mm_nixl:pickle_dumps", color="magenta"):
                pickled_item = pickle.dumps(feat.data)
            t_pickle = time.perf_counter() - t0

            t0 = time.perf_counter()
            with _nvtx.annotate("mm_nixl:register_descriptor", color="magenta"):
                if not self._pool.empty() and len(pickled_item) <= self._max_item_bytes:
                    # Fast path: borrow pre-registered slot, just memcpy + re-arm.
                    slot = self._pool.get()
                    self._slot_load(slot, pickled_item)
                    _buf, _desc, op, _cap = slot
                    readable_meta = op.metadata()
                    completion = op.wait_for_completion()
                else:
                    # Slow path: dynamic allocation (pool exhausted or item too large).
                    slot = None
                    pickled_tensor = torch.frombuffer(
                        bytearray(pickled_item), dtype=torch.uint8
                    )
                    descriptor = self._nixl_connect.Descriptor(pickled_tensor)
                    readable_op = await self._connector.create_readable(descriptor)
                    readable_meta = readable_op.metadata()
                    completion = readable_op.wait_for_completion()
            t_register = time.perf_counter() - t0

            total_bytes += len(pickled_item)
            print(
                f"[TIMING][NIXL-Sender] feature[{i}]: pickle={t_pickle*1e3:.2f}ms register={t_register*1e3:.2f}ms size={len(pickled_item)/1024:.1f}KB",
                flush=True,
            )

            spec = TensorTransferSpec(
                field_name="__pickled_kwargs_item__",
                shape=[len(pickled_item)],
                dtype_str="uint8",
                serialized_request=readable_meta.model_dump(),
            )
            tensor_specs.append(spec)
            cleanup_items.append((slot, completion))

        _nvtx.end_range(rng)
        t_prepare_total = time.perf_counter() - t_prepare_start
        if not tensor_specs:
            return None, []

        print(
            f"[TIMING][NIXL-Sender] prepare total={t_prepare_total*1e3:.2f}ms n_items={len(tensor_specs)} total_bytes={total_bytes/1024:.1f}KB",
            flush=True,
        )

        metadata = MmKwargsTransferMetadata(
            modality=modality,
            tensor_specs=tensor_specs,
            mm_hashes=mm_hashes,
        )
        return {"mm_kwargs_nixl": metadata.model_dump()}, cleanup_items

    async def cleanup(self, items: list[Any]) -> None:
        """Await NIXL completion and return pool slots."""
        if not items:
            return
        try:
            t0 = time.perf_counter()
            for slot, completion in items:
                await completion
                if slot is not None:
                    self._slot_restore(slot)
                    self._pool.put(slot)
            print(
                f"[TIMING][NIXL-Sender] cleanup (wait_for_completion) total={(time.perf_counter()-t0)*1e3:.2f}ms n_items={len(items)}",
                flush=True,
            )
        except Exception:
            logger.warning("[NIXL-Sender] transfer completion failed", exc_info=True)


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

    async def receive(self, metadata: MmKwargsTransferMetadata) -> dict[str, Any]:
        """Pull all data described in metadata via NIXL READ.

        Returns:
            Dict mapping field_name to received data (tensor or bytes).
        """
        if not self._available:
            raise RuntimeError("NIXL not available for mm_kwargs reception")

        t_receive_start = time.perf_counter()
        rng = _nvtx.start_range("mm_nixl:receiver_read", color="magenta")
        # Pre-allocate result slots to preserve spec order regardless of
        # completion order from asyncio.gather.
        read_tasks = []
        # Track acquired descriptors for release after reads complete.
        acquired: list[tuple[Any, bool, int | None]] = []
        # Store (spec_index, field_name, tensor_view, size_bytes) per task.
        task_meta: list[tuple[int, str, Any, int]] = []
        # Per-spec timing collected inside each coroutine.
        spec_timings: list[dict[str, float]] = [{} for _ in metadata.tensor_specs]

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

            async def _do_read(rm=rdma_metadata, d=desc, i=idx, sz=size_bytes):
                t0 = time.perf_counter()
                read_op = await self._connector.begin_read(rm, d)
                t_begin = time.perf_counter() - t0
                t0 = time.perf_counter()
                await read_op.wait_for_completion()
                t_wait = time.perf_counter() - t0
                spec_timings[i] = {
                    "begin_read_ms": t_begin * 1e3,
                    "wait_ms": t_wait * 1e3,
                    "size_kb": sz / 1024,
                }

            read_tasks.append(_do_read())

        t0 = time.perf_counter()
        await asyncio.gather(*read_tasks)
        t_gather = time.perf_counter() - t0

        for i, st in enumerate(spec_timings):
            if st:
                print(
                    f"[TIMING][NIXL-Receiver] spec[{i}]: begin_read={st['begin_read_ms']:.2f}ms wait={st['wait_ms']:.2f}ms size={st['size_kb']:.1f}KB",
                    flush=True,
                )
        print(
            f"[TIMING][NIXL-Receiver] gather={t_gather*1e3:.2f}ms n_specs={len(metadata.tensor_specs)} total={(time.perf_counter()-t_receive_start)*1e3:.2f}ms",
            flush=True,
        )

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

        _nvtx.end_range(rng)
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

    async def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        """Pickle mm_kwargs into shared memory.

        Returns:
            (extra_args_update, shm_handles) or (None, []) if nothing to send.
            Pass shm_handles to cleanup() after streaming completes.
        """
        if not mm_features:
            return None, []

        t_prepare_start = time.perf_counter()
        rng = _nvtx.start_range("mm_shm:sender_prepare", color="cyan")
        items: list[dict[str, Any]] = []
        handles: list[shm.SharedMemory] = []
        mm_hashes: list[str] = []
        total_bytes = 0

        for i, feat in enumerate(mm_features):
            if feat.mm_hash:
                mm_hashes.append(feat.mm_hash)
            if feat.data is None:
                continue

            t0 = time.perf_counter()
            with _nvtx.annotate("mm_shm:pickle_dumps", color="cyan"):
                pickled = pickle.dumps(feat.data)
            t_pickle = time.perf_counter() - t0

            name = f"mm_kwargs_{os.getpid()}_{uuid.uuid4().hex[:12]}_{i}"
            t0 = time.perf_counter()
            with _nvtx.annotate("mm_shm:create_and_write", color="cyan"):
                sm = shm.SharedMemory(name=name, create=True, size=len(pickled))
                sm.buf[: len(pickled)] = pickled
            t_write = time.perf_counter() - t0

            total_bytes += len(pickled)
            print(
                f"[TIMING][SHM-Sender] feature[{i}]: pickle={t_pickle*1e3:.2f}ms create_write={t_write*1e3:.2f}ms size={len(pickled)/1024:.1f}KB",
                flush=True,
            )
            handles.append(sm)
            items.append({"name": name, "size": len(pickled)})

        _nvtx.end_range(rng)
        if not items:
            return None, []

        print(
            f"[TIMING][SHM-Sender] prepare total={(time.perf_counter()-t_prepare_start)*1e3:.2f}ms n_items={len(items)} total_bytes={total_bytes/1024:.1f}KB",
            flush=True,
        )
        meta: dict[str, Any] = {
            "modality": modality,
            "items": items,
            "mm_hashes": mm_hashes,
        }
        return {"mm_kwargs_shm": meta}, handles

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


class MmKwargsTcpSender(MmKwargsSender):
    """Transfers pickled mm_kwargs via Dynamo's TCP request plane.

    Embeds base64-encoded pickled data directly in extra_args["mm_kwargs_tcp"].
    No separate memory segment or RDMA channel needed — the data travels with
    the request payload through the existing Dynamo TCP plane.

    Suitable for same-node or cross-node transfers up to ~50 MB. Latency is
    dominated by serialization (pickle + base64) and TCP framing.

    Usage::

        sender = MmKwargsTcpSender()
        extra_args, _ = await sender.prepare(mm_features, "image")
        # No cleanup needed — data lives in the request payload.
    """

    import base64 as _base64

    async def prepare(
        self,
        mm_features: list[Any],
        modality: str = "image",
    ) -> tuple[dict[str, Any] | None, list[Any]]:
        if not mm_features:
            return None, []

        import base64

        t_prepare_start = time.perf_counter()
        items_b64: list[str] = []
        mm_hashes: list[str] = []
        total_bytes = 0

        for i, feat in enumerate(mm_features):
            if feat.mm_hash:
                mm_hashes.append(feat.mm_hash)
            if feat.data is None:
                continue

            t0 = time.perf_counter()
            pickled = pickle.dumps(feat.data)
            t_pickle = time.perf_counter() - t0

            t0 = time.perf_counter()
            encoded = base64.b64encode(pickled).decode("ascii")
            t_encode = time.perf_counter() - t0

            total_bytes += len(pickled)
            print(
                f"[TIMING][TCP-Sender] feature[{i}]: pickle={t_pickle*1e3:.2f}ms encode={t_encode*1e3:.2f}ms size={len(pickled)/1024:.1f}KB",
                flush=True,
            )
            items_b64.append(encoded)

        if not items_b64:
            return None, []

        t_prepare_total = time.perf_counter() - t_prepare_start
        print(
            f"[TIMING][TCP-Sender] prepare total={t_prepare_total*1e3:.2f}ms n_items={len(items_b64)} total_bytes={total_bytes/1024:.1f}KB",
            flush=True,
        )
        meta: dict[str, Any] = {
            "modality": modality,
            "items_b64": items_b64,
            "mm_hashes": mm_hashes,
        }
        return {"mm_kwargs_tcp": meta}, []

    async def cleanup(self, items: list[Any]) -> None:
        pass  # Nothing to release — data is embedded in the request payload.


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
        t_receive_start = time.perf_counter()
        rng = _nvtx.start_range("mm_shm:receiver_read", color="cyan")
        results: dict[str, Any] = {}

        for i, item in enumerate(meta.get("items", [])):
            name = item["name"]
            size = item["size"]
            t0 = time.perf_counter()
            with _nvtx.annotate("mm_shm:open_and_read", color="cyan"):
                sm = shm.SharedMemory(name=name, create=False)
                data = bytes(sm.buf[:size])
                sm.close()
            print(
                f"[TIMING][SHM-Receiver] item[{i}]: open_read={(time.perf_counter()-t0)*1e3:.2f}ms size={size/1024:.1f}KB",
                flush=True,
            )
            results.setdefault("__pickled_kwargs_item__", []).append(data)

        print(
            f"[TIMING][SHM-Receiver] total={(time.perf_counter()-t_receive_start)*1e3:.2f}ms n_items={len(meta.get('items', []))}",
            flush=True,
        )
        _nvtx.end_range(rng)
        return results
