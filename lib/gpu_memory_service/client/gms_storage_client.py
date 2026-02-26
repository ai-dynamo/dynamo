# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS storage client: save GMS state to disk and load it back.

Exports GMS state (all allocations + metadata) to a compact sharded format for
offline analysis, backup, or migration, then loads it back into a fresh GMS
server.  Optionally uses NIXL's GDS backend for GPU-direct I/O.

File format::

    save_dir/
    ├── manifest.json        # version, timestamp, layout_hash, device, use_gds,
    │                        #   gds_available, allocations[]
    ├── gms_metadata.json    # {key: {allocation_id, offset_bytes, value (base64)}}
    └── shards/
        ├── shard_0000.bin   # allocations packed contiguously (raw bytes, no headers)
        ├── shard_0001.bin   # next batch
        └── ...

Each allocation's ``AllocationEntry`` records which shard file it lives in
(``tensor_file``) and its byte offset within that file (``tensor_offset``).
Shards are written sequentially during save and **read sequentially** during
load — no ``seek()`` calls are issued within a shard file.  Parallelism
across shard files is provided via ``ThreadPoolExecutor``.  During restore,
GMS VAs are pre-allocated serially (Phase A) then filled in parallel using
per-thread CUDA streams (Phase B).

Sizing: with the default 4 GiB shard limit, a 100 GB model with 100k tensors
produces roughly 25 shard files rather than 100 000 individual files.

Usage::

    # Save running GMS → disk
    client = GMSStorageClient("/tmp/save_dir", socket_path="/tmp/gms.sock", device=0)
    manifest = client.save()

    # Load disk → fresh GMS server (RW → commit)
    id_map = client.load_to_gms("/tmp/save_dir")
    # id_map: {old_allocation_id: new_allocation_id, ...}

    # Load tensor data only (no GMS write-back)
    tensors, metadata = GMSStorageClient.load_tensors("/tmp/save_dir", device=0)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GMS imports (module-level so they are patchable in tests)
# ---------------------------------------------------------------------------

try:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
    from gpu_memory_service.common.types import RequestedLockType

    _GMS_IMPORTS_AVAILABLE = True
except ImportError:
    _GMS_IMPORTS_AVAILABLE = False
    GMSClientMemoryManager = None  # type: ignore[assignment,misc]
    _tensor_from_pointer = None  # type: ignore[assignment]
    RequestedLockType = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# NIXL availability check
# ---------------------------------------------------------------------------

try:
    from nixl._api import nixl_agent, nixl_agent_config  # type: ignore[import]

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False
    nixl_agent = None  # type: ignore[assignment]
    nixl_agent_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Lazy PyTorch import (allows unit tests to run without CUDA)
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Lazy GMS imports (allow tests to mock them)
# ---------------------------------------------------------------------------

_CURRENT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationEntry:
    """Immutable record of one dumped allocation.

    ``tensor_file`` is a path relative to the dump directory pointing to the
    shard file that contains this allocation's bytes.  ``tensor_offset`` is
    the byte offset within that shard file where the data starts.

    Older dumps (version 1.0 before sharding) may not have ``tensor_offset``
    in their JSON; ``SaveManifest.from_dict`` defaults it to ``0``.
    """

    allocation_id: str
    size: int
    aligned_size: int
    tag: str
    tensor_file: str  # relative path inside dump_dir (e.g. "shards/shard_0000.bin")
    tensor_offset: int = 0  # byte offset within tensor_file


@dataclass
class SaveManifest:
    """Manifest for a GMS dump directory."""

    version: str
    timestamp: float
    layout_hash: str
    device: int
    use_gds: bool
    gds_available: bool
    allocations: List[AllocationEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "layout_hash": self.layout_hash,
            "device": self.device,
            "use_gds": self.use_gds,
            "gds_available": self.gds_available,
            "allocations": [asdict(a) for a in self.allocations],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SaveManifest":
        # Construct AllocationEntry explicitly so we can default tensor_offset=0
        # for manifests written before the sharding feature was added.
        allocations = [
            AllocationEntry(
                allocation_id=a["allocation_id"],
                size=a["size"],
                aligned_size=a["aligned_size"],
                tag=a["tag"],
                tensor_file=a["tensor_file"],
                tensor_offset=a.get("tensor_offset", 0),
            )
            for a in d.get("allocations", [])
        ]
        return cls(
            version=d["version"],
            timestamp=d["timestamp"],
            layout_hash=d["layout_hash"],
            device=d["device"],
            use_gds=d["use_gds"],
            gds_available=d["gds_available"],
            allocations=allocations,
        )


# ---------------------------------------------------------------------------
# NIXL GDS writer  (kept for GPU-direct I/O; not used in the main shard path)
# ---------------------------------------------------------------------------


class _NixlGdsWriter:
    """Wraps a nixl_agent to write/read GPU tensors via GDS (or POSIX fallback).

    The backend is selected at construction time:
    - Tries GDS first (requires cuFile / NVMe-direct support)
    - Falls back to POSIX if GDS backend creation fails
    - Falls back to no-NIXL (torch.save) if POSIX also fails

    Attributes:
        is_gds: True if the GDS backend is active.
        is_active: True if any NIXL backend is active.
    """

    def __init__(self, agent_name: str = "gms_dump") -> None:
        self._agent = None
        self._backend: Optional[str] = None

        if not _NIXL_AVAILABLE:
            return

        try:
            self._agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
        except Exception as exc:
            warnings.warn(
                f"Failed to create NIXL agent: {exc}; disabling NIXL"  # noqa: E702
            )
            return

        # Try GDS first, then POSIX
        for backend in ("GDS", "POSIX"):
            try:
                self._agent.create_backend(backend)
                self._backend = backend
                logger.debug("NIXL backend selected: %s", backend)
                break
            except Exception as exc:
                logger.debug("NIXL backend %s unavailable: %s", backend, exc)

        if self._backend is None:
            warnings.warn("All NIXL backends failed; disabling NIXL")
            self._agent = None

    @property
    def is_gds(self) -> bool:
        return self._backend == "GDS"

    @property
    def is_active(self) -> bool:
        return self._agent is not None and self._backend is not None

    def write_tensor(self, tensor: "torch.Tensor", path: str) -> None:
        """Write *tensor* to *path* using NIXL (GPU → NVMe direct if GDS).

        The file is created/overwritten. The tensor must be contiguous and on GPU.
        """
        if not self.is_active:
            raise RuntimeError("_NixlGdsWriter is not active")

        size = tensor.numel() * tensor.element_size()
        # Open (or create) file for writing
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            # Pre-allocate file to full size
            os.ftruncate(fd, size)

            gpu_reg = self._agent.register_memory(tensor)
            file_reg = self._agent.register_memory([(0, size, fd, path)], "FILE")

            src = self._agent.get_xfer_descs([tensor])
            dst = self._agent.get_xfer_descs([(0, size, fd, path)], "FILE")

            handle = self._agent.initialize_xfer("WRITE", src, dst, self._agent.name)
            try:
                # Kick off transfer and poll until done
                self._agent.transfer(handle)
                while True:
                    state = self._agent.check_xfer_state(handle)
                    if state == "DONE":
                        break
                    if state == "ERR":
                        raise RuntimeError(f"NIXL transfer error writing {path}")
            finally:
                self._agent.release_xfer_handle(handle)

            self._agent.deregister_memory(gpu_reg)
            self._agent.deregister_memory(file_reg)
        finally:
            os.close(fd)

    def read_tensor(
        self,
        path: str,
        size: int,
        dtype: "torch.dtype",
        shape: List[int],
        device_index: int,
    ) -> "torch.Tensor":
        """Read tensor from *path* using NIXL (NVMe → GPU direct if GDS).

        Falls back to ``open().readinto(cpu_buffer)`` if NIXL is unavailable.
        """
        if not self.is_active:
            raise RuntimeError("_NixlGdsWriter is not active")

        tensor = torch.empty(
            shape, dtype=dtype, device=f"cuda:{device_index}"  # noqa: E231
        )
        fd = os.open(path, os.O_RDONLY)
        try:
            gpu_reg = self._agent.register_memory(tensor)
            file_reg = self._agent.register_memory([(0, size, fd, path)], "FILE")

            dst = self._agent.get_xfer_descs([tensor])
            src = self._agent.get_xfer_descs([(0, size, fd, path)], "FILE")

            handle = self._agent.initialize_xfer("READ", src, dst, self._agent.name)
            try:
                self._agent.transfer(handle)
                while True:
                    state = self._agent.check_xfer_state(handle)
                    if state == "DONE":
                        break
                    if state == "ERR":
                        raise RuntimeError(f"NIXL transfer error reading {path}")
            finally:
                self._agent.release_xfer_handle(handle)

            self._agent.deregister_memory(gpu_reg)
            self._agent.deregister_memory(file_reg)
        finally:
            os.close(fd)

        return tensor

    def read_into_tensor(
        self,
        tensor: "torch.Tensor",
        path: str,
        file_offset: int,
        size: int,
    ) -> None:
        """Read *size* bytes from *path* at *file_offset* into *tensor* in-place.

        Unlike :meth:`read_tensor`, this method writes into a **pre-allocated**
        tensor rather than allocating a new one.  This is used during GMS
        restore to read shard data directly into the GMS virtual-address
        mapping, eliminating any intermediate GPU or CPU copy.

        Args:
            tensor: Pre-allocated destination tensor (must be contiguous).
            path: Path to the shard file.
            file_offset: Byte offset within the shard file to start reading.
            size: Number of bytes to transfer (must equal ``tensor.nbytes``).
        """
        if not self.is_active:
            raise RuntimeError("_NixlGdsWriter is not active")

        fd = os.open(path, os.O_RDONLY)
        try:
            gpu_reg = self._agent.register_memory(tensor)
            file_reg = self._agent.register_memory(
                [(file_offset, size, fd, path)], "FILE"
            )

            dst = self._agent.get_xfer_descs([tensor])
            src = self._agent.get_xfer_descs([(file_offset, size, fd, path)], "FILE")

            handle = self._agent.initialize_xfer("READ", src, dst, self._agent.name)
            try:
                self._agent.transfer(handle)
                while True:
                    state = self._agent.check_xfer_state(handle)
                    if state == "DONE":
                        break
                    if state == "ERR":
                        raise RuntimeError(
                            f"NIXL transfer error reading {path}@{file_offset}"
                        )
            finally:
                self._agent.release_xfer_handle(handle)

            self._agent.deregister_memory(gpu_reg)
            self._agent.deregister_memory(file_reg)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class _ShardWriter:
    """Packs allocation bytes sequentially into large binary shard files.

    Each allocation is appended back-to-back with no inter-allocation padding
    (``aligned_size`` is already aligned to CUDA VMM granularity).  This
    layout enables restore to read each shard **front-to-back with zero
    seeking**: entries sorted by ``tensor_offset`` are contiguous in the file,
    so sequential ``f.read(aligned_size)`` calls naturally advance the file
    pointer to the next entry.

    Shard files are named ``shard_{n:04d}.bin`` inside *shards_dir* and are
    referenced with paths like ``shards/shard_0000.bin`` relative to the dump
    root.

    A new shard is started whenever the next write would cause the current
    shard to exceed *shard_size_bytes*, **unless** the current shard is still
    empty (in which case an oversized allocation is written as the sole entry
    in that shard).

    Args:
        shards_dir: Absolute path to the directory that will hold shard files.
            Created automatically if absent.
        shard_size_bytes: Soft upper bound per shard (default 4 GiB).
    """

    def __init__(self, shards_dir: str, shard_size_bytes: int = 4 * 1024**3) -> None:
        self._shards_dir = shards_dir
        self._shard_size = shard_size_bytes
        self._shard_idx = -1
        self._current_offset = 0
        self._current_file: Optional[Any] = None
        self._current_rel_path: str = ""
        os.makedirs(shards_dir, exist_ok=True)

    def _roll_shard(self) -> None:
        """Close the current shard (if open) and open the next one."""
        if self._current_file is not None:
            self._current_file.close()
        self._shard_idx += 1
        filename = f"shard_{self._shard_idx:04d}.bin"  # noqa: E231
        abs_path = os.path.join(self._shards_dir, filename)
        self._current_file = open(abs_path, "wb")
        self._current_rel_path = os.path.join("shards", filename)
        self._current_offset = 0

    def write(self, tensor: "torch.Tensor") -> Tuple[str, int]:
        """Append *tensor* bytes to the current shard.

        Rolls to the next shard if the current one would overflow
        *shard_size_bytes* (but never leaves an empty shard: an oversized
        single allocation always starts its own shard).

        Args:
            tensor: Any dtype tensor (GPU or CPU).  Moved to CPU and written
                as a contiguous raw byte stream.

        Returns:
            ``(rel_path, byte_offset)`` where *rel_path* is the path of the
            shard file relative to the dump root directory and *byte_offset*
            is the byte offset at which this allocation's data starts within
            that file.
        """
        cpu = tensor.cpu() if hasattr(tensor, "is_cuda") and tensor.is_cuda else tensor
        if hasattr(cpu, "is_contiguous") and not cpu.is_contiguous():
            cpu = cpu.contiguous()
        arr = cpu.numpy()
        size = arr.nbytes

        # Roll to next shard if this write would overflow the current one
        # (but always write at least one allocation per shard)
        if self._current_file is None or (
            self._current_offset > 0 and self._current_offset + size > self._shard_size
        ):
            self._roll_shard()

        offset = self._current_offset
        arr.tofile(self._current_file)
        self._current_offset += size

        return self._current_rel_path, offset

    def close(self) -> None:
        """Flush and close the current shard file."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def __enter__(self) -> "_ShardWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Sequential shard reader
# ---------------------------------------------------------------------------


def _read_shard_sequential(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    device: int,
) -> Dict[str, "torch.Tensor"]:
    """Read one shard file **front-to-back without seeking**.

    ``sorted_entries`` must be sorted by ``tensor_offset`` in ascending order.
    Because :class:`_ShardWriter` writes allocations contiguously with no gaps,
    reading them in offset order is equivalent to a pure sequential scan: each
    ``f.read(aligned_size)`` call advances the file pointer exactly to the
    start of the next entry.

    Legacy single-allocation ``*.pt`` files (written before the sharding
    feature) are handled transparently via ``torch.load``.

    Args:
        abs_path: Absolute path to the shard (or legacy ``.pt``) file.
        sorted_entries: Entries belonging to this file, sorted by
            ``tensor_offset`` ascending.
        device: CUDA device index.  Pass ``-1`` to keep tensors on CPU
            (used by :func:`load_to_gms` to avoid holding two GPU copies of
            the model simultaneously).

    Returns:
        ``{allocation_id: tensor}`` dict for all entries in this shard.
    """
    result: Dict[str, "torch.Tensor"] = {}
    device_str = f"cuda:{device}" if device >= 0 else "cpu"  # noqa: E231

    if abs_path.endswith(".pt"):
        # Legacy format: one .pt file contains exactly one allocation.
        assert len(sorted_entries) == 1, (
            f"Expected exactly 1 entry for legacy .pt file, got "
            f"{len(sorted_entries)}: {abs_path}"
        )
        entry = sorted_entries[0]
        t = torch.load(abs_path, weights_only=True, map_location=device_str)
        result[entry.allocation_id] = t
        return result

    # Binary shard: read sequentially, one allocation at a time.
    # Entries are sorted by tensor_offset so f.read() advances naturally.
    with open(abs_path, "rb") as f:
        for entry in sorted_entries:
            raw = f.read(entry.aligned_size)
            if len(raw) != entry.aligned_size:
                raise RuntimeError(
                    f"Short read from {abs_path} at offset {entry.tensor_offset}: "
                    f"expected {entry.aligned_size} bytes, got {len(raw)}"
                )
            # np.frombuffer returns a read-only view; .copy() makes it writable
            # so torch.from_numpy() can take ownership without a second copy.
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
            t = torch.from_numpy(arr)
            if device >= 0:
                t = t.to(device_str)
            result[entry.allocation_id] = t

    return result


def _decode_metadata(
    raw_meta: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Decode a raw metadata dict (as loaded from JSON) into Python types.

    Base64-encoded ``value`` fields are decoded to ``bytes``.
    """
    return {
        key: {
            "allocation_id": entry["allocation_id"],
            "offset_bytes": int(entry["offset_bytes"]),
            "value": base64.b64decode(entry["value"]),
        }
        for key, entry in raw_meta.items()
    }


def _group_entries_by_shard(
    allocations: List[AllocationEntry],
) -> Dict[str, List[AllocationEntry]]:
    """Group allocation entries by shard file and sort each group by offset.

    The resulting per-shard lists are sorted by ``tensor_offset`` ascending,
    which is the order required for sequential (seek-free) reads.
    """
    groups: Dict[str, List[AllocationEntry]] = defaultdict(list)
    for entry in allocations:
        groups[entry.tensor_file].append(entry)
    for entries_in_shard in groups.values():
        entries_in_shard.sort(key=lambda e: e.tensor_offset)
    return dict(groups)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GMSStorageClient:
    """Dump and restore GMS state to/from disk.

    Can be used for dump-only, restore-only, or both:

    * **Dump**: pass ``output_dir``; call :meth:`dump`.
    * **Restore**: ``output_dir`` may be ``None``; call :meth:`load_to_gms`
      with the dump directory path.
    * **Both**: pass ``output_dir`` and use the same instance for both.

    The dump format packs all allocations into a small number of large binary
    shard files (default 4 GiB each).  For a 100 GB model with 100k tensors
    this produces ~25 shard files instead of 100 000 individual files.
    Restore reads each shard **sequentially** (no seeking); shard files are
    processed in parallel via a thread pool.

    Args:
        output_dir: Directory in which to create the dump (created if absent).
            Pass ``None`` when using this client for restore only.
        socket_path: Unix socket path for the GMS server.  If ``None``, the
            default UUID-based path for *device* is used.
        device: CUDA device index.
        use_gds: Request the GDS I/O backend.  Silently falls back to POSIX
            or ``torch.save`` if NIXL/GDS is unavailable.
        timeout_ms: Timeout in milliseconds for lock acquisition.
        shard_size_bytes: Soft upper bound per shard file (default 4 GiB).
            Decrease for faster parallel restore on systems with many I/O
            lanes; increase to reduce file count.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        socket_path: Optional[str] = None,
        device: int = 0,
        *,
        use_gds: bool = False,
        timeout_ms: Optional[int] = None,
        shard_size_bytes: int = 4 * 1024**3,
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self._timeout_ms = timeout_ms
        self._shard_size = shard_size_bytes

        # Resolve socket path lazily to avoid importing pynvml in tests
        if socket_path is None:
            from gpu_memory_service.common.utils import get_socket_path

            socket_path = get_socket_path(device)
        self._socket_path = socket_path

        # Set up NIXL writer if requested (reserved for future GDS shard I/O)
        self._nixl_writer: Optional[_NixlGdsWriter] = None
        if use_gds:
            if not _NIXL_AVAILABLE:
                warnings.warn(
                    "use_gds=True requested but nixl is not installed; "
                    "falling back to torch.save",
                    stacklevel=2,
                )
            else:
                writer = _NixlGdsWriter()
                if writer.is_active:
                    self._nixl_writer = writer
                else:
                    warnings.warn(
                        "NIXL agent initialised but no backend available; "
                        "falling back to torch.save",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Public: dump
    # ------------------------------------------------------------------

    def save(self) -> SaveManifest:
        """Connect to GMS in RO mode and save all allocations + metadata to disk.

        All allocation bytes are packed sequentially into shard files under
        ``{output_dir}/shards/``.  Metadata is written to
        ``{output_dir}/gms_metadata.json`` and a manifest to
        ``{output_dir}/manifest.json``.

        Returns:
            :class:`SaveManifest` describing the saved state.

        Raises:
            ConnectionError: If GMS server is not running at *socket_path*.
            RuntimeError: If GMS has no committed weights.
            ValueError: If *output_dir* was not provided at construction time.
        """
        import time

        if not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                "GMS client imports unavailable (missing cuda-python or torch)"
            )
        if self.output_dir is None:
            raise ValueError(
                "output_dir must be set to call dump(); pass it to GMSStorageClient()"
            )

        os.makedirs(self.output_dir, exist_ok=True)
        shards_dir = os.path.join(self.output_dir, "shards")

        with GMSClientMemoryManager(
            self._socket_path,
            mode=RequestedLockType.RO,
            device=self.device,
            timeout_ms=self._timeout_ms,
        ) as mm:
            if not mm._client_rpc.committed:
                raise RuntimeError(
                    "GMS server has no committed weights; nothing to dump"
                )

            layout_hash = mm._client_rpc.get_memory_layout_hash()
            allocations_info = mm.list_allocations()

            entries: List[AllocationEntry] = []
            with _ShardWriter(shards_dir, self._shard_size) as writer:
                for alloc in allocations_info:
                    alloc_id = alloc["allocation_id"]
                    size = int(alloc["size"])
                    aligned_size = int(alloc["aligned_size"])
                    tag = str(alloc.get("tag", "default"))

                    va = mm.import_allocation(alloc_id)
                    tensor = _tensor_from_pointer(
                        va, [aligned_size], [1], torch.uint8, self.device
                    )

                    rel_path, offset = writer.write(tensor)
                    entries.append(
                        AllocationEntry(
                            allocation_id=alloc_id,
                            size=size,
                            aligned_size=aligned_size,
                            tag=tag,
                            tensor_file=rel_path,
                            tensor_offset=offset,
                        )
                    )
                    logger.info(
                        "Dumped allocation %s (%d bytes) → %s@%d",
                        alloc_id,
                        aligned_size,
                        rel_path,
                        offset,
                    )

            metadata = self._save_metadata(mm)

        # Write metadata file
        metadata_path = os.path.join(self.output_dir, "gms_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Wrote metadata to %s (%d keys)", metadata_path, len(metadata))

        gds_available = self._nixl_writer is not None and self._nixl_writer.is_gds
        manifest = SaveManifest(
            version=_CURRENT_VERSION,
            timestamp=time.time(),
            layout_hash=layout_hash,
            device=self.device,
            use_gds=self._nixl_writer is not None,
            gds_available=gds_available,
            allocations=entries,
        )

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.info(
            "Wrote manifest to %s (%d allocations)", manifest_path, len(entries)
        )

        return manifest

    # ------------------------------------------------------------------
    # Public: load_to_gms
    # ------------------------------------------------------------------

    def load_to_gms(
        self,
        input_dir: str,
        *,
        max_workers: int = 4,
        clear_existing: bool = True,
    ) -> Dict[str, str]:
        """Load a saved GMS state back into a running GMS server.

        Connects in **RW mode**, allocates GMS memory for each saved
        allocation, loads the tensor bytes from disk and copies them into GMS
        memory, restores all metadata (remapping old → new allocation IDs),
        then commits.

        Disk I/O is parallel across shard files (``max_workers`` threads),
        with each thread reading its shard **front-to-back without seeking**.
        In the non-GDS path, GMS VAs are pre-allocated serially (Phase A) and
        then filled in parallel using per-thread CUDA streams (Phase B).

        Args:
            input_dir: Directory previously created by :meth:`dump`.
            max_workers: Thread pool size for parallel shard reads.
            clear_existing: If ``True`` (default) call ``clear_all()`` on the
                server before restoring, so the result is an exact replica of
                the dump.  Set to ``False`` to add allocations on top of any
                existing state (advanced use).

        Returns:
            Mapping of ``{old_allocation_id: new_allocation_id}`` — the IDs
            assigned by GMS during restore.  Use this if callers cache the
            old allocation IDs and need to look up the new ones.

        Raises:
            ConnectionError: If GMS server is not running at *socket_path*.
            RuntimeError: If GMS imports are unavailable or restore fails.
        """
        if not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                "GMS client imports unavailable (missing cuda-python or torch)"
            )

        # ---- Load manifest and metadata from disk ----------------------
        manifest_path = os.path.join(input_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = SaveManifest.from_dict(json.load(f))

        metadata_path = os.path.join(input_dir, "gms_metadata.json")
        raw_meta: Dict[str, Any] = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                raw_meta = json.load(f)

        saved_metadata = _decode_metadata(raw_meta)

        id_map: Dict[str, str] = {}  # old_alloc_id → new_alloc_id

        if self._nixl_writer is not None and self._nixl_writer.is_active:
            # ------------------------------------------------------------------
            # GDS path: allocate GMS VA first, then read from disk directly
            # into it via NIXL (disk → GPU, bypassing CPU).  Because the data
            # lands straight into the GMS allocation, no intermediate GPU copy
            # is needed and peak VRAM equals exactly one copy of the model.
            # ------------------------------------------------------------------
            logger.info(
                "Loading %d allocations via NIXL %s (GDS=%s); connecting to GMS in RW mode",
                len(manifest.allocations),
                self._nixl_writer._backend,
                self._nixl_writer.is_gds,
            )
            with GMSClientMemoryManager(
                self._socket_path,
                mode=RequestedLockType.RW,
                device=self.device,
                timeout_ms=self._timeout_ms,
            ) as mm:
                if clear_existing:
                    cleared = mm.clear_all()
                    if cleared:
                        logger.info("Cleared %d pre-existing allocations", cleared)

                for entry in manifest.allocations:
                    old_id = entry.allocation_id

                    # Allocate GMS physical memory and map it into the server VA.
                    va = mm.allocate_and_map(entry.aligned_size, entry.tag)
                    new_id = mm._mappings[va].allocation_id
                    id_map[old_id] = new_id

                    # Zero-copy view of the GMS allocation — NIXL writes here.
                    dst_tensor = _tensor_from_pointer(
                        va, [entry.aligned_size], [1], torch.uint8, self.device
                    )

                    # Read directly from the shard file into the GMS VA (no
                    # intermediate CPU or GPU staging buffer).
                    abs_path = os.path.join(input_dir, entry.tensor_file)
                    self._nixl_writer.read_into_tensor(
                        dst_tensor, abs_path, entry.tensor_offset, entry.aligned_size
                    )
                    logger.info(
                        "Restored allocation %s → %s (%d bytes, tag=%s)",
                        old_id,
                        new_id,
                        entry.aligned_size,
                        entry.tag,
                    )

                self._restore_metadata(mm, saved_metadata, id_map)
                ok = mm.commit()
                if not ok:
                    raise RuntimeError("GMS commit failed after restore")

        else:
            # ------------------------------------------------------------------
            # Non-GDS path: parallel CPU staging → two-phase parallel copy to GMS.
            #
            # Shard files are read in parallel into CPU (host) memory.  We
            # deliberately stage through CPU (device=-1) so that the combined
            # footprint of the staging tensors + GMS allocations never exceeds
            # available VRAM.
            #
            # Phase A (serial): pre-allocate every GMS VA upfront.  Batching
            # all cuMemCreate calls before any cudaMemcpy keeps the CUDA
            # virtual-address space clean and avoids interleaving slow RPCs
            # with DMA transfers.
            #
            # Phase B (parallel): copy CPU staging buffers to their GMS VAs
            # concurrently using one CUDA stream per worker thread.
            # Non-blocking copy_() lets multiple DMA engines run in parallel;
            # torch.cuda.synchronize() ensures all transfers complete before
            # commit.
            # ------------------------------------------------------------------
            groups = _group_entries_by_shard(manifest.allocations)
            loaded: Dict[str, "torch.Tensor"] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _read_shard_sequential,
                        os.path.join(input_dir, rel_path),
                        sorted_entries,
                        -1,  # CPU — avoids double GPU allocation during restore
                    ): rel_path
                    for rel_path, sorted_entries in groups.items()
                }
                for future in as_completed(futures):
                    rel_path = futures[future]
                    try:
                        shard_tensors = future.result()
                        loaded.update(shard_tensors)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load shard {rel_path}: {exc}"
                        ) from exc

            logger.info(
                "Loaded %d allocations from disk (CPU staging); connecting to GMS in RW mode",
                len(loaded),
            )

            with GMSClientMemoryManager(
                self._socket_path,
                mode=RequestedLockType.RW,
                device=self.device,
                timeout_ms=self._timeout_ms,
            ) as mm:
                if clear_existing:
                    cleared = mm.clear_all()
                    if cleared:
                        logger.info("Cleared %d pre-existing allocations", cleared)

                # ---- Phase A: allocate all GMS VAs (serial cuMemCreate) ----
                vas: Dict[str, int] = {}  # old_alloc_id → va
                for entry in manifest.allocations:
                    old_id = entry.allocation_id
                    va = mm.allocate_and_map(entry.aligned_size, entry.tag)
                    new_id = mm._mappings[va].allocation_id
                    id_map[old_id] = new_id
                    vas[old_id] = va

                logger.info(
                    "Phase A complete: allocated %d GMS VAs; starting parallel copy",
                    len(vas),
                )

                # ---- Phase B: parallel CPU → GPU copy, one stream/worker ----
                _use_streams = _TORCH_AVAILABLE and torch.cuda.is_available()
                streams = (
                    [torch.cuda.Stream(device=self.device) for _ in range(max_workers)]
                    if _use_streams
                    else []
                )

                def _copy_entry(idx: int, entry: AllocationEntry) -> None:
                    va = vas[entry.allocation_id]
                    src = loaded[entry.allocation_id]
                    dst = _tensor_from_pointer(
                        va, [entry.aligned_size], [1], torch.uint8, self.device
                    )
                    if streams:
                        with torch.cuda.stream(streams[idx % max_workers]):
                            dst.copy_(src, non_blocking=True)
                    else:
                        dst.copy_(src)

                with ThreadPoolExecutor(max_workers=max_workers) as copy_pool:
                    copy_futs = [
                        copy_pool.submit(_copy_entry, i, entry)
                        for i, entry in enumerate(manifest.allocations)
                    ]
                    for fut in copy_futs:
                        fut.result()  # propagate any exceptions

                # Flush all async copies before commit.
                if _use_streams:
                    torch.cuda.synchronize(device=self.device)
                loaded.clear()

                logger.info(
                    "Phase B complete: copied %d allocations to GMS memory",
                    len(manifest.allocations),
                )

                self._restore_metadata(mm, saved_metadata, id_map)
                ok = mm.commit()
                if not ok:
                    raise RuntimeError("GMS commit failed after restore")

        logger.info(
            "load_to_gms complete: %d allocations, %d metadata keys",
            len(id_map),
            len(saved_metadata),
        )
        return id_map

    def _restore_metadata(
        self,
        mm: Any,
        saved_metadata: Dict[str, Dict[str, Any]],
        id_map: Dict[str, str],
    ) -> None:
        """Write saved metadata back to GMS, remapping old → new allocation IDs."""
        for key, meta in saved_metadata.items():
            old_alloc_id = meta["allocation_id"]
            new_alloc_id = id_map.get(old_alloc_id, old_alloc_id)
            mm.metadata_put(key, new_alloc_id, meta["offset_bytes"], meta["value"])
            logger.debug("Restored metadata key=%s → alloc=%s", key, new_alloc_id)
        logger.info("Restored %d metadata keys; committing", len(saved_metadata))

    # ------------------------------------------------------------------
    # Public: load_tensors  (disk-only, no GMS write-back)
    # ------------------------------------------------------------------

    @staticmethod
    def load_tensors(
        input_dir: str,
        device: int = 0,
        *,
        max_workers: int = 4,
    ) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Dict[str, Any]]]:
        """Load tensors and metadata from a dump directory into GPU memory.

        This is a **disk-only** operation — it does NOT connect to GMS.
        Use :meth:`load_to_gms` to write data back into a running GMS server.

        Shard files are read in parallel (``max_workers`` threads), each thread
        reading its shard **front-to-back without seeking**.

        Args:
            input_dir: Directory created by :meth:`dump`.
            device: CUDA device index to restore tensors onto.
            max_workers: Thread pool size for parallel shard reads.

        Returns:
            ``(tensors, metadata)`` where *tensors* maps allocation ID →
            ``torch.Tensor`` (uint8, on device) and *metadata* maps metadata
            key → ``{allocation_id, offset_bytes, value}`` (value as bytes).
        """
        manifest_path = os.path.join(input_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = SaveManifest.from_dict(json.load(f))

        metadata_path = os.path.join(input_dir, "gms_metadata.json")
        raw_meta: Dict[str, Any] = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                raw_meta = json.load(f)

        metadata = _decode_metadata(raw_meta)
        groups = _group_entries_by_shard(manifest.allocations)

        # One thread per shard; each thread reads its shard sequentially
        tensors: Dict[str, "torch.Tensor"] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _read_shard_sequential,
                    os.path.join(input_dir, rel_path),
                    sorted_entries,
                    device,
                ): rel_path
                for rel_path, sorted_entries in groups.items()
            }
            for future in as_completed(futures):
                rel_path = futures[future]
                try:
                    shard_tensors = future.result()
                    tensors.update(shard_tensors)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load shard {rel_path}: {exc}"
                    ) from exc

        logger.info("Loaded %d allocations from %s", len(tensors), input_dir)
        return tensors, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_metadata(self, mm: Any) -> Dict[str, Any]:
        """Read all metadata entries from GMS and return a JSON-serialisable dict.

        Values (bytes) are base64-encoded so the result is JSON-safe.
        """
        result: Dict[str, Any] = {}
        for key in mm.metadata_list():
            got = mm.metadata_get(key)
            if got is None:
                logger.warning("Metadata key disappeared during dump: %s", key)
                continue
            allocation_id, offset_bytes, value = got
            result[key] = {
                "allocation_id": str(allocation_id),
                "offset_bytes": int(offset_bytes),
                "value": base64.b64encode(value).decode("ascii"),
            }
        return result
