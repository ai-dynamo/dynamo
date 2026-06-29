# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage-backend abstraction.

The daemon's storage tier is a pluggable thing: today's `StorageTier`
writes local files; tomorrow's `NixlBackend` ships bytes over a
NIXL transport (POSIX plugin for disk, OBJ plugin for S3) and a
`MooncakeBackend` uses the Mooncake transfer engine.

All backends implement the same `StorageBackend` interface so the
daemon shell, the engine RPCs, and the sweeper can treat them
uniformly. The slot dataclass `StorageSlot` is the abstract record;
backends are free to subclass it to attach backend-specific resource
handles (e.g., a file path for the local backend, an object key for
NIXL OBJ).

Crash semantics: a backend method may raise. The daemon wraps backend
calls in a `BackendSupervisor` (see `daemon/supervisor.py`) that
catches Python exceptions, increments a metric, and re-creates the
backend instance after N consecutive failures. A C-level SIGSEGV
from a transport library will still kill the whole daemon — surviving
that requires subprocess workers, which is a separate evolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageSlot:
    """Abstract record for one offloaded block. Backends MAY subclass
    to attach resource handles (file path, NIXL region id, S3 key);
    this base shape carries only what the cleanup primitives use."""

    size: int
    # CRC32 (IEEE) of the payload bytes. Stored at demote, verified
    # on promote. Single source of truth across tier hops.
    crc: int
    # Last-write time, seconds since epoch. Set by demote() and
    # backend reconcile(). Drives LRU + TTL eviction without
    # per-slot stat() syscalls.
    mtime: float = 0.0


class StorageBackend(ABC):
    """Pluggable backend for the storage tier.

    Lifecycle:
      - Backends are constructed at daemon startup. The constructor
        SHOULD probe its runtime deps and raise a clear error if
        they're missing (typically via `cls.is_available()` first).
      - `reconcile()` is called at construction to rebuild the index
        from durable state (disk files / remote registry / etc).
      - The daemon calls `demote/promote/get/release_*` during
        normal operation.
      - Cleanup primitives (`prune_older_than`, `enforce_byte_quota`,
        `enforce_per_engine_byte_quota`) are driven by the sweeper
        thread and operator-driven RPCs.

    Slot identity is (engine_id, layer, offset). The backend owns the
    mapping from key → backend-specific resource."""

    #: Short identifier exported via /metrics + logs. Override.
    name: str = "abstract"

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff this backend's runtime dependencies are
        importable + usable in the current process. Default: True
        (no deps). Backends that wrap external libs override this to
        probe their imports lazily — so a daemon shipping with no
        NIXL installed still imports cleanly."""
        return True

    # ----- core slot lifecycle -----

    @abstractmethod
    def demote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        host_ptr: int,
        size: int,
        crc: int,
    ) -> StorageSlot:
        """Write `size` bytes from pinned-host `host_ptr` into durable
        backend storage, stamped with the provided CRC. Returns the
        slot record (subclass-specific resource handle attached).

        Caller MUST guarantee the source bytes are stable for the
        duration of the call (typically by having synced the engine's
        evict stream before calling demote)."""

    @abstractmethod
    def promote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        dest_host_ptr: int,
        max_size: int,
    ) -> Optional[int]:
        """Read the backend's stored bytes into pinned-host
        `dest_host_ptr` and verify integrity. Returns the verified
        CRC on success. Returns None if:
          - the slot is unknown
          - reading the backend failed
          - the on-backend size doesn't fit in max_size
          - the CRC verify failed (at-rest corruption)
        Implementations MUST be safe to call concurrently with
        demote() of other keys."""

    @abstractmethod
    def get(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> Optional[StorageSlot]:
        """Index lookup. Cheap, in-memory."""

    @abstractmethod
    def release_slot(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        """Drop the slot and free its backend resource. Idempotent."""

    def release_if_current(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        expected: StorageSlot,
    ) -> bool:
        """Release only when ``expected`` is still indexed at the key.

        Backends should override this with an atomic implementation. The
        conservative default never risks deleting a replacement slot.
        """
        return False

    @abstractmethod
    def release_engine(self, engine_id: str) -> int:
        """Free every slot for `engine_id`. Returns count released."""

    # ----- introspection / observability -----

    @abstractmethod
    def n_slots(self) -> int:
        ...

    @abstractmethod
    def total_bytes(self) -> int:
        ...

    @abstractmethod
    def bytes_by_engine(self) -> dict[str, int]:
        ...

    @abstractmethod
    def stats(self) -> dict:
        ...

    # ----- cleanup primitives (driven by sweeper + RPCs) -----

    @abstractmethod
    def prune_older_than(
        self,
        max_age_seconds: float,
        now: Optional[float] = None,
    ) -> int:
        ...

    @abstractmethod
    def enforce_byte_quota(self, max_bytes: int) -> int:
        ...

    @abstractmethod
    def enforce_per_engine_byte_quota(
        self,
        max_bytes_per_engine: int,
    ) -> int:
        ...

    # ----- optional: key snapshot for scrub -----

    def snapshot_keys(self) -> list[tuple[str, int, int]]:
        """Return a copy of the current slot-key list. Used by the
        daemon's background backend-scrub thread, which doesn't
        want to hold the backend's lock across per-slot read +
        CRC compute (slow when reading large slots from disk).

        Default implementation reads `self._slots` under
        `self._lock` if both attributes exist — true for every
        first-party backend (Local, NIXL, Mooncake). Backends that
        don't follow this convention return an empty list, opting
        out of scrub support."""
        slots = getattr(self, "_slots", None)
        if slots is None:
            return []
        lock = getattr(self, "_lock", None)
        if lock is None:
            return list(slots.keys())
        with lock:
            return list(slots.keys())

    # ----- optional GPU-direct data plane (GDS-style backends) -----

    def supports_gpu_direct(self) -> bool:
        """True iff the backend has `demote_from_gpu` /
        `promote_into_gpu` data-plane methods (i.e., it can accept a
        GPU pointer as source/dest and skip CPU staging). Default
        False — most backends route through pinned host."""
        return False

    # ----- optional sidecar manifest (mooncake-style backends) -----

    def manifest_size_bytes(self) -> int:
        """Bytes on disk for any sidecar metadata file the backend
        maintains (e.g., MooncakeBackend's JSONL manifest). Default 0
        — backends that walk durable storage on reconcile (Local,
        NIXL POSIX) don't have a sidecar."""
        return 0

    def compact_manifest(self) -> int:
        """Rewrite the sidecar manifest to drop dead records.
        Returns bytes freed. Default 0 (no-op) — only backends that
        maintain an append-only manifest implement this."""
        return 0

    def _sweep_once(
        self,
        max_age_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_bytes_per_engine: Optional[int] = None,
    ) -> tuple[int, int, int]:
        """Run one cleanup pass.
        Returns (ttl_evicted, per_engine_evicted, global_quota_evicted).

        Order: TTL → per-engine → global. Each phase sees state from
        the prior. Backends may override if they want a different
        order, but the default works for any in-memory `_slots` shape."""
        ttl_n = 0
        per_eng_n = 0
        global_n = 0
        if max_age_seconds is not None:
            ttl_n = self.prune_older_than(float(max_age_seconds))
        if max_bytes_per_engine is not None:
            per_eng_n = self.enforce_per_engine_byte_quota(
                int(max_bytes_per_engine),
            )
        if max_bytes is not None:
            global_n = self.enforce_byte_quota(int(max_bytes))
        return ttl_n, per_eng_n, global_n
