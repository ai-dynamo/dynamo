# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage tier — filesystem-backed offload below host_tier.

ROLE IN THE BACKEND STRATEGY: this is the **dev / test / CI**
backend. Use it when a zero-extra-deps path is required (pure
Python + local filesystem only) — unit tests, laptop dev, CI
without NIXL installed. **PRODUCTION** should use `NixlBackend`
(POSIX plugin is a strict superset functionally, plus brings
GDS/OBJ/UCX). **Mooncake** is OPTIONAL for users already running
a Mooncake-store fabric. New GMS features land in NIXL first;
Local and Mooncake only get parity on explicit demand. The state
machine + sweeper + CRC + manifest infrastructure is shared via
`backend.StorageBackend`, so the behavioral contract is uniform
across all three.

State machine (per (engine_id, layer, offset) block):

    EMPTY -> HBM_FRESH -> EVICT_PENDING_HOST -> AT_REST_HOST
        ^                                            |
        |                                            v
        |                                   SPILL_PENDING_STORAGE  <-- (this module)
        |                                            |
        |                                            v
        +---- RESTORE_PENDING <-- AT_REST_HOST <-- AT_REST_STORAGE
                                  (promote)        (demote)

Demote (AT_REST_HOST -> AT_REST_STORAGE):
    1. Read host_tier slot's pinned bytes + its stored CRC.
    2. Write a versioned file: header (magic+version+crc+size) + payload.
    3. fsync the payload so a crash doesn't leave a torn file looking valid.
    4. Atomic rename into place (open temp, rename) so a partial file
       is never indexable.
    5. Free the host_tier slot.

Promote (AT_REST_STORAGE -> AT_REST_HOST):
    1. Open the file, verify magic/version, read header.
    2. Allocate a host_tier slot of size == header.size.
    3. Read payload bytes into the slot.
    4. Recompute CRC over the bytes. If it doesn't match the header's
       stored CRC, FAIL — the bytes were corrupted at rest.
    5. mark_ready the host_tier slot with the (now-verified) CRC.

The CRC pattern matches host_tier exactly — same CRC32 (IEEE), same
helper (`common/checksum.py`). A demoted block thus survives any
number of tier-to-tier hops with end-to-end integrity: each
transition re-verifies and re-stamps.

This module is intentionally minimal: synchronous file I/O, no
worker thread, no LRU. SSD bandwidth (~2-3 GB/s for NVMe) is the
dominant cost; a worker pool can be layered above if needed."""

from __future__ import annotations

import logging
import os
import struct
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

from gms_kv_ring.daemon.backend import StorageBackend

logger = logging.getLogger(__name__)


_MAGIC = 0x47535453  # 'STSG' little-endian read as "GSTS"
_VERSION = 1
_HEADER_FMT = "<IIIIQ"  # magic, version, crc, size, _pad
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


@dataclass
class _StorageSlot:
    path: str
    size: int
    # CRC32 of the payload bytes, captured at demote-time and
    # re-verified on promote. Mismatch on promote == at-rest corruption.
    crc: int
    # Last-write time, seconds since epoch. Set at demote (time.time())
    # and at reconcile (os.stat). Drives LRU quota eviction and TTL
    # prune without per-iteration stat() syscalls.
    mtime: float = 0.0


class StorageTier(StorageBackend):
    """Filesystem-backed offload pool, keyed by (engine_id, layer, offset).

    All files live under `base_dir` in a deterministic per-engine
    subtree. Files are written atomically (temp + rename + fsync) so a
    crash mid-demote never leaves a torn file the index would trust.

    Implements the `StorageBackend` interface — see daemon/backend.py
    for the contract. Other backends (NIXL, Mooncake) sit beside this
    one as siblings."""

    name = "local"

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self._slots: dict[tuple[str, int, int], _StorageSlot] = {}
        self._lock = threading.Lock()
        # On startup, rebuild the index from disk and clean up any
        # torn temp files left over from a kill -9 mid-write.
        self._reconcile_from_disk()

    # ----- startup reconcile (crash-recovery) -----

    def _reconcile_from_disk(self) -> None:
        """Walk `base_dir` and rebuild `_slots` from on-disk files.

        Daemon restart invariant: in-memory state is gone, but storage
        files persist. After this runs:
          - every valid file is indexed with its header CRC
          - every torn .tmp-* file is unlinked (atomic rename never
            ran => caller's write was incomplete => file is suspect)
          - files with bad magic/version are LEFT on disk but not
            indexed (out-of-band recovery / forensics)

        CRC of the payload is verified lazily on promote — scanning
        every byte at startup would be O(disk size). The header CRC
        stamp gives us index-level integrity; the lazy verify gives
        us payload-level integrity. Both must agree on promote."""
        n_indexed = 0
        n_torn = 0
        n_bad = 0
        for dirpath, _dirnames, filenames in os.walk(self.base_dir):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if fn.startswith(".tmp-"):
                    try:
                        os.unlink(full)
                        n_torn += 1
                    except OSError:
                        pass
                    continue
                if not fn.endswith(".bin"):
                    continue
                # Derive (engine_id, layer, offset) from the path.
                rel = os.path.relpath(full, self.base_dir)
                parts = rel.split(os.sep)
                if len(parts) != 3:
                    continue
                eid_quoted, layer_str, offset_bin = parts
                try:
                    layer = int(layer_str)
                    offset = int(offset_bin[: -len(".bin")])
                except ValueError:
                    continue
                from urllib.parse import unquote

                engine_id = unquote(eid_quoted)
                slot = self._read_header(full)
                if slot is None:
                    n_bad += 1
                    continue
                self._slots[(engine_id, layer, offset)] = slot
                n_indexed += 1
        if n_indexed or n_torn or n_bad:
            logger.info(
                "storage_tier reconcile: indexed=%d torn-tmp-cleaned=%d "
                "bad-header=%d",
                n_indexed,
                n_torn,
                n_bad,
            )

    def _read_header(self, path: str) -> Optional[_StorageSlot]:
        """Open `path`, read+validate the 24-byte header. Returns a
        slot describing the indexed entry, or None on any error.
        Does NOT read the payload (lazy CRC verify happens on promote)."""
        try:
            with open(path, "rb") as f:
                hdr = f.read(_HEADER_SIZE)
                if len(hdr) != _HEADER_SIZE:
                    return None
                magic, version, file_crc, file_size, _pad = struct.unpack(
                    _HEADER_FMT,
                    hdr,
                )
                if magic != _MAGIC or version != _VERSION:
                    return None
                # Sanity-check the file is at least header + payload.
                # Don't read the payload — that's promote's job.
                f.seek(0, 2)
                disk_size = f.tell()
                if disk_size < _HEADER_SIZE + file_size:
                    # Truncated file — write got killed after the
                    # rename but before fsync completed flushing the
                    # payload. (Unlikely with our write order, but
                    # possible on power loss between fsync and dirent
                    # commit.) Discard the index entry.
                    logger.warning(
                        "reconcile: truncated file %s " "(disk_size=%d header.size=%d)",
                        path,
                        disk_size,
                        file_size,
                    )
                    return None
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = time.time()
            return _StorageSlot(
                path=path,
                size=int(file_size),
                crc=int(file_crc) & 0xFFFFFFFF,
                mtime=mtime,
            )
        except OSError as exc:
            logger.warning("reconcile: read %s failed: %s", path, exc)
            return None

    # ----- key/path helpers -----

    def _path_for(self, engine_id: str, layer: int, offset: int) -> str:
        # engine_id may contain user-provided characters; quote for fs.
        safe_eid = quote(str(engine_id), safe="")
        return os.path.join(
            self.base_dir,
            safe_eid,
            str(int(layer)),
            f"{int(offset)}.bin",
        )

    # ----- demote: host_tier slot -> filesystem -----

    def demote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        host_ptr: int,
        size: int,
        crc: int,
    ) -> _StorageSlot:
        """Write `size` bytes from `host_ptr` to a file, stamped with
        the provided CRC. Replaces any prior file at the same key.

        Caller is responsible for ensuring the host_tier slot's CRC
        matches the bytes being written — typically by passing the
        slot's stored CRC directly. We do NOT recompute here because:
          a) we want to detect drift across tier boundaries (i.e., if
             this CRC doesn't match the file's bytes on promote, that
             tells us *where* the corruption occurred), and
          b) it's a single source of truth — the host_tier CRC.

        Raises OSError on filesystem failure."""
        import ctypes

        key = (str(engine_id), int(layer), int(offset))
        path = self._path_for(engine_id, layer, offset)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write to a temp file in the same dir, fsync, rename.
        # Same-dir rename is atomic on POSIX.
        header = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            int(crc) & 0xFFFFFFFF,
            int(size),
            0,
        )
        buf = (ctypes.c_ubyte * size).from_address(int(host_ptr))
        payload = bytes(buf)  # one copy: pinned RAM -> bytes

        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-",
            suffix=".bin",
            dir=os.path.dirname(path),
        )
        try:
            with os.fdopen(fd, "wb", closefd=True) as f:
                f.write(header)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, path)
            tmp_path = None  # signal: rename succeeded, don't unlink
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        slot = _StorageSlot(
            path=path,
            size=int(size),
            crc=int(crc) & 0xFFFFFFFF,
            mtime=time.time(),
        )
        with self._lock:
            self._slots[key] = slot
        return slot

    # ----- promote: filesystem -> host_tier slot -----

    def promote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        dest_host_ptr: int,
        max_size: int,
    ) -> Optional[int]:
        """Read the storage file into `dest_host_ptr` and verify CRC.

        Returns the verified CRC on success (caller stores it on the
        host_tier slot via mark_ready). Returns None if:
          - no storage slot at this key
          - file header is malformed (magic/version mismatch)
          - file size doesn't fit in max_size
          - CRC mismatch (at-rest corruption)

        Logs the failure cause at WARNING level."""
        import ctypes

        from gms_kv_ring.common.checksum import crc32_at_ptr

        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
        if slot is None:
            logger.debug("promote: no slot at %r", key)
            return None
        if slot.size > int(max_size):
            logger.warning(
                "promote: slot.size=%d > max_size=%d for %r",
                slot.size,
                max_size,
                key,
            )
            return None

        try:
            with open(slot.path, "rb") as f:
                hdr_bytes = f.read(_HEADER_SIZE)
                if len(hdr_bytes) != _HEADER_SIZE:
                    logger.warning(
                        "promote: short header for %r (%d bytes)",
                        key,
                        len(hdr_bytes),
                    )
                    return None
                magic, version, file_crc, file_size, _pad = struct.unpack(
                    _HEADER_FMT,
                    hdr_bytes,
                )
                if magic != _MAGIC or version != _VERSION:
                    logger.warning(
                        "promote: bad magic/version for %r: %#x/%d",
                        key,
                        magic,
                        version,
                    )
                    return None
                if file_size != slot.size or file_crc != slot.crc:
                    # Index and on-disk header disagree — that's a
                    # write-side bug. Don't trust the file.
                    logger.warning(
                        "promote: index/header mismatch for %r: "
                        "size %d/%d crc %#x/%#x",
                        key,
                        slot.size,
                        file_size,
                        slot.crc,
                        file_crc,
                    )
                    return None
                payload = f.read(file_size)
                if len(payload) != file_size:
                    logger.warning(
                        "promote: short payload for %r (%d/%d bytes)",
                        key,
                        len(payload),
                        file_size,
                    )
                    return None
        except OSError as exc:
            logger.warning("promote: open/read failed for %r: %s", key, exc)
            return None

        # Write into the destination host_tier slot.
        dst = (ctypes.c_ubyte * file_size).from_address(int(dest_host_ptr))
        ctypes.memmove(dst, payload, file_size)

        # Re-verify CRC from the bytes we just wrote — catches both
        # at-rest disk corruption AND any in-flight munging through
        # the read path.
        actual_crc = crc32_at_ptr(int(dest_host_ptr), file_size)
        if actual_crc != slot.crc:
            logger.warning(
                "promote: CRC mismatch for %r: stored=%#x actual=%#x "
                "(at-rest corruption)",
                key,
                slot.crc,
                actual_crc,
            )
            return None
        return slot.crc

    # ----- introspection / lifecycle -----

    def get(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> Optional[_StorageSlot]:
        with self._lock:
            return self._slots.get(
                (str(engine_id), int(layer), int(offset)),
            )

    def release_slot(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        with self._lock:
            slot = self._slots.pop(
                (str(engine_id), int(layer), int(offset)),
                None,
            )
        if slot is None:
            return False
        try:
            os.unlink(slot.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("release_slot: unlink failed: %s", exc)
        return True

    def release_if_current(self, engine_id, layer, offset, expected) -> bool:
        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            if self._slots.get(key) is not expected:
                return False
            slot = self._slots.pop(key)
        try:
            os.unlink(slot.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("release_if_current: unlink failed: %s", exc)
        return True

    def release_engine(self, engine_id: str) -> int:
        eid = str(engine_id)
        to_release: list[_StorageSlot] = []
        with self._lock:
            keys = [k for k in self._slots if k[0] == eid]
            for k in keys:
                to_release.append(self._slots.pop(k))
        for slot in to_release:
            try:
                os.unlink(slot.path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        # Best-effort cleanup of the engine's subdirectory tree.
        eng_dir = os.path.join(self.base_dir, quote(eid, safe=""))
        if os.path.isdir(eng_dir):
            for dirpath, _dirnames, filenames in os.walk(
                eng_dir,
                topdown=False,
            ):
                for fn in filenames:
                    try:
                        os.unlink(os.path.join(dirpath, fn))
                    except OSError:
                        pass
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
        return len(to_release)

    def n_slots(self) -> int:
        with self._lock:
            return len(self._slots)

    def total_bytes(self) -> int:
        """Sum of payload sizes across all indexed slots. Does NOT
        include the 24-byte header per file or the filesystem's
        block-size overhead — those are typically a small fraction."""
        with self._lock:
            return sum(s.size for s in self._slots.values())

    def bytes_by_engine(self) -> dict[str, int]:
        """Bytes resident per engine_id. O(n_slots), in-memory."""
        out: dict[str, int] = {}
        with self._lock:
            for (eid, _layer, _offset), s in self._slots.items():
                out[eid] = out.get(eid, 0) + s.size
        return out

    def stats(self) -> dict:
        """Snapshot for /metrics + operator dashboards. Cheap — all
        in-memory, no syscalls. Includes per-engine bytes so operators
        can size per-engine quotas."""
        with self._lock:
            slots_list = list(self._slots.items())
        if not slots_list:
            return {
                "n_slots": 0,
                "total_bytes": 0,
                "oldest_mtime": 0.0,
                "newest_mtime": 0.0,
                "bytes_by_engine": {},
            }
        by_eng: dict[str, int] = {}
        for (eid, _l, _o), s in slots_list:
            by_eng[eid] = by_eng.get(eid, 0) + s.size
        return {
            "n_slots": len(slots_list),
            "total_bytes": sum(s.size for _k, s in slots_list),
            "oldest_mtime": min(s.mtime for _k, s in slots_list),
            "newest_mtime": max(s.mtime for _k, s in slots_list),
            "bytes_by_engine": by_eng,
        }

    # ----- operator-driven cleanup -----

    def prune_older_than(
        self,
        max_age_seconds: float,
        now: Optional[float] = None,
    ) -> int:
        """TTL eviction. Unlinks every slot whose mtime is older than
        `now - max_age_seconds`. Returns the count removed.

        Driven by an operator-facing RPC or a periodic policy thread.
        Concretely useful when (a) an engine was retired and its
        stale spills should age out, or (b) a daemon is approaching
        a disk-pressure threshold and operators want a coarse purge."""
        if now is None:
            now = time.time()
        threshold = float(now) - float(max_age_seconds)
        to_remove: list[tuple[tuple[str, int, int], _StorageSlot]] = []
        with self._lock:
            for k, s in list(self._slots.items()):
                if s.mtime < threshold:
                    to_remove.append((k, s))
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for _k, slot in to_remove:
            try:
                os.unlink(slot.path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("prune: unlink %s failed: %s", slot.path, exc)
        return len(to_remove)

    def _sweep_once(
        self,
        max_age_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_bytes_per_engine: Optional[int] = None,
    ) -> tuple[int, int, int]:
        """Run one cleanup pass.
        Returns (ttl_evicted, per_engine_evicted, global_quota_evicted).

        Order matters: TTL first (clears obvious junk), then per-engine
        (catches noisy individual engines), then global quota
        (mops up aggregate pressure). Each phase reads the state left
        by the prior — so a per-engine eviction may make global a
        no-op, which is the right behavior."""
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

    def enforce_per_engine_byte_quota(
        self,
        max_bytes_per_engine: int,
    ) -> int:
        """LRU eviction PER ENGINE: any engine over `max_bytes_per_engine`
        loses its oldest slots until under. Returns total evictions
        across all engines.

        Why per-engine in addition to global quota: shared storage
        without per-engine bounds means one noisy engine can fill
        the disk and the global LRU then evicts other engines'
        legitimate (smaller, older) data. With per-engine quotas,
        a misbehaving tenant only evicts ITS OWN data."""
        cap = int(max_bytes_per_engine)
        n_evicted = 0
        with self._lock:
            # Group slots by engine_id.
            by_eng: dict[str, list[tuple[tuple[str, int, int], _StorageSlot]]] = {}
            for k, s in self._slots.items():
                by_eng.setdefault(k[0], []).append((k, s))
            to_remove: list[tuple[tuple[str, int, int], _StorageSlot]] = []
            for eid, items in by_eng.items():
                current = sum(s.size for _, s in items)
                if current <= cap:
                    continue
                # Oldest-mtime-first within this engine.
                items.sort(key=lambda kv: kv[1].mtime)
                for k, s in items:
                    if current <= cap:
                        break
                    to_remove.append((k, s))
                    current -= s.size
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for _k, slot in to_remove:
            try:
                os.unlink(slot.path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "enforce_per_engine_quota: unlink %s failed: %s",
                    slot.path,
                    exc,
                )
            n_evicted += 1
        return n_evicted

    def enforce_byte_quota(self, max_bytes: int) -> int:
        """LRU eviction until total_bytes <= max_bytes. Returns the
        count of slots evicted.

        Oldest-mtime-first. Ties are broken arbitrarily (dict order).
        This is best-effort: if a concurrent demote happens during
        enforcement, the post-condition `total_bytes <= max_bytes`
        may not strictly hold — caller can re-run if precise bound
        matters. For typical operator use (background sweeper), the
        eventual-consistency is fine."""
        max_bytes = int(max_bytes)
        n_evicted = 0
        with self._lock:
            current = sum(s.size for s in self._slots.values())
            if current <= max_bytes:
                return 0
            # Sort by mtime ascending — oldest first.
            ordered = sorted(
                self._slots.items(),
                key=lambda kv: kv[1].mtime,
            )
            to_remove: list[tuple[tuple[str, int, int], _StorageSlot]] = []
            for k, s in ordered:
                if current <= max_bytes:
                    break
                to_remove.append((k, s))
                current -= s.size
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for _k, slot in to_remove:
            try:
                os.unlink(slot.path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("enforce_quota: unlink %s failed: %s", slot.path, exc)
            n_evicted += 1
        return n_evicted


class StorageSweeper:
    """Background thread that periodically runs cleanup against a
    StorageTier. Configured with TTL and/or byte quota; either or
    both may be None (disabled).

    Lifecycle: tied to the Daemon's serve() — started when the daemon
    starts listening, stopped when serve() returns. The thread is a
    plain daemon thread (not asyncio); we don't share state with the
    control loop beyond the StorageTier instance, which is already
    lock-protected.

    Why a thread, not an asyncio task: cleanup is synchronous
    filesystem I/O. Doing it on the asyncio loop would block control
    RPCs for the duration of the sweep — fine when there's nothing
    to evict, but unbounded when the index is large or disk is slow.
    A separate thread keeps the control plane responsive."""

    def __init__(
        self,
        storage_tier: StorageBackend,
        *,
        interval_s: float,
        max_age_s: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_bytes_per_engine: Optional[int] = None,
        compact_manifest_at_bytes: Optional[int] = None,
        on_sweep: Optional[callable] = None,
    ) -> None:
        """`compact_manifest_at_bytes`: if set, after each sweep the
        sweeper checks `backend.manifest_size_bytes()` and calls
        `compact_manifest()` when the file is over the threshold.
        Backends without a manifest (Local, NIXL POSIX) return 0
        from both, so the check is a free no-op for them."""
        if interval_s <= 0:
            raise ValueError(f"sweeper interval_s must be > 0, got {interval_s}")
        if (
            max_age_s is None
            and max_bytes is None
            and max_bytes_per_engine is None
            and compact_manifest_at_bytes is None
        ):
            raise ValueError(
                "sweeper needs at least one of max_age_s / max_bytes / "
                "max_bytes_per_engine / compact_manifest_at_bytes",
            )
        self.storage_tier = storage_tier
        self.interval_s = float(interval_s)
        self.max_age_s = max_age_s
        self.max_bytes = max_bytes
        self.max_bytes_per_engine = max_bytes_per_engine
        self.compact_manifest_at_bytes = compact_manifest_at_bytes
        # Test hook: called after each sweep as
        # (ttl_n, per_engine_n, global_n).
        self._on_sweep = on_sweep
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Counters for introspection. Read by tests + the daemon's
        # exported gauge. No lock needed for ints under GIL.
        self.sweeps_run = 0
        self.last_ttl_evicted = 0
        self.last_per_engine_evicted = 0
        self.last_quota_evicted = 0
        self.compactions_run = 0
        self.last_compaction_savings = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sweeper already started")
        self._thread = threading.Thread(
            target=self._run,
            name="storage-sweeper",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal + join. Safe to call multiple times."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "storage sweeper failed to exit within %.1fs",
                    timeout,
                )
        self._thread = None

    def _run(self) -> None:
        logger.info(
            "storage sweeper: interval=%.1fs max_age=%s max_bytes=%s "
            "max_bytes_per_engine=%s",
            self.interval_s,
            self.max_age_s,
            self.max_bytes,
            self.max_bytes_per_engine,
        )
        # Wait first so we don't race the daemon's startup. Use the
        # stop event so shutdown is immediate, not blocked on a sleep.
        while not self._stop.wait(self.interval_s):
            try:
                ttl_n, per_eng_n, quota_n = self.storage_tier._sweep_once(
                    max_age_seconds=self.max_age_s,
                    max_bytes=self.max_bytes,
                    max_bytes_per_engine=self.max_bytes_per_engine,
                )
                self.sweeps_run += 1
                self.last_ttl_evicted = ttl_n
                self.last_per_engine_evicted = per_eng_n
                self.last_quota_evicted = quota_n
                # Optional manifest compaction. Cheap when no manifest
                # (default impl returns 0); when configured, triggers
                # a rewrite once the file bloats past the threshold.
                # Runs AFTER eviction so any release-driven dead
                # records are already gone from the in-memory index
                # — compaction will drop them from the manifest too.
                if self.compact_manifest_at_bytes is not None:
                    size = self.storage_tier.manifest_size_bytes()
                    if size > self.compact_manifest_at_bytes:
                        savings = self.storage_tier.compact_manifest()
                        self.compactions_run += 1
                        self.last_compaction_savings = savings
                        logger.info(
                            "storage sweeper: manifest compaction "
                            "fired (size=%d > threshold=%d, "
                            "savings=%d)",
                            size,
                            self.compact_manifest_at_bytes,
                            savings,
                        )
                if ttl_n or per_eng_n or quota_n:
                    logger.info(
                        "storage sweeper: ttl=%d per_engine=%d quota=%d " "(sweep #%d)",
                        ttl_n,
                        per_eng_n,
                        quota_n,
                        self.sweeps_run,
                    )
                if self._on_sweep is not None:
                    try:
                        self._on_sweep(ttl_n, per_eng_n, quota_n)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "storage sweeper: on_sweep hook failed",
                            exc_info=True,
                        )
            except Exception:  # noqa: BLE001
                # Sweeper must never crash — a one-off failure (disk
                # full, permission glitch) is recoverable; we'll try
                # again next interval. Crashing leaves storage growing
                # silently.
                logger.warning(
                    "storage sweeper: sweep failed, will retry",
                    exc_info=True,
                )
