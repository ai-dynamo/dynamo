# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake-backed storage tier.

ROLE IN THE BACKEND STRATEGY: this is the **optional /
integration** backend. Use it when an existing Mooncake-store
fabric is already deployed and a customer wants to integrate
KV-offload bytes into that fabric. The feature surface here is
**frozen**: new GMS features (restore-on-hit, async ring,
`dest_offset` remap, GPU-direct paths) land in `NixlBackend`
first; this backend gets parity on explicit demand only. For
PRODUCTION on greenfield deployments use `NixlBackend` (its UCX
plugin covers RDMA-fabric transfers, OBJ for object storage,
GDS for GPU-direct file I/O). For DEV use `StorageTier` (Local).

Mooncake (https://github.com/kvcache-ai/Mooncake) is a distributed
key-value store with:
  - DRAM pool across nodes (the headline capability — engines on
    node A can pull from RAM on node B)
  - Optional SSD offload for cold data
  - Pluggable transport (TCP or RDMA)

What this backend delivers:
  - Real demote/promote via `MooncakeDistributedStore.put_from` /
    `get_into` with a real (external) Mooncake master.
  - Single-blob layout: [24-byte CRC header | payload] per slot
    key. Same header format as the Local and NIXL backends, so
    the CRC contract is identical (single source of truth).

Reconcile after restart: handled via a SIDECAR MANIFEST. Mooncake's
Python binding doesn't expose a list-keys API, so we can't
enumerate the master's metadata directly. Instead, the backend
appends one JSON-lines record per demote to a local file (path
passed at construction). On restart, we read the manifest and for
each record verify the key still exists via `is_exist()` — keys
that the master lost (its own restart, eviction) are silently
skipped. The manifest is append-only; compaction is a follow-up.
If `manifest_path` is None (default), reconcile is disabled and
the backend starts with an empty index — fine for single-process
test fixtures and ephemeral deployments.

Manifest compaction: the file is append-only and grows over time
with released-key records. `compact_manifest()` snapshots the
current in-memory index, atomically rewrites the file with just
those records, and swaps the append fp. Operators trigger
explicitly or via the daemon's sweeper; the backend exposes
manifest size + record-count so a policy can be data-driven.

What's deferred:
  - Pre-registered host buffer pool. Today demote/promote allocate
    a fresh staging buffer per call (header + payload bytes). A
    production version would reuse a per-engine pool to avoid the
    malloc/memcpy.

Connection model: this backend is a CLIENT of an external Mooncake
master process. The master is spun up out-of-band (systemd, k8s
pod, dev script). `master_server_addr` in the constructor points
at the master's RPC port; `metadata_server` (or
`enable_http_metadata_server`) handles cluster-wide naming."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import struct
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

from gms_kv_ring.daemon.backend import StorageBackend, StorageSlot

logger = logging.getLogger(__name__)


# Same header format as Local + NIXL backends — file/key
# interoperability across backends, single CRC source of truth.
_MAGIC = 0x47535453  # 'STSG'
_VERSION = 1
_HEADER_FMT = "<IIIIQ"  # magic, version, crc, size, _pad
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


@dataclass
class _MooncakeSlot(StorageSlot):
    """Slot record. `mc_key` is the Mooncake-side key used for
    put_from / get_into. We construct it deterministically from
    (engine_id, layer, offset) so the key is recoverable from
    the slot's identity."""

    mc_key: str = ""


def _build_key(engine_id: str, layer: int, offset: int) -> str:
    """Stable, hierarchical Mooncake key. Slashes are typical
    Mooncake convention for grouping (lets us potentially use
    `remove_by_regex` for engine-scoped wipes)."""
    return f"gms/{engine_id}/{int(layer)}/{int(offset)}"


class MooncakeBackend(StorageBackend):
    """Mooncake-backed storage tier. Requires an external master."""

    name = "mooncake"

    @classmethod
    def is_available(cls) -> bool:
        """True iff the `mooncake.store` extension imports cleanly.
        Whether a master is reachable is a separate constructor
        concern."""
        from importlib.util import find_spec

        if find_spec("mooncake") is None:
            return False
        # The .so modules under mooncake/ link against libibverbs +
        # cudart. Importing `mooncake.store` lazily checks this.
        try:
            import mooncake.store  # noqa: F401
        except ImportError as exc:
            logger.debug("mooncake.store import failed: %s", exc)
            return False
        return True

    def __init__(
        self,
        *,
        local_hostname: str = "localhost",
        metadata_server: str = "",
        master_server_addr: str = "127.0.0.1:50051",
        global_segment_size: int = 1 << 30,  # 1 GiB
        local_buffer_size: int = 256 << 20,  # 256 MiB
        protocol: str = "tcp",
        rdma_devices: str = "",
        enable_ssd_offload: bool = False,
        ssd_offload_path: str = "",
        manifest_path: Optional[str] = None,
        sync_manifest: bool = True,
    ) -> None:
        """`master_server_addr`: address of the running Mooncake
        master process (`host:port`).
        `metadata_server`: HTTP metadata server (e.g.
        "host:port") OR `"etcd://host:port"`. Pass an empty string
        to use the master as its own metadata source if the master
        was launched with `--enable_http_metadata_server`.
        `protocol`: "tcp" or "rdma". TCP works without IB hardware.
        `sync_manifest`: when True, fsync each manifest update before
        returning so a process crash cannot resurrect released keys.
        """
        if not self.is_available():
            raise RuntimeError(
                "MooncakeBackend requires the `mooncake` Python "
                "package. Install per Mooncake docs and ensure "
                "libibverbs.so.1 is on LD_LIBRARY_PATH."
            )
        from mooncake.store import MooncakeDistributedStore

        self.local_hostname = local_hostname
        self.master_server_addr = master_server_addr
        self.protocol = protocol
        self._slots: dict[tuple[str, int, int], _MooncakeSlot] = {}
        self._lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._sync_manifest = bool(sync_manifest)

        self._mds = MooncakeDistributedStore()
        # The config-dict overload is cleaner than the positional one.
        cfg = {
            "local_hostname": local_hostname,
            "metadata_server": metadata_server,
            "global_segment_size": int(global_segment_size),
            "local_buffer_size": int(local_buffer_size),
            "protocol": protocol,
            "rdma_devices": rdma_devices,
            "master_server_addr": master_server_addr,
        }
        if enable_ssd_offload:
            cfg["enable_ssd_offload"] = True
            if ssd_offload_path:
                cfg["ssd_offload_path"] = ssd_offload_path
        rc = self._mds.setup(cfg)
        if rc != 0:
            raise RuntimeError(f"MooncakeBackend setup failed: rc={rc} cfg={cfg!r}")

        # Sidecar manifest for restart-replay. None = disabled (no
        # reconcile across restarts, ephemeral mode). When set, every
        # demote appends one JSON line; reconcile replays and filters
        # via is_exist() so keys the master no longer has are
        # transparently dropped.
        self.manifest_path = manifest_path
        self._manifest_fp = None
        if manifest_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
            self._replay_manifest()
            # Open for append AFTER replay so new records land at EOF.
            self._manifest_fp = open(manifest_path, "ab", buffering=0)

        logger.info(
            "MooncakeBackend ready: master=%s protocol=%s " "manifest=%s indexed=%d",
            master_server_addr,
            protocol,
            manifest_path or "<none>",
            len(self._slots),
        )

    # ----- manifest replay (restart-replay) -----

    def _replay_manifest(self) -> None:
        """Read the JSON-lines manifest, dedupe by key (latest wins),
        verify each surviving key is still in Mooncake, and populate
        `_slots`. Missing/corrupt lines are skipped with a warning;
        the manifest is append-only and never rewritten during
        replay (compaction is a follow-up)."""
        if not os.path.exists(self.manifest_path):
            return
        candidates: dict[tuple[str, int, int], dict] = {}
        n_records = 0
        n_bad = 0
        try:
            with open(self.manifest_path, "rb") as f:
                for raw in f:
                    n_records += 1
                    try:
                        rec = json.loads(raw)
                        key = (
                            str(rec["eid"]),
                            int(rec["layer"]),
                            int(rec["offset"]),
                        )
                        if bool(rec.get("deleted", False)):
                            candidates.pop(key, None)
                            continue
                        # Validate required fields up front so a
                        # truncated line caught later doesn't propagate.
                        _ = rec["mc_key"]
                        _ = rec["size"]
                        _ = rec["crc"]
                        _ = rec["mtime"]
                        candidates[key] = rec
                    except (ValueError, KeyError, TypeError):
                        n_bad += 1
                        continue
        except OSError as exc:
            logger.warning(
                "MooncakeBackend: manifest read failed: %s — starting "
                "with empty index",
                exc,
            )
            return

        # Now verify each candidate is still present in Mooncake.
        # `is_exist` returns 1 if present, 0 if not, negative on
        # transport error. We treat anything != 1 as "skip this key."
        n_surviving = 0
        n_lost = 0
        for tkey, rec in candidates.items():
            mc_key = str(rec["mc_key"])
            try:
                rc = self._mds.is_exist(mc_key)
            except Exception:  # noqa: BLE001
                rc = -1
            if rc != 1:
                n_lost += 1
                continue
            self._slots[tkey] = _MooncakeSlot(
                size=int(rec["size"]),
                crc=int(rec["crc"]) & 0xFFFFFFFF,
                mtime=float(rec["mtime"]),
                mc_key=mc_key,
            )
            n_surviving += 1
        logger.info(
            "MooncakeBackend manifest replay: records=%d bad=%d "
            "candidates=%d surviving=%d lost=%d",
            n_records,
            n_bad,
            len(candidates),
            n_surviving,
            n_lost,
        )

    def _write_manifest_record(self, rec: dict) -> None:
        if self._manifest_fp is None:
            return
        try:
            with self._manifest_lock:
                self._manifest_fp.write(
                    (json.dumps(rec) + "\n").encode("utf-8"),
                )
                if self._sync_manifest:
                    os.fsync(self._manifest_fp.fileno())
        except OSError as exc:
            logger.warning(
                "MooncakeBackend: manifest append failed: %s",
                exc,
            )

    def _append_manifest(
        self, slot: _MooncakeSlot, engine_id: str, layer: int, offset: int
    ) -> None:
        """Append one durable live-slot record."""
        rec = {
            "eid": str(engine_id),
            "layer": int(layer),
            "offset": int(offset),
            "mc_key": slot.mc_key,
            "size": int(slot.size),
            "crc": int(slot.crc),
            "mtime": float(slot.mtime),
        }
        self._write_manifest_record(rec)

    def _append_tombstone(self, engine_id: str, layer: int, offset: int) -> None:
        """Append a deletion record so manifest replay does not
        resurrect released, pruned, or quota-evicted keys."""
        self._write_manifest_record(
            {
                "eid": str(engine_id),
                "layer": int(layer),
                "offset": int(offset),
                "deleted": True,
                "mtime": time.time(),
            }
        )

    # ----- manifest introspection / compaction -----

    def manifest_size_bytes(self) -> int:
        """Bytes on disk for the manifest file. 0 if no manifest
        configured or the file doesn't exist yet."""
        if self.manifest_path is None:
            return 0
        try:
            return os.path.getsize(self.manifest_path)
        except OSError:
            return 0

    def compact_manifest(self) -> int:
        """Rewrite the manifest to contain only records for slots
        currently in the in-memory index. Returns the byte savings
        (old size - new size); negative if compaction grew the file
        (shouldn't happen but defended against).

        Atomic: writes a temp file in the same directory, fsyncs,
        then renames over the existing manifest. The append fp is
        swapped to the new file under the same lock so concurrent
        demote()'s either land in the new file (preserved) or
        block briefly during the swap. No-op if manifest disabled.

        Compaction is the right move when the manifest has grown
        many multiples of the live record count — typical heuristic:
        compact when size > 4 * live_record_count * avg_record_bytes.
        Driver lives in operator policy / the sweeper, not here."""
        if self.manifest_path is None or self._manifest_fp is None:
            return 0
        with self._lock, self._manifest_lock:
            live_items = list(self._slots.items())
            old_size = self.manifest_size_bytes()

            # Write to a temp file in the SAME directory so the
            # rename is atomic on POSIX (cross-fs rename would copy).
            manifest_dir = os.path.dirname(
                os.path.abspath(self.manifest_path),
            )
            fd, tmp_path = tempfile.mkstemp(
                prefix=".manifest-compact-",
                suffix=".jsonl",
                dir=manifest_dir,
            )
            try:
                with os.fdopen(fd, "wb") as out:
                    for (eid, layer, offset), slot in live_items:
                        rec = {
                            "eid": eid,
                            "layer": int(layer),
                            "offset": int(offset),
                            "mc_key": slot.mc_key,
                            "size": int(slot.size),
                            "crc": int(slot.crc),
                            "mtime": float(slot.mtime),
                        }
                        out.write(
                            (json.dumps(rec) + "\n").encode("utf-8"),
                        )
                    out.flush()
                    os.fsync(out.fileno())
                # Close the old append fp BEFORE the rename so we're
                # not holding a stale inode. On Linux, rename over an
                # open file is legal — the open fd keeps pointing at
                # the old (now-unlinked) inode — but that's exactly
                # the wrong behavior here.
                try:
                    self._manifest_fp.close()
                except OSError:
                    pass
                os.rename(tmp_path, self.manifest_path)
                tmp_path = None  # rename committed
                # Reopen for append at the new file.
                self._manifest_fp = open(
                    self.manifest_path,
                    "ab",
                    buffering=0,
                )
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            new_size = self.manifest_size_bytes()
        savings = old_size - new_size
        from gms_kv_ring.common import metrics

        metrics.mooncake_manifest_compactions.inc()
        logger.info(
            "MooncakeBackend manifest compaction: %d -> %d bytes "
            "(saved %d), live records=%d",
            old_size,
            new_size,
            savings,
            len(live_items),
        )
        return savings

    def close(self) -> None:
        """Drop the Mooncake connection. Caller's responsibility to
        ensure no concurrent operations. Idempotent."""
        with self._manifest_lock:
            if self._manifest_fp is not None:
                try:
                    if self._sync_manifest:
                        os.fsync(self._manifest_fp.fileno())
                    self._manifest_fp.close()
                except OSError:
                    pass
                self._manifest_fp = None
        if self._mds is not None:
            try:
                self._mds.close()
            except Exception:  # noqa: BLE001
                pass
            self._mds = None

    # ----- data plane -----

    def demote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        host_ptr: int,
        size: int,
        crc: int,
    ) -> _MooncakeSlot:
        """Stage [header | payload] into a contiguous buffer and put
        it as a single Mooncake key. The header carries the CRC +
        size; the slot dict mirrors it so promote can verify
        without a separate metadata round trip."""
        eid = str(engine_id)
        key = _build_key(eid, layer, offset)
        total_size = _HEADER_SIZE + int(size)

        # Staging buffer: header bytes + payload bytes contiguous.
        # Each demote allocates one; production would reuse a pool.
        staging = (ctypes.c_ubyte * total_size)()
        header = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            int(crc) & 0xFFFFFFFF,
            int(size),
            0,
        )
        ctypes.memmove(
            ctypes.addressof(staging),
            header,
            _HEADER_SIZE,
        )
        ctypes.memmove(
            ctypes.addressof(staging) + _HEADER_SIZE,
            int(host_ptr),
            int(size),
        )

        rc = self._mds.put_from(
            key,
            ctypes.addressof(staging),
            total_size,
        )
        if rc != 0:
            raise RuntimeError(f"MooncakeBackend.demote: put_from rc={rc} key={key!r}")

        slot = _MooncakeSlot(
            size=int(size),
            crc=int(crc) & 0xFFFFFFFF,
            mtime=time.time(),
            mc_key=key,
        )
        with self._lock:
            self._slots[(eid, int(layer), int(offset))] = slot
        # Append after the index is committed so an exception in
        # _append_manifest can't leave the index empty but the
        # manifest claiming the slot exists.
        self._append_manifest(slot, eid, layer, offset)
        return slot

    def promote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        dest_host_ptr: int,
        max_size: int,
    ) -> Optional[int]:
        """get_into a temp [header | payload] buffer, validate
        header against the slot, memmove payload into the
        engine's destination, recompute CRC to detect any
        in-transit/at-rest corruption."""
        from gms_kv_ring.common.checksum import crc32_at_ptr

        eid = str(engine_id)
        with self._lock:
            slot = self._slots.get((eid, int(layer), int(offset)))
        if slot is None:
            return None
        if slot.size > int(max_size):
            logger.warning(
                "MooncakeBackend.promote: slot.size=%d > max_size=%d",
                slot.size,
                max_size,
            )
            return None
        total_size = _HEADER_SIZE + int(slot.size)

        staging = (ctypes.c_ubyte * total_size)()
        rc = self._mds.get_into(
            slot.mc_key,
            ctypes.addressof(staging),
            total_size,
        )
        # get_into returns bytes read on success (>= 0) and a
        # negative error code on failure. We just need not-negative
        # AND the expected total_size to make it back.
        if rc < 0:
            logger.warning(
                "MooncakeBackend.promote: get_into rc=%d key=%s",
                rc,
                slot.mc_key,
            )
            return None
        if rc != total_size:
            logger.warning(
                "MooncakeBackend.promote: short read rc=%d expected=%d " "key=%s",
                rc,
                total_size,
                slot.mc_key,
            )
            return None

        # Validate header bytes match the slot.
        hdr_bytes = bytes(
            ctypes.string_at(
                ctypes.addressof(staging),
                _HEADER_SIZE,
            )
        )
        magic, version, file_crc, file_size, _pad = struct.unpack(
            _HEADER_FMT,
            hdr_bytes,
        )
        if (
            magic != _MAGIC
            or version != _VERSION
            or file_size != slot.size
            or file_crc != slot.crc
        ):
            logger.warning(
                "MooncakeBackend.promote: header/index mismatch for %s",
                slot.mc_key,
            )
            return None

        # Copy payload portion to the engine's destination.
        ctypes.memmove(
            int(dest_host_ptr),
            ctypes.addressof(staging) + _HEADER_SIZE,
            int(slot.size),
        )
        # Re-CRC at the engine's dest to catch any munging through
        # the copy path or at-rest bit-flips.
        actual = crc32_at_ptr(int(dest_host_ptr), int(slot.size))
        if actual != slot.crc:
            logger.warning(
                "MooncakeBackend.promote: CRC mismatch for %s: "
                "stored=%#x actual=%#x",
                slot.mc_key,
                slot.crc,
                actual,
            )
            return None
        return slot.crc

    # ----- index + lifecycle -----

    def get(self, engine_id, layer, offset) -> Optional[_MooncakeSlot]:
        with self._lock:
            return self._slots.get(
                (str(engine_id), int(layer), int(offset)),
            )

    def release_slot(self, engine_id, layer, offset) -> bool:
        eid = str(engine_id)
        with self._lock:
            slot = self._slots.pop(
                (eid, int(layer), int(offset)),
                None,
            )
        if slot is None:
            return False
        self._free_resource(slot)
        self._append_tombstone(eid, layer, offset)
        return True

    def release_if_current(self, engine_id, layer, offset, expected) -> bool:
        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            if self._slots.get(key) is not expected:
                return False
            slot = self._slots.pop(key)
        self._free_resource(slot)
        self._append_tombstone(key[0], key[1], key[2])
        return True

    def release_engine(self, engine_id) -> int:
        eid = str(engine_id)
        with self._lock:
            keys = [k for k in self._slots if k[0] == eid]
            to_free = [self._slots.pop(k) for k in keys]
        for key, slot in zip(keys, to_free):
            self._free_resource(slot)
            self._append_tombstone(key[0], key[1], key[2])
        return len(to_free)

    def _free_resource(self, slot: _MooncakeSlot) -> None:
        try:
            self._mds.remove(slot.mc_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MooncakeBackend._free_resource: remove(%s) failed: %s",
                slot.mc_key,
                exc,
            )

    # ----- observability -----

    def n_slots(self) -> int:
        with self._lock:
            return len(self._slots)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(s.size for s in self._slots.values())

    def bytes_by_engine(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for (eid, _l, _o), s in self._slots.items():
                out[eid] = out.get(eid, 0) + s.size
        return out

    def stats(self) -> dict:
        with self._lock:
            slots = list(self._slots.items())
        if not slots:
            return {
                "n_slots": 0,
                "total_bytes": 0,
                "oldest_mtime": 0.0,
                "newest_mtime": 0.0,
                "bytes_by_engine": {},
            }
        by_eng: dict[str, int] = {}
        for (eid, _l, _o), s in slots:
            by_eng[eid] = by_eng.get(eid, 0) + s.size
        return {
            "n_slots": len(slots),
            "total_bytes": sum(s.size for _k, s in slots),
            "oldest_mtime": min(s.mtime for _k, s in slots),
            "newest_mtime": max(s.mtime for _k, s in slots),
            "bytes_by_engine": by_eng,
        }

    # ----- cleanup (same shape as Local / NIXL) -----

    def prune_older_than(self, max_age_seconds, now=None) -> int:
        if now is None:
            now = time.time()
        threshold = float(now) - float(max_age_seconds)
        with self._lock:
            to_remove = [
                (k, s) for k, s in list(self._slots.items()) if s.mtime < threshold
            ]
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for key, slot in to_remove:
            self._free_resource(slot)
            self._append_tombstone(key[0], key[1], key[2])
        return len(to_remove)

    def enforce_byte_quota(self, max_bytes) -> int:
        max_bytes = int(max_bytes)
        with self._lock:
            current = sum(s.size for s in self._slots.values())
            if current <= max_bytes:
                return 0
            ordered = sorted(
                self._slots.items(),
                key=lambda kv: kv[1].mtime,
            )
            to_remove = []
            for k, s in ordered:
                if current <= max_bytes:
                    break
                to_remove.append((k, s))
                current -= s.size
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for key, slot in to_remove:
            self._free_resource(slot)
            self._append_tombstone(key[0], key[1], key[2])
        return len(to_remove)

    def enforce_per_engine_byte_quota(
        self,
        max_bytes_per_engine,
    ) -> int:
        cap = int(max_bytes_per_engine)
        with self._lock:
            by_eng: dict[str, list] = {}
            for k, s in self._slots.items():
                by_eng.setdefault(k[0], []).append((k, s))
            to_remove = []
            for _eid, items in by_eng.items():
                current = sum(s.size for _, s in items)
                if current <= cap:
                    continue
                items.sort(key=lambda kv: kv[1].mtime)
                for k, s in items:
                    if current <= cap:
                        break
                    to_remove.append((k, s))
                    current -= s.size
            for k, _ in to_remove:
                self._slots.pop(k, None)
        for key, slot in to_remove:
            self._free_resource(slot)
            self._append_tombstone(key[0], key[1], key[2])
        return len(to_remove)
