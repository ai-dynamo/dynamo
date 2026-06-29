# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL-backed storage tier.

ROLE IN THE BACKEND STRATEGY: this is the **production** backend.
Today it is the only path that supports GPU-direct storage I/O
(GDS / GDS_MT plugins), and via its plugin matrix it functionally
supersedes the Local backend (POSIX plugin = same byte-level
capabilities) and a subset of Mooncake (UCX for RDMA fabric,
OBJ/AZURE_BLOB for object stores). New GMS features
(restore-on-hit, async ring, `dest_offset` remap) land here first;
Local is for dev/test/CI, and Mooncake is for integrators with
an existing Mooncake-store deployment (feature-frozen surface;
new features port on demand only).

NIXL is NVIDIA's unified transport abstraction over UCX, GDS, POSIX,
GDS_MT, OBJ (S3), AZURE_BLOB and GUSLI plugins. This backend supports
several plugin modes:

  POSIX (default): local/networked files. Files live under `base_dir`
    in a `<engine>/<layer>/<offset>.bin` tree. Reconcile walks the
    filesystem on startup. No manifest needed; the directory is the
    source of truth.

  GDS / GDS_MT: GPUDirect Storage. Same file layout as POSIX, but
    the data plane can DMA directly between GPU HBM and the file
    bypassing CPU staging — when the source is a VRAM_SEG (GPU
    pointer). Use `demote_from_gpu` / `promote_into_gpu` to take
    that path; plain `demote` / `promote` still work with a host
    pointer (then GDS plugin falls back to a CPU-staged path
    equivalent to POSIX). Requires a GDS-capable filesystem
    (NVMe with cuFile-enabled mount, BeeGFS, Lustre, etc.) for
    the actual GPU-direct benefit; on a non-GDS mount the
    transfer succeeds but routes through host RAM.

  OBJ: S3-compatible object storage. Slots are addressed by string
    key, not filesystem path. NIXL doesn't expose list-objects via
    its Python API, so OBJ requires a sidecar JSONL manifest (same
    shape and replay semantics as the Mooncake backend). Pass
    `manifest_path` at construction. The bucket is configured via
    `AWS_DEFAULT_BUCKET` env var or `backend_init_params={"bucket":
    "..."}`.

File format on disk is bit-identical to `StorageTier` (LocalStorage):
24-byte CRC32-stamped header + payload. A NIXL-written file can be
read by a Local backend and vice versa — useful for migration.

Atomic write: like Local, demote writes to a temp file, fsyncs, then
renames into place. Reconcile (called from the inherited
`__init__`) ignores any `.tmp-*` leftovers from a crash.

Performance: for the POSIX plugin specifically, NIXL adds agent
overhead vs raw pread/pwrite — perf parity with the Local backend,
maybe a hair slower. The architectural win is plugin substitution:
swapping POSIX for GDS (GPUDirect Storage), OBJ (S3), or GDS_MT
later is a configuration change, not a code rewrite. NIXL is also
where the Mooncake-style distributed RAM path lands when we
introduce a UCX plugin selection.

Concurrency: NIXL agents are thread-safe for concurrent transfers
on different xfer handles. We hold `self._lock` only for the slot
index update; the NIXL transfer happens outside the lock.

Optimization deferred: per-transfer register/deregister is paying a
sys call cost per block. A real production version would
pre-register a host-buffer pool once and reuse the registration
across transfers. Same for the file fd: open-on-demote, close-on-
done is simpler; a per-engine fd cache would be faster."""

from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, unquote

from gms_kv_ring.daemon.backend import StorageBackend, StorageSlot

logger = logging.getLogger(__name__)


# Same header format as LocalStorageBackend — files are
# interchangeable between Local and NIXL backends.
_MAGIC = 0x47535453  # 'STSG' ('GSTS' little-endian)
_VERSION = 1
_HEADER_FMT = "<IIIIQ"  # magic, version, crc, size, _pad
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Transfer polling cadence. Short enough to keep latency low,
# long enough to avoid spin-burning a core under load.
_POLL_S = 1e-4

# Plugins that store data in files on a (possibly distributed)
# filesystem — POSIX semantics for atomic write + reconcile.
# GDS/GDS_MT use the same file layout; what differs is whether
# the source register is DRAM_SEG (CPU-staged) or VRAM_SEG (GPU
# direct), driven by which demote/promote method the caller uses.
_FILE_PLUGINS = frozenset({"POSIX", "GDS", "GDS_MT"})


@dataclass
class _NixlSlot(StorageSlot):
    """Slot record for NIXL backend. Same fields as Local — file
    path is the resource handle. We could attach a cached fd /
    registration here for repeat-access optimization (deferred)."""

    path: str = ""


class NixlBackend(StorageBackend):
    """NIXL-backed storage tier. POSIX plugin by default."""

    name = "nixl"

    @classmethod
    def is_available(cls) -> bool:
        """True iff `nixl` Python binding imports cleanly. Doesn't
        guarantee a specific plugin works — that's checked at
        construction time."""
        from importlib.util import find_spec

        return find_spec("nixl") is not None

    def supports_gpu_direct(self) -> bool:
        """GPU-direct data plane is available only under the GDS /
        GDS_MT plugins. POSIX/OBJ go through CPU pinned host."""
        return self.plugin in ("GDS", "GDS_MT")

    def __init__(
        self,
        base_dir: str,
        *,
        plugin: str = "POSIX",
        agent_name: Optional[str] = None,
        backend_init_params: Optional[dict] = None,
        manifest_path: Optional[str] = None,
        bucket: Optional[str] = None,
        sync_manifest: bool = True,
    ) -> None:
        """`base_dir`: filesystem path for POSIX-backed files. Ignored
        in OBJ mode (objects live in the bucket, not on local disk),
        but still used as the parent dir for the sidecar manifest by
        default.
        `plugin`: NIXL plugin to use. Must be in `get_plugin_list()`.
        Today: POSIX (files) and OBJ (S3-compatible) are validated;
        other plugins (GDS, UCX, ...) are accepted but data-plane
        code paths haven't been exercised.
        `agent_name`: optional explicit name; default is a uuid.
        `backend_init_params`: forwarded to `agent.create_backend()`.
        For OBJ, `bucket` is required either here or via
        `AWS_DEFAULT_BUCKET` env or the `bucket=` kwarg.
        `manifest_path`: required for OBJ (no list-objects API).
        Optional for POSIX (FS walk is the source of truth).
        `bucket`: convenience kwarg that maps to
        backend_init_params["bucket"]. Overrides env."""
        if not self.is_available():
            raise RuntimeError(
                "NixlBackend requires the `nixl` Python binding. "
                "Install per NIXL docs and ensure the desired plugin "
                "is on LD_LIBRARY_PATH."
            )
        from nixl._api import nixl_agent, nixl_agent_config

        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.plugin = plugin
        self.agent_name = agent_name or f"gms-kvr-nixl-{uuid.uuid4().hex[:8]}"
        self._slots: dict[tuple[str, int, int], _NixlSlot] = {}
        self._lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._sync_manifest = bool(sync_manifest)

        # OBJ-specific config: bucket. Resolve before create_backend
        # so the error message points at the right knob.
        init_params = dict(backend_init_params or {})
        if plugin == "OBJ":
            if bucket is not None:
                init_params["bucket"] = bucket
            if "bucket" not in init_params:
                env_bucket = os.environ.get("AWS_DEFAULT_BUCKET")
                if not env_bucket:
                    raise RuntimeError(
                        "NixlBackend(plugin='OBJ') requires a bucket. "
                        "Pass `bucket=` kwarg, set "
                        "`backend_init_params={'bucket': '...'}`, or "
                        "set AWS_DEFAULT_BUCKET in the environment."
                    )
                init_params["bucket"] = env_bucket
            if manifest_path is None:
                raise RuntimeError(
                    "NixlBackend(plugin='OBJ') requires `manifest_path` "
                    "because NIXL doesn't expose list-objects — the "
                    "manifest is the only restart-replay source."
                )

        # Construct the agent and instantiate the plugin. Constructing
        # the agent always brings up UCX (used for control); we add
        # POSIX (or whatever) for the data plane.
        self._agent = nixl_agent(
            self.agent_name,
            nixl_agent_config(backends=[]),
        )
        plugins = self._agent.get_plugin_list()
        if plugin not in plugins:
            raise RuntimeError(
                f"NIXL plugin {plugin!r} not in available plugins " f"{plugins!r}"
            )
        self._agent.create_backend(plugin, init_params)

        # Sidecar manifest (OBJ requires it; POSIX may also enable it
        # for cross-backend interop or extra-fast reconcile).
        self.manifest_path = manifest_path
        self._manifest_fp = None
        if manifest_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
            self._replay_manifest()
            self._manifest_fp = open(manifest_path, "ab", buffering=0)

        # File-based plugins (POSIX, GDS, GDS_MT) additionally walk
        # the filesystem. (If manifest was also enabled, FS walk runs
        # second and merges — manifest entries that point at missing
        # files get dropped here.)
        if plugin in _FILE_PLUGINS:
            self._reconcile_from_disk()
        logger.info(
            "NixlBackend ready: plugin=%s base_dir=%s manifest=%s " "indexed=%d",
            plugin,
            self.base_dir,
            manifest_path or "<none>",
            len(self._slots),
        )

    # ----- path / reconcile helpers (same shape as Local) -----

    def _path_for(self, engine_id: str, layer: int, offset: int) -> str:
        safe = quote(str(engine_id), safe="")
        return os.path.join(
            self.base_dir,
            safe,
            str(int(layer)),
            f"{int(offset)}.bin",
        )

    def _reconcile_from_disk(self) -> None:
        """Walk base_dir and rebuild `_slots` by reading each file's
        24-byte header. Cleans up torn `.tmp-*` files. Same logic +
        file format as LocalStorageBackend, so a NIXL backend can
        adopt files written by Local and vice versa."""
        n_indexed = n_torn = n_bad = 0
        seen: set[tuple[str, int, int]] = set()
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
                rel = os.path.relpath(full, self.base_dir)
                parts = rel.split(os.sep)
                if len(parts) != 3:
                    continue
                try:
                    eid = unquote(parts[0])
                    layer = int(parts[1])
                    offset = int(parts[2][: -len(".bin")])
                except ValueError:
                    continue
                slot = self._read_header(full)
                if slot is None:
                    n_bad += 1
                    continue
                key = (eid, layer, offset)
                self._slots[key] = slot
                seen.add(key)
                n_indexed += 1
        n_missing = 0
        if self.manifest_path is not None:
            for key, slot in list(self._slots.items()):
                if key in seen and os.path.exists(slot.path):
                    continue
                self._slots.pop(key, None)
                n_missing += 1
        if n_indexed or n_torn or n_bad or n_missing:
            logger.info(
                "NixlBackend reconcile: indexed=%d torn=%d bad=%d "
                "manifest_missing=%d",
                n_indexed,
                n_torn,
                n_bad,
                n_missing,
            )

    def _read_header(self, path: str) -> Optional[_NixlSlot]:
        try:
            with open(path, "rb") as f:
                hdr = f.read(_HEADER_SIZE)
                if len(hdr) != _HEADER_SIZE:
                    return None
                magic, version, crc, size, _pad = struct.unpack(
                    _HEADER_FMT,
                    hdr,
                )
                if magic != _MAGIC or version != _VERSION:
                    return None
                f.seek(0, 2)
                if f.tell() < _HEADER_SIZE + size:
                    logger.warning(
                        "NixlBackend reconcile: truncated %s",
                        path,
                    )
                    return None
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = time.time()
            return _NixlSlot(
                size=int(size),
                crc=int(crc) & 0xFFFFFFFF,
                mtime=mtime,
                path=path,
            )
        except OSError as exc:
            logger.warning("NixlBackend reconcile: %s: %s", path, exc)
            return None

    # ----- sidecar manifest (used by OBJ, optional for POSIX) -----

    def _replay_manifest(self) -> None:
        """Read the JSONL manifest. Same shape as MooncakeBackend's:
        one record per slot, keyed (eid, layer, offset). For OBJ we
        can't double-check existence via NIXL (no list/head op), so
        we accept every well-formed record. POSIX additionally runs
        `_reconcile_from_disk` which drops slots whose files are
        missing — that's the existence check."""
        if not os.path.exists(self.manifest_path):
            return
        candidates: dict[tuple[str, int, int], dict] = {}
        n_records = n_bad = 0
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
                        _ = rec["path"]
                        _ = rec["size"]
                        _ = rec["crc"]
                        _ = rec["mtime"]
                        candidates[key] = rec
                    except (ValueError, KeyError, TypeError):
                        n_bad += 1
        except OSError as exc:
            logger.warning(
                "NixlBackend: manifest read failed: %s",
                exc,
            )
            return
        for tkey, rec in candidates.items():
            self._slots[tkey] = _NixlSlot(
                size=int(rec["size"]),
                crc=int(rec["crc"]) & 0xFFFFFFFF,
                mtime=float(rec["mtime"]),
                path=str(rec["path"]),
            )
        logger.info(
            "NixlBackend manifest replay: records=%d bad=%d " "indexed=%d",
            n_records,
            n_bad,
            len(candidates),
        )

    def _write_manifest_record(self, rec: dict) -> None:
        if self._manifest_fp is None:
            return
        with self._manifest_lock:
            self._manifest_fp.write(
                (json.dumps(rec) + "\n").encode("utf-8"),
            )
            if self._sync_manifest:
                os.fsync(self._manifest_fp.fileno())

    def _append_manifest(
        self,
        slot: _NixlSlot,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> None:
        """Append one durable live-slot record."""
        rec = {
            "eid": str(engine_id),
            "layer": int(layer),
            "offset": int(offset),
            "path": slot.path,
            "size": int(slot.size),
            "crc": int(slot.crc),
            "mtime": float(slot.mtime),
        }
        self._write_manifest_record(rec)

    def _append_tombstone(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> None:
        """Append a deletion record so manifest replay does not
        resurrect released slots. Required for OBJ, which has no delete
        API in NIXL, and useful for file-plugin manifests too."""
        self._write_manifest_record(
            {
                "eid": str(engine_id),
                "layer": int(layer),
                "offset": int(offset),
                "deleted": True,
                "mtime": time.time(),
            }
        )

    def manifest_size_bytes(self) -> int:
        if self.manifest_path is None:
            return 0
        try:
            return os.path.getsize(self.manifest_path)
        except OSError:
            return 0

    def compact_manifest(self) -> int:
        """Atomic rewrite-then-rename. Same shape as Mooncake's."""
        if self.manifest_path is None or self._manifest_fp is None:
            return 0
        with self._lock, self._manifest_lock:
            live = list(self._slots.items())
            old_size = self.manifest_size_bytes()
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
                    for (eid, layer, offset), slot in live:
                        rec = {
                            "eid": eid,
                            "layer": int(layer),
                            "offset": int(offset),
                            "path": slot.path,
                            "size": int(slot.size),
                            "crc": int(slot.crc),
                            "mtime": float(slot.mtime),
                        }
                        out.write(
                            (json.dumps(rec) + "\n").encode("utf-8"),
                        )
                    out.flush()
                    os.fsync(out.fileno())
                try:
                    self._manifest_fp.close()
                except OSError:
                    pass
                os.rename(tmp_path, self.manifest_path)
                tmp_path = None
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
        logger.info(
            "NixlBackend manifest compaction: %d -> %d bytes "
            "(saved %d), live records=%d",
            old_size,
            new_size,
            savings,
            len(live),
        )
        return savings

    # ----- internal: one NIXL transfer (POSIX or OBJ) -----

    def _do_nixl_xfer_posix(
        self,
        direction: str,
        host_ptr: int,
        payload_size: int,
        fd: int,
        file_offset: int,
    ) -> None:
        """POSIX-plugin transfer: source/dest is an open file fd."""
        self._do_nixl_xfer(
            direction,
            host_descs_3tup=[(int(host_ptr), int(payload_size), 0)],
            storage_descs_3tup=[(int(file_offset), int(payload_size), int(fd))],
            storage_reg_4tup=[
                (
                    int(file_offset),
                    int(payload_size),
                    int(fd),
                    "",
                )
            ],
            storage_mem_type="FILE",
            host_reg_4tup=[(int(host_ptr), int(payload_size), 0, "")],
        )

    def _do_nixl_xfer_gds(
        self,
        direction: str,
        gpu_ptr: int,
        payload_size: int,
        fd: int,
        file_offset: int,
    ) -> None:
        """GDS-plugin transfer with a VRAM source. Register source as
        VRAM_SEG so the GDS plugin sees a GPU pointer; destination
        stays a file fd. On a GDS-capable filesystem this becomes a
        true GPUDirect Storage DMA bypassing CPU. On non-GDS mounts
        the transfer succeeds but routes through host RAM (cuFile's
        compat-mode fallback)."""
        self._do_nixl_xfer(
            direction,
            host_descs_3tup=[(int(gpu_ptr), int(payload_size), 0)],
            storage_descs_3tup=[(int(file_offset), int(payload_size), int(fd))],
            storage_reg_4tup=[
                (
                    int(file_offset),
                    int(payload_size),
                    int(fd),
                    "",
                )
            ],
            storage_mem_type="FILE",
            host_reg_4tup=[(int(gpu_ptr), int(payload_size), 0, "")],
            host_mem_type="VRAM",
        )

    def _do_nixl_xfer_obj(
        self,
        direction: str,
        host_ptr: int,
        payload_size: int,
        obj_key: str,
    ) -> None:
        """OBJ-plugin transfer: source/dest is an S3-compatible object
        addressed by string key. NIXL tuple shape differs from POSIX:
          register tuple: (0, 0, key_string, "") — key in slot 2
          xfer descriptor: (0, size, key_string)
        Per the NIXL Python binding's convention (mirrored from
        SGLang's hicache_nixl)."""
        self._do_nixl_xfer(
            direction,
            host_descs_3tup=[(int(host_ptr), int(payload_size), 0)],
            storage_descs_3tup=[(0, int(payload_size), obj_key)],
            storage_reg_4tup=[(0, 0, obj_key, "")],
            storage_mem_type="OBJ",
            host_reg_4tup=[(int(host_ptr), int(payload_size), 0, "")],
        )

    def _do_nixl_xfer(
        self,
        direction: str,
        host_descs_3tup: list,
        storage_descs_3tup: list,
        storage_reg_4tup: list,
        storage_mem_type: str,
        host_reg_4tup: list,
        host_mem_type: str = "DRAM",
    ) -> None:
        """Generic NIXL transfer. Tuple shapes are plugin-specific
        (see _do_nixl_xfer_posix / _do_nixl_xfer_obj /
        _do_nixl_xfer_gds). `host_mem_type` selects DRAM_SEG (CPU
        pinned host) or VRAM_SEG (GPU pinned). Blocks until DONE;
        raises on ERR / timeout."""
        host_reg = None
        storage_reg = None
        xfer = None
        try:
            host_reg = self._agent.register_memory(
                self._agent.get_reg_descs(host_reg_4tup, host_mem_type),
            )
            storage_reg = self._agent.register_memory(
                self._agent.get_reg_descs(
                    storage_reg_4tup,
                    storage_mem_type,
                ),
            )
            host_descs = self._agent.get_xfer_descs(
                host_descs_3tup,
                host_mem_type,
            )
            storage_descs = self._agent.get_xfer_descs(
                storage_descs_3tup,
                storage_mem_type,
            )
            xfer = self._agent.initialize_xfer(
                direction,
                host_descs,
                storage_descs,
                self.agent_name,
            )
            state = self._agent.transfer(xfer)
            deadline = time.monotonic() + 60.0
            while state not in ("DONE", "ERR"):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"NIXL {direction} timed out after 60s")
                state = self._agent.check_xfer_state(xfer)
                if state not in ("DONE", "ERR"):
                    time.sleep(_POLL_S)
            if state == "ERR":
                raise RuntimeError(f"NIXL {direction} returned ERR state")
        finally:
            if xfer is not None:
                try:
                    self._agent.release_xfer_handle(xfer)
                except Exception:
                    pass  # noqa: BLE001
            if host_reg is not None:
                try:
                    self._agent.deregister_memory(host_reg)
                except Exception:
                    pass  # noqa: BLE001
            if storage_reg is not None:
                try:
                    self._agent.deregister_memory(storage_reg)
                except Exception:
                    pass  # noqa: BLE001

    # ----- data plane -----

    def _obj_key_for(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> str:
        """Object key for OBJ mode. Hierarchical for grouping;
        bucket prefix is set in the agent's plugin config."""
        return f"gms/{engine_id}/{int(layer)}/{int(offset)}"

    def demote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        host_ptr: int,
        size: int,
        crc: int,
    ) -> _NixlSlot:
        """Write pinned-host bytes via the selected NIXL plugin.

        POSIX: atomic-rename file write with 24-byte header pwritten
        at offset 0, payload via NIXL.
        OBJ: stage [header | payload] in a contiguous buffer, push
        the whole blob as one object via NIXL. (S3 has no analog
        to fsync-then-rename; the object becomes visible only on
        success.)"""
        key = (str(engine_id), int(layer), int(offset))
        if self.plugin in _FILE_PLUGINS:
            final_path = self._path_for(engine_id, layer, offset)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp-",
                suffix=".bin",
                dir=os.path.dirname(final_path),
            )
            try:
                header = struct.pack(
                    _HEADER_FMT,
                    _MAGIC,
                    _VERSION,
                    int(crc) & 0xFFFFFFFF,
                    int(size),
                    0,
                )
                os.pwrite(fd, header, 0)
                os.ftruncate(fd, _HEADER_SIZE + int(size))
                self._do_nixl_xfer_posix(
                    "WRITE",
                    host_ptr=host_ptr,
                    payload_size=int(size),
                    fd=fd,
                    file_offset=_HEADER_SIZE,
                )
                os.fsync(fd)
                os.close(fd)
                fd = -1
                os.rename(tmp_path, final_path)
                tmp_path = None
            finally:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            slot = _NixlSlot(
                size=int(size),
                crc=int(crc) & 0xFFFFFFFF,
                mtime=time.time(),
                path=final_path,
            )
        elif self.plugin == "OBJ":
            import ctypes

            obj_key = self._obj_key_for(engine_id, layer, offset)
            total_size = _HEADER_SIZE + int(size)
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
            self._do_nixl_xfer_obj(
                "WRITE",
                host_ptr=ctypes.addressof(staging),
                payload_size=total_size,
                obj_key=obj_key,
            )
            slot = _NixlSlot(
                size=int(size),
                crc=int(crc) & 0xFFFFFFFF,
                mtime=time.time(),
                path=obj_key,
            )
        else:
            raise NotImplementedError(
                f"NixlBackend.demote: plugin {self.plugin!r} "
                "data-plane not implemented in this commit. "
                "POSIX and OBJ are supported."
            )
        with self._lock:
            self._slots[key] = slot
        self._append_manifest(slot, engine_id, layer, offset)
        return slot

    def promote(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        dest_host_ptr: int,
        max_size: int,
    ) -> Optional[int]:
        """Read back via the selected NIXL plugin, verify CRC.

        POSIX: pread+validate the 24-byte header, NIXL READ for
        payload into the engine's dest, re-CRC at dest.
        OBJ: NIXL READ the whole [header | payload] blob into a
        staging buffer, validate header, memmove payload to dest,
        re-CRC at dest."""
        from gms_kv_ring.common.checksum import crc32_at_ptr

        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
        if slot is None:
            return None
        if slot.size > int(max_size):
            logger.warning(
                "NixlBackend promote: slot.size=%d > max_size=%d",
                slot.size,
                max_size,
            )
            return None

        if self.plugin in _FILE_PLUGINS:
            try:
                fd = os.open(slot.path, os.O_RDONLY)
            except OSError as exc:
                logger.warning(
                    "NixlBackend promote: open(%s) failed: %s",
                    slot.path,
                    exc,
                )
                return None
            try:
                hdr = os.pread(fd, _HEADER_SIZE, 0)
                if len(hdr) != _HEADER_SIZE:
                    return None
                magic, version, file_crc, file_size, _pad = struct.unpack(
                    _HEADER_FMT, hdr
                )
                if (
                    magic != _MAGIC
                    or version != _VERSION
                    or file_size != slot.size
                    or file_crc != slot.crc
                ):
                    logger.warning(
                        "NixlBackend promote: header/index mismatch " "for %r",
                        key,
                    )
                    return None
                try:
                    self._do_nixl_xfer_posix(
                        "READ",
                        host_ptr=dest_host_ptr,
                        payload_size=int(file_size),
                        fd=fd,
                        file_offset=_HEADER_SIZE,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NixlBackend promote: NIXL READ failed: %s",
                        exc,
                    )
                    return None
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
        elif self.plugin == "OBJ":
            import ctypes

            total_size = _HEADER_SIZE + int(slot.size)
            staging = (ctypes.c_ubyte * total_size)()
            try:
                self._do_nixl_xfer_obj(
                    "READ",
                    host_ptr=ctypes.addressof(staging),
                    payload_size=total_size,
                    obj_key=slot.path,  # OBJ key stored in slot.path
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NixlBackend promote (OBJ): READ failed: %s",
                    exc,
                )
                return None
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
                    "NixlBackend promote (OBJ): header/index " "mismatch for %r",
                    key,
                )
                return None
            ctypes.memmove(
                int(dest_host_ptr),
                ctypes.addressof(staging) + _HEADER_SIZE,
                int(slot.size),
            )
        else:
            raise NotImplementedError(
                f"NixlBackend.promote: plugin {self.plugin!r} " "not implemented"
            )

        actual = crc32_at_ptr(int(dest_host_ptr), int(slot.size))
        if actual != slot.crc:
            logger.warning(
                "NixlBackend promote: CRC mismatch for %r: " "stored=%#x actual=%#x",
                key,
                slot.crc,
                actual,
            )
            return None
        return slot.crc

    # ----- GPU-direct data plane (GDS only) -----

    @dataclass
    class _DeferredDemoteHandle:
        """Opaque token returned by demote_from_gpu_deferred_crc.
        Caller passes it to commit_deferred_crc once the real CRC
        is available."""

        engine_id: str
        layer: int
        offset: int
        size: int
        tmp_path: str
        final_path: str

    def demote_from_gpu_deferred_crc(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        gpu_ptr: int,
        size: int,
    ) -> "NixlBackend._DeferredDemoteHandle":
        """Like `demote_from_gpu` but with a CRC=0 placeholder and
        NO commit to the in-memory slot index. The file stays under
        its `.tmp-*` name; nothing observable changes until
        `commit_deferred_crc()` patches the real CRC and renames.

        Use this pattern when you want to overlap the GDS transfer
        with the CRC compute (e.g., on a pinned scratch via async
        D2H). Without the deferred-commit, a reader racing between
        the rename and the CRC patch would see CRC=0 in both the
        header and the slot record, and fail verification.

        Only valid for GDS / GDS_MT plugins."""
        if self.plugin not in ("GDS", "GDS_MT"):
            raise NotImplementedError(
                "demote_from_gpu_deferred_crc requires GDS/GDS_MT; "
                f"got {self.plugin!r}"
            )
        final_path = self._path_for(engine_id, layer, offset)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-",
            suffix=".bin",
            dir=os.path.dirname(final_path),
        )
        try:
            # Placeholder header (CRC=0). Real CRC patched later.
            placeholder = struct.pack(
                _HEADER_FMT,
                _MAGIC,
                _VERSION,
                0,
                int(size),
                0,
            )
            os.pwrite(fd, placeholder, 0)
            os.ftruncate(fd, _HEADER_SIZE + int(size))
            self._do_nixl_xfer_gds(
                "WRITE",
                gpu_ptr=gpu_ptr,
                payload_size=int(size),
                fd=fd,
                file_offset=_HEADER_SIZE,
            )
            # IMPORTANT: do NOT fsync, do NOT rename, do NOT add to
            # _slots. That's the commit step's job.
            os.close(fd)
            fd = -1
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
        return NixlBackend._DeferredDemoteHandle(
            engine_id=str(engine_id),
            layer=int(layer),
            offset=int(offset),
            size=int(size),
            tmp_path=tmp_path,
            final_path=final_path,
        )

    def commit_deferred_crc(
        self,
        handle: "NixlBackend._DeferredDemoteHandle",
        real_crc: int,
    ) -> _NixlSlot:
        """Patch the temp file's header with the real CRC, fsync,
        atomic rename to the final path, and add to the slot index.
        After this returns the slot is visible to promote/get."""
        real_header = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            int(real_crc) & 0xFFFFFFFF,
            int(handle.size),
            0,
        )
        # Open the temp file, patch the header, fsync, close.
        fd = os.open(handle.tmp_path, os.O_RDWR)
        try:
            os.pwrite(fd, real_header, 0)
            os.fsync(fd)
        finally:
            os.close(fd)
        # Atomic rename — file becomes visible at final_path NOW.
        os.rename(handle.tmp_path, handle.final_path)
        slot = _NixlSlot(
            size=int(handle.size),
            crc=int(real_crc) & 0xFFFFFFFF,
            mtime=time.time(),
            path=handle.final_path,
        )
        with self._lock:
            self._slots[(handle.engine_id, handle.layer, handle.offset)] = slot
        self._append_manifest(
            slot,
            handle.engine_id,
            handle.layer,
            handle.offset,
        )
        return slot

    def demote_from_gpu(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        gpu_ptr: int,
        size: int,
        crc: int,
    ) -> _NixlSlot:
        """GDS-direct demote: source is a GPU pointer (VRAM_SEG),
        destination is a file on a GDS-capable mount. The NVMe
        controller DMAs directly to GPU memory via PCIe, bypassing
        host RAM staging.

        Caller passes the pre-computed CRC over the GPU bytes (the
        engine has the data on GPU; CRC is up to the engine, possibly
        computed on GPU via a kernel). On promote_into_gpu we
        recompute the CRC directly over destination HBM.

        Only valid for GDS/GDS_MT plugins; raises for POSIX/OBJ.
        On a non-GDS filesystem the NIXL plugin still functions but
        falls back to cuFile's compat mode (CPU-staged); the data
        gets through, you just don't get the GPU-direct benefit."""
        if self.plugin not in ("GDS", "GDS_MT"):
            raise NotImplementedError(
                "demote_from_gpu requires plugin in {GDS, GDS_MT}; "
                f"got {self.plugin!r}"
            )
        final_path = self._path_for(engine_id, layer, offset)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-",
            suffix=".bin",
            dir=os.path.dirname(final_path),
        )
        try:
            header = struct.pack(
                _HEADER_FMT,
                _MAGIC,
                _VERSION,
                int(crc) & 0xFFFFFFFF,
                int(size),
                0,
            )
            # Header: a 24-byte pwrite, far cheaper than a separate
            # GDS transfer for such a small payload.
            os.pwrite(fd, header, 0)
            os.ftruncate(fd, _HEADER_SIZE + int(size))
            # Payload: VRAM -> FILE via GDS.
            self._do_nixl_xfer_gds(
                "WRITE",
                gpu_ptr=gpu_ptr,
                payload_size=int(size),
                fd=fd,
                file_offset=_HEADER_SIZE,
            )
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.rename(tmp_path, final_path)
            tmp_path = None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        slot = _NixlSlot(
            size=int(size),
            crc=int(crc) & 0xFFFFFFFF,
            mtime=time.time(),
            path=final_path,
        )
        with self._lock:
            self._slots[(str(engine_id), int(layer), int(offset))] = slot
        self._append_manifest(slot, engine_id, layer, offset)
        return slot

    def promote_into_gpu(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        dest_gpu_ptr: int,
        max_size: int,
    ) -> Optional[int]:
        """Symmetric to demote_from_gpu: file -> VRAM via GDS.
        Returns the verified CRC, or None on any failure.

        CRC verification runs directly over destination HBM. A GDS
        restore fails closed when the CUDA verifier is unavailable; copying
        the payload through host memory would violate the GPU-direct contract."""

        if self.plugin not in ("GDS", "GDS_MT"):
            raise NotImplementedError(
                "promote_into_gpu requires plugin in {GDS, GDS_MT}; "
                f"got {self.plugin!r}"
            )
        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
        if slot is None or slot.size > int(max_size):
            return None

        try:
            fd = os.open(slot.path, os.O_RDONLY)
        except OSError as exc:
            logger.warning("promote_into_gpu: open(%s): %s", slot.path, exc)
            return None
        try:
            hdr = os.pread(fd, _HEADER_SIZE, 0)
            if len(hdr) != _HEADER_SIZE:
                return None
            magic, version, file_crc, file_size, _pad = struct.unpack(
                _HEADER_FMT,
                hdr,
            )
            if (
                magic != _MAGIC
                or version != _VERSION
                or file_size != slot.size
                or file_crc != slot.crc
            ):
                return None
            try:
                self._do_nixl_xfer_gds(
                    "READ",
                    gpu_ptr=dest_gpu_ptr,
                    payload_size=int(file_size),
                    fd=fd,
                    file_offset=_HEADER_SIZE,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "promote_into_gpu: NIXL READ failed: %s",
                    exc,
                )
                return None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        try:
            from gms_kv_ring.common.crc32_gpu import crc32_gpu

            actual = crc32_gpu(int(dest_gpu_ptr), int(slot.size))
        except Exception as exc:  # noqa: BLE001
            logger.warning("promote_into_gpu: GPU CRC verify failed: %s", exc)
            return None
        if actual != slot.crc:
            logger.warning(
                "promote_into_gpu: CRC mismatch for %r",
                key,
            )
            return None
        return slot.crc

    # ----- index + lifecycle -----

    def get(self, engine_id, layer, offset) -> Optional[_NixlSlot]:
        with self._lock:
            return self._slots.get(
                (str(engine_id), int(layer), int(offset)),
            )

    def _commit_removals_locked(self, items):
        """Persist tombstones before removing the same indexed slots.

        Caller holds ``_lock``. A manifest failure leaves the in-memory
        index unchanged, preventing released OBJ entries from resurrecting.
        """
        for key, _slot in items:
            self._append_tombstone(key[0], key[1], key[2])
        removed = []
        for key, slot in items:
            if self._slots.get(key) is slot:
                self._slots.pop(key)
                removed.append((key, slot))
        return removed

    def release_if_current(self, engine_id, layer, offset, expected) -> bool:
        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            if self._slots.get(key) is not expected:
                return False
            removed = self._commit_removals_locked([(key, expected)])
        self._free_file(removed[0][1])
        return True

    def release_slot(self, engine_id, layer, offset) -> bool:
        key = (str(engine_id), int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return False
            removed = self._commit_removals_locked([(key, slot)])
        self._free_file(removed[0][1])
        return True

    def release_engine(self, engine_id) -> int:
        eid = str(engine_id)
        with self._lock:
            items = [(key, slot) for key, slot in self._slots.items() if key[0] == eid]
            removed = self._commit_removals_locked(items)
        for _key, slot in removed:
            self._free_file(slot)
        return len(removed)

    def _free_file(self, slot: _NixlSlot) -> None:
        """Free the slot's backend resource. POSIX: unlink the file.
        OBJ: delete the object — NIXL doesn't have a delete op in
        its Python API, so we log + leak (operator's bucket-side
        lifecycle policy is expected to GC eventually)."""
        if self.plugin in _FILE_PLUGINS:
            try:
                os.unlink(slot.path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "NixlBackend._free_file: unlink %s failed: %s",
                    slot.path,
                    exc,
                )
        elif self.plugin == "OBJ":
            # No NIXL delete API; rely on bucket lifecycle / TTL
            # policy. We've already dropped the slot from the
            # in-memory index, so it's invisible to promote — the
            # leak is bytes on S3, not a correctness issue.
            logger.debug(
                "NixlBackend._free_file (OBJ): %s removed from "
                "index; bytes remain in bucket until lifecycle GC",
                slot.path,
            )

    def close(self) -> None:
        """Close manifest resources. Idempotent."""
        with self._manifest_lock:
            if self._manifest_fp is not None:
                try:
                    if self._sync_manifest:
                        os.fsync(self._manifest_fp.fileno())
                    self._manifest_fp.close()
                except OSError:
                    pass
                self._manifest_fp = None

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

    # ----- cleanup (same logic shape as Local) -----

    def prune_older_than(self, max_age_seconds, now=None) -> int:
        if now is None:
            now = time.time()
        threshold = float(now) - float(max_age_seconds)
        with self._lock:
            to_remove = [
                (k, s) for k, s in list(self._slots.items()) if s.mtime < threshold
            ]
            removed = self._commit_removals_locked(to_remove)
        for _key, slot in removed:
            self._free_file(slot)
        return len(removed)

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
            removed = self._commit_removals_locked(to_remove)
        for _key, slot in removed:
            self._free_file(slot)
        return len(removed)

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
            removed = self._commit_removals_locked(to_remove)
        for _key, slot in removed:
            self._free_file(slot)
        return len(removed)
