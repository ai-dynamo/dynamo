# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-engine `_PrefixIndex` + snapshot persistence.

The connector-side hash → (engine_id, block_id, generation) mapping
that powers cache-hit detection for vLLM, SGLang (indirectly via the
radix tree), and TRT-LLM. Lives in `common/` so all three connectors
share one implementation and snapshot format.

Invariants the class enforces (referenced by the cross-engine
parity work):
  - I-1 (prune-on-record): older hashes pointing at a slot are
    deleted before the new mapping is inserted.
  - I-3 (generation tagging): every slot mapping carries a
    monotonically increasing generation; daemon-side enforces
    match on restore (closes Race #3).
  - LRU bound: oldest-record-first eviction at `max_entries`.
  - Persistence: optional snapshot/load tied to daemon_epoch.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import time
import zlib
from collections import OrderedDict
from typing import Optional

from gms_kv_ring.common.prefix_hashes import prefix_block_hashes as _prefix_block_hashes

logger = logging.getLogger(__name__)


# Snapshot file format. Layout:
#   magic   (8 bytes)  : b"GMSPIDX\x00"
#   version (2 bytes)  : uint16 LE
#   pkl_len (4 bytes)  : uint32 LE (length of the pickle payload)
#   pkl     (pkl_len)  : pickle.dumps(payload_dict, protocol=4)
#   crc32   (4 bytes)  : zlib.crc32 over the pickle bytes
_SNAPSHOT_MAGIC = b"GMSPIDX\x00"
_SNAPSHOT_VERSION = 2
_SNAPSHOT_HEADER_FMT = "<8sHI"  # 8s + uint16 + uint32 = 14 bytes
_SNAPSHOT_HEADER_LEN = struct.calcsize(_SNAPSHOT_HEADER_FMT)


def _serialize_snapshot(
    max_entries: int,
    table: "OrderedDict[bytes, tuple[str, int, int]]",
    slot_to_hashes: "dict[tuple[str, int], set[bytes]]",
    slot_generations: "dict[tuple[str, int], int]",
    daemon_epoch: Optional[int] = None,
) -> bytes:
    payload = {
        "max": int(max_entries),
        "table": table,
        "slot_to_hashes": slot_to_hashes,
        "slot_generations": slot_generations,
        "daemon_epoch": (None if daemon_epoch is None else int(daemon_epoch)),
    }
    pkl = pickle.dumps(payload, protocol=4)
    header = struct.pack(
        _SNAPSHOT_HEADER_FMT,
        _SNAPSHOT_MAGIC,
        _SNAPSHOT_VERSION,
        len(pkl),
    )
    crc = struct.pack("<I", zlib.crc32(pkl) & 0xFFFFFFFF)
    return header + pkl + crc


def _deserialize_snapshot(blob: bytes) -> Optional[dict]:
    """Returns the payload dict, or None if the blob is malformed or
    of an unrecognized version. Never raises — corrupt files must
    not break connector init."""
    if len(blob) < _SNAPSHOT_HEADER_LEN + 4:
        return None
    try:
        magic, version, pkl_len = struct.unpack(
            _SNAPSHOT_HEADER_FMT,
            blob[:_SNAPSHOT_HEADER_LEN],
        )
    except struct.error:
        return None
    if magic != _SNAPSHOT_MAGIC:
        return None
    if version != _SNAPSHOT_VERSION:
        logger.warning(
            "PrefixIndex snapshot version %d; this build expects "
            "%d. Ignoring snapshot.",
            version,
            _SNAPSHOT_VERSION,
        )
        return None
    expected_end = _SNAPSHOT_HEADER_LEN + pkl_len + 4
    if len(blob) != expected_end:
        return None
    pkl = blob[_SNAPSHOT_HEADER_LEN : _SNAPSHOT_HEADER_LEN + pkl_len]
    (stored_crc,) = struct.unpack(
        "<I",
        blob[_SNAPSHOT_HEADER_LEN + pkl_len : expected_end],
    )
    if zlib.crc32(pkl) & 0xFFFFFFFF != stored_crc:
        return None
    try:
        payload = pickle.loads(pkl)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class PrefixIndex:
    """Scheduler-local content-hash → (engine_id, src_block_id)
    mapping. Populated at eviction time, queried at cache-hit time.

    See the module docstring for invariants.

    Each connector (vLLM, SGLang, TRT-LLM) instantiates one and feeds
    it spilled `(prompt_tokens, block_ids, block_size, cache_salt,
    engine_id)` tuples via `record()`. Cache-hit detection at request
    arrival walks `lookup()` for matching prefix hits."""

    _DEFAULT_MAX_ENTRIES = 100_000
    _DEFAULT_SNAPSHOT_THRESHOLD = 256
    _DEFAULT_SNAPSHOT_INTERVAL_S = 30.0

    def __init__(
        self,
        max_entries: Optional[int] = None,
        snapshot_path: Optional[str] = None,
        snapshot_interval_s: Optional[float] = None,
        snapshot_threshold: Optional[int] = None,
        expected_daemon_epoch: Optional[int] = None,
    ) -> None:
        if max_entries is None:
            try:
                max_entries = int(
                    os.environ.get(
                        "GMS_KVR_PREFIX_INDEX_MAX",
                        self._DEFAULT_MAX_ENTRIES,
                    ),
                )
            except ValueError:
                max_entries = self._DEFAULT_MAX_ENTRIES
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._max = int(max_entries)
        self._table: "OrderedDict[bytes, tuple[str, int, int]]" = OrderedDict()
        self._slot_to_hashes: dict[tuple[str, int], set[bytes]] = {}
        self._slot_generations: dict[tuple[str, int], int] = {}

        if snapshot_path is None:
            snapshot_path = os.environ.get(
                "GMS_KVR_PREFIX_INDEX_SNAPSHOT",
                "",
            )
        self._snap_path: str = snapshot_path or ""
        if snapshot_interval_s is None:
            try:
                snapshot_interval_s = float(
                    os.environ.get(
                        "GMS_KVR_PREFIX_INDEX_SNAPSHOT_INTERVAL",
                        self._DEFAULT_SNAPSHOT_INTERVAL_S,
                    ),
                )
            except ValueError:
                snapshot_interval_s = self._DEFAULT_SNAPSHOT_INTERVAL_S
        if snapshot_threshold is None:
            try:
                snapshot_threshold = int(
                    os.environ.get(
                        "GMS_KVR_PREFIX_INDEX_SNAPSHOT_THRESHOLD",
                        self._DEFAULT_SNAPSHOT_THRESHOLD,
                    ),
                )
            except ValueError:
                snapshot_threshold = self._DEFAULT_SNAPSHOT_THRESHOLD
        self._snap_interval_s: float = max(0.0, float(snapshot_interval_s))
        self._snap_threshold: int = max(1, int(snapshot_threshold))
        self._dirty: int = 0
        self._last_snap_t: float = time.monotonic()
        self._daemon_epoch: Optional[int] = (
            None if expected_daemon_epoch is None else int(expected_daemon_epoch)
        )

        if self._snap_path:
            self._load_snapshot()

    def set_daemon_epoch(self, epoch: Optional[int]) -> None:
        self._daemon_epoch = None if epoch is None else int(epoch)

    def __len__(self) -> int:
        return len(self._table)

    def _drop_hash(self, h: bytes) -> None:
        entry = self._table.pop(h, None)
        if entry is None:
            return
        slot_key = (entry[0], entry[1])
        slot_hashes = self._slot_to_hashes.get(slot_key)
        if slot_hashes is not None:
            slot_hashes.discard(h)
            if not slot_hashes:
                del self._slot_to_hashes[slot_key]

    def _evict_lru(self) -> None:
        try:
            oldest_hash, oldest_entry = next(iter(self._table.items()))
        except StopIteration:
            return
        del self._table[oldest_hash]
        slot_key = (oldest_entry[0], oldest_entry[1])
        slot_hashes = self._slot_to_hashes.get(slot_key)
        if slot_hashes is not None:
            slot_hashes.discard(oldest_hash)
            if not slot_hashes:
                del self._slot_to_hashes[slot_key]

    def record(
        self,
        prompt_token_ids: "list[int]",
        block_ids: "list[int]",
        block_size: int,
        cache_salt: Optional[str],
        engine_id: str,
    ) -> "list[int]":
        """Index a finished request's prompt prefix block-by-block.
        Returns per-block generations parallel to `block_ids`."""
        hashes = _prefix_block_hashes(
            prompt_token_ids,
            block_size,
            cache_salt,
        )
        n = min(len(hashes), len(block_ids))
        assigned_generations: list[int] = []
        for i in range(n):
            h = hashes[i]
            bid = int(block_ids[i])
            slot_key = (engine_id, bid)
            new_gen = self._slot_generations.get(slot_key, 0) + 1
            self._slot_generations[slot_key] = new_gen
            assigned_generations.append(new_gen)
            stale = self._slot_to_hashes.pop(slot_key, None)
            if stale is not None:
                for old in stale:
                    self._table.pop(old, None)
            self._table.pop(h, None)
            self._table[h] = (engine_id, bid, new_gen)
            self._slot_to_hashes[slot_key] = {h}
            while len(self._table) > self._max:
                self._evict_lru()
            self._dirty += 1
        self._maybe_snapshot()
        return assigned_generations

    def drop_slots(
        self,
        engine_id: str,
        block_ids: "list[int]",
    ) -> int:
        """Drop every hash currently pointing at any (engine_id,
        block_id) in `block_ids`. Used by the post-forward failure
        path."""
        n_dropped = 0
        for bid in block_ids:
            slot_key = (str(engine_id), int(bid))
            hashes = self._slot_to_hashes.pop(slot_key, None)
            if not hashes:
                continue
            for h in hashes:
                if self._table.pop(h, None) is not None:
                    n_dropped += 1
        return n_dropped

    def invalidate(self) -> None:
        """Drop every entry. Used by the daemon-epoch-change path."""
        self._table.clear()
        self._slot_to_hashes.clear()
        self._slot_generations.clear()
        self._dirty = 0
        if self._snap_path:
            try:
                self.snapshot()
            except Exception:
                logger.warning(
                    "PrefixIndex invalidate: snapshot rewrite failed; "
                    "stale on-disk view will be discarded on next "
                    "load via the epoch-mismatch guard.",
                    exc_info=True,
                )

    def _load_snapshot(self) -> None:
        path = self._snap_path
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(
                "PrefixIndex snapshot %s: read failed (%s); " "starting cold.",
                path,
                e,
            )
            return
        payload = _deserialize_snapshot(blob)
        if payload is None:
            logger.warning(
                "PrefixIndex snapshot %s: malformed; starting cold.",
                path,
            )
            return
        try:
            table = payload["table"]
            slot_to_hashes = payload["slot_to_hashes"]
            slot_generations = payload["slot_generations"]
        except (KeyError, TypeError):
            logger.warning(
                "PrefixIndex snapshot %s: schema mismatch; starting " "cold.",
                path,
            )
            return
        snap_epoch = payload.get("daemon_epoch")
        if (
            self._daemon_epoch is not None
            and snap_epoch is not None
            and int(snap_epoch) != int(self._daemon_epoch)
        ):
            logger.info(
                "PrefixIndex snapshot %s: daemon_epoch %d != "
                "current %d (daemon restarted since snapshot); "
                "starting cold.",
                path,
                int(snap_epoch),
                int(self._daemon_epoch),
            )
            return
        if not isinstance(table, OrderedDict):
            try:
                table = OrderedDict(table)
            except (TypeError, ValueError):
                logger.warning(
                    "PrefixIndex snapshot %s: table not iterable as "
                    "mapping; starting cold.",
                    path,
                )
                return
        self._table = table
        self._slot_to_hashes = dict(slot_to_hashes)
        self._slot_generations = dict(slot_generations)
        while len(self._table) > self._max:
            self._evict_lru()
        logger.info(
            "PrefixIndex loaded %d entries from snapshot %s " "(daemon_epoch=%r)",
            len(self._table),
            path,
            snap_epoch,
        )

    def snapshot(self) -> None:
        path = self._snap_path
        if not path:
            return
        blob = _serialize_snapshot(
            self._max,
            self._table,
            self._slot_to_hashes,
            self._slot_generations,
            daemon_epoch=self._daemon_epoch,
        )
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            parent = os.path.dirname(path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            try:
                os.write(fd, blob)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(tmp_path, path)
        except OSError as e:
            logger.warning(
                "PrefixIndex snapshot %s: write failed (%s); "
                "in-memory index unaffected.",
                path,
                e,
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return
        self._dirty = 0
        self._last_snap_t = time.monotonic()

    def _maybe_snapshot(self) -> None:
        if not self._snap_path:
            return
        if self._dirty < self._snap_threshold:
            return
        if time.monotonic() - self._last_snap_t < self._snap_interval_s:
            return
        self.snapshot()

    def lookup(
        self,
        prompt_token_ids: "list[int]",
        block_size: int,
        cache_salt: Optional[str],
        engine_id: str,
    ) -> "list[tuple[int, int]]":
        """Return the longest contiguous prefix's (src_block_id,
        expected_generation) pairs for this engine."""
        hashes = _prefix_block_hashes(
            prompt_token_ids,
            block_size,
            cache_salt,
        )
        out: list[tuple[int, int]] = []
        for h in hashes:
            entry = self._table.get(h)
            if entry is None or entry[0] != engine_id:
                break
            out.append((entry[1], entry[2]))
        return out
