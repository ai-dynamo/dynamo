# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist + rehydrate the vLLM prefix-cache index across a shadow failover.

M1 makes the KV *bytes* survive an engine crash and be reattached by the standby.
A prefix-cache HIT also needs the *index* (``block_hash -> block_id``), which lives
in the scheduler's ``BlockPool`` process RAM and cannot be regenerated from the bytes
(the hash is over token ids). This module carries the index across the crash:

  * write-through: after ``BlockPool.cache_full_blocks`` runs, append each newly
    cached ``(block_hash_with_group_id, block_id, num_tokens)`` to a shared,
    append-only log beside the KV region.
  * rehydrate: on the first cache lookup after a failover wake, replay the log into
    the standby's ``BlockPool`` so the re-sent prefix HITs the reattached bytes.
    Because both engines share identical, pinned geometry, ``block_id`` N on the
    standby is the same reattached physical block as on the primary, so the
    ``hash -> block_id`` map transfers verbatim.

Determinism (B3): both engines must hash identically, so the launcher pins
``PYTHONHASHSEED`` -- otherwise vLLM's ``NONE_HASH`` (root of the hash chain) is
``os.urandom(32)`` per process and no hash would ever match.

POC simplifications (documented, deferred):
  * Eager write-through (not gated on the KV-write barrier) is safe for the North
    Star because the test warms the prefix and only *then* crashes, so every
    persisted block's bytes are already durable. A production build would stage
    records and publish them at the step barrier (B2).
  * Evictions/tombstones are not tracked; the pinned pool never evicts during the
    test. On replay we keep the LAST record per block_id, which reconstructs each
    block's final cached content.

Enabled only when ``GMS_KV_INDEX_PATH`` is set (opt-in; no effect otherwise).
"""

import logging
import os
import pickle
import struct
import threading

logger = logging.getLogger(__name__)

_log_path: str | None = None
_log_lock = threading.Lock()

# Set by GMSWorker.wake_up when it reattaches a prior engine's KV (a failover
# takeover). Read by the patched BlockPool lookup, which rehydrates once then clears
# it. Both run in the same EngineCore process (TP=1), so a module global suffices.
REHYDRATE_PENDING = False

# DEBUG: reference to this engine's KV tensors (set by the worker after wake) so the
# persist hook can log a per-layer signature for winner-vs-standby byte comparison.
_kv_caches = None
_kv_manager = None
_dbg_persist_calls = 0


def set_kv_caches(kv_caches) -> None:
    global _kv_caches
    _kv_caches = kv_caches


def set_kv_manager(mgr) -> None:
    global _kv_manager
    _kv_manager = mgr


def kv_fingerprint(tensor, nchunks: int = 1024):
    """POSITION-SENSITIVE fingerprint of a KV tensor. Split the flat tensor into `nchunks`
    ordered spans, abs-sum each -> a vector, and sha over the ORDERED vector. Unlike a
    whole-tensor abssum (permutation-invariant), any block permutation or offset changes
    the vector and hence the hash. Cheap (nchunks reductions) and localizable (the first
    divergent chunk points at where two allocations differ)."""
    import hashlib

    import torch

    t = tensor.detach().reshape(-1)
    n = int(t.numel())
    csz = (n + nchunks - 1) // nchunks
    chunks = []
    for c in range(nchunks):
        a = c * csz
        if a >= n:
            chunks.append(0.0)
            continue
        b = min(n, a + csz)
        chunks.append(float(t[a:b].float().abs().sum().item()))
    torch.cuda.synchronize()
    body = ",".join(f"{v:.3f}" for v in chunks)
    return {
        "vec_sha": hashlib.sha256(body.encode()).hexdigest()[:16],
        "chunks": chunks,
        "abssum": float(sum(chunks)),
        "numel": n,
        "nchunks": nchunks,
    }


def _locate_layer(kv_caches, mappings, want_alloc=None):
    """Return (layer_index, tensor, off, alloc) for the tensor on `want_alloc` (or L0's
    containing allocation if want_alloc is None), found by data_ptr containment."""
    maps = list(mappings.items())

    def alloc_of(dp):
        for _va, _m in maps:
            _span = max(getattr(_m, "va_reserved_size", 0), _m.aligned_size)
            if _va <= dp < _va + _span:
                return str(_m.allocation_id), (dp - _va)
        return None, -1

    if want_alloc is not None:
        for li, t in enumerate(kv_caches or []):
            a, off = alloc_of(t.data_ptr())
            if a == want_alloc:
                return li, t, off, a
        return None, None, None, None
    if kv_caches:
        a, off = alloc_of(kv_caches[0].data_ptr())
        return 0, kv_caches[0], off, a
    return None, None, None, None


def _fresh_import_probe(maps, kv_caches):
    """WINNER side, at a STABLE point (all writes done, nothing unmapped): compute a
    POSITION-SENSITIVE fingerprint of L0's KV and dump it (+ allocation identity) so the
    shadow -- after it reattaches the SAME allocation -- can compare byte-for-byte,
    including position. Order matters: read the model's OWN tensor (always safe) and dump
    FIRST, then a best-effort raw re-import cross-check (which can fault on some layouts
    and must never gate the dump)."""
    import json as _json

    import torch
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
    from gpu_memory_service.common.locks import GrantedLockType

    mdict = dict(maps)
    torch.cuda.synchronize()

    # Fingerprint EVERY layer (position-sensitive) + record its allocation, so the shadow
    # can prove the WHOLE reattach is byte-perfect, not just L0 -- and catch a per-layer
    # cross-wire (permutation-invariant abssum would miss it).
    layers = []
    for li, t in enumerate(kv_caches or []):
        _li2, _t, off, alloc = _locate_layer([t], mdict)
        lf = kv_fingerprint(t)
        layers.append(
            {
                "layer": li,
                "alloc": alloc or "UNKNOWN",
                "off": off,
                "vec_sha": lf["vec_sha"],
                "abssum": lf["abssum"],
                "numel": lf["numel"],
            }
        )
    l0 = (
        layers[0]
        if layers
        else {"alloc": "UNKNOWN", "off": -1, "vec_sha": "", "abssum": 0.0}
    )
    l0_chunks = kv_fingerprint(kv_caches[0])["chunks"] if kv_caches else []
    logger.info(
        "[GMS][dbg] STABLE KV: %d layers | L0 alloc=%s off=%s abssum=%.1f vec_sha=%s | "
        "per-layer-vecsha-hash=%s",
        len(layers),
        str(l0["alloc"])[:8],
        l0["off"],
        l0["abssum"],
        l0["vec_sha"],
        __import__("hashlib")
        .sha256(",".join(x["vec_sha"] for x in layers).encode())
        .hexdigest()[:12],
    )

    tf = os.getenv("GMS_KV_TARGET_FILE")
    if tf and layers and l0["off"] == 0:
        _json.dump(
            {
                "dtype": str(kv_caches[0].dtype),
                "nchunks": 1024,
                "layers": layers,
                "l0_chunks": l0_chunks,
            },
            open(tf, "w"),
        )
        logger.info(
            "[GMS][dbg] dumped winner KV target (%d layers, L0 vec_sha=%s) to %s",
            len(layers),
            l0["vec_sha"],
            tf,
        )
    fp = {"abssum": l0["abssum"]}  # for the raw cross-check below

    # Best-effort raw cross-check: re-import the same handle at a FRESH VA and read it.
    # Same value => the model's VA really is on the server allocation. May fault on some
    # layouts; wrapped so a failure never loses the fingerprint above.
    try:
        _l0 = None
        for _va, _m in maps:
            _span = max(getattr(_m, "va_reserved_size", 0), _m.aligned_size)
            if _va <= kv_caches[0].data_ptr() < _va + _span:
                _l0 = (_va, _m, kv_caches[0].data_ptr() - _va)
                break
        if _l0 is not None and _l0[2] == 0:
            _va, m0, _o = _l0
            asz = m0.aligned_size
            fd = _kv_manager.export_handle(m0.allocation_id)
            h = _kv_manager._vmm.import_shareable_handle_close_fd(fd)
            fva = _kv_manager._vmm.address_reserve(asz, _kv_manager.granularity)
            _kv_manager._vmm.map(fva, asz, h)
            _kv_manager._vmm.set_access(
                fva, asz, _kv_manager.device, GrantedLockType.RW
            )
            _kv_manager._vmm.synchronize()
            ft = _tensor_from_pointer(
                fva, [kv_caches[0].numel()], [1], kv_caches[0].dtype, 0
            )
            fresh = float(ft.float().abs().sum().item())
            logger.info(
                "[GMS][dbg] WINNER fresh-import L0=%s fresh_read=%.1f model_abssum=%.1f match=%s",
                str(alloc)[:8],
                fresh,
                fp["abssum"],
                abs(fresh - fp["abssum"]) < max(1.0, fp["abssum"] * 1e-6),
            )
    except Exception as _fe:
        logger.warning("[GMS][dbg] fresh-import cross-check skipped: %s", _fe)


def compare_to_winner_target(kv_caches, mappings):
    """SHADOW side, at reattach (post-remap, PRE-forward): read the layer that sits on the
    winner-dumped allocation and compare a position-sensitive fingerprint.

      MATCH    => the shadow's reattached VA holds the winner's exact bytes at the exact
                  positions. Reattach is byte-perfect; any garbage output is DOWNSTREAM
                  (prefix-index rehydrate or the continuation forward pass).
      MISMATCH => the reattach binds different physical / a positional offset -- and the
                  first divergent chunk localizes where.
    """
    import json as _json

    tf = os.getenv("GMS_KV_TARGET_FILE")
    if not tf or not os.path.exists(tf):
        logger.info(
            "[GMS][dbg] no winner target (%s); skipping reattach byte-compare", tf
        )
        return None
    tgt = _json.load(open(tf))
    wlayers = tgt.get("layers") or []
    nchunks = int(tgt.get("nchunks", 1024))

    matched = 0
    alloc_missing = 0
    diverged = []  # (layer, first_divergent_chunk)
    for w in wlayers:
        want = w["alloc"]
        li, t, off, alloc = _locate_layer(kv_caches, mappings, want_alloc=want)
        if t is None:
            alloc_missing += 1
            diverged.append((w["layer"], "ALLOC-NOT-BOUND"))
            continue
        fp = kv_fingerprint(t, nchunks=nchunks)
        if fp["vec_sha"] == w["vec_sha"]:
            matched += 1
        else:
            # localize using L0's full chunk vector when this is L0
            fd = "?"
            if w["layer"] == 0 and tgt.get("l0_chunks"):
                wc = tgt["l0_chunks"]
                for i in range(min(len(wc), len(fp["chunks"]))):
                    if abs(wc[i] - fp["chunks"][i]) > max(1.0, abs(wc[i]) * 1e-4):
                        fd = i
                        break
            diverged.append((w["layer"], fd))

    all_match = matched == len(wlayers) and wlayers
    logger.info(
        "[GMS][dbg] REATTACH BYTE-COMPARE (ALL LAYERS): %d/%d layers byte-identical "
        "(alloc-not-bound=%d) | ALL_MATCH=%s | diverged=%s",
        matched,
        len(wlayers),
        alloc_missing,
        all_match,
        diverged[:8],
    )
    return all_match


def _path() -> str | None:
    global _log_path
    if _log_path is None:
        _log_path = os.getenv("GMS_KV_INDEX_PATH")
    return _log_path


def is_enabled() -> bool:
    return _path() is not None


def mark_rehydrate_pending() -> None:
    """Called by the standby's wake path once its KV bytes are reattached."""
    global REHYDRATE_PENDING
    if is_enabled():
        REHYDRATE_PENDING = True
        logger.info("[GMS] KV index rehydrate armed (failover takeover)")


def _append_records(records: list) -> None:
    if not records:
        return
    buf = bytearray()
    for rec in records:
        payload = pickle.dumps(rec, protocol=4)
        buf += struct.pack("<I", len(payload)) + payload
    with _log_lock:
        with open(_path(), "ab") as f:
            f.write(buf)


def enable_kv_index_persistence() -> None:
    """Monkey-patch ``BlockPool`` for write-through persist + lazy rehydrate.

    No-op unless ``GMS_KV_INDEX_PATH`` is set. Patches the class (not an instance),
    so it applies to whatever ``BlockPool`` the scheduler builds.
    """
    if not is_enabled():
        return

    from vllm.v1.core.block_pool import BlockPool

    if getattr(BlockPool, "_gms_kv_index_patched", False):
        return

    orig_cache_full_blocks = BlockPool.cache_full_blocks
    orig_get_cached_block = BlockPool.get_cached_block

    def cache_full_blocks(
        self, request, blocks, num_cached_blocks, num_full_blocks, *args, **kwargs
    ):
        orig_cache_full_blocks(
            self, request, blocks, num_cached_blocks, num_full_blocks, *args, **kwargs
        )
        try:
            records = []
            for blk in blocks[num_cached_blocks:num_full_blocks]:
                bh = blk.block_hash
                if bh is not None:
                    records.append((bytes(bh), blk.block_id, blk.block_hash_num_tokens))
            _append_records(records)
        except Exception as e:  # never let persistence break serving
            logger.warning("[GMS] KV index persist failed: %s", e)
        if os.getenv("GMS_KV_DEBUG") and _kv_caches:
            global _dbg_persist_calls
            _dbg_persist_calls += 1
            try:
                first3 = [
                    round(float(_kv_caches[i].float().abs().sum().item()), 1)
                    for i in range(min(3, len(_kv_caches)))
                ]
                logger.info(
                    "[GMS][dbg] winner KV per-layer abssum first3=%s (persist call %d)",
                    first3,
                    _dbg_persist_calls,
                )
                # WINNER-D1: is the model's KV tensor (the one attention wrote these
                # bytes into) actually sitting on the GMS mapping VA, or did it escape
                # the GMS scratch pool to plain torch memory?
                if _kv_manager is not None and _dbg_persist_calls <= 2:
                    _maps = list(_kv_manager.mappings.items())
                    for _i in range(min(3, len(_kv_caches))):
                        _dp = _kv_caches[_i].data_ptr()
                        _match = "OFF-GMS (escaped!)"
                        for _va, _m in _maps:
                            _span = max(
                                getattr(_m, "va_reserved_size", 0), _m.aligned_size
                            )
                            if _va <= _dp < _va + _span:
                                _match = "GMS va=0x%x off=%d alloc=%s" % (
                                    _va,
                                    _dp - _va,
                                    str(_m.allocation_id)[:8],
                                )
                                break
                        logger.info(
                            "[GMS][dbg] WINNER-D1 L%d data_ptr=0x%x -> %s",
                            _i,
                            _dp,
                            _match,
                        )
                    if _dbg_persist_calls == 2:
                        try:
                            _fresh_import_probe(_maps, _kv_caches)
                        except Exception as _fe:
                            logger.warning(
                                "[GMS][dbg] fresh-import probe failed: %s", _fe
                            )
            except Exception as _e:
                logger.warning("[GMS][dbg] winner D1 failed: %s", _e)

    def get_cached_block(self, *args, **kwargs):
        global REHYDRATE_PENDING
        if REHYDRATE_PENDING:
            REHYDRATE_PENDING = False
            try:
                rehydrate_block_pool(self)
            except Exception as e:
                logger.warning("[GMS] KV index rehydrate failed: %s", e)
        return orig_get_cached_block(self, *args, **kwargs)

    BlockPool.cache_full_blocks = cache_full_blocks
    BlockPool.get_cached_block = get_cached_block
    BlockPool._gms_kv_index_patched = True
    logger.info("[GMS] KV index persistence enabled (log=%s)", _path())


def rehydrate_block_pool(block_pool) -> None:
    """Replay the persisted index into ``block_pool`` (standby, post-reattach)."""
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    path = _path()
    if not path or not os.path.exists(path):
        logger.info("[GMS] KV index log absent (%s); nothing to rehydrate", path)
        return

    with open(path, "rb") as f:
        data = f.read()

    # Keep the LAST record per block_id -> each block's final cached content.
    latest: dict[int, tuple[bytes, int | None]] = {}
    off, total = 0, len(data)
    while off + 4 <= total:
        (ln,) = struct.unpack_from("<I", data, off)
        off += 4
        if off + ln > total:
            break
        hash_bytes, block_id, num_tokens = pickle.loads(data[off : off + ln])
        off += ln
        latest[block_id] = (hash_bytes, num_tokens)

    null_id = block_pool.null_block.block_id
    n = 0
    for block_id, (hash_bytes, num_tokens) in latest.items():
        if block_id == null_id or block_id < 0 or block_id >= len(block_pool.blocks):
            continue
        block = block_pool.blocks[block_id]
        if block.block_hash is not None or block.ref_cnt != 0:
            # Only rehydrate blocks that are free and unhashed on the standby;
            # never disturb a block the standby is already using.
            continue
        block_pool._insert_block_hash(
            BlockHashWithGroupId(hash_bytes), block, num_tokens
        )
        n += 1
    logger.info(
        "[GMS] KV index rehydrated: replayed %d block entries into BlockPool "
        "(%d records on log)",
        n,
        len(latest),
    )
