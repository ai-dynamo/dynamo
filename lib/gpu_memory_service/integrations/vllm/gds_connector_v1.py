# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM KVCacheConnectorV1 implementation for the GMS GDS-direct path.

VLLM VERSION COMPATIBILITY
==========================

This connector is developed against vLLM 0.20.x. The V1
`KVConnectorBase_V1` API has stabilized in 0.20+ but specific
fields the connector reads may shift in future minor releases:

  - `KVTransferConfig.kv_role` — required on construction
    (values: "kv_producer" / "kv_consumer" / "kv_both"; we use
    "kv_both"). Verified present in 0.20.2.
  - `KVTransferConfig.engine_id` — required.
  - `KVTransferConfig.kv_connector_extra_config` — dict for
    user-supplied extras (we read `gms_daemon_socket` and
    `gms_async_restore`).
  - `Request.prompt_token_ids` — used for prefix hash computation
    in the scheduler-side `_PrefixIndex`. Present in v1.
  - `Request.cache_salt` — read with `getattr(request,
    "cache_salt", None)` (defensive against absence).
  - `EngineArgs.worker_cls` — required as a string class path for
    GMSWorker registration. Verified present in 0.20.2.
  - The `register_kv_caches(dict[str, torch.Tensor])` hook.

When upgrading vLLM, run
`tests/real_engine/test_vllm_v1_connector.py::test_smoke_one_generation`
to catch API drift early.



Routes per-request KV block eviction through the gms_kv_ring
`VllmGdsConnector`, which calls `demote_hbm_to_storage` on every
(layer, offset, size) of every block via the daemon's GPU-direct
storage backend (NIXL GDS/GDS_MT today).

Deployment topology — **one GMS daemon instance per GPU**:

  Each GPU on the host has its own GMS stack:
    - one weights GMS server  (legacy; for model weights)
    - one kv_cache GMS server (legacy; for cuMemMap-allocated KV)
    - one gms_kv_ring daemon  (this connector's offload/restore peer)
  Socket paths are keyed by GPU UUID via `get_socket_path(device,
  tag)`. In a TP=N deployment, rank K's worker process talks ONLY
  to device K's daemon — daemon-side state is naturally per-GPU,
  no cross-rank routing in the daemon. The `engine_id` field
  identifies the LOGICAL engine (one across all ranks); each rank's
  daemon happens to be told the same engine_id but is a different
  process binding a different GPU's KV pool.

Two roles, modeled on KVBM's `DynamoConnector`:

  SCHEDULER (leader process)
    - request_finished(req, block_ids) → stash (req_id, block_ids)
      on the next step's metadata; index the request's prompt
      prefix block-by-block in the in-process prefix-hash table so
      that a subsequent request hitting the same prefix can be
      answered by `get_num_new_matched_tokens`.
    - get_num_new_matched_tokens(req, num_computed_tokens) → walk
      the request's prompt token ids, compute per-block prefix
      hashes, look them up in the index. Return the longest matched
      prefix length (in tokens) past `num_computed_tokens`.
    - update_state_after_alloc(req, blocks, n_ext) → vLLM has just
      allocated `dst_block_ids` for the matched tokens. Pair them
      with the indexed `src_block_ids` and queue (src, dst) on the
      restore stash.
    - build_connector_meta → JSON-encode both evict + restore
      stashes and reset.

  WORKER (per-device worker process)
    - register_kv_caches(kv_caches) → lazily build the GMSKvRing
      handle and VllmGdsConnector from the kv tensor dict.
    - bind_connector_metadata(meta) → decode evict + restore
      queues. Evict via VllmGdsConnector.evict_blocks_to_storage;
      restore via VllmGdsConnector.restore_blocks_remap (storage
      key = src_block_id, dest VA = dst_block_id).
    - get_finished(finished) → return the req_ids whose evictions
      have settled (vLLM frees those blocks now).

If the daemon's storage backend does NOT advertise
`supports_gpu_direct`, the worker-side path is a clean no-op:
`is_available()` returns False, evict_blocks_to_storage is skipped,
and we still return req_ids as "finished" so vLLM frees blocks
normally. That makes this connector safe to enable unconditionally —
the no-GDS configuration just behaves like no connector at all.

Restore-on-hit prefix index:
  The scheduler maintains an in-process dict mapping
  `(cache_salt, prompt_token_ids[:n*block_size]_hash) → src_block_id`.
  Hash is sha256 over `(cache_salt, prefix_token_ids_packed)` —
  intentionally NOT vLLM's BlockHash, since that's version-coupled.
  Eviction populates it; cache-hit lookup queries it. Persistence
  across scheduler restart is opt-in via
  `GMS_KVR_PREFIX_INDEX_SNAPSHOT=<path>` (atomic tmp+rename writes
  on a throttled cadence; corrupt/missing snapshots → cold start).
  See `_PrefixIndex` docstring invariant 5 for the safety argument.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from gpu_memory_service.common.utils import get_socket_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# vLLM is imported lazily so this module can be loaded for tests
# without vLLM installed. The class body references the V1 base
# class, so the import happens at class-definition time inside a
# helper function rather than at module import time.


def _import_v1_base():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    return KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole


_KVConnectorBase_V1, _KVConnectorMetadata, _KVConnectorRole = _import_v1_base()


class _GmsGdsMetadata(_KVConnectorMetadata):
    """JSON-encoded payload carrying both queues from scheduler to
    worker each step:
      - evict_queue: list[(req_id, block_ids)] for spill
      - restore_queue: list[(req_id, [(src_block, dst_block), ...])]
        for cache-hit restore
    Small payloads (per-step), JSON over the wire is fine."""

    def __init__(self, payload: bytes):
        assert isinstance(payload, bytes)
        self.payload = payload


# Schema versioning so a worker built against an older scheduler
# (and vice versa) fails loudly instead of misinterpreting bytes.
#
# v3 splits the restore queue into:
#   - restore_async : safe for the ring path (storage src_block_id
#                      is NOT in this step's evict set)
#   - restore_sync  : MUST run synchronously in bind, before the
#                      evict, so the storage read sees pre-evict
#                      bytes.
#
# v4 carries generations through both queues to close Race #3
# (cross-step async-restore vs evict). Evict entries gain a
# parallel `generations: list[int]` (one per block_id). Restore
# entries become triples (src, dst, src_generation) so the daemon
# can enforce `slot_generation == src_generation` before reading.
#
# v6 adds `restore_staging_async`: per-block content hashes already
# present in the destination daemon's StagingTier. The worker records
# these through the restore ring with FLAG_SOURCE_STAGING.
_META_SCHEMA_VERSION = 6


def _decode_meta(
    payload: bytes,
) -> tuple[
    list[tuple[str, list[int], list[int], list[bytes]]],
    list[tuple[str, list[tuple[int, int, int]]]],
    list[tuple[str, list[tuple[int, int, int]]]],
    list[tuple[str, list[tuple[bytes, int, int]]]],
]:
    """Returns queues decoded from scheduler metadata.

    Evict queue entries:    (req_id, [block_id, ...], [generation, ...], [hash, ...])
    Local restore entries:  (req_id, [(src, dst, src_gen), ...])
    Staging entries:        (req_id, [(content_hash, dst, generation), ...])

    The 4th element of each evict tuple is per-block content hashes
    (one per block_id, may be empty for blocks beyond the indexed
    prefix or when cross-node is disabled). Hashes are 32-byte bytes
    objects; empty list means "no cross-node registration for this
    evict batch". The worker uses these to call
    `register_content_addresses_batch` on the daemon so a peer can
    later `fetch_remote` (or NIXL-read directly) these hashes.

    Backward compat: v1/v2/v3/v4/v5 decode into the v6 shape with
    generations defaulted to 0, hashes defaulted to [], and no staging
    restore entries. A v5 scheduler talking to a v6 worker is a silent
    rolling-upgrade scenario; we don't break it.
    """
    if not payload:
        return [], [], [], []
    raw = json.loads(payload.decode("utf-8"))
    if isinstance(raw, list):
        # v1: bare evict list.
        return (
            [
                (str(rid), [int(b) for b in bids], [0] * len(bids), [])
                for rid, bids in raw
            ],
            [],
            [],
            [],
        )
    v = raw.get("v")
    if v not in (2, 3, 4, 5, 6):
        raise ValueError(
            f"GMSKVCacheConnectorV1: metadata schema version "
            f"{v!r} not supported (this build expects "
            f"v in {{2, 3, 4, 5, 6}})."
        )
    if v in (2, 3):
        evict = [
            (str(rid), [int(b) for b in bids], [0] * len(bids), [])
            for rid, bids in raw.get("evict", [])
        ]
    else:
        evict = []
        # v5+ stores per-evict hashes alongside the evict array. The
        # `hashes_per_evict` list is parallel to `evict` and contains
        # one list[str-hex] per entry. Missing -> cross-node disabled.
        hashes_per_evict = raw.get("hashes_per_evict") or []
        for i, entry in enumerate(raw.get("evict", [])):
            rid, bids, gens = entry
            ids = [int(b) for b in bids]
            generations = [int(g) for g in gens]
            if len(generations) < len(ids):
                generations += [0] * (len(ids) - len(generations))
            block_hashes: list[bytes] = []
            if i < len(hashes_per_evict):
                for hx in hashes_per_evict[i]:
                    try:
                        block_hashes.append(bytes.fromhex(hx))
                    except (ValueError, TypeError):
                        # Malformed hash -- skip rather than fail the
                        # whole step. Hash registration is best-effort
                        # cross-node hint, not a correctness contract.
                        block_hashes.append(b"")
            evict.append((str(rid), ids, generations, block_hashes))
    if v == 2:
        restore_async = []
        restore_sync = [
            (str(rid), [(int(s), int(d), 0) for s, d in pairs])
            for rid, pairs in raw.get("restore", [])
        ]
    elif v == 3:
        restore_async = [
            (str(rid), [(int(s), int(d), 0) for s, d in pairs])
            for rid, pairs in raw.get("restore_async", [])
        ]
        restore_sync = [
            (str(rid), [(int(s), int(d), 0) for s, d in pairs])
            for rid, pairs in raw.get("restore_sync", [])
        ]
    else:
        restore_async = [
            (str(rid), [(int(s), int(d), int(g)) for s, d, g in pairs])
            for rid, pairs in raw.get("restore_async", [])
        ]
        restore_sync = [
            (str(rid), [(int(s), int(d), int(g)) for s, d, g in pairs])
            for rid, pairs in raw.get("restore_sync", [])
        ]

    restore_staging_async = []
    if v >= 6:
        for rid, entries in raw.get("restore_staging_async", []):
            parsed = []
            for entry in entries:
                try:
                    h_hex, dst, generation = entry
                    h = bytes.fromhex(str(h_hex))
                    parsed.append((h, int(dst), int(generation)))
                except (ValueError, TypeError):
                    continue
            restore_staging_async.append((str(rid), parsed))

    return evict, restore_async, restore_sync, restore_staging_async


def _encode_meta(
    evict_queue: "list[tuple]",
    restore_async: Optional[list[tuple[str, list[tuple[int, int, int]]]]] = None,
    restore_sync: Optional[list[tuple[str, list[tuple[int, int, int]]]]] = None,
    restore_staging_async: Optional[
        list[tuple[str, list[tuple[bytes, int, int]]]]
    ] = None,
) -> bytes:
    """Encodes evict + restore queues into a JSON payload (v6).

    `evict_queue` accepts either 3-tuples `(req_id, ids, gens)` or
    4-tuples `(req_id, ids, gens, hashes)`. When any 4-tuple has
    a non-empty `hashes` list, it is written into a sibling
    `hashes_per_evict` array (parallel to `evict`). When all entries
    are 3-tuples or have empty hashes, the field is omitted to keep
    the steady-state payload small (the dominant case).
    """
    evict_out: list[tuple[str, list[int], list[int]]] = []
    hashes_per_evict: list[list[str]] = []
    any_hashes = False
    for entry in evict_queue:
        if len(entry) == 4:
            rid, ids, gens, hashes = entry
            hashes_hex = [
                h.hex() if isinstance(h, (bytes, bytearray)) and h else ""
                for h in hashes
            ]
            if any(h for h in hashes_hex):
                any_hashes = True
        else:
            rid, ids, gens = entry
            hashes_hex = []
        evict_out.append((rid, ids, gens))
        hashes_per_evict.append(hashes_hex)
    out = {
        "v": _META_SCHEMA_VERSION,
        "evict": evict_out,
        "restore_async": restore_async or [],
        "restore_sync": restore_sync or [],
    }
    if restore_staging_async:
        out["restore_staging_async"] = [
            (
                rid,
                [(h.hex(), int(dst), int(gen)) for h, dst, gen in entries],
            )
            for rid, entries in restore_staging_async
        ]
    if any_hashes:
        out["hashes_per_evict"] = hashes_per_evict
    return json.dumps(out).encode("utf-8")


def _extend_match_via_staging(
    block_hashes: "list[bytes]",
    local_match_blocks: int,
    staging_hits: "dict[bytes, dict]",
) -> int:
    """Compute how many additional contiguous blocks beyond
    `local_match_blocks` are available in the daemon's staging tier.

    Cross-node design Phase 2 (see `docs/CROSS_NODE_DESIGN.md` §4.3).
    The local prefix-index match runs first; this function extends
    that match by walking the remaining block hashes in order and
    counting how many CONSECUTIVE hashes are present in `staging_hits`.
    A gap stops the extension — vLLM's connector contract requires
    contiguous block coverage starting from `num_computed_tokens`.

    `staging_hits` is a `{content_hash: metadata_dict}` mapping
    returned by `GMSKvRing.staging_scan`. Only key membership matters
    for this function; the metadata is consumed elsewhere when the
    actual restore record is emitted (Phase 3).

    Returns the count of additional contiguous blocks. Zero when the
    local match already covers all blocks, when staging has no
    relevant hits, or when the very next block after the local match
    is absent from staging.
    """
    if local_match_blocks >= len(block_hashes):
        return 0
    if not staging_hits:
        return 0
    n_extra = 0
    for i in range(local_match_blocks, len(block_hashes)):
        if block_hashes[i] not in staging_hits:
            break
        n_extra += 1
    return n_extra


def _split_restore_by_conflict(  # Invariant I-2 (see docs/ARCHITECTURE.md)
    restore_queue: list[tuple[str, list[tuple[int, int, int]]]],
    evict_block_ids: set[int],
) -> tuple[
    list[tuple[str, list[tuple[int, int, int]]]],
    list[tuple[str, list[tuple[int, int, int]]]],
]:
    """Partition restore triples into (async-safe, sync-required).

    Enforces **invariant I-2** (in-step restore conflict-split is
    atomic w.r.t. evict): blocks that this scheduler step both
    restores AND evicts are returned in the sync lane, which the
    bind-time copy runs BEFORE the evict ring drains.

    A triple `(src, dst, gen)` is sync-required iff `src` is in
    the same step's `evict_block_ids` — running it asynchronously
    would let the synchronous evict overwrite storage at
    `(engine_id, src)` before the daemon consumer reads it.
    Generations carry through unchanged."""
    out_async: list[tuple[str, list[tuple[int, int, int]]]] = []
    out_sync: list[tuple[str, list[tuple[int, int, int]]]] = []
    for req_id, triples in restore_queue:
        a_triples = []
        s_triples = []
        for entry in triples:
            # Tolerate either (src, dst) or (src, dst, gen) — older
            # callers / tests may inject 2-tuples; default gen=0.
            if len(entry) == 3:
                s, d, g = entry
            else:
                s, d = entry
                g = 0
            if int(s) in evict_block_ids:
                s_triples.append((int(s), int(d), int(g)))
            else:
                a_triples.append((int(s), int(d), int(g)))
        if a_triples:
            out_async.append((req_id, a_triples))
        if s_triples:
            out_sync.append((req_id, s_triples))
    return out_async, out_sync


def _resolve_engine_id(vllm_config: "VllmConfig") -> str:
    """Engine id the daemon identifies us by — required field of
    `kv_transfer_config` (KVBM precedent). vLLM's V1 connector base
    class rejects construction without kv_transfer_config, so we
    don't fall back to a default."""
    eid = vllm_config.kv_transfer_config.engine_id
    return str(eid)


def _resolve_daemon_socket(vllm_config: "VllmConfig", device: int) -> str:
    """Daemon socket path. User may override via kv_connector_extra_config;
    default matches GMSWorker's kv_cache socket so we ride along the same
    GMS-managed pool the worker already attached to."""
    extra = {}
    cfg = getattr(vllm_config, "kv_transfer_config", None)
    if cfg is not None:
        extra = getattr(cfg, "kv_connector_extra_config", None) or {}
    sock = extra.get("gms_daemon_socket")
    if sock:
        return str(sock)
    return get_socket_path(device, "kv_cache")


def _power_of_2_env(
    var: str,
    default: int,
    *,
    floor: int = 16,
    ceiling: int = 1 << 20,
) -> int:
    """Read `os.environ[var]` as a power-of-2 int; fall back to
    `default` if missing or invalid. Ring capacities must be
    power-of-2 (SPSC ring's mask-based slot indexing assumes it);
    we round invalid values down to the nearest valid power-of-2.

    `floor` / `ceiling` clamp the result into a sane range. Below
    `floor` (default 16): a 1-slot ring is syntactically valid but
    breaks the producer/consumer protocol the moment it sees one
    in-flight op. Above `ceiling` (default 1 Mi): pinned shared
    memory cost balloons (each slot is tens of bytes; 1 Mi slots
    = tens of MiB). Defensive clamping pattern borrowed from
    Pegaflow's `_LOAD_TIMEOUT_FLOOR_SECONDS` guard."""
    if floor & (floor - 1) or ceiling & (ceiling - 1):
        raise ValueError(
            f"_power_of_2_env: floor={floor} and ceiling={ceiling} "
            "must themselves be powers of 2"
        )
    raw = os.environ.get(var)
    if raw is None:
        return int(default)
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "[GMS GDS Connector] %s=%r is not an int; using default %d",
            var,
            raw,
            default,
        )
        return int(default)
    if n <= 0:
        return int(default)
    # Round down to nearest power-of-2.
    if n & (n - 1):
        # Not power-of-2; find largest power-of-2 <= n.
        rounded = 1 << (n.bit_length() - 1)
        logger.warning(
            "[GMS GDS Connector] %s=%d is not a power-of-2; " "rounding down to %d",
            var,
            n,
            rounded,
        )
        n = rounded
    if n < floor:
        logger.warning(
            "[GMS GDS Connector] %s=%d below floor %d; clamped up",
            var,
            n,
            floor,
        )
        return floor
    if n > ceiling:
        logger.warning(
            "[GMS GDS Connector] %s=%d above ceiling %d; clamped down",
            var,
            n,
            ceiling,
        )
        return ceiling
    return n


def _layer_index_from_name(layer_name: str) -> int:
    """Extract a layer index from a vLLM-style layer name. We
    intentionally do NOT import `vllm.model_executor.models.utils`
    here — that module's transitive import chain (fused_moe →
    nixl_ep → libcudart) is fragile in test environments. The
    logic we need is just "find the integer in the dotted name."
    """
    ints = []
    for sub in layer_name.split("."):
        try:
            ints.append(int(sub))
        except ValueError:
            continue
    if not ints:
        raise ValueError(
            f"GMSKVCacheConnectorV1: cannot extract layer index "
            f"from name {layer_name!r}"
        )
    # First integer is the layer index. vLLM's extract_layer_index
    # asserts on multiple ints unless num_attn_module > 1; we are
    # lenient since attention layers in current vLLM models all
    # follow the single-integer convention.
    return ints[0]


@dataclass(frozen=True)
class _LayerRegistration:
    """Per-layer registration info inferred from a vLLM KV tensor.

    `physical_blocks_per_logical_block > 1` indicates MLA-style split:
    vLLM's scheduler hashes at a logical block boundary
    (`block_size`), but the attention kernel materializes each logical
    block as N contiguous physical rows. We must spill/restore all
    N rows together to preserve byte-correctness for one logical
    block id.

    `kv_stride_bytes > 0` indicates KV-first layout
    (`shape == (2, num_blocks, ...)`): K and V live in separate
    contiguous segments. Restore must copy both segments for each
    logical block id.

    Mirrors Pegaflow's `_infer_kv_cache_registration` in
    `python/pegaflow/connector/worker.py` (see Pegaflow comparison in
    `docs/IMPLEMENTATION.md`)."""

    num_blocks: int  # logical blocks
    bytes_per_block: int  # bytes per logical block (per layer)
    kv_stride_bytes: int  # >0 → K-V split (KV-first layout)
    physical_blocks_per_logical_block: int  # >1 → MLA split-block


def _infer_layer_registration(
    kv_cache: "torch.Tensor",
    logical_block_size: int,
    *,
    is_mla: bool,
) -> _LayerRegistration:
    """Infer GMS registration shape from one layer's KV tensor.

    Two-axis non-MLA layouts: `(2, num_blocks, ...)` is KV-first;
    anything else with blocks on `shape[0]` is blocks-first.

    MLA layouts: `shape[0]` is the physical block count and
    `shape[1]` is the physical block size. When kernel block size
    (FlashMLA: 64) differs from scheduler block size (vLLM manager:
    128), `physical_blocks_per_logical_block = logical // physical`.

    Returns a `_LayerRegistration` shaped for the closure built by
    `_build_layers_and_layout`."""
    shape = tuple(kv_cache.shape)
    stride = tuple(kv_cache.stride())
    element_size = kv_cache.element_size()
    if logical_block_size <= 0:
        raise ValueError(f"logical block size must be > 0, got {logical_block_size}")

    if not is_mla:
        if len(shape) >= 2 and shape[0] == 2:
            num_blocks = shape[1]
            bytes_per_block = stride[1] * element_size
            kv_stride_bytes = stride[0] * element_size
        else:
            num_blocks = shape[0]
            bytes_per_block = stride[0] * element_size
            kv_stride_bytes = 0
        if num_blocks <= 0:
            raise ValueError(f"physical block count must be > 0, got {num_blocks}")
        if bytes_per_block == 0:
            raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")
        return _LayerRegistration(
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            kv_stride_bytes=kv_stride_bytes,
            physical_blocks_per_logical_block=1,
        )

    # MLA: layout is blocks-first with `shape[1]` = physical block size.
    physical_num_blocks = shape[0]
    physical_block_size = shape[1] if len(shape) >= 2 else logical_block_size
    physical_bytes_per_block = stride[0] * element_size
    if physical_num_blocks <= 0:
        raise ValueError(f"physical block count must be > 0, got {physical_num_blocks}")
    if physical_block_size <= 0:
        raise ValueError(f"physical block size must be > 0, got {physical_block_size}")
    if logical_block_size % physical_block_size != 0:
        raise ValueError(
            "logical block size must be a multiple of physical block size "
            f"(logical={logical_block_size}, "
            f"physical={physical_block_size})"
        )
    ratio = logical_block_size // physical_block_size
    if physical_num_blocks % ratio != 0:
        raise ValueError(
            "physical block count must be divisible by physical/logical "
            f"split ratio (physical_blocks={physical_num_blocks}, "
            f"ratio={ratio})"
        )
    bytes_per_block = physical_bytes_per_block * ratio
    if bytes_per_block == 0:
        raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")
    return _LayerRegistration(
        num_blocks=physical_num_blocks // ratio,
        bytes_per_block=bytes_per_block,
        kv_stride_bytes=0,
        physical_blocks_per_logical_block=ratio,
    )


def _build_layers_and_layout(
    kv_caches: "dict[str, torch.Tensor]",
    *,
    logical_block_size: Optional[int] = None,
    is_mla: bool = False,
):
    """From vLLM's per-layer kv tensor dict, derive the (layers,
    block_layout_fn) pair GMSKvRing needs.

    Assumes uniform per-layer shape (hybrid models with mixed shapes
    are not handled — KVBM also rejects them).

    `logical_block_size` is vLLM's scheduler block size in tokens.
    Required only for MLA models with split kernel blocks (FlashMLA
    on DeepSeek-V3). For non-MLA models the value is unused; passing
    None falls back to the legacy axis-max heuristic for
    backward-compat with callers that don't have block-size info."""
    ordered = sorted(kv_caches.items(), key=lambda it: _layer_index_from_name(it[0]))
    if not ordered:
        raise ValueError("register_kv_caches: empty kv_caches dict")
    first = ordered[0][1]
    shape = first.shape
    if not all(t.shape == shape for t in kv_caches.values()):
        raise NotImplementedError(
            "GMSKVCacheConnectorV1 does not support hybrid models "
            "with non-uniform per-layer KV cache shapes."
        )

    if logical_block_size is None:
        # Legacy path: no MLA awareness, fall back to KVBM heuristic.
        num_blocks = shape[0] if len(shape) < 2 else max(shape[0], shape[1])
        layer_bytes = first.numel() * first.element_size()
        if num_blocks <= 0 or layer_bytes % num_blocks != 0:
            raise ValueError(
                f"GMSKVCacheConnectorV1: cannot infer block size from "
                f"shape={tuple(shape)} num_blocks={num_blocks} "
                f"layer_bytes={layer_bytes}"
            )
        block_bytes = layer_bytes // num_blocks
        reg = _LayerRegistration(
            num_blocks=num_blocks,
            bytes_per_block=block_bytes,
            kv_stride_bytes=0,
            physical_blocks_per_logical_block=1,
        )
    else:
        reg = _infer_layer_registration(
            first,
            logical_block_size,
            is_mla=is_mla,
        )

    n_layers = len(ordered)
    block_bytes = reg.bytes_per_block
    kv_stride = reg.kv_stride_bytes
    layers = [
        {
            "layer_idx": L,
            "va": int(t.data_ptr()),
            "size": int(t.numel() * t.element_size()),
            "stride": int(block_bytes),
        }
        for L, (_name, t) in enumerate(ordered)
    ]

    # block_layout: list of (layer_idx, byte_offset, byte_size) entries
    # describing the bytes for one logical block id. For non-split
    # non-KV-first layouts this is one entry per layer. KV-first
    # layouts add a second entry per layer for the V segment. MLA
    # split-block layouts coalesce N physical rows into one contiguous
    # range (they are contiguous in memory by construction).
    if kv_stride > 0:

        def block_layout(block_id: int):
            out = []
            base = int(block_id) * block_bytes
            for L in range(n_layers):
                out.append((L, base, block_bytes))  # K
                out.append((L, kv_stride + base, block_bytes))  # V
            return out

    else:
        # block_bytes already accounts for the MLA split ratio
        # (`reg.physical_blocks_per_logical_block × physical_bytes`),
        # so the byte range for one logical block id is contiguous
        # and a single per-layer entry suffices.
        def block_layout(block_id: int):
            return [
                (L, int(block_id) * block_bytes, block_bytes) for L in range(n_layers)
            ]

    return layers, block_layout, n_layers, block_bytes


# Re-export the shared cross-engine prefix-block hash. Both vLLM and
# SGLang connectors must use the SAME implementation so cross-node
# transfers between heterogeneous engines work — see
# `gms_kv_ring/common/prefix_hashes.py` for the wire-format definition.
from gms_kv_ring.common.prefix_hashes import (  # noqa: E402
    prefix_block_hashes as _prefix_block_hashes,
)

# Snapshot file format for `_PrefixIndex.snapshot/_load_snapshot`.
# Layout:
#   magic   (8 bytes)  : b"GMSPIDX\x00"
#   version (2 bytes)  : uint16 LE
#   pkl_len (4 bytes)  : uint32 LE (length of the pickle payload)
#   pkl     (pkl_len)  : pickle.dumps(payload_dict, protocol=4)
#   crc32   (4 bytes)  : zlib.crc32 over the pickle bytes
#
# `payload_dict` schema (version=2):
#   {
#     "max":              int,
#     "table":            OrderedDict[bytes, (str, int, int)],
#     "slot_to_hashes":   dict[(str, int), set[bytes]],
#     "slot_generations": dict[(str, int), int],
#     "daemon_epoch":     Optional[int]   # added in v2
#   }
#
# v2 binds the snapshot to the daemon instance it was written
# against. On load, if `expected_daemon_epoch` is supplied and
# differs from the snapshot's embedded value, the snapshot is
# discarded (cold start). Without this, a daemon restart between
# snapshot write and load would leave the connector with stale
# slot mappings pointing at a daemon whose host-tier is empty —
# every restore from the loaded index would fail.
#
# A version bump requires bumping `_SNAPSHOT_VERSION` and adding a
# decoder branch. Old files with a higher version load as empty
# (logged) so a downgrade is safe.
# _PrefixIndex + snapshot helpers are shared across vLLM, SGLang, and
# TRT-LLM connectors. The implementation lives in
# `gms_kv_ring/common/prefix_index.py`; we re-export under the same
# names used historically by this module so existing test imports
# keep working.
from gms_kv_ring.common.prefix_index import (  # noqa: E402,F401
    PrefixIndex as _PrefixIndex,
)


class GMSKVCacheConnectorV1(_KVConnectorBase_V1):
    """vLLM KVCacheConnectorV1 wired to gms_kv_ring's VllmGdsConnector.

    Construct via vLLM by setting `--kv-transfer-config '{
        "kv_connector":
            "gpu_memory_service.integrations.vllm.gds_connector_v1"
            ".GMSKVCacheConnectorV1",
        "engine_id": "0"
    }'`. Optional `kv_connector_extra_config.gms_daemon_socket` overrides
    the default socket (`get_socket_path(device, "kv_cache")`).

    See module docstring for SCHEDULER / WORKER role behavior."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.engine_id = _resolve_engine_id(vllm_config)
        self._vllm_config = vllm_config
        self._role = role
        try:
            self._extra_config = (
                vllm_config.kv_transfer_config.kv_connector_extra_config or {}
            )
        except AttributeError:
            self._extra_config = {}
        src_kw = self._extra_config.get("gms_source_engine_id")
        self._source_engine_id = str(
            src_kw
            or os.environ.get("GMS_VLLM_SOURCE_ENGINE_ID")
            or os.environ.get("GMS_KVR_SOURCE_ENGINE_ID")
            or self.engine_id
        )
        host_kw = self._extra_config.get("gms_host_tier_fallback")
        if host_kw is None:
            host_kw = os.environ.get("GMS_KVR_HOST_TIER_FALLBACK", "0")
        self._host_tier_fallback = str(host_kw).lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        mirror_kw = self._extra_config.get("gms_host_tier_mirror")
        if mirror_kw is None:
            mirror_kw = os.environ.get("GMS_KVR_HOST_TIER_MIRROR", "0")
        self._host_tier_mirror = str(mirror_kw).lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )

        # Scheduler state. Each step: collected by request_finished,
        # drained by build_connector_meta.
        # v4: each entry is (req_id, block_ids, generations) — gens
        # are parallel to block_ids; 0 means "no gen tag" for legacy
        # paths.
        # 4th element is per-block content hashes when cross-node is
        # enabled (see GMS_KVR_CROSS_NODE env var); otherwise [].
        self._sched_evict_queue: list[
            tuple[str, list[int], list[int], list[bytes]]
        ] = []
        # Cross-node hint registration: when set, the scheduler
        # attaches content hashes to evict entries so the worker can
        # call `register_content_addresses_batch` on the daemon. The
        # daemon then knows how to fulfill `fetch_remote` (or a
        # direct NIXL read) for these hashes. Default OFF — opt-in
        # per deployment.
        self._cross_node_register = os.environ.get(
            "GMS_KVR_CROSS_NODE",
            "0",
        ).lower() not in ("0", "false", "no", "off", "")
        # Worker-side counterpart of the scheduler flag. Starts True
        # iff the env var is set; auto-disabled on the first batched
        # call that the daemon reports `skipped=True` for (the daemon
        # has no transport). This avoids paying the RPC cost forever
        # in single-node deployments that happen to set the env var.
        self._cross_node_xfer_enabled = self._cross_node_register
        # Restore queue: per req_id, list of (src_block_id,
        # dst_block_id, generation) triples to remap from storage to
        # HBM on the worker. Built by update_state_after_alloc;
        # drained by build_connector_meta.
        self._sched_restore_queue: list[tuple[str, list[tuple[int, int, int]]]] = []
        # Staging restore queue: per req_id, list of
        # (content_hash, dst_block_id, staging_generation) entries.
        # These are hits in the destination daemon's StagingTier,
        # usually produced by a decode-to-decode GMS transfer.
        self._sched_staging_restore_queue: list[
            tuple[str, list[tuple[bytes, int, int]]]
        ] = []
        # Requests we told vLLM are "finishing async" — they stay
        # here until the worker side reports them via get_finished.
        # Bookkeeping only; vLLM holds blocks based on the True return
        # from request_finished, not on this set.
        self._sched_pending: set[str] = set()
        # Daemon-epoch state, used to invalidate `_prefix_index` when
        # the daemon process is replaced (its in-memory host_tier and
        # slot_generations are zeroed; every cached hash is stale).
        # Set lazily on the first poll; the index is constructed below
        # with this current value so the snapshot-load path can
        # cold-start if the embedded epoch differs. See `_epoch_loop`
        # and `_check_daemon_epoch_once`.
        self._daemon_epoch_seen: Optional[int] = None
        self._epoch_thread: Optional[threading.Thread] = None
        self._epoch_stop = threading.Event()
        self._epoch_client = None
        self._staging_scan_client = None
        # Best-effort startup epoch probe. Open a thin DaemonClient
        # JUST for epoch reads (the worker still owns the hot-path
        # client via the GMSKvRing handle). If the daemon isn't up
        # yet at scheduler construction (common in tests), we leave
        # `_daemon_epoch_seen = None` and the background thread will
        # pick it up later.
        self._epoch_socket: Optional[str] = None
        try:
            _ = vllm_config.cache_config
            # Engine device 0 is the scheduler-side fallback when no
            # specific GPU is attached at this point. The connector
            # talks to the SAME daemon socket the workers do (one
            # daemon per GPU on this host).
            self._epoch_socket = _resolve_daemon_socket(
                vllm_config,
                0,
            )
        except Exception:  # noqa: BLE001
            self._epoch_socket = None
        # NOTE: we deliberately do NOT open a DaemonClient at __init__.
        # The background poll thread handles connect + epoch read,
        # so connector construction never pays a daemon-RPC latency
        # cost. A snapshot loaded at __init__ runs with epoch=None
        # (no embedded-epoch check); if the daemon has restarted
        # since the snapshot was written, the first poll observes
        # epoch=current vs seen=None, sets the baseline, and the
        # next poll (5 s later) sees mismatch and invalidates. The
        # cost is a brief window of stale-but-soon-invalidated
        # entries — same property as "no persistence" in that
        # window, never wrong KV bytes (daemon's slot-generation
        # check catches stale entries on actual restore attempts).
        # Content-hash → (engine_id, src_block_id) mapping populated
        # at request_finished, queried at get_num_new_matched_tokens.
        # `expected_daemon_epoch` is the value the snapshot must
        # carry to be loaded — None disables the check (back-compat /
        # daemon-not-up).
        self._prefix_index = _PrefixIndex(
            expected_daemon_epoch=self._daemon_epoch_seen,
        )
        # Background poller: cheaply checks for daemon-epoch changes
        # so the connector can invalidate proactively. Cadence is
        # configurable; default 5 s is fine for an event that only
        # happens on operator-driven restarts. Lives ONLY in the
        # scheduler-side connector — the worker side already detects
        # changes implicitly via its hot-path RPCs.
        try:
            poll_s = float(
                os.environ.get("GMS_KVR_EPOCH_POLL_S", "5.0"),
            )
        except ValueError:
            poll_s = 5.0
        self._epoch_poll_s: float = max(0.5, poll_s)
        if self._epoch_socket:
            self._epoch_thread = threading.Thread(
                target=self._epoch_loop,
                name=f"gms-epoch-{self.engine_id}",
                daemon=True,
            )
            self._epoch_thread.start()
        # Per-request: src_block_ids pinned by get_num_new_matched_tokens
        # so update_state_after_alloc can pair them with the
        # dst_block_ids vLLM just allocated.
        self._pending_hit: dict[str, list[tuple]] = {}
        # Block size in tokens — read once from vllm_config at init;
        # used by prefix-hash computation.
        try:
            self._block_size_tokens = int(
                vllm_config.cache_config.block_size,
            )
        except AttributeError:
            # cache_config not always present in test stubs.
            self._block_size_tokens = 0

        # Worker state — populated lazily on register_kv_caches.
        self._handle = None
        self._gds_conn = None
        self._block_layout = None
        self._n_layers = 0
        self._block_bytes = 0
        # Per-bind: req_ids the worker considers finished-saving after
        # processing the metadata. get_finished drains this.
        self._worker_finished_saves: set[str] = set()
        # Per-bind: list of (req_id, slot, target) from
        # handle.record_restore_gds. start_load_kv drains this and
        # issues cuStreamWaitValue32 on the compute stream;
        # clear_connector_metadata host-checks restore_succeeded()
        # after the forward and bumps failure metrics per req_id.
        # Per-bind entries: (req_id, counter_slot, target,
        # src_block_ids). The 4th element lets clear_connector_metadata
        # drop the failed slots from _PrefixIndex on detected failure
        # so the next request doesn't re-claim the same broken hit.
        self._pending_restore_waits: list[tuple[str, int, int, list[int]]] = []
        # start_load_kv → clear_connector_metadata handoff (per
        # forward pass): hooks for the post-forward host-side
        # restore_succeeded check.
        self._pending_restore_checks: list[tuple[str, int, int, list[int]]] = []
        # Knob: opt out of async restore (force the synchronous
        # remap path, useful for benchmarking or rollback). Read
        # once from kv_connector_extra_config or
        # GMS_KVR_ASYNC_RESTORE env var (default: True).
        async_kw = None
        async_kw = self._extra_config.get("gms_async_restore")
        if async_kw is None:
            async_kw = os.environ.get("GMS_KVR_ASYNC_RESTORE", "1")
        self._async_restore = str(async_kw).lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    # ------------------------------------------------------------------
    # Daemon-epoch invalidation (scheduler-side, background thread)
    # ------------------------------------------------------------------

    def _epoch_loop(self) -> None:  # Invariant I-4 (see docs/ARCHITECTURE.md)
        """Background loop. Polls the daemon's epoch on a low cadence
        (`GMS_KVR_EPOCH_POLL_S`, default 5 s) and invalidates the
        prefix index on observed change. Runs on its own thread so
        the scheduler hot path pays NOTHING for this check.

        Enforces **invariant I-4** (connector observes daemon restart
        eagerly): an epoch change triggers `_PrefixIndex.invalidate()`
        before any subsequent lookup can return a stale entry.

        Resilient to daemon down/up: if a ping raises, we sleep and
        retry without resetting last-seen epoch — so a transient
        socket error doesn't trigger a false invalidation."""
        while not self._epoch_stop.wait(self._epoch_poll_s):
            try:
                self._check_daemon_epoch_once()
            except Exception:  # noqa: BLE001
                # Don't let a poll error tear down the thread.
                logger.debug(
                    "GMS epoch poll error",
                    exc_info=True,
                )

    def _check_daemon_epoch_once(self) -> None:
        """One poll. If the daemon's current epoch differs from
        `_daemon_epoch_seen`, drop the prefix index and update the
        seen value. Called from `_epoch_loop`; also test-callable.

        First-time observation (seen=None, daemon-up) sets the
        baseline without invalidating — that's the normal startup
        case, not a restart event."""
        if self._epoch_client is None:
            # Try to (re)connect best-effort. Daemon may have come
            # up after scheduler init.
            try:
                from gms_kv_ring.daemon.client import DaemonClient

                self._epoch_client = DaemonClient(
                    self._epoch_socket,
                    connect_timeout=0.5,
                    op_timeout=2.0,
                )
            except Exception:  # noqa: BLE001
                return
        try:
            self._epoch_client._call({"op": "ping"})
        except Exception:  # noqa: BLE001
            # Socket may have died after a daemon restart. Close the
            # stale client; the next poll will reconnect (and observe
            # the new epoch).
            try:
                self._epoch_client.close()
            except Exception:  # noqa: BLE001
                pass
            self._epoch_client = None
            return
        current = self._epoch_client.current_daemon_epoch()
        if current is None:
            return
        if self._daemon_epoch_seen is None:
            self._daemon_epoch_seen = int(current)
            self._prefix_index.set_daemon_epoch(int(current))
            return
        if int(current) != int(self._daemon_epoch_seen):
            from gms_kv_ring.common import metrics

            old = self._daemon_epoch_seen
            self._daemon_epoch_seen = int(current)
            self._prefix_index.set_daemon_epoch(int(current))
            self._prefix_index.invalidate()
            metrics.connector_daemon_epoch_changes.inc(
                engine_id=str(self.engine_id),
            )
            logger.warning(
                "[GMS connector] daemon epoch changed %d → %d "
                "(daemon was restarted). Dropped prefix index; "
                "engine reverts to cold cache until re-warmed.",
                int(old),
                int(current),
            )

    def shutdown(self) -> None:
        """Stop the epoch poll thread and release the dedicated
        DaemonClient. Idempotent. Called from `__del__` and tests.

        Tight join timeout (200 ms) because the poll thread sleeps
        on an `Event.wait()` that wakes immediately when we `set()`
        the stop event. A longer timeout only matters if the thread
        is mid-RPC, which is at most a few-ms `ping` on a healthy
        socket. Slow process exit hurts unit-test teardown more
        than fast exit risks leaking a daemon thread."""
        try:
            self._epoch_stop.set()
        except AttributeError:
            return
        t = getattr(self, "_epoch_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=0.2)
        try:
            if self._epoch_client is not None:
                self._epoch_client.close()
                self._epoch_client = None
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._staging_scan_client is not None:
                self._staging_scan_client.close()
                self._staging_scan_client = None
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # SCHEDULER role
    # ------------------------------------------------------------------

    def _staging_scan(self, block_hashes: "list[bytes]") -> "dict[bytes, dict]":
        """Best-effort scheduler-side StagingTier lookup.

        The scheduler only needs this when cross-node GMS transfer is
        enabled. Failure is non-fatal: returning no hits makes vLLM
        recompute the prefix rather than consuming remote KV.
        """
        if not block_hashes or not self._cross_node_register:
            return {}
        if not self._epoch_socket:
            return {}
        try:
            if self._staging_scan_client is None:
                from gms_kv_ring.daemon.client import DaemonClient

                self._staging_scan_client = DaemonClient(
                    self._epoch_socket,
                    connect_timeout=0.5,
                    op_timeout=2.0,
                )
            return self._staging_scan_client.staging_scan(block_hashes)
        except Exception:  # noqa: BLE001
            try:
                if self._staging_scan_client is not None:
                    self._staging_scan_client.close()
            except Exception:  # noqa: BLE001
                pass
            self._staging_scan_client = None
            logger.debug("GMS staging_scan failed", exc_info=True)
            return {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Return the longest local-or-staged prefix match.

        Local prefix-index hits are preferred. When cross-node GMS is
        enabled, the connector extends that contiguous match with
        READY entries from the destination daemon's StagingTier, so a
        router-orchestrated decode-to-decode transfer can be consumed
        by the same vLLM external-prefix path.
        """
        if self._block_size_tokens <= 0:
            return (0, False)
        prompt = list(getattr(request, "prompt_token_ids", None) or [])
        if not prompt:
            return (0, False)
        salt = getattr(request, "cache_salt", None)
        block_hashes = _prefix_block_hashes(
            prompt,
            self._block_size_tokens,
            salt,
        )
        if not block_hashes:
            return (0, False)
        src_entries = self._prefix_index.lookup(
            prompt,
            self._block_size_tokens,
            salt,
            self._source_engine_id,
        )
        local_match_blocks = len(src_entries)
        staging_hits: dict[bytes, dict] = {}
        staging_extra = 0
        if local_match_blocks < len(block_hashes):
            staging_hits = self._staging_scan(
                block_hashes[local_match_blocks:],
            )
            staging_extra = _extend_match_via_staging(
                block_hashes,
                local_match_blocks,
                staging_hits,
            )
        matched_blocks = local_match_blocks + staging_extra
        if matched_blocks <= 0:
            return (0, False)
        matched_tokens = matched_blocks * self._block_size_tokens
        # Subtract whatever vLLM already has computed internally --
        # we only advertise the DELTA past the internal prefix-cache
        # hit. Round down to the previous block boundary so we don't
        # claim a partial block.
        already = (
            num_computed_tokens // self._block_size_tokens
        ) * self._block_size_tokens
        if matched_tokens <= already:
            return (0, False)
        full_entries: list[tuple] = []
        for src_block_id, generation in src_entries:
            full_entries.append(("local", int(src_block_id), int(generation)))
        for idx in range(local_match_blocks, matched_blocks):
            h = block_hashes[idx]
            hit = staging_hits.get(h) or {}
            full_entries.append(
                (
                    "staging",
                    h,
                    int(hit.get("generation", 0)),
                )
            )
        # Slice to the actually-claimed prefix beyond what's computed.
        skip_blocks = already // self._block_size_tokens
        claimable = full_entries[skip_blocks:]
        delta_tokens = len(claimable) * self._block_size_tokens
        self._pending_hit[str(request.request_id)] = claimable
        return (delta_tokens, False)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Pair matched source entries with vLLM destination blocks.

        Local entries become storage/GDS restores. Staging entries
        become FLAG_SOURCE_STAGING restores from the destination
        daemon's StagingTier.
        """
        req_id = str(request.request_id)
        src_entries = self._pending_hit.pop(req_id, None)
        if not src_entries or num_external_tokens <= 0:
            return
        if self._block_size_tokens <= 0:
            return
        try:
            dst_groups = blocks.get_block_ids()
        except AttributeError:
            return
        if not dst_groups:
            return
        dst_ids = list(dst_groups[0])
        n_matched_blocks = num_external_tokens // self._block_size_tokens
        n_matched_blocks = min(
            n_matched_blocks,
            len(src_entries),
            len(dst_ids),
        )
        if n_matched_blocks <= 0:
            return
        # vLLM places the externally-matched blocks at the END of
        # the allocated list (after the locally-computed prefix).
        dst_slice = dst_ids[-n_matched_blocks:]
        local_triples: list[tuple[int, int, int]] = []
        staging_triples: list[tuple[bytes, int, int]] = []
        for src_entry, d in zip(src_entries[:n_matched_blocks], dst_slice):
            kind = src_entry[0]
            if kind == "local":
                _kind, src_block_id, generation = src_entry
                local_triples.append(
                    (int(src_block_id), int(d), int(generation)),
                )
            elif kind == "staging":
                _kind, content_hash, generation = src_entry
                if not isinstance(content_hash, (bytes, bytearray)):
                    continue
                staging_triples.append(
                    (bytes(content_hash), int(d), int(generation)),
                )
        if local_triples:
            self._sched_restore_queue.append((req_id, local_triples))
        if staging_triples:
            self._sched_staging_restore_queue.append(
                (req_id, staging_triples),
            )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ):
        """Drain this step's queues. The restore queue is
        partitioned into `async-safe` and `sync-required` lanes
        based on whether the restore's `src_block_id` is also in
        the same step's evict set:

          - sync-required: an evict for the same block_id this
            step will overwrite storage before the daemon can
            read it. The worker must restore SYNCHRONOUSLY in
            bind, BEFORE the evict runs, so the read sees
            pre-evict bytes.
          - async-safe: no conflict; push to the restore ring,
            daemon consumer pops + cuFile-reads in parallel with
            the forward pass.

        Conflict is uncommon (only when the hash index has a
        cross-step mapping to a block that's also being re-evicted
        this step), so the async perf win is preserved for the
        typical case."""
        evict_block_ids: set[int] = set()
        for entry in self._sched_evict_queue:
            # Tuple is (req_id, ids, gens) for v4 or
            # (req_id, ids, gens, hashes) for v5; both unpack via [1].
            for b in entry[1]:
                evict_block_ids.add(int(b))
        restore_async, restore_sync = _split_restore_by_conflict(
            self._sched_restore_queue,
            evict_block_ids,
        )
        payload = _encode_meta(
            self._sched_evict_queue,
            restore_async,
            restore_sync,
            self._sched_staging_restore_queue,
        )
        self._sched_evict_queue = []
        self._sched_restore_queue = []
        self._sched_staging_restore_queue = []
        return _GmsGdsMetadata(payload)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Index this request's prompt prefix block-by-block so a
        # future request with a matching prefix can hit. Done first
        # (cheap; in-process) regardless of whether we're keeping
        # blocks pinned for async-save. The record() call returns
        # the per-block generation assigned to this evict; we carry
        # those into the evict queue so the worker can stamp them
        # on the daemon's slots — closing Race #3 for any restore
        # that later hits this hash.
        gen_list: list[int] = []
        if (
            block_ids
            and self._block_size_tokens > 0
            and getattr(request, "prompt_token_ids", None)
        ):
            gen_list = self._prefix_index.record(
                list(request.prompt_token_ids),
                list(block_ids),
                self._block_size_tokens,
                getattr(request, "cache_salt", None),
                self.engine_id,
            )
        # Stash for this step's worker-side eviction. Each entry is
        # parallel arrays: block_ids[i] is tagged with gens[i] (zero
        # for blocks beyond the indexed prefix — those still get
        # spilled but without generation tracking).
        if block_ids:
            ids = [int(b) for b in block_ids]
            gens = list(gen_list) + [0] * max(
                0,
                len(ids) - len(gen_list),
            )
            # When cross-node is enabled, compute per-block content
            # hashes in parallel with ids. _prefix_block_hashes is
            # deterministic over the same (tokens, block_size, salt)
            # triple that record() just hashed; recomputing is a few
            # SHA-256 chains and lets us avoid changing record()'s
            # public signature. Blocks beyond the full-prefix range
            # get empty bytes (no content_hash → not registerable).
            hashes: list[bytes] = []
            if (
                self._cross_node_register
                and self._block_size_tokens > 0
                and getattr(request, "prompt_token_ids", None)
            ):
                pref_hashes = _prefix_block_hashes(
                    list(request.prompt_token_ids),
                    self._block_size_tokens,
                    getattr(request, "cache_salt", None),
                )
                hashes = pref_hashes + [b""] * max(
                    0,
                    len(ids) - len(pref_hashes),
                )
            self._sched_evict_queue.append(
                (str(request.request_id), ids, gens, hashes),
            )
            self._sched_pending.add(str(request.request_id))
            # True → vLLM keeps the blocks pinned until get_finished
            # reports this req_id (worker confirms write landed).
            return (True, None)
        return (False, None)

    def reclaim_cached_kv(self, target_blocks: int) -> int:
        """Best-effort shadow-transition reclaim hook.

        vLLM's GMS connector already mirrors finished request KV in
        request_finished() and reports completion through get_finished(), after
        which vLLM releases those blocks through its own BlockPool. There is no
        additional connector-owned live-HBM cache to free safely here; shadow
        headroom is enforced by the shared GMS lease reservation layer.
        """
        _ = target_blocks
        return 0

    # ------------------------------------------------------------------
    # WORKER role
    # ------------------------------------------------------------------

    def register_kv_caches(
        self,
        kv_caches: "dict[str, torch.Tensor]",
    ) -> None:
        from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector
        from gms_kv_ring.engines.vllm.install_kv_ring import install_for_vllm

        # Best-effort MLA + logical-block-size detection from vLLM config.
        # Falls back to the legacy axis-max heuristic if neither is
        # available, preserving non-MLA behavior unchanged.
        logical_block_size = None
        is_mla = False
        cache_cfg = getattr(self._vllm_config, "cache_config", None)
        if cache_cfg is not None:
            logical_block_size = getattr(cache_cfg, "block_size", None)
        model_cfg = getattr(self._vllm_config, "model_config", None)
        if model_cfg is not None:
            is_mla = bool(getattr(model_cfg, "use_mla", False))
        layers, layout, n_layers, block_bytes = _build_layers_and_layout(
            kv_caches,
            logical_block_size=logical_block_size,
            is_mla=is_mla,
        )
        # Device id from the first tensor — every layer is on the
        # same device by construction (vLLM places kv per worker).
        # In a TP=N deployment, rank K's worker process runs only
        # this device-K codepath; it talks ONLY to device K's
        # daemon (per-GPU-UUID socket via get_socket_path). No
        # rank disambiguation is needed because the daemon binds
        # one-to-one with this GPU — cross-rank collisions are
        # impossible by construction.
        first = next(iter(kv_caches.values()))
        device = int(first.device.index)
        socket = _resolve_daemon_socket(self._vllm_config, device)

        # Ring capacities are tunable via env var; defaults match
        # install_for_vllm. Sustained-eviction workloads with high
        # cache-hit rates may push the restore ring; bump
        # GMS_KVR_RESTORE_RING_CAPACITY to a higher power-of-2 if
        # you see `connector_restore_ring_full` increasing in
        # production. Same for evict ring under heavy spill.
        evict_cap = _power_of_2_env(
            "GMS_KVR_EVICT_RING_CAPACITY",
            default=4096,
        )
        restore_cap = _power_of_2_env(
            "GMS_KVR_RESTORE_RING_CAPACITY",
            default=4096,
        )
        self._handle = install_for_vllm(
            engine_id=self.engine_id,
            daemon_socket=socket,
            layers=layers,
            evict_ring_capacity=evict_cap,
            restore_ring_capacity=restore_cap,
        )
        self._gds_conn = VllmGdsConnector(self._handle, layout)
        self._block_layout = layout
        self._n_layers = n_layers
        self._block_bytes = block_bytes
        if not self._gds_conn.is_available():
            logger.info(
                "[GMS GDS Connector] daemon backend lacks GPU-direct "
                "support (engine=%s socket=%s) — connector will be a "
                "pass-through; evictions are reported finished but "
                "no spill to storage occurs.",
                self.engine_id,
                socket,
            )
        else:
            logger.info(
                "[GMS GDS Connector] registered: engine=%s socket=%s "
                "n_layers=%d block_bytes=%d",
                self.engine_id,
                socket,
                n_layers,
                block_bytes,
            )

    def _can_use_host_tier(self) -> bool:
        return (
            self._host_tier_fallback
            and self._gds_conn is not None
            and hasattr(self._gds_conn, "evict_blocks_to_host")
            and hasattr(self._gds_conn, "restore_blocks_remap_from_host")
        )

    def _should_restore_from_host(self, available: bool, eid: str) -> bool:
        return self._can_use_host_tier() and (
            not available or str(self._source_engine_id) != str(eid)
        )

    def _restore_local_triples_sync(self, triples, *, available: bool, eid: str):
        if self._should_restore_from_host(available, eid):
            return self._gds_conn.restore_blocks_remap_from_host(
                self._source_engine_id,
                triples,
            )
        if available:
            return self._gds_conn.restore_blocks_remap(triples)
        return None

    def _should_evict_to_host(self, available: bool) -> bool:
        return self._can_use_host_tier() and (self._host_tier_mirror or not available)

    def bind_connector_metadata(self, connector_metadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        meta = connector_metadata
        payload = getattr(meta, "payload", None)
        if payload is None:
            return
        (
            evict_queue,
            restore_async,
            restore_sync,
            restore_staging_async,
        ) = _decode_meta(payload)
        if not (evict_queue or restore_async or restore_sync or restore_staging_async):
            return
        if self._gds_conn is None:
            # register_kv_caches hasn't been called yet — drop and
            # report finished so vLLM frees blocks instead of hanging.
            for entry in evict_queue:
                self._worker_finished_saves.add(entry[0])
            return
        from gms_kv_ring.common import metrics

        eid = self._handle.engine_id if self._handle else self.engine_id
        available = self._gds_conn.is_available()

        # ORDERING IS LOAD-BEARING.
        #
        # The original sync-everything path drained `restore`
        # BEFORE `evict` so a within-step conflict (an evict for
        # the same block_id a restore was reading) would still see
        # pre-evict storage. With async restore that argument breaks
        # because the daemon consumer doesn't pop the ring record
        # until LATER — by then evict has already overwritten
        # storage. The scheduler partitions restores into:
        #
        #   restore_sync : src_block_id ∈ this step's evict set →
        #                  MUST run synchronously, before evict,
        #                  same correctness as the original reorder.
        #   restore_async: no conflict → push to ring, daemon
        #                  consumer overlaps cuFile read with forward.
        #
        # Within bind: drain sync restores → push async restores →
        # drain evicts. start_load_kv issues cuStreamWaitValue32
        # for the async ones on the compute stream.

        # --- 1. SYNC restores (conflicting with this step's evict) ---
        # `triples` are (src, dst, expected_generation). The
        # sync path passes them through restore_blocks_remap which
        # threads `expected_generation` into the daemon RPC for
        # Race #3 enforcement.
        for req_id, triples in restore_sync:
            metrics.connector_restore_conflict_sync.inc(
                engine_id=eid,
                n=len(triples),
            )
            try:
                out = self._restore_local_triples_sync(
                    triples,
                    available=available,
                    eid=eid,
                )
                if out is None:
                    continue
                fails = [d for d, ok in out.items() if not ok]
                if fails:
                    metrics.connector_restore_failures.inc(
                        engine_id=eid,
                        n=len(fails),
                    )
                    logger.warning(
                        "[GMS GDS Connector] partial sync-restore "
                        "for req=%s: %d of %d blocks failed "
                        "(dst_ids=%s)",
                        req_id,
                        len(fails),
                        len(out),
                        fails,
                    )
            except Exception:  # noqa: BLE001
                metrics.connector_restore_failures.inc(
                    engine_id=eid,
                    n=len(triples),
                )
                logger.warning(
                    "[GMS GDS Connector] sync-restore raised for "
                    "req=%s triple_count=%d",
                    req_id,
                    len(triples),
                    exc_info=True,
                )

        # --- 2. ASYNC restores (push to ring; daemon pops later) ---
        for req_id, triples in restore_async:
            if self._should_restore_from_host(available, eid):
                try:
                    out = self._gds_conn.restore_blocks_remap_from_host(
                        self._source_engine_id,
                        triples,
                    )
                    fails = [d for d, ok in out.items() if not ok]
                    if fails:
                        metrics.connector_restore_failures.inc(
                            engine_id=eid,
                            n=len(fails),
                        )
                        logger.warning(
                            "[GMS GDS Connector] partial host-tier "
                            "restore req=%s: %d of %d blocks failed",
                            req_id,
                            len(fails),
                            len(out),
                        )
                except Exception:  # noqa: BLE001
                    metrics.connector_restore_failures.inc(
                        engine_id=eid,
                        n=len(triples),
                    )
                    logger.warning(
                        "[GMS GDS Connector] host-tier restore "
                        "raised for req=%s triple_count=%d",
                        req_id,
                        len(triples),
                        exc_info=True,
                    )
                continue
            if not available:
                continue
            slot_target = None
            if self._async_restore and hasattr(
                self._handle,
                "record_restore_gds",
            ):
                try:
                    # Async ring's binary record format doesn't (yet)
                    # carry per-pair generations; ring path is
                    # therefore Race-#3-protected only when the
                    # consumer can derive the generation from the
                    # daemon's slot state itself, which it CAN —
                    # it reads pool.current_block_generation(src)
                    # and compares against the SCHEDULER's view
                    # captured at queue time. For correctness we
                    # downgrade triples whose src is being evicted
                    # this step to the sync lane (already done via
                    # _split_restore_by_conflict). The remaining
                    # async pairs only race with FUTURE step evicts;
                    # the daemon's per-block monotonic generation
                    # catches that.
                    slot_target = self._handle.record_restore_gds(
                        eid,
                        [(int(s), int(d)) for s, d, _g in triples],
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[GMS GDS Connector] async restore push "
                        "raised for req=%s — falling back to sync",
                        req_id,
                        exc_info=True,
                    )
                    slot_target = None
            if slot_target is not None:
                # Tag with req_id so the post-forward host-side
                # check can attribute failures to specific requests.
                # Also carry the src_block_ids being restored — on
                # detected failure, clear_connector_metadata uses
                # these to drop the matching (engine_id, block_id)
                # entries from `_PrefixIndex`, preventing the next
                # request from re-claiming the same broken slots
                # and re-failing.
                src_bids = [int(s) for s, _d, _g in triples]
                self._pending_restore_waits.append(
                    (req_id, slot_target[0], slot_target[1], src_bids),
                )
                continue
            # Fall through: ring full / opted out / legacy handle.
            metrics.connector_restore_ring_full.inc(engine_id=eid)
            try:
                out = self._gds_conn.restore_blocks_remap(triples)
                fails = [d for d, ok in out.items() if not ok]
                if fails:
                    metrics.connector_restore_failures.inc(
                        engine_id=eid,
                        n=len(fails),
                    )
                    logger.warning(
                        "[GMS GDS Connector] partial async-fallback "
                        "restore req=%s: %d of %d blocks failed",
                        req_id,
                        len(fails),
                        len(out),
                    )
            except Exception:  # noqa: BLE001
                metrics.connector_restore_failures.inc(
                    engine_id=eid,
                    n=len(triples),
                )
                logger.warning(
                    "[GMS GDS Connector] async-fallback restore "
                    "raised for req=%s triple_count=%d",
                    req_id,
                    len(triples),
                    exc_info=True,
                )

        # --- 3. STAGING restores (destination daemon already has bytes) ---
        for req_id, triples in restore_staging_async:
            if self._handle is None or not hasattr(
                self._handle,
                "record_restore_staging",
            ):
                metrics.connector_restore_failures.inc(
                    engine_id=eid,
                    n=len(triples),
                )
                continue
            try:
                slot_target = self._handle.record_restore_staging(triples)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMS GDS Connector] staging restore push raised "
                    "for req=%s triple_count=%d",
                    req_id,
                    len(triples),
                    exc_info=True,
                )
                slot_target = None
            if slot_target is None:
                metrics.connector_restore_failures.inc(
                    engine_id=eid,
                    n=len(triples),
                )
                logger.warning(
                    "[GMS GDS Connector] staging restore unavailable "
                    "for req=%s triple_count=%d; request will fall "
                    "back through restore failure handling",
                    req_id,
                    len(triples),
                )
                continue
            self._pending_restore_waits.append(
                (req_id, slot_target[0], slot_target[1], []),
            )

        # --- 3. EVICTS ---
        # Pass per-block generations into the connector so each
        # per-layer demote_hbm_to_storage RPC stamps the daemon's
        # slot_generations table with the right value. A subsequent
        # restore-on-hit's expected_generation will then match.
        # Aggregate cross-node hash registrations across all evict
        # entries in this step into a single batched RPC at the end
        # (one RPC per bind instead of one per request).
        cross_node_batch: list[dict] = []
        for entry in evict_queue:
            # Tolerate v4 (no hashes) and v5 (hashes) shapes.
            if len(entry) == 4:
                req_id, block_ids, gens, block_hashes = entry
            else:
                req_id, block_ids, gens = entry
                block_hashes = []
            failed_ids: set[int] = set()
            if available or self._should_evict_to_host(available):
                gens_map = {
                    int(b): int(g) for b, g in zip(block_ids, gens) if int(g) != 0
                }
                try:
                    if self._should_evict_to_host(available):
                        res = self._gds_conn.evict_blocks_to_host(
                            block_ids,
                            generations=gens_map if gens_map else None,
                        )
                    else:
                        res = self._gds_conn.evict_blocks_to_storage(
                            block_ids,
                            generations=gens_map if gens_map else None,
                        )
                    if res.failed:
                        metrics.connector_evict_failures.inc(
                            engine_id=eid,
                            n=res.failed,
                        )
                        logger.warning(
                            "[GMS GDS Connector] partial spill for "
                            "req=%s: %d of %d blocks failed (ids=%s)",
                            req_id,
                            res.failed,
                            res.failed + res.succeeded,
                            res.failed_ids,
                        )
                        failed_ids = {int(b) for b in res.failed_ids}
                except Exception:  # noqa: BLE001
                    metrics.connector_evict_failures.inc(
                        engine_id=eid,
                        n=len(block_ids),
                    )
                    logger.warning(
                        "[GMS GDS Connector] spill raised for req=%s "
                        "block_count=%d; reporting finished anyway to "
                        "avoid block-pin deadlock",
                        req_id,
                        len(block_ids),
                        exc_info=True,
                    )
                    failed_ids = {int(b) for b in block_ids}
            # Build cross-node registrations for blocks that (a) have
            # a non-empty content_hash (only the indexed prefix does),
            # (b) didn't fail the spill, and (c) actually spilled
            # (we don't register addresses the daemon doesn't have).
            # Skip entirely when the daemon has no transport.
            if (
                self._cross_node_xfer_enabled
                and available
                and block_hashes
                and self._block_layout is not None
            ):
                for bid, h in zip(block_ids, block_hashes):
                    if not h or int(bid) in failed_ids:
                        continue
                    cross_node_batch.append(
                        {
                            "content_hash": h,
                            "engine_id": eid,
                            "ranges": self._block_layout(int(bid)),
                        }
                    )
            self._worker_finished_saves.add(req_id)

        if cross_node_batch and self._handle is not None:
            try:
                _total, skipped = self._handle._client.register_content_addresses_batch(
                    cross_node_batch,
                )
                if skipped:
                    # Daemon has no transport — disable for the
                    # remainder of this connector's lifetime so we
                    # don't pay the RPC cost on every bind.
                    self._cross_node_xfer_enabled = False
            except Exception:  # noqa: BLE001
                # Best-effort registration. A failure here doesn't
                # corrupt the local cache — at worst, a peer router
                # can't issue cross-node transfers for these hashes
                # until the next request_finished refreshes them.
                logger.warning(
                    "[GMS GDS Connector] cross-node batch "
                    "registration raised (size=%d) — disabling",
                    len(cross_node_batch),
                    exc_info=True,
                )
                self._cross_node_xfer_enabled = False

    def clear_connector_metadata(self) -> None:
        """vLLM calls this AFTER the forward pass for the step. We
        use the hook to do the host-side restore_succeeded() check
        on every async restore we queued in bind. The compute stream
        has run past the cuStreamWaitValue32 sentinels by this point
        (model output produced), so the counter values are settled.

        On any failure we bump `connector_restore_failures`. We do
        NOT retry — vLLM's V1 connector protocol has no in-flight
        retry hook, and the request has already consumed (possibly
        wrong) HBM bytes. The metric is the alert path; ops should
        investigate sustained increases."""
        if self._pending_restore_checks and self._handle is not None:
            from gms_kv_ring.common import metrics

            eid = self._handle.engine_id
            failed = 0
            # Aggregate failed src_block_ids across all failed
            # restores in this step. We drop them from _PrefixIndex
            # in one pass at the end — subsequent requests can't
            # re-claim the same broken slots and re-fail, which would
            # otherwise multiply the wrong-output blast radius from
            # one corrupted slot to every cache-hit that crosses it.
            failed_src_bids: list[int] = []
            for entry in self._pending_restore_checks:
                # Tolerate 3-tuple legacy entries (no src_bids).
                if len(entry) == 4:
                    req_id, slot, target, src_bids = entry
                else:
                    req_id, slot, target = entry
                    src_bids = []
                try:
                    if not self._handle.restore_succeeded(slot, target):
                        failed += 1
                        failed_src_bids.extend(src_bids)
                        logger.warning(
                            "[GMS GDS Connector] async restore "
                            "failed for req=%s slot=%d target=%d "
                            "src_blocks=%r — engine consumed wrong "
                            "HBM bytes for these blocks; dropping "
                            "them from the prefix index to bound "
                            "the blast radius to this request.",
                            req_id,
                            slot,
                            target,
                            src_bids,
                        )
                except Exception:  # noqa: BLE001
                    failed += 1
                    failed_src_bids.extend(src_bids)
                    logger.warning(
                        "[GMS GDS Connector] restore_succeeded "
                        "raised slot=%d target=%d",
                        slot,
                        target,
                        exc_info=True,
                    )
            if failed:
                metrics.connector_restore_failures.inc(
                    engine_id=eid,
                    n=failed,
                )
            if failed_src_bids:
                # Invariant I-5 (see docs/ARCHITECTURE.md): drop-on-
                # failure invalidates the PrefixIndex entries pointing
                # at the failed slot before the next scheduler step.
                n_dropped = self._prefix_index.drop_slots(
                    str(eid),
                    failed_src_bids,
                )
                if n_dropped:
                    metrics.connector_prefix_invalidated_on_failure.inc(
                        engine_id=str(eid),
                        n=n_dropped,
                    )
        self._pending_restore_checks = []
        super().clear_connector_metadata()

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs,
    ) -> None:
        """Drain the per-bind pending-restore-wait list by queueing
        cuStreamWaitValue32 on the current compute stream. Each
        wait gates downstream kernels until the daemon's restore
        consumer signals the counter (= cuFile read into HBM has
        committed).

        This runs ONCE per forward pass. Synchronous-fallback
        restores from bind_connector_metadata don't appear here —
        their bytes are already in HBM by the time bind returned.

        The wait is FREE on the CPU (cuStreamWaitValue32 is a
        GPU-stream op) so this method returns immediately; the
        forward pass kicks off; attention kernels block only when
        they reach the wait sentinel in the stream. That's the
        overlap that makes async restore a perf win.

        Pending-wait entries move from `_pending_restore_waits` to
        `_pending_restore_checks` so clear_connector_metadata can
        host-check them after the forward."""
        if not self._pending_restore_waits:
            return
        if self._handle is None:
            self._pending_restore_waits = []
            return
        try:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        except Exception:  # noqa: BLE001
            logger.warning(
                "[GMS GDS Connector] start_load_kv: could not get "
                "current CUDA stream — dropping restore waits, "
                "engine may read stale HBM",
                exc_info=True,
            )
            self._pending_restore_waits = []
            return
        for entry in self._pending_restore_waits:
            # 4-tuple shape carries src_block_ids for drop-on-failure.
            if len(entry) == 4:
                req_id, slot, target, _src_bids = entry
            else:
                req_id, slot, target = entry
            try:
                self._handle.wait_restore(stream, slot, target)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMS GDS Connector] wait_restore raised "
                    "req=%s slot=%d target=%d; engine may read "
                    "stale HBM",
                    req_id,
                    slot,
                    target,
                    exc_info=True,
                )
        # Hand off to the post-forward host-side check.
        self._pending_restore_checks = list(self._pending_restore_waits)
        self._pending_restore_waits = []

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata,
        **kwargs,
    ) -> None:
        # We spill block-batched in bind_connector_metadata, not per
        # layer per step — KVBM-style per-layer save would duplicate
        # work and miss the per-block CRC framing. Intentional no-op.
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # We only do "saves" (evictions). Loads aren't wired yet.
        done = self._worker_finished_saves
        self._worker_finished_saves = set()
        return (done if done else None, None)


_registered = False


def register_gms_gds_connector() -> None:
    """Register `GMSKVCacheConnectorV1` with vLLM's KVConnectorFactory
    under the short name `GMSKVCacheConnectorV1`. Idempotent; safe to
    call from module-import side effects.

    After this runs, users can opt in via:
        --kv-transfer-config '{"kv_connector": "GMSKVCacheConnectorV1",
                                "engine_id": "0"}'
    instead of spelling out the full module path."""
    global _registered
    if _registered:
        return
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    try:
        KVConnectorFactory.register_connector(
            "GMSKVCacheConnectorV1",
            "gpu_memory_service.integrations.vllm.gds_connector_v1",
            "GMSKVCacheConnectorV1",
        )
    except ValueError:
        # Already registered (e.g. worker reloads, or test reruns).
        pass
    _registered = True
    logger.info(
        "[GMS GDS Connector] registered with vLLM "
        "KVConnectorFactory as 'GMSKVCacheConnectorV1'"
    )
