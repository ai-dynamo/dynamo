# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMSRadixCache — SGLang RadixCache subclass with daemon-storage tier.

ARCHITECTURE
============

SGLang's `RadixCache` is a token-sequence radix tree. Nodes hold
`value: torch.Tensor[int64]` — flat KV indices into the
`token_to_kv_pool_allocator.pool`. On eviction (LRU pressure),
`RadixCache.evict()` frees `node.value` to the allocator and
DELETES the leaf — the tree forgets the prefix existed.

For GMS daemon-storage integration we want the opposite: when
SGLang would delete a leaf, we instead SPILL its bytes to the
daemon and KEEP the tree node with `value=None` plus a marker
recording which daemon block IDs hold the bytes. A later
`match_prefix()` that walks into that node restores the bytes
from the daemon back into freshly-allocated KV slots and the
request proceeds normally.

The pattern mirrors `HiRadixCache` (which adds a CPU tier) but
the "tier" here is the gms_kv_ring daemon — engine-agnostic
storage backed by NIXL+cuFile in production, local FS in dev.

DESIGN CHOICE: block_id == SGLang slot index
--------------------------------------------

`BlockGdsConnector` takes `block_id` as an opaque identifier
and uses `block_layout_fn(block_id) → [(layer, byte_offset,
byte_size), ...]` to find the HBM byte ranges to read/write.

We use SGLang's flat KV-slot indices (the values inside
`node.value`) directly as block_ids. That makes
`block_layout_fn` a pure function of `slot_idx → byte_ranges`,
and the daemon stores data keyed by `(engine_id, slot_idx)`.

Restore allocates FRESH slot indices; the connector's
`restore_blocks_remap([(src=old_slot, dst=new_slot, gen), ...])`
reads from `(engine_id, old_slot)` and writes via
`block_layout_fn(new_slot)` to the new HBM location.

INTEGRATION POINTS OVERRIDDEN
-----------------------------

  - `evict(EvictParams)`: redirect eviction through daemon
    spill; retain leaf with `value=None`.
  - `match_prefix(MatchPrefixParams)`: detect spilled nodes
    along the matched path, restore each from daemon, rebuild
    the MatchResult.

PHASE 1 SCOPE
-------------

  - Happy-path evict + restore byte-correctness.
  - Unit tests with mocked allocator/pool shapes.
  - No Phase A-D analogs yet — Phase 2.
  - No real-engine validation — Phase 3.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


def _import_sglang():
    """Lazy import — keeps this module loadable without SGLang.
    Returns (RadixCache, MatchResult, new_api_symbols_or_None).

    Two SGLang API generations are supported:
      - **Latest upstream main**: `evict(EvictParams)` /
        `match_prefix(MatchPrefixParams)` / `insert(InsertParams)`
        return structured result types. Detected by importing
        `EvictParams` from `cache_init_params`.
      - **Released 0.5.x**: `evict(num_tokens: int)` /
        `match_prefix(key, **kwargs)` / `insert(key, value=...,
        ...)`. Simpler signatures, evict returns None.

    The subclass dispatches to whichever API is present. Code
    targets the latest API; the 0.5.x path is a fallback so
    tests can run against the installed package until it is upgraded."""
    from sglang.srt.mem_cache.radix_cache import RadixCache

    # MatchResult may live in base_prefix_cache (older) or
    # radix_cache (newer). Try both.
    try:
        from sglang.srt.mem_cache.radix_cache import MatchResult
    except ImportError:
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult
    new_api = None
    # Try both module locations — upstream main moved these
    # symbols from `cache_init_params` to `base_prefix_cache`.
    for mod_path in (
        "sglang.srt.mem_cache.base_prefix_cache",
        "sglang.srt.mem_cache.cache_init_params",
    ):
        try:
            mod = __import__(
                mod_path,
                fromlist=[
                    "EvictParams",
                    "EvictResult",
                    "MatchPrefixParams",
                ],
            )
            new_api = (
                mod.EvictParams,
                mod.EvictResult,
                mod.MatchPrefixParams,
            )
            break
        except (ImportError, AttributeError):
            continue
    return RadixCache, MatchResult, new_api


# Layout callback signature: given a `slot_idx` (an int from
# `node.value`), return per-layer HBM byte ranges. Used by
# `BlockGdsConnector` to map slot → bytes.
BlockLayoutFn = Callable[[int], "list[tuple[int, int, int]]"]


@dataclass
class _SpillInfo:
    """Per-node spill record. Holds the original slot indices
    (used as block IDs in the daemon) so restore can ask for
    them by key."""

    # The original SGLang slot indices that hold the data on the
    # daemon side. Daemon storage key = (engine_id, slot_idx).
    block_ids: list[int]
    # Generations parallel to block_ids, captured at spill time.
    # Zero today (Phase 1); Phase 2 wires real generations.
    generations: list[int]
    # Length of the original value tensor — used to allocate the
    # right number of fresh slots on restore.
    value_len: int
    # Cumulative prefix tokens from root → this node. Populated at
    # spill time ONLY when prefix persistence OR cross-node is
    # enabled (cross-node also walks the chain). Reused by the
    # background snapshot thread so it never has to touch the live
    # radix tree under lock. None when neither feature is enabled.
    prefix_tokens: Optional[list] = None


# ----------------------------------------------------------------------
# Snapshot file format (warm-restart persistence).
# Layout:
#   magic   (8 bytes)  : b"GMSSGLP\x00"
#   version (2 bytes)  : uint16 LE
#   epoch   (8 bytes)  : int64 LE (daemon epoch when written; -1 if
#                       no epoch client). On load, mismatch ⇒ discard.
#   pkl_len (4 bytes)  : uint32 LE
#   pkl     (pkl_len)  : pickle of `{"entries": [{...}, ...]}`
#   crc32   (4 bytes)  : zlib.crc32 over pkl
#
# Each entry: {"prefix_tokens": list[int], "block_ids": list[int],
#              "generations": list[int]}.
# ----------------------------------------------------------------------
_SNAPSHOT_MAGIC = b"GMSSGLP\x00"
_SNAPSHOT_VERSION = 1
_SNAPSHOT_HEADER_FMT = "<8sHqI"
_SNAPSHOT_HEADER_LEN = struct.calcsize(_SNAPSHOT_HEADER_FMT)


def make_gms_radix_cache_class():
    """Build the actual `GMSRadixCache(RadixCache)` subclass.
    Called once at install time, after SGLang is importable."""
    _RadixCache, _MatchResult, _new_api = _import_sglang()
    import torch  # noqa: F401 — needed by methods below

    if _new_api is not None:
        _EvictParams, _EvictResult, _MatchPrefixParams = _new_api
    else:
        _EvictParams = _EvictResult = _MatchPrefixParams = None

    class _GMSRadixCache(_RadixCache):
        _DEFAULT_MAX_SPILLED_NODES = 100_000

        def __init__(
            self,
            params: "CacheInitParams",
            *,
            gds: Optional["BlockGdsConnector"] = None,
            block_layout_fn: Optional[BlockLayoutFn] = None,
            engine_id: Optional[str] = None,
            max_spilled_nodes: Optional[int] = None,
            daemon_socket: Optional[str] = None,
        ) -> None:
            """Args:
            params: passed to RadixCache.__init__ unchanged.
            gds: a `BlockGdsConnector`. If None or
                 `gds.is_available()=False`, this class
                 behaves identically to RadixCache.
            block_layout_fn: maps slot_idx → per-layer byte
                 ranges. Required iff `gds` is active.
            engine_id: logging label.
            max_spilled_nodes: cap on the spill table.
            daemon_socket: unix-socket path. When set, enables
                 Phase A — a background thread polls the
                 daemon's epoch and invalidates all spilled
                 state on observed change (daemon restart).
                 None disables epoch polling.
            """
            super().__init__(params)
            self._gds = gds
            self._block_layout_fn = block_layout_fn
            self._engine_id = str(engine_id) if engine_id else "0"
            self._source_engine_id = (
                os.environ.get("GMS_SGLANG_SOURCE_ENGINE_ID")
                or os.environ.get("GMS_KVR_SOURCE_ENGINE_ID")
                or self._engine_id
            )
            self._host_tier_fallback = os.environ.get(
                "GMS_KVR_HOST_TIER_FALLBACK", "0"
            ).lower() not in ("0", "false", "no", "off", "")
            self._host_tier_mirror = os.environ.get(
                "GMS_KVR_HOST_TIER_MIRROR", "0"
            ).lower() not in ("0", "false", "no", "off", "")
            if max_spilled_nodes is None:
                try:
                    max_spilled_nodes = int(
                        os.environ.get(
                            "GMS_SGLANG_MAX_SPILLED_NODES",
                            self._DEFAULT_MAX_SPILLED_NODES,
                        ),
                    )
                except ValueError:
                    max_spilled_nodes = self._DEFAULT_MAX_SPILLED_NODES
            self._max_spilled = max(1, int(max_spilled_nodes))
            # Per-node spill state keyed by id(TreeNode). The
            # tree itself bounds live-node count; this dict caps
            # spilled-node count separately.
            self._spilled: "dict[int, _SpillInfo]" = {}
            # Phase A — daemon-epoch invalidation. Background
            # thread polls the daemon on a slow cadence; on
            # epoch change we drop every spill record because
            # the daemon's storage tier has been zeroed. No
            # impact on the SGLang scheduler hot path — all
            # work happens on a daemon-only thread.
            self._daemon_socket = daemon_socket
            self._daemon_epoch_seen: Optional[int] = None
            self._epoch_thread: Optional[threading.Thread] = None
            self._epoch_stop = threading.Event()
            self._epoch_client = None
            self._staging_scan_client = None
            try:
                self._epoch_poll_s = max(
                    0.5,
                    float(
                        os.environ.get(
                            "GMS_SGLANG_EPOCH_POLL_S",
                            "5.0",
                        )
                    ),
                )
            except ValueError:
                self._epoch_poll_s = 5.0
            if self._daemon_socket:
                self._epoch_thread = threading.Thread(
                    target=self._epoch_loop,
                    name=f"gms-sglang-epoch-{self._engine_id}",
                    daemon=True,
                )
                self._epoch_thread.start()
            # Cross-node hash registration (P4d parity with vLLM).
            # When enabled, _spill_node computes per-page content
            # hashes by walking from the node up to the root, then
            # batch-registers (hash, engine_id, ranges) with the
            # daemon so a peer can `fetch_remote` (or NIXL-read
            # directly) these hashes. Default OFF (single-node
            # deployments pay nothing).
            self._cross_node_register = os.environ.get(
                "GMS_KVR_CROSS_NODE",
                "0",
            ).lower() not in ("0", "false", "no", "off", "")
            # Optional cross-engine cache salt — must match what the
            # peer engine uses so hashes line up. Default empty
            # (matches vLLM's `cache_salt=None` path).
            self._cross_node_salt = os.environ.get(
                "GMS_KVR_CROSS_NODE_SALT",
                "",
            )
            self._cross_node_staging_enabled = self._cross_node_register

            # ---------- Warm-restart persistence (SGL-PERSIST) ----------
            # Background snapshot of the spilled-node table so a
            # restarted engine can rehydrate the radix tree from
            # daemon-resident bytes without first servicing requests
            # to re-warm. Opt-in via env. Zero hot-path overhead
            # by design:
            #   - prefix_tokens captured at spill time (reusing the
            #     walk that cross-node already does)
            #   - snapshot serialization runs on a background thread
            #     with only microseconds of lock-held time per cycle
            #   - match/insert paths are unchanged
            self._persist_enabled = os.environ.get(
                "GMS_SGLANG_PERSIST_PREFIX",
                "0",
            ).lower() not in ("0", "false", "no", "off", "")
            self._persist_path = os.environ.get(
                "GMS_SGLANG_PERSIST_PATH",
                "",
            ).strip()
            try:
                self._persist_interval_s = max(
                    1.0,
                    float(
                        os.environ.get(
                            "GMS_SGLANG_SNAPSHOT_INTERVAL_S",
                            "30.0",
                        )
                    ),
                )
            except ValueError:
                self._persist_interval_s = 30.0
            # Cap on persisted prefix length per entry (in tokens).
            # Above this, the spill record is kept in memory but
            # NOT persisted — limits worst-case snapshot file size.
            try:
                self._persist_max_prefix_tokens = max(
                    1,
                    int(
                        os.environ.get(
                            "GMS_SGLANG_PERSIST_MAX_PREFIX_TOKENS",
                            "8192",
                        )
                    ),
                )
            except ValueError:
                self._persist_max_prefix_tokens = 8192
            # Lock around `_spilled` mutations to give the snapshot
            # thread a consistent read view. Scheduler-side ops grab
            # it for microseconds.
            self._spill_lock = threading.Lock()
            # Race #3 closure (parity with vLLM): per-slot
            # monotonic generation counter. Bumped on each spill;
            # the daemon stores the new generation alongside the
            # bytes and refuses a restore whose `expected_generation`
            # doesn't match. This catches cross-step async restore
            # vs same-slot re-evict — without it, a stale async
            # restore could read post-overwrite bytes.
            self._slot_generations: dict[int, int] = {}
            self._slot_gen_lock = threading.Lock()
            self._persist_dirty = False
            self._persist_thread: Optional[threading.Thread] = None
            self._persist_stop = threading.Event()

            # On startup, attempt to load any existing snapshot. The
            # snapshot is epoch-tagged: a mismatch silently discards
            # so we cold-start cleanly when the daemon was replaced
            # between the previous engine run and this one.
            if self._persist_enabled and self._persist_path:
                try:
                    n_loaded = self._load_snapshot()
                    if n_loaded > 0:
                        logger.info(
                            "[GMSRadixCache eng=%s] loaded %d "
                            "spilled-prefix entries from %s",
                            self._engine_id,
                            n_loaded,
                            self._persist_path,
                        )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[GMSRadixCache eng=%s] failed to load "
                        "snapshot %s; cold-starting",
                        self._engine_id,
                        self._persist_path,
                        exc_info=True,
                    )

            # Start the background snapshot thread last so the load
            # path above isn't racing against it.
            if self._persist_enabled and self._persist_path:
                self._persist_thread = threading.Thread(
                    target=self._snapshot_loop,
                    name=f"gms-sglang-snap-{self._engine_id}",
                    daemon=True,
                )
                self._persist_thread.start()

        # --------------------------------------------------------
        # State helpers
        # --------------------------------------------------------

        def _gms_child_key(self, key):
            """SGLang moved child-key logic from RadixKey to
            RadixCache helper functions. Prefer the instance helper
            when present and keep the older API as fallback."""
            get_child_key = getattr(self, "get_child_key_fn", None)
            if get_child_key is not None:
                return get_child_key(key)
            return key.child_key(self.page_size)

        def _gms_key_match(self, node_key, lookup_key) -> int:
            key_match = getattr(self, "key_match_fn", None)
            if key_match is not None:
                return int(key_match(node_key, lookup_key))
            return int(node_key.match(lookup_key, page_size=self.page_size))

        def _is_active(self) -> bool:
            """True iff spill/restore should engage. False
            collapses to vanilla RadixCache behavior."""
            if self._gds is None or self._block_layout_fn is None:
                return False
            if self._gds.is_available():
                return True
            return (
                self._host_tier_fallback
                and hasattr(self._gds, "evict_blocks_to_host")
                and hasattr(self._gds, "restore_blocks_remap_from_host")
            )

        def _cap_spilled(self) -> None:
            """Cap the spill table by oldest-first eviction."""
            while len(self._spilled) > self._max_spilled:
                oldest = next(iter(self._spilled))
                self._spilled.pop(oldest)

        @staticmethod
        def _value_to_int_list(value) -> "list[int]":
            """Convert a `node.value` (torch.Tensor[int64] or
            list) to a list of Python ints."""
            if hasattr(value, "tolist"):
                return [int(v) for v in value.tolist()]
            return [int(v) for v in value]

        @staticmethod
        def _key_to_int_list(key) -> "list[int]":
            """Convert a TreeNode `key` (SGLang token-id container)
            to a flat list of Python ints. SGLang's key API has
            evolved over versions — `.token_ids`, `.ids`,
            `tolist()`, or plain-iterable all show up. Defensive
            extraction so the cross-node hook is version-agnostic."""
            for attr in ("token_ids", "ids"):
                v = getattr(key, attr, None)
                if v is not None:
                    return [int(t) for t in v]
            if hasattr(key, "tolist"):
                return [int(t) for t in key.tolist()]
            try:
                return [int(t) for t in key]
            except TypeError:
                return []

        def _collect_prefix_tokens(self, node) -> "list[int]":
            """Walk node → root collecting the cumulative prefix
            tokens. The root's key is conventionally empty in
            SGLang; we skip nodes with no key to be robust against
            that. Returns the full token sequence ending at this
            node's last token."""
            chain: list = []
            n = node
            # Guard against infinite loops on unexpected tree shapes
            # (cycles or detached subtrees) by bounding the walk
            # depth at a generous limit.
            for _ in range(10_000):
                if n is None:
                    break
                k = getattr(n, "key", None)
                if k is not None:
                    chain.append(k)
                parent = getattr(n, "parent", None)
                if parent is n or parent is None:
                    break
                n = parent
            chain.reverse()
            tokens: list[int] = []
            for k in chain:
                tokens.extend(self._key_to_int_list(k))
            return tokens

        def _maybe_register_cross_node(
            self,
            node,
            slot_indices: "list[int]",
            prefix_tokens: "Optional[list[int]]" = None,
        ) -> None:
            """After a successful spill, batch-register per-page
            content hashes with the daemon so a peer router can
            request these bytes via cross-node `transfer_block`.

            One hash per `page_size`-chunk of slot_indices; each
            hash's content is the cumulative prefix sha256 over
            (cache_salt, all-tokens-up-to-and-including this
            chunk's last token) — identical algorithm to vLLM
            so the two engines can share daemon entries.

            No-op when the daemon has no transport (`skipped=True`
            from the RPC); we self-disable to avoid the per-spill
            RPC cost in single-node deployments."""
            if not self._cross_node_register:
                return
            page_size = int(getattr(self, "page_size", 0))
            if page_size <= 0 or len(slot_indices) % page_size != 0:
                # Mis-aligned spill (not page-multiple) — skip
                # rather than ship a hash that won't match what a
                # peer would compute on lookup. Eventual chunks
                # may still align in later spills.
                return
            if prefix_tokens is None:
                try:
                    prefix_tokens = self._collect_prefix_tokens(node)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[GMSRadixCache eng=%s] cross-node prefix walk "
                        "raised — disabling cross-node registration",
                        self._engine_id,
                        exc_info=True,
                    )
                    self._cross_node_register = False
                    return
            if not prefix_tokens:
                return
            try:
                from gms_kv_ring.common.prefix_hashes import prefix_block_hashes

                all_hashes = prefix_block_hashes(
                    prefix_tokens,
                    page_size,
                    self._cross_node_salt,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] prefix-hash compute "
                    "raised — disabling cross-node registration",
                    self._engine_id,
                    exc_info=True,
                )
                self._cross_node_register = False
                return
            n_local = len(slot_indices) // page_size
            if n_local == 0 or len(all_hashes) < n_local:
                return
            local_hashes = all_hashes[-n_local:]
            items: list[dict] = []
            for i, h in enumerate(local_hashes):
                chunk_slots = slot_indices[i * page_size : (i + 1) * page_size]
                ranges = []
                for sl in chunk_slots:
                    ranges.extend(self._block_layout_fn(sl))
                items.append(
                    {
                        "content_hash": h,
                        "engine_id": self._engine_id,
                        "ranges": ranges,
                    }
                )
            if not items:
                return
            client = getattr(getattr(self._gds, "handle", None), "_client", None)
            if client is None:
                return
            try:
                _total, skipped = client.register_content_addresses_batch(
                    items,
                )
                if skipped:
                    # Daemon has no transport — self-disable so we
                    # don't pay the per-spill RPC cost forever.
                    self._cross_node_register = False
                    self._cross_node_staging_enabled = False
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] register_content_"
                    "addresses_batch raised (size=%d) — disabling",
                    self._engine_id,
                    len(items),
                    exc_info=True,
                )
                self._cross_node_register = False
                self._cross_node_staging_enabled = False

        def _staging_scan(self, block_hashes: "list[bytes]") -> "dict[bytes, dict]":
            """Best-effort lookup against the local daemon's staging tier."""
            if not block_hashes or not self._daemon_socket:
                return {}
            try:
                from gms_kv_ring.daemon.client import DaemonClient

                if self._staging_scan_client is None:
                    self._staging_scan_client = DaemonClient(
                        self._daemon_socket,
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
                logger.debug(
                    "[GMSRadixCache eng=%s] staging_scan failed",
                    self._engine_id,
                    exc_info=True,
                )
                return {}

        def _restore_staged_suffix_for_key(self, key, matched_value) -> bool:
            """Restore a contiguous staged suffix and insert it in the tree.

            The SGLang GMS mapping uses one daemon content hash per
            ``page_size`` token slots, while local spill/restore still uses
            individual SGLang slot ids. This method allocates fresh slots for
            staged pages, asks the daemon to copy each staged payload into the
            explicit slot ranges, then inserts the resulting prefix into the
            radix tree so vanilla matching can consume it.
            """
            if not self._cross_node_staging_enabled:
                return False
            page_size = int(getattr(self, "page_size", 0))
            if page_size <= 0:
                return False
            try:
                key, _ = key.maybe_to_bigram_view(self.is_eagle)
                key = key.page_aligned(page_size)
            except Exception:  # noqa: BLE001
                return False
            tokens = self._key_to_int_list(key)
            if not tokens:
                return False
            matched_len = int(len(matched_value)) if matched_value is not None else 0
            if matched_len >= len(tokens) or matched_len % page_size != 0:
                return False
            try:
                from gms_kv_ring.common.prefix_hashes import prefix_block_hashes

                block_hashes = prefix_block_hashes(
                    tokens,
                    page_size,
                    self._cross_node_salt,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GMSRadixCache eng=%s] staged prefix hash " "compute failed",
                    self._engine_id,
                    exc_info=True,
                )
                return False
            start_block = matched_len // page_size
            if start_block >= len(block_hashes):
                return False
            candidate_hashes = block_hashes[start_block:]
            hits = self._staging_scan(candidate_hashes)
            staged: list[tuple[bytes, int]] = []
            for h in candidate_hashes:
                hit = hits.get(h)
                if hit is None:
                    break
                try:
                    generation = int(hit.get("generation", 0))
                except (TypeError, ValueError):
                    break
                staged.append((h, generation))
            if not staged:
                return False

            n_slots = len(staged) * page_size
            try:
                fresh = self.token_to_kv_pool_allocator.alloc(n_slots)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] alloc failed during " "staged restore",
                    self._engine_id,
                    exc_info=True,
                )
                return False
            if fresh is None or len(fresh) < n_slots:
                if fresh is not None:
                    self.token_to_kv_pool_allocator.free(fresh)
                return False

            fresh_list = self._value_to_int_list(fresh)
            range_hits: list[tuple[bytes, int, list[tuple[int, int, int]]]] = []
            try:
                for i, (content_hash, generation) in enumerate(staged):
                    chunk_slots = fresh_list[i * page_size : (i + 1) * page_size]
                    ranges: list[tuple[int, int, int]] = []
                    for slot_idx in chunk_slots:
                        ranges.extend(self._block_layout_fn(int(slot_idx)))
                    range_hits.append((content_hash, generation, ranges))
                handle = getattr(self._gds, "handle", None)
                restore_ranges = getattr(
                    handle,
                    "restore_staging_ranges_sync",
                    None,
                )
                if restore_ranges is None:
                    logger.warning(
                        "[GMSRadixCache eng=%s] staged restore ranges "
                        "not supported by GMS handle",
                        self._engine_id,
                    )
                    self.token_to_kv_pool_allocator.free(fresh)
                    return False
                if not restore_ranges(range_hits):
                    self.token_to_kv_pool_allocator.free(fresh)
                    return False

                import torch

                prefix_len = (start_block + len(staged)) * page_size
                restore_key = key[:prefix_len]
                if matched_len > 0:
                    prefix_value = matched_value[:matched_len]
                    if getattr(prefix_value, "device", None) != getattr(
                        fresh,
                        "device",
                        None,
                    ):
                        prefix_value = prefix_value.to(fresh.device)
                    value = torch.cat([prefix_value, fresh])
                else:
                    value = fresh
                self._insert_helper(self.root_node, restore_key, value, 0, False)
                return True
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] staged restore failed",
                    self._engine_id,
                    exc_info=True,
                )
                try:
                    self.token_to_kv_pool_allocator.free(fresh)
                except Exception:  # noqa: BLE001
                    pass
                return False

        # --------------------------------------------------------
        # Spill / restore
        # --------------------------------------------------------

        def _spill_node(self, node, *, force_host_tier: bool = False) -> bool:
            """Read node.value bytes from HBM into daemon
            storage keyed by the slot indices. Returns True on
            success. On success the spill record is stored and
            the caller MUST free node.value via the allocator
            and set node.value=None to mark the node "evicted
            from device, present in daemon."""
            value = node.value
            if value is None:
                return False
            slot_indices = self._value_to_int_list(value)
            if not slot_indices:
                return False
            # Race #3: bump per-slot generation counter and tag the
            # demote RPC with the new gens. The daemon stores the
            # generation alongside the bytes; a subsequent async
            # restore carrying an OLDER expected_generation will be
            # refused. Prevents cross-step "async restore vs
            # re-evict" reads of post-overwrite bytes. Mirrors
            # vLLM's Race #3 closure (RACE3 task).
            new_gens: list[int] = []
            with self._slot_gen_lock:
                for sl in slot_indices:
                    g = self._slot_generations.get(sl, 0) + 1
                    self._slot_generations[sl] = g
                    new_gens.append(g)
            gens_map = {int(sl): int(g) for sl, g in zip(slot_indices, new_gens)}
            try:
                host_spill = (
                    (force_host_tier or self._host_tier_mirror)
                    and self._host_tier_fallback
                    and hasattr(self._gds, "evict_blocks_to_host")
                )
                if self._gds.is_available() and not host_spill:
                    res = self._gds.evict_blocks_to_storage(
                        slot_indices,
                        generations=gens_map,
                    )
                else:
                    res = self._gds.evict_blocks_to_host(
                        slot_indices,
                        generations=gens_map,
                    )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] KV spill raised; "
                    "vanilla-evicting this node",
                    self._engine_id,
                    exc_info=True,
                )
                return False
            if res.failed:
                logger.warning(
                    "[GMSRadixCache eng=%s] partial-spill: %d/%d "
                    "blocks failed (failed_ids=%r); vanilla-"
                    "evicting this node",
                    self._engine_id,
                    res.failed,
                    res.failed + res.succeeded,
                    list(res.failed_ids)[:8],
                )
                return False
            # If persistence OR cross-node is on we want the
            # cumulative prefix tokens for this spill. Walk once
            # here so neither feature has to walk independently;
            # the work is bounded by tree depth and runs on the
            # already-slow evict path.
            prefix_tokens: Optional[list] = None
            if self._persist_enabled or self._cross_node_register:
                try:
                    prefix_tokens = self._collect_prefix_tokens(node)
                except Exception:  # noqa: BLE001
                    prefix_tokens = None
            with self._spill_lock:
                self._spilled[id(node)] = _SpillInfo(
                    block_ids=slot_indices,
                    generations=new_gens,
                    value_len=len(slot_indices),
                    prefix_tokens=prefix_tokens,
                )
                self._cap_spilled()
                self._persist_dirty = True
            # P4d: cross-node hash registration. Off by default; the
            # call is a no-op when the env knob is unset. When the
            # daemon has no transport the first call returns
            # skipped=True and self-disables. Pre-walked tokens
            # passed to avoid a second walk.
            self._maybe_register_cross_node(
                node,
                slot_indices,
                prefix_tokens=prefix_tokens,
            )
            return True

        # --------------------------------------------------------
        # Warm-restart snapshot (background thread + load)
        # --------------------------------------------------------

        def _serialize_snapshot(
            self,
            entries: "list[dict]",
        ) -> bytes:
            payload = {"entries": entries}
            pkl = pickle.dumps(payload, protocol=4)
            epoch = -1
            if self._daemon_epoch_seen is not None:
                epoch = int(self._daemon_epoch_seen)
            header = struct.pack(
                _SNAPSHOT_HEADER_FMT,
                _SNAPSHOT_MAGIC,
                _SNAPSHOT_VERSION,
                epoch,
                len(pkl),
            )
            crc = struct.pack("<I", zlib.crc32(pkl) & 0xFFFFFFFF)
            return header + pkl + crc

        def _deserialize_snapshot(
            self,
            blob: bytes,
        ) -> "Optional[tuple[int, list[dict]]]":
            """Returns `(daemon_epoch, entries)` or None if the blob
            is malformed / wrong-version / CRC-fails."""
            if len(blob) < _SNAPSHOT_HEADER_LEN + 4:
                return None
            try:
                magic, version, epoch, pkl_len = struct.unpack(
                    _SNAPSHOT_HEADER_FMT,
                    blob[:_SNAPSHOT_HEADER_LEN],
                )
            except struct.error:
                return None
            if magic != _SNAPSHOT_MAGIC:
                return None
            if version != _SNAPSHOT_VERSION:
                logger.warning(
                    "[GMSRadixCache eng=%s] snapshot version=%d, "
                    "this build expects %d; ignoring",
                    self._engine_id,
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
                logger.warning(
                    "[GMSRadixCache eng=%s] snapshot CRC mismatch; " "ignoring",
                    self._engine_id,
                )
                return None
            try:
                payload = pickle.loads(pkl)
            except Exception:  # noqa: BLE001
                return None
            entries = payload.get("entries") or []
            return (epoch, list(entries))

        def _atomic_write(self, path: str, blob: bytes) -> None:
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        def _take_snapshot(self) -> int:
            """Copy `_spilled` under the lock, serialize + write
            atomically OUTSIDE the lock. Returns the number of
            entries written. The lock-held duration is bounded by
            the size of `_spilled` (microseconds at production
            scale)."""
            if not (self._persist_enabled and self._persist_path):
                return 0
            with self._spill_lock:
                if not self._persist_dirty:
                    return 0
                cap = self._persist_max_prefix_tokens
                snap = []
                for info in self._spilled.values():
                    if (
                        info.prefix_tokens is None
                        or len(info.prefix_tokens) == 0
                        or len(info.prefix_tokens) > cap
                    ):
                        continue
                    snap.append(
                        {
                            "prefix_tokens": list(info.prefix_tokens),
                            "block_ids": list(info.block_ids),
                            "generations": list(info.generations),
                        }
                    )
                self._persist_dirty = False
            blob = self._serialize_snapshot(snap)
            try:
                self._atomic_write(self._persist_path, blob)
            except OSError:
                logger.warning(
                    "[GMSRadixCache eng=%s] snapshot write to %s " "failed",
                    self._engine_id,
                    self._persist_path,
                    exc_info=True,
                )
                return 0
            return len(snap)

        def _snapshot_loop(self) -> None:
            while not self._persist_stop.is_set():
                self._persist_stop.wait(self._persist_interval_s)
                if self._persist_stop.is_set():
                    break
                try:
                    n = self._take_snapshot()
                    if n:
                        logger.debug(
                            "[GMSRadixCache eng=%s] snapshot %d " "entries",
                            self._engine_id,
                            n,
                        )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[GMSRadixCache eng=%s] snapshot raised",
                        self._engine_id,
                        exc_info=True,
                    )

        def _insert_spilled_chain(
            self,
            prefix_tokens: list,
            block_ids: list,
            generations: list,
        ) -> bool:
            """Custom insert used only during warm-restart load.
            Creates tree nodes for `prefix_tokens` without
            allocating HBM (value=None) and registers the spill
            record so a future match_prefix triggers a daemon
            restore. Returns True on success.

            Mirrors the structural moves vanilla RadixCache makes
            in `_insert_helper`, minus the `value.clone()` step
            (we have no live tensor)."""
            try:
                # Import lazily — RadixKey lives inside SGLang.
                from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
            except Exception:  # noqa: BLE001
                return False
            key = RadixKey(token_ids=list(prefix_tokens), extra_key=None)
            node = self.root_node
            child_key = self._gms_child_key(key)
            access_time = time.monotonic()
            while len(key) > 0 and child_key in node.children:
                child = node.children[child_key]
                child.last_access_time = access_time
                prefix_len = self._gms_key_match(child.key, key)
                if prefix_len < len(child.key):
                    # The existing edge diverges; split before
                    # continuing. _split_node returns the new
                    # intermediate node that now matches `prefix_len`.
                    new_intermediate = self._split_node(
                        child.key,
                        child,
                        prefix_len,
                    )
                    node = new_intermediate
                    key = key[prefix_len:]
                    if len(key) == 0:
                        break
                    child_key = self._gms_child_key(key)
                else:
                    node = child
                    key = key[prefix_len:]
                    if len(key) == 0:
                        break
                    child_key = self._gms_child_key(key)
            if len(key) > 0:
                new_node = TreeNode(priority=0)
                new_node.parent = node
                new_node.key = key
                new_node.value = None  # spilled placeholder
                new_node.last_access_time = access_time
                node.children[child_key] = new_node
                node = new_node
            # `node` is now the leaf representing this prefix.
            # Tag as spilled — record carries the prefix tokens too
            # so the NEXT snapshot can re-persist them without an
            # explicit re-walk.
            self._spilled[id(node)] = _SpillInfo(
                block_ids=list(block_ids),
                generations=list(generations),
                value_len=len(block_ids),
                prefix_tokens=list(prefix_tokens),
            )
            # Race #3: rehydrate per-slot generation counter from
            # the snapshot so a post-restart spill BUMPS from the
            # right base — otherwise it'd start at 1 and could
            # collide with the daemon's stored generation for a
            # slot that survived the restart.
            with self._slot_gen_lock:
                for sl, g in zip(block_ids, generations):
                    sl_i = int(sl)
                    if int(g) > self._slot_generations.get(sl_i, 0):
                        self._slot_generations[sl_i] = int(g)
            return True

        def _load_snapshot(self) -> int:
            """Read `self._persist_path` and rehydrate spilled
            entries into the radix tree. Returns the count loaded.
            Failures are logged and treated as cold-start."""
            if not (self._persist_enabled and self._persist_path):
                return 0
            if not os.path.exists(self._persist_path):
                return 0
            with open(self._persist_path, "rb") as f:
                blob = f.read()
            parsed = self._deserialize_snapshot(blob)
            if parsed is None:
                return 0
            file_epoch, entries = parsed
            # Daemon-epoch validation: if the daemon was replaced
            # between the previous run and this one, every spilled
            # slot has been zeroed. Discard rather than rehydrate
            # phantom mappings.
            if self._daemon_socket and file_epoch >= 0:
                try:
                    from gms_kv_ring.daemon.client import DaemonClient

                    with DaemonClient(self._daemon_socket) as c:
                        current = int(c.current_daemon_epoch())
                    if current != file_epoch:
                        logger.info(
                            "[GMSRadixCache eng=%s] snapshot epoch "
                            "%d != current %d; cold-starting",
                            self._engine_id,
                            file_epoch,
                            current,
                        )
                        return 0
                except Exception:  # noqa: BLE001
                    # Couldn't validate — be conservative and skip.
                    return 0
            n = 0
            for e in entries:
                ok = self._insert_spilled_chain(
                    e["prefix_tokens"],
                    e["block_ids"],
                    e["generations"],
                )
                if ok:
                    n += 1
            return n

        def _restore_node(self, node) -> bool:
            """Read bytes back from daemon into fresh HBM slots
            and set node.value to the new tensor. Returns True
            on success."""
            info = self._spilled.get(id(node))
            if info is None:
                return False
            try:
                fresh = self.token_to_kv_pool_allocator.alloc(
                    info.value_len,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] alloc failed during " "restore",
                    self._engine_id,
                    exc_info=True,
                )
                return False
            if fresh is None or len(fresh) < info.value_len:
                if fresh is not None:
                    self.token_to_kv_pool_allocator.free(fresh)
                return False
            # Build (src_slot_idx, dst_slot_idx, gen) triples.
            fresh_list = self._value_to_int_list(fresh)
            triples = [
                (info.block_ids[i], fresh_list[i], info.generations[i])
                for i in range(info.value_len)
            ]
            try:
                host_restore = (
                    self._host_tier_fallback
                    and str(self._source_engine_id) != str(self._engine_id)
                    and hasattr(self._gds, "restore_blocks_remap_from_host")
                )
                if self._gds.is_available() and not host_restore:
                    results = self._gds.restore_blocks_remap(triples)
                else:
                    results = self._gds.restore_blocks_remap_from_host(
                        self._source_engine_id, triples
                    )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[GMSRadixCache eng=%s] KV restore raised",
                    self._engine_id,
                    exc_info=True,
                )
                self.token_to_kv_pool_allocator.free(fresh)
                return False
            failed = [d for d, ok in results.items() if not ok]
            if failed:
                logger.warning(
                    "[GMSRadixCache eng=%s] restore partial-"
                    "failure for node id=%d: %d/%d blocks failed; "
                    "dropping spill record so future requests "
                    "don't re-attempt the same broken slots "
                    "(Phase B drop-on-failure)",
                    self._engine_id,
                    id(node),
                    len(failed),
                    info.value_len,
                )
                self.token_to_kv_pool_allocator.free(fresh)
                # Phase B: drop the spill record so subsequent
                # match_prefix walks won't keep retrying the
                # same broken (engine, slot) keys. The tree node
                # stays with value=None; _match_prefix_helper's
                # `value is None and id not in spilled → break`
                # branch handles it.
                with self._spill_lock:
                    self._spilled.pop(id(node), None)
                    self._persist_dirty = True
                try:
                    from gms_kv_ring.common import metrics

                    metrics.connector_prefix_invalidated_on_failure.inc(
                        engine_id=self._engine_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return False
            node.value = fresh
            with self._spill_lock:
                self._spilled.pop(id(node), None)
                self._persist_dirty = True
            return True

        # --------------------------------------------------------
        # Phase A — daemon-epoch invalidation
        # --------------------------------------------------------

        def _epoch_loop(self) -> None:
            """Background loop. Polls daemon's epoch at
            `_epoch_poll_s` cadence; on change, invalidates
            every spill record because the daemon's storage
            has been zeroed. Resilient to daemon down/up."""
            while not self._epoch_stop.wait(self._epoch_poll_s):
                try:
                    self._check_daemon_epoch_once()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[GMSRadixCache eng=%s] epoch poll error",
                        self._engine_id,
                        exc_info=True,
                    )

        def _check_daemon_epoch_once(self) -> None:
            """One epoch poll. First-time observation sets the
            baseline; subsequent change triggers invalidation."""
            if self._epoch_client is None:
                try:
                    from gms_kv_ring.daemon.client import DaemonClient

                    self._epoch_client = DaemonClient(
                        self._daemon_socket,
                        connect_timeout=0.5,
                        op_timeout=2.0,
                    )
                except Exception:  # noqa: BLE001
                    return
            try:
                self._epoch_client._call({"op": "ping"})
            except Exception:  # noqa: BLE001
                # Socket died — close so the next poll tries
                # to reconnect (which will then see the new
                # epoch).
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
                return
            if int(current) != int(self._daemon_epoch_seen):
                old = self._daemon_epoch_seen
                self._daemon_epoch_seen = int(current)
                self.invalidate_all_spilled()
                try:
                    from gms_kv_ring.common import metrics

                    metrics.connector_daemon_epoch_changes.inc(
                        engine_id=self._engine_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                logger.warning(
                    "[GMSRadixCache eng=%s] daemon epoch "
                    "changed %d → %d; dropped %d spilled nodes",
                    self._engine_id,
                    int(old),
                    int(current),
                    len(self._spilled),
                )

        def invalidate_all_spilled(self) -> None:
            """Drop every spill record. The corresponding tree
            nodes still exist with value=None, but they're now
            unreachable from match_prefix (since they're no
            longer in `_spilled`, the helper breaks the walk
            at them). Future evictions of sibling nodes will
            naturally clean them up via SGLang's normal eviction
            of empty-parent paths."""
            with self._spill_lock:
                self._spilled.clear()
                self._persist_dirty = True

        def shutdown(self) -> None:
            """Stop background threads and close clients.
            Idempotent. Called from `__del__` and tests."""
            try:
                self._epoch_stop.set()
            except AttributeError:
                return
            # Persistence thread: signal stop, drain one final
            # snapshot synchronously so engine restart sees the
            # latest state.
            try:
                self._persist_stop.set()
            except AttributeError:
                pass
            watcher = getattr(self, "_gms_transition_reclaim_watcher", None)
            if watcher is not None:
                try:
                    watcher.stop(timeout=0.5)
                except Exception:  # noqa: BLE001
                    pass
                self._gms_transition_reclaim_watcher = None
            t_persist = getattr(self, "_persist_thread", None)
            if t_persist is not None and t_persist.is_alive():
                t_persist.join(timeout=0.5)
            try:
                if getattr(self, "_persist_enabled", False) and getattr(
                    self, "_persist_path", ""
                ):
                    self._take_snapshot()
            except Exception:  # noqa: BLE001
                pass
            t = getattr(self, "_epoch_thread", None)
            if t is not None and t.is_alive():
                t.join(timeout=0.2)
            try:
                if self._staging_scan_client is not None:
                    self._staging_scan_client.close()
                    self._staging_scan_client = None
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._epoch_client is not None:
                    self._epoch_client.close()
                    self._epoch_client = None
            except Exception:  # noqa: BLE001
                pass

        def __del__(self):
            try:
                self.shutdown()
            except Exception:  # noqa: BLE001
                pass

        # --------------------------------------------------------
        # RadixCache method overrides
        # --------------------------------------------------------

        # The actual eviction + match logic lives in
        # `_do_evict_core` / `_do_match_prefix_core`. The
        # version-specific `evict` / `match_prefix` overrides
        # below are thin dispatchers that adapt to whichever
        # SGLang signature is active.

        def reclaim_cached_kv(self, target_blocks: int) -> int:
            """Best-effort shadow-transition reclaim.

            Only evictable radix-cache leaves are considered. Each selected
            leaf is synchronously spilled to the GMS daemon before its HBM
            slots are returned to SGLang's allocator, so live decode state is
            untouched and later cache hits restore from the CPU/storage tier.
            """
            if not self._is_active() or self.disable:
                return 0
            return int(
                self._do_evict_core(
                    max(0, int(target_blocks)),
                    force_host_tier=self._host_tier_fallback,
                )
            )

        def _do_evict_core(
            self,
            num_tokens: int,
            *,
            force_host_tier: bool = False,
        ) -> int:
            """Core eviction logic shared by both API
            generations. Returns the number of tokens actually
            evicted (or spilled + retained). Side-effects on
            tree state are the same as vanilla `evict`."""
            import heapq
            import time

            start_time = time.perf_counter()
            # The leaves source differs by API version: newer
            # uses `evictable_leaves` (a set); older uses
            # `_collect_leaves()`. Detect at runtime.
            if hasattr(self, "evictable_leaves") and self.evictable_leaves:
                leaves = list(self.evictable_leaves)
            elif hasattr(self, "_collect_leaves"):
                leaves = self._collect_leaves()
            else:
                leaves = []
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            num_evicted = 0
            while num_evicted < num_tokens and eviction_heap:
                _prio, x = heapq.heappop(eviction_heap)
                # 1. Spill BEFORE freeing HBM (daemon needs to
                #    read the live bytes).
                spilled = self._spill_node(
                    x,
                    force_host_tier=force_host_tier,
                )
                # 2. Free HBM back to allocator either way.
                value_len = len(x.value) if x.value is not None else 0
                if x.value is not None:
                    self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += value_len
                # 3. Spilled → retain leaf with value=None.
                #    Not spilled → vanilla delete path.
                if spilled:
                    x.value = None
                    if hasattr(self, "evictable_leaves"):
                        self.evictable_leaves.discard(x)
                else:
                    self._delete_leaf(x)
                    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                        new_prio = self.eviction_strategy.get_priority(x.parent)
                        heapq.heappush(
                            eviction_heap,
                            (new_prio, x.parent),
                        )
                    self._record_remove_event(x)

            if hasattr(self, "update_eviction_metrics"):
                self.update_eviction_metrics(num_evicted, start_time)
            return num_evicted

        def _match_prefix_helper(self, node, key):
            """Override SGLang's tree-walk to restore spilled
            children inline before accessing their `value`. This
            is the critical hook: vanilla
            `_match_prefix_helper` does `value.append(child.value)`
            and `torch.cat(value)` afterwards — both of which
            crash on `value=None`. By restoring before reading,
            the rest of the walk (and the subsequent torch.cat)
            sees the same in-HBM tensors it always did.

            On restore failure we break out of the walk early,
            naturally truncating the match at the last
            successfully-restored node."""
            if not self._is_active():
                return super()._match_prefix_helper(node, key)
            import time as _t

            access_time = _t.monotonic()
            node.last_access_time = access_time
            child_key = self._gms_child_key(key)
            value = []
            while len(key) > 0 and child_key in node.children:
                child = node.children[child_key]
                child.last_access_time = access_time
                # CRITICAL: restore spilled children before
                # reading their value tensor.
                if child.value is None and id(child) in self._spilled:
                    if not self._restore_node(child):
                        break
                if child.value is None:
                    # value is None but not in our spill table —
                    # something else cleared it. Treat as cache
                    # miss past this point.
                    break
                prefix_len = self._gms_key_match(child.key, key)
                if prefix_len < len(child.key):
                    new_node = self._split_node(
                        child.key,
                        child,
                        prefix_len,
                    )
                    value.append(new_node.value)
                    node = new_node
                    break
                else:
                    value.append(child.value)
                    node = child
                    key = key[prefix_len:]
                    if len(key):
                        child_key = self._gms_child_key(key)
            return value, node

        # API dispatch: define evict/match_prefix matching the
        # SGLang version present. Both call into the _do_*_core
        # helpers above.

        # Restoration happens inline in `_match_prefix_helper`
        # (overridden above). super().match_prefix() handles
        # tensor concatenation correctly because all spilled
        # children have been restored by the time we touch
        # `child.value`. So we ONLY need to override evict() at
        # the public level — match_prefix() works via the helper.

        if _new_api is not None:
            # Latest upstream main: structured params + results.

            def evict(self, params):  # type: ignore[override]
                if not self._is_active() or self.disable:
                    return super().evict(params)
                num = self._do_evict_core(params.num_tokens)
                return _EvictResult(num_tokens_evicted=num)

            def match_prefix(self, params):  # type: ignore[override]
                if not self._is_active() or self.disable:
                    return super().match_prefix(params)
                result = super().match_prefix(params)
                if self._restore_staged_suffix_for_key(
                    params.key,
                    result.device_indices,
                ):
                    return super().match_prefix(params)
                return result

        else:
            # Released 0.5.x: simple signatures, evict returns None.

            def evict(self, num_tokens: int):  # type: ignore[override]
                if not self._is_active() or self.disable:
                    return super().evict(num_tokens)
                self._do_evict_core(int(num_tokens))

            def match_prefix(self, key, **kwargs):  # type: ignore[override]
                if not self._is_active() or self.disable:
                    return super().match_prefix(key, **kwargs)
                result = super().match_prefix(key, **kwargs)
                matched_value = getattr(result, "device_indices", None)
                if matched_value is None and isinstance(result, tuple) and result:
                    matched_value = result[0]
                if self._restore_staged_suffix_for_key(key, matched_value):
                    return super().match_prefix(key, **kwargs)
                return result

    return _GMSRadixCache
