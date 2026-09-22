# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang UnifiedRadixCache adapter for persistent GMS HBM.

SGLang remains the sole owner of prefix policy and tree structure. GMS stores
only the survivable content-hash-to-page directory needed by a replacement
engine to adopt immutable pages after a crash.
"""

from __future__ import annotations

import logging
import os
from hashlib import sha256
from typing import TYPE_CHECKING

from gms_kv_ring.common.content_directory import ContentDirectory
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang.install_kv_leases import (
    adopt_hbm_pages,
    demote_hbm_pages_local,
    hint_hbm_page_release,
    retain_hbm_pages,
    rollback_adopted_hbm_pages,
)
from gpu_memory_service.integrations.sglang.tp_consistency import (
    GmsTPConsistencyError,
    TPConsistency,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult

logger = logging.getLogger(__name__)

_publication_gate_installed = False
_original_process_batch_result = None
_original_stream_output = None


def _install_publication_gate() -> None:
    """Batch completed-prefix commits and fence client-visible output on them."""
    global _publication_gate_installed
    global _original_process_batch_result, _original_stream_output
    if _publication_gate_installed:
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

    _original_process_batch_result = Scheduler.process_batch_result
    _original_stream_output = SchedulerOutputStreamer.stream_output

    def process_batch_result(scheduler, *args, **kwargs):
        cache = getattr(scheduler, "tree_cache", None)
        begin = getattr(cache, "_gms_begin_publication_batch", None)
        end = getattr(cache, "_gms_end_publication_batch", None)
        if not callable(begin) or not callable(end):
            return _original_process_batch_result(scheduler, *args, **kwargs)
        begin()
        try:
            return _original_process_batch_result(scheduler, *args, **kwargs)
        finally:
            # Normal generation output flushes through the streamer hook below.
            # This final fence covers uncommon completion paths with no output.
            end()

    def stream_output(streamer, *args, **kwargs):
        flush = getattr(streamer.tree_cache, "_gms_flush_publications", None)
        if callable(flush):
            flush()
        return _original_stream_output(streamer, *args, **kwargs)

    Scheduler.process_batch_result = process_batch_result
    SchedulerOutputStreamer.stream_output = stream_output
    _publication_gate_installed = True


def _directory_hashes(key, page_size: int, get_hash_str) -> list[bytes]:
    """Return native page hashes scoped like SGLang's radix key."""
    prior_hash = None
    cache_salt = getattr(key, "cache_salt", None)
    if cache_salt is not None:
        prior_hash = sha256(
            b"sglang-cache-salt-v1\0" + cache_salt.encode("utf-8")
        ).hexdigest()
    hashes = get_hash_str(key, prior_hash, page_size=page_size)
    assert isinstance(hashes, list)
    result = [bytes.fromhex(value) for value in hashes]
    extra_key = getattr(key, "extra_key", None)
    if extra_key is not None:
        namespace = sha256(
            b"sglang-extra-key-v1\0" + extra_key.encode("utf-8")
        ).digest()
        result = [sha256(namespace + value).digest() for value in result]
    return result


def _logical_layout_digest(items: list[dict]) -> bytes:
    """Digest generation-independent fields used for cross-rank agreement."""
    digest = sha256(b"sglang-gms-layout-v1\0")
    for item in items:
        content_hash = bytes(item["content_hash"])
        engine_id = str(item["engine_id"]).encode("utf-8")
        tier = str(item["tier"]).encode("utf-8")
        digest.update(len(content_hash).to_bytes(4, "little"))
        digest.update(content_hash)
        digest.update(len(engine_id).to_bytes(4, "little"))
        digest.update(engine_id)
        slots = item["slot_ids"]
        digest.update(len(slots).to_bytes(4, "little"))
        for slot in slots:
            digest.update(int(slot).to_bytes(8, "little", signed=False))
        digest.update(len(tier).to_bytes(4, "little"))
        digest.update(tier)
        digest.update(bytes([bool(item["active"])]))
    return digest.digest()


def _engine_id() -> str:
    return str(
        os.environ.get("GMS_SGLANG_ENGINE_ID")
        or os.environ.get("GMS_KVR_ENGINE_ID")
        or "0"
    )


def _standby() -> bool | None:
    """Derive writer passivity from the failover lock, never engine ordering."""
    if os.environ.get("DYN_GMS_FAILOVER_SHADOW_MODE") is None:
        return None
    explicit = os.environ.get("GMS_KV_DIRECTORY_STANDBY")
    if explicit is not None:
        return explicit.strip().lower() not in ("0", "false", "no", "off", "")
    active_lock = os.environ.get("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD")
    if active_lock is not None:
        return active_lock.strip().lower() in ("0", "false", "no", "off", "")
    # A process in failover mode is not allowed to claim directory ownership
    # until the orchestrator explicitly says it holds the external lock.
    return True


def _make_directory(page_size: int) -> ContentDirectory:
    return ContentDirectory(
        os.environ.get("GMS_KV_DIRECTORY_SOCKET")
        or os.environ.get("GMS_SGLANG_DAEMON_SOCKET")
        or "",
        engine="sglang",
        block_size=int(page_size),
        mode=os.environ.get("GMS_KV_DIRECTORY_MODE"),
        keyspace="sglang-native-hbm-v1",
        standby=_standby(),
    )


def _invalidate_and_verify(directory: ContentDirectory, items: list[dict]) -> None:
    """Generation-conditionally remove a failed publication from one daemon."""
    directory.publish(
        [
            {
                "content_hash": item["content_hash"],
                "engine_id": item["engine_id"],
                "slot_ids": item["slot_ids"],
                "generations": item["generations"],
                "tier": "hbm",
                "sealed": False,
            }
            for item in items
        ]
    )
    hashes = [item["content_hash"] for item in items]
    entries = directory.lookup_authoritative(hashes)
    if len(entries) != len(hashes) or any(entry is not None for entry in entries):
        raise RuntimeError(
            "failed SGLang HBM publication remains visible after invalidation"
        )


def make_gms_unified_cache_class():
    """Return the cache subclass after SGLang is importable."""
    import torch
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.utils import get_hash_str
    from sglang.srt.session.streaming_session import is_virtual_node

    class GMSUnifiedRadixCache(UnifiedRadixCache):
        """Add crash-safe HBM discovery without replacing SGLang's tree."""

        def __init__(self, params):
            super().__init__(params)
            _install_publication_gate()
            self._gms_directory = _make_directory(self.page_size)
            self._gms_engine_id = _engine_id()
            self.token_to_kv_pool_allocator._gms_kv_directory = self._gms_directory
            # An active writer otherwise leases every native-free page when it
            # enters the lock-free steady state. In failover mode, keep exactly
            # one shared-free page per rank for the current or next sleeping
            # standbys bounded EXTEND warmup. Ordinary workers retain the full
            # pool. This is based on topology, not the startup role: a sleeping
            # standby becomes the active writer without reconstructing the cache.
            self.token_to_kv_pool_allocator._gms_standby_headroom_pages = (
                1 if _standby() is not None else 0
            )
            self._gms_tp = TPConsistency(self.tp_group, self.tp_world_size)
            self.token_to_kv_pool_allocator._gms_tp_consistency = self._gms_tp
            self._gms_steady_state = False
            self._gms_recovery_candidates: set[bytes] = set()
            self.token_to_kv_pool_allocator._gms_recovery_candidates = (
                self._gms_recovery_candidates
            )
            self._gms_capture_finished_insert = False
            self._gms_finished_insert = None
            # (kv record, CPU page IDs) armed by claim_kv_row for the one
            # release free of a request's own pages.
            self._gms_release_hint = None
            self._gms_publication_batch_depth = 0
            self._gms_pending_publications = []
            self._gms_local_pages_by_hash: dict[bytes, int] = {}
            self._gms_local_hashes_by_page: dict[int, set[bytes]] = {}
            self._gms_retained_order: dict[int, None] = {}
            self.token_to_kv_pool_allocator._gms_local_pages_by_hash = (
                self._gms_local_pages_by_hash
            )
            self.token_to_kv_pool_allocator._gms_local_hashes_by_page = (
                self._gms_local_hashes_by_page
            )
            self._gms_directory.start_async_read()

        def _gms_hidden_recoverable_tokens(self) -> int:
            from gpu_memory_service.integrations.sglang.install_kv_leases import (
                hidden_recoverable_tokens,
            )

            return hidden_recoverable_tokens(self.token_to_kv_pool_allocator)

        def protected_size(self) -> int:
            # A promoted writer keeps predecessor SEALED pages outside native
            # free_pages until a hash match adopts them. They are intentionally
            # unavailable, not leaked, and belong in SGLang's protected-capacity
            # term so the native conservation invariant remains meaningful.
            return super().protected_size() + self._gms_hidden_recoverable_tokens()

        def full_protected_size(self) -> int:
            return super().full_protected_size() + self._gms_hidden_recoverable_tokens()

        def _maybe_enter_steady_state(self) -> bool:
            """Establish one common recovery inventory, once per writer epoch."""
            if self._gms_steady_state:
                return True
            ready = self._gms_tp.all_true(
                "steady:writer-ready",
                bool(
                    getattr(self._gms_directory, "read_view_is_current_writer", False)
                ),
            )
            if not ready:
                return False
            _local, common = self._gms_tp.run_intersection(
                "steady:inventory",
                lambda: [
                    content_hash
                    for content_hash, _entry in self._gms_directory.read_view_items(
                        tier="hbm", state=""
                    )
                ],
            )
            if not self._gms_tp.all_true(
                "steady:writer-recheck",
                bool(
                    getattr(self._gms_directory, "read_view_is_current_writer", False)
                ),
            ):
                return False
            from gpu_memory_service.integrations.sglang.install_kv_leases import (
                _STATE,
                enter_exclusive_steady_state,
            )

            writable_pages = enter_exclusive_steady_state(
                self.token_to_kv_pool_allocator
            )
            self._gms_recovery_candidates.update(bytes(value) for value in common)
            if not self._gms_directory.freeze_current_writer_view():
                raise RuntimeError(
                    "could not freeze SGLang directory view after writer handoff"
                )
            self._gms_steady_state = True
            state = _STATE.get(id(self.token_to_kv_pool_allocator))
            if state is not None:
                state["steady_state"] = True
            logger.info(
                "[GMS-KVLease] SGLang entered exclusive steady state "
                "candidates=%d writable_pages=%d",
                len(self._gms_recovery_candidates),
                writable_pages,
            )
            return True

        def evict_for_alloc(self, params):
            # Native pressure eviction otherwise frees just one request's
            # shortfall. That leaves no eligible pages over which the GMS
            # reservation protocol can amortize its TP transaction. Ask the
            # native policy for bounded headroom; it still chooses the LRU
            # victims and protects every locked/in-flight prefix.
            from dataclasses import replace

            from gpu_memory_service.integrations.sglang.install_kv_leases import (
                activate_hidden_recovery_capacity,
            )

            activated = 0
            if self._gms_steady_state and params.num_tokens > 0:
                activated = activate_hidden_recovery_capacity(
                    self.token_to_kv_pool_allocator, params.num_tokens
                )
                if activated:
                    params = replace(
                        params,
                        num_tokens=max(0, params.num_tokens - activated),
                    )

            if (
                params.num_tokens > 0
                and self._gms_directory.authoritative
                and not self._gms_steady_state
            ):
                size = int(getattr(self.token_to_kv_pool_allocator, "size", 0))
                page_size = int(self.page_size)
                headroom = (size // 16 // page_size) * page_size
                params = replace(params, num_tokens=max(params.num_tokens, headroom))
            result = super().evict_for_alloc(params)
            if activated:
                result.num_tokens_evicted += activated
            return result

        @staticmethod
        def _hashes_for_key(key, page_size: int) -> list[bytes]:
            return _directory_hashes(key, page_size, get_hash_str)

        def _remember_local_pages(self, hashes, pages) -> None:
            for content_hash, page in zip(hashes, pages):
                content_hash = bytes(content_hash)
                page = int(page)
                previous = self._gms_local_pages_by_hash.get(content_hash)
                if previous is not None and previous != page:
                    old_hashes = self._gms_local_hashes_by_page.get(previous)
                    if old_hashes is not None:
                        old_hashes.discard(content_hash)
                        if not old_hashes:
                            self._gms_local_hashes_by_page.pop(previous, None)
                self._gms_local_pages_by_hash[content_hash] = page
                self._gms_local_hashes_by_page.setdefault(page, set()).add(content_hash)

        def _match_device_indices(self, params, result):
            """Return the matched prefix's KV indices, or None if unreadable.

            SGLang's MatchResult carries only the matched length; the indices
            lie on the path to the matched node, except for a streaming
            session's match, which lives in the request's lent row.
            """
            matched_len = int(result.device_prefix_len)
            if matched_len == 0:
                return torch.empty((0,), dtype=torch.int64, device=self.device)
            node = result.last_device_node
            req = getattr(params, "req", None)
            if is_virtual_node(node):
                if req is None:
                    return None
                return self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :matched_len
                ]
            indices = self.path_device_indices(node)
            if indices is None or len(indices) < matched_len:
                return None
            return indices[:matched_len]

        def _native_prefix_indices(self, key):
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

            params = MatchPrefixParams(key=key)
            return self._match_device_indices(params, super().match_prefix(params))

        def _attach_request_pages(self, params, result) -> None:
            req = getattr(params, "req", None)
            if req is None:
                return
            matched_pages = int(result.device_prefix_len) // int(self.page_size)
            if matched_pages == 0:
                # match_prefix can run again after allocation.  Do not erase
                # the allocator's CPU page record when the second lookup still
                # has an empty prefix.
                if not hasattr(req, "_gms_kv_page_ids"):
                    req._gms_kv_page_ids = []
                return
            key = params.key.page_aligned(self.page_size)
            hashes = self._hashes_for_key(key, self.page_size)[:matched_pages]
            pages = [self._gms_local_pages_by_hash.get(value) for value in hashes]
            if len(pages) == matched_pages and all(page is not None for page in pages):
                req._gms_kv_page_ids = [int(page) for page in pages]
                return
            known = getattr(req, "_gms_kv_page_ids", None)
            if known is None or len(known) < matched_pages:
                req._gms_kv_page_ids = None

        def claim_kv_row(self, req) -> bool:
            # release_kv_cache enters here first, before any checkpoint or
            # free. Arm a hint from the lease allocator's request-local CPU
            # page record so the release never waits for a GPU tensor.
            self._gms_release_hint = (req.kv, getattr(req, "_gms_kv_page_ids", None))
            try:
                claimed = super().claim_kv_row(req)
            except BaseException:
                self._gms_release_hint = None
                raise
            if claimed:
                self._gms_release_hint = None
            return claimed

        def on_release(self, req, *, checkpointed: bool) -> None:
            self._gms_release_hint = None
            return super().on_release(req, checkpointed=checkpointed)

        def free_kv_row(self, kv, ranges) -> None:
            """Pass exact CPU page IDs to the lease allocator when available."""
            hint, self._gms_release_hint = self._gms_release_hint, None
            # Only the release free of the request's own row is hinted. A
            # later over-allocation free or another record's free falls back
            # to the native device-tensor release.
            pages = hint[1] if hint is not None and hint[0] is kv else None
            if pages is not None and kv.get_evicted_seqlen(ComponentType.SWA) == 0:
                page_size = int(self.page_size)
                valid = True
                hinted = []
                for start, end in ranges:
                    start, end = int(start), int(end)
                    page_end = (end + page_size - 1) // page_size
                    if (
                        start < 0
                        or end < start
                        or start % page_size
                        or page_end > len(pages)
                    ):
                        valid = False
                        break
                    hinted.append(pages[start // page_size : page_end])
                if valid:
                    for group in hinted:
                        hint_hbm_page_release(self.token_to_kv_pool_allocator, group)
            return super().free_kv_row(kv, ranges)

        def _prepare_finished_prefix(
            self, key, last_device_node=None, request_pages=None
        ):
            if request_pages is not None:
                indices = None
                expected_pages = len(key) // int(self.page_size)
                if len(request_pages) < expected_pages:
                    raise RuntimeError(
                        "completed SGLang prefix has incomplete CPU page metadata"
                    )
                # The request record includes its final partial page, while
                # the radix key contains only the page-aligned cached prefix.
                request_pages = request_pages[:expected_pages]
                resident_len = len(request_pages) * int(self.page_size)
            elif last_device_node is None:
                # Kept for direct callers and tests. The normal completion path
                # captures the insert result and avoids walking the same radix
                # path a second time.
                indices = self._native_prefix_indices(key)
                if indices is None:
                    raise RuntimeError(
                        "completed SGLang prefix has no resident device indices"
                    )
                resident_len = len(indices)
            else:
                indices = self.tree_core.collect_full_device_indices(
                    last_device_node,
                    self.tree_core.root_node_handle(getattr(key, "extra_key", None)),
                )
                resident_len = len(indices)
            if resident_len != len(key):
                raise RuntimeError("completed SGLang prefix is not fully resident")
            page_size = int(self.page_size)
            if page_size <= 0 or resident_len % page_size:
                raise RuntimeError("completed SGLang prefix is not page-aligned")
            expected = resident_len // page_size
            hashes = self._hashes_for_key(key, self.page_size)
            if len(hashes) != expected:
                raise RuntimeError(
                    "completed SGLang prefix has incomplete native hashes"
                )
            if len(set(hashes)) != len(hashes):
                raise RuntimeError(
                    "completed SGLang prefix has duplicate native hashes"
                )
            if self._gms_steady_state and request_pages is not None:
                pages = [
                    int(self._gms_local_pages_by_hash.get(content_hash, page))
                    for content_hash, page in zip(hashes, request_pages)
                ]
            elif self._gms_steady_state:
                # SGLang's paged allocator stores every page as one contiguous
                # page_size run. Copy only the page representatives to the CPU;
                # copying the whole prefix synchronizes hundreds of GPU values
                # on every completion and stalls unrelated decode streams.
                pages = [
                    int(value) // page_size
                    for value in indices[::page_size].detach().cpu().tolist()
                ]
            else:
                # Bootstrap/recovery validates the full physical layout before
                # establishing the invariant used by the steady-state fast path.
                slots = [int(value) for value in indices.detach().cpu().tolist()]
                pages = []
                for offset in range(expected):
                    chunk = slots[offset * page_size : (offset + 1) * page_size]
                    page = chunk[0] // page_size
                    if chunk != list(range(page * page_size, (page + 1) * page_size)):
                        raise RuntimeError(
                            "completed SGLang prefix does not contain contiguous KV pages"
                        )
                    pages.append(page)
            lease_map = self.token_to_kv_pool_allocator._gms_kv_leases_by_page
            items = []
            for page, content_hash in zip(pages, hashes):
                lease = lease_map.get(page)
                if lease is None or int(lease.block_id) != page:
                    raise RuntimeError("completed SGLang HBM page has no lease")
                items.append(
                    {
                        "content_hash": content_hash,
                        "engine_id": self._gms_engine_id,
                        "slot_ids": [page],
                        "generations": [int(lease.generation)],
                        "tier": "hbm",
                        "active": False,
                    }
                )
            if len(set(pages)) != len(pages):
                raise RuntimeError("completed SGLang prefix reuses a physical KV page")
            return hashes, pages, items

        def _commit_finished_prefixes(self, prepared) -> None:
            if not prepared:
                return
            # A mapped standby or a newly promoted writer may complete work
            # before its directory observes writer authority. Keep these
            # just-computed pages in SGLang's native radix cache, but do not
            # seal or advertise them until writer authority is visible. They
            # remain protected by their ordinary LEASED ownership and become
            # reusable through the native eviction path.
            if not getattr(self._gms_directory, "authoritative", False):
                logger.debug("[GMS-KVDirectory] deferring durability during takeover")
                return
            lease_map = self.token_to_kv_pool_allocator._gms_kv_leases_by_page
            retained = self.token_to_kv_pool_allocator._gms_retained_pages
            item_by_hash = {}
            unique_pages = []
            seen_pages = set()
            for hashes, pages, items in prepared:
                for item in items:
                    content_hash = bytes(item["content_hash"])
                    previous = item_by_hash.setdefault(content_hash, item)
                    if previous != item:
                        raise RuntimeError(
                            "one completed SGLang hash maps to divergent HBM pages"
                        )
                for page in pages:
                    if page not in seen_pages:
                        seen_pages.add(page)
                        unique_pages.append(page)
            all_items = list(item_by_hash.values())
            # A sealed exact-generation mapping is already durable. Repeated
            # native prefix hits should touch its retention order, not resend
            # the same hash and re-seal the same page on every completion.
            items = [
                item
                for item in all_items
                if not (
                    int(item["slot_ids"][0]) in retained
                    and self._gms_local_pages_by_hash.get(bytes(item["content_hash"]))
                    == int(item["slot_ids"][0])
                    and int(lease_map[int(item["slot_ids"][0])].generation)
                    == int(item["generations"][0])
                )
            ]
            pages_to_seal = list(
                dict.fromkeys(int(item["slot_ids"][0]) for item in items)
            )
            newly_retained = {page for page in pages_to_seal if page not in retained}
            publication_attempted = False
            retired_hashes: set[bytes] = set()
            retirement_items = []

            def seal_and_publish():
                nonlocal publication_attempted
                leases = retain_hbm_pages(
                    self.token_to_kv_pool_allocator, pages_to_seal
                )
                retained_leases = {
                    int(lease.block_id): int(lease.generation) for lease in leases
                }
                expected_leases = {
                    page: int(lease_map[page].generation) for page in pages_to_seal
                }
                if (
                    len(leases) != len(pages_to_seal)
                    or retained_leases != expected_leases
                ):
                    raise RuntimeError("could not seal every completed SGLang HBM page")

                # Keep a bounded crash-recoverable window without a second
                # large directory selection RPC. Oldest locally published
                # pages are made writable by advancing their lease generation.
                # Their old directory records can remain as bounded candidates:
                # adoption validates the exact generation and therefore turns
                # a racing/stale lookup into a safe miss. Reusing the physical
                # slot removes the old record atomically with its replacement.
                allocator = self.token_to_kv_pool_allocator
                total_pages = int(getattr(allocator, "size", 0)) // max(
                    1, int(getattr(allocator, "page_size", 1))
                )
                cap = max(1, total_pages * 3 // 4) if total_pages > 2 else 0
                excess = max(0, len(retained) - cap) if cap else 0
                current_pages = set(unique_pages)
                retire_pages = [
                    page
                    for page in self._gms_retained_order
                    if page in retained and page not in current_pages
                ][:excess]
                if retire_pages:
                    retired_leases = demote_hbm_pages_local(allocator, retire_pages)
                    if len(retired_leases) != len(retire_pages):
                        raise RuntimeError("incomplete SGLang HBM page demotion")
                    for page, retired_lease in zip(retire_pages, retired_leases):
                        hashes = self._gms_local_hashes_by_page.get(page, set())
                        if not hashes:
                            raise RuntimeError(
                                "retained SGLang page has no content identity"
                            )
                        for content_hash in hashes:
                            content_hash = bytes(content_hash)
                            retired_hashes.add(content_hash)
                            # A new publication for the same content hash
                            # replaces the old slot atomically; adding a
                            # tombstone would create a duplicate batch key.
                            if content_hash not in item_by_hash:
                                retirement_items.append(
                                    {
                                        "content_hash": content_hash,
                                        "engine_id": self._gms_engine_id,
                                        "slot_ids": [page],
                                        "generations": [int(retired_lease.generation)],
                                        "tier": "hbm",
                                        "sealed": False,
                                    }
                                )
                        # Retention and native residency have different
                        # lifetimes. Demotion removes crash-recovery authority,
                        # but SGLang still owns this radix page until its LRU
                        # frees it. Keep the CPU content identity so native
                        # eviction need not synchronize a GPU tensor. The
                        # allocator release path removes both maps before reuse.
                        self._gms_retained_order.pop(page, None)

                publication_attempted = True
                mutations = retirement_items + items
                if self._gms_directory.publish_deferred(mutations) != len(mutations):
                    raise RuntimeError("incomplete SGLang HBM directory publication")
                return leases

            try:
                # Layout validation, lease sealing, ordered publication enqueue,
                # and the peer vote form one compensatable transaction. The old
                # path used three serialized Python-object collectives here,
                # stalling other streams whenever one request finished.
                if self._gms_steady_state:
                    # The native TP scheduler supplies identical completed
                    # layouts after the one-time inventory agreement. Recovery
                    # accepts only the intersection of READY shard records, so
                    # a local failure is a safe miss and kills that cohort.
                    seal_and_publish()
                else:
                    self._gms_tp.transact_digest(
                        "publish:commit",
                        _logical_layout_digest(items),
                        seal_and_publish,
                    )
                for hashes, pages, _items in prepared:
                    self._remember_local_pages(hashes, pages)
                self._gms_recovery_candidates.difference_update(retired_hashes)
                for page in unique_pages:
                    self._gms_retained_order.pop(page, None)
                    if page in retained:
                        self._gms_retained_order[page] = None
            except Exception:
                if not publication_attempted:
                    # No directory mutation was attempted. Removing only new
                    # flags lets native eviction release its still-live lease.
                    retained.difference_update(newly_retained)
                    raise
                try:
                    # Preserve queue order: a synchronous tombstone must not
                    # race ahead of the publication it compensates.
                    if not self._gms_directory.flush_deferred(timeout=2.0):
                        raise RuntimeError("timed out draining failed publication")
                    _invalidate_and_verify(self._gms_directory, items)
                except Exception as cleanup_error:
                    # An ambiguous record is safe only while its exact lease
                    # generation stays sealed. The failed cohort is fenced,
                    # and a replacement accepts only a common TP prefix.
                    raise RuntimeError(
                        "could not verify failed SGLang HBM publication cleanup; "
                        "retaining leases and failing closed"
                    ) from cleanup_error
                retained.difference_update(newly_retained)
                raise

        def _gms_begin_publication_batch(self) -> None:
            self._gms_publication_batch_depth += 1

        def _gms_end_publication_batch(self) -> None:
            if self._gms_publication_batch_depth <= 0:
                raise RuntimeError("unbalanced SGLang GMS publication batch")
            self._gms_publication_batch_depth -= 1
            if self._gms_publication_batch_depth == 0:
                self._gms_flush_publications()

        def _gms_flush_publications(self) -> None:
            if not self._gms_pending_publications:
                return
            pending = self._gms_pending_publications
            self._gms_pending_publications = []
            prepared = [
                self._prepare_finished_prefix(*publication) for publication in pending
            ]
            self._commit_finished_prefixes(prepared)

        def _publish_finished_prefix(
            self, key, last_device_node=None, request_pages=None
        ) -> None:
            self._gms_pending_publications.append(
                (key, last_device_node, request_pages)
            )
            if self._gms_publication_batch_depth == 0:
                self._gms_flush_publications()

        def reset(self) -> None:
            """Atomically retire GMS-derived cache state before native reset."""
            # UnifiedRadixCache.__init__ dispatches to self.reset() before this
            # subclass has installed any GMS metadata. That constructor reset
            # is purely native and must remain so.
            if not hasattr(self, "_gms_local_pages_by_hash"):
                super().reset()
                return
            lease_map = self.token_to_kv_pool_allocator._gms_kv_leases_by_page
            items = []
            for content_hash, page in self._gms_local_pages_by_hash.items():
                lease = lease_map.get(int(page))
                if lease is not None:
                    items.append(
                        {
                            "content_hash": bytes(content_hash),
                            "engine_id": self._gms_engine_id,
                            "slot_ids": [int(page)],
                            "generations": [int(lease.generation)],
                            "tier": "hbm",
                            "sealed": False,
                        }
                    )
            if items:
                if not self._gms_directory.flush_deferred(timeout=2.0):
                    raise RuntimeError("timed out draining SGLang HBM publications")
                _invalidate_and_verify(self._gms_directory, items)
            super().reset()
            self._gms_steady_state = False
            self._gms_recovery_candidates.clear()
            self._gms_pending_publications.clear()
            self._gms_publication_batch_depth = 0
            self._gms_local_pages_by_hash.clear()
            self._gms_local_hashes_by_page.clear()
            self._gms_retained_order.clear()
            self._gms_finished_insert = None
            self._gms_release_hint = None

        def insert(self, params):
            result = super().insert(params)
            if self._gms_capture_finished_insert:
                self._gms_finished_insert = (params, result)
            return result

        def _node_directory_hashes(self, node) -> list[bytes]:
            """Rebuild one node's public hashes from CPU-resident radix keys."""
            path = []
            current = node
            while current is not None and getattr(current, "key", None) is not None:
                if len(current.key):
                    path.append(current)
                current = current.parent
            path.reverse()
            if not path:
                return []
            extra_key = node.key.extra_key
            cache_salt = node.key.cache_salt
            prior_hash = None
            if cache_salt is not None:
                prior_hash = sha256(
                    b"sglang-cache-salt-v1\0" + cache_salt.encode("utf-8")
                ).hexdigest()
            own_hashes = []
            for current in path:
                if (
                    current.key.extra_key != extra_key
                    or current.key.cache_salt != cache_salt
                ):
                    return []
                hashes = get_hash_str(
                    current.key, prior_hash, page_size=int(self.page_size)
                )
                if not isinstance(hashes, list):
                    return []
                if hashes:
                    prior_hash = hashes[-1]
                if current is node:
                    own_hashes = hashes
            result = [bytes.fromhex(value) for value in own_hashes]
            if extra_key is not None:
                namespace = sha256(
                    b"sglang-extra-key-v1\0" + extra_key.encode("utf-8")
                ).digest()
                result = [sha256(namespace + value).digest() for value in result]
            return result

        def _evict_device_leaf(self, node_id, tracker):
            """Hint native LRU page IDs from the durable content directory."""
            try:
                node = self.tree_core.node_by_id(node_id)
            except NotImplementedError:
                # SGLang's default Rust TreeCore does not expose node keys.
                # The hint is optional: without it the allocator reads the
                # released page IDs from the native device tensor.
                return super()._evict_device_leaf(node_id, tracker)
            pages = [
                self._gms_local_pages_by_hash.get(content_hash)
                for content_hash in self._node_directory_hashes(node)
            ]
            # An unbacked write-back leaf is backed up before it is freed, so
            # it must not leave a release hint on the FIFO yet.
            will_free = bool(node.backuped or not self.is_write_back)
            if will_free and pages and all(page is not None for page in pages):
                hint_hbm_page_release(
                    self.token_to_kv_pool_allocator,
                    [int(page) for page in pages],
                )
            return super()._evict_device_leaf(node_id, tracker)

        def checkpoint_into_tree(self, req, *, up_to: int, **kwargs) -> None:
            # Every tree insert of a request, from both checkpoint_kv_cache and
            # a checkpointing release_kv_cache, lands here. SGLang itself tells
            # the final insert from a running checkpoint by req.finished().
            if req.finished():
                return self._gms_checkpoint_finished(req, up_to=up_to, **kwargs)
            return super().checkpoint_into_tree(req, up_to=up_to, **kwargs)

        def _gms_checkpoint_finished(self, req, *, up_to: int, **kwargs) -> None:
            self._gms_finished_insert = None
            # Reuse the lease allocator's request-local CPU page record so
            # completion never waits for a GPU tensor.
            request_pages = getattr(req, "_gms_kv_page_ids", None)
            self._gms_capture_finished_insert = not self.disable
            try:
                super().checkpoint_into_tree(req, up_to=up_to, **kwargs)
            finally:
                self._gms_capture_finished_insert = False
            captured = self._gms_finished_insert
            self._gms_finished_insert = None
            if captured is None:
                return
            insert_params, result = captured
            key = insert_params.key
            if key is not None and len(key) and result.last_device_node is not None:
                self._publish_finished_prefix(
                    key,
                    result.last_device_node,
                    request_pages=request_pages,
                )

        def _adopt_directory_suffix(
            self, params: MatchPrefixParams, result: MatchResult
        ) -> bool:
            key = params.key.page_aligned(self.page_size)
            matched_len = int(result.device_prefix_len)
            if matched_len >= len(key):
                return False

            if self._gms_steady_state and not self._gms_recovery_candidates:
                return False
            hashes = self._hashes_for_key(key, self.page_size)
            suffix_hashes = hashes[matched_len // self.page_size :]
            if self._gms_steady_state:
                if not any(
                    content_hash in self._gms_recovery_candidates
                    for content_hash in suffix_hashes
                ):
                    return False
            elif not self._gms_tp.leader_true(
                "adopt:candidate",
                self._gms_directory.may_have_hbm_candidate(suffix_hashes),
            ):
                return False
            claim_token = None
            leases = []
            staged_records = []

            def release_claim() -> None:
                nonlocal claim_token
                if claim_token is None:
                    return
                try:
                    self._gms_directory.release_claim(claim_token)
                except Exception:
                    logger.warning(
                        "failed to release unused SGLang directory claim",
                        exc_info=True,
                    )
                finally:
                    claim_token = None

            def invalidate_staged() -> None:
                self._gms_directory.publish(
                    [
                        {
                            "content_hash": content_hash,
                            "engine_id": self._gms_engine_id,
                            "slot_ids": [page],
                            # The public record still stores the claimed source
                            # generation; the successor is only pending.
                            "generations": [source_generation],
                            "tier": "hbm",
                            "sealed": False,
                        }
                        for (
                            content_hash,
                            page,
                            source_generation,
                            _successor_generation,
                        ) in staged_records
                    ]
                )

            def lookup_usable():
                nonlocal claim_token
                entries, claim_token = self._gms_directory.lookup_and_claim(
                    suffix_hashes
                )
                usable = []
                for entry in entries:
                    if (
                        entry is None
                        or entry.get("state") not in ("ready", "active")
                        or entry.get("tier") != "hbm"
                    ):
                        break
                    slots = entry.get("slot_ids") or []
                    generations = entry.get("generations") or []
                    if len(slots) != 1 or len(generations) != 1:
                        break
                    usable.append((int(slots[0]), int(generations[0])))
                return usable, [page for page, _generation in usable]

            try:
                # A single vote covers both local lookup failure and the
                # common logical page prefix. Generations remain rank-local
                # fencing tokens and deliberately need not compare equal.
                usable, common_pages = self._gms_tp.run_common_prefix(
                    "adopt:lookup", lookup_usable
                )
                usable = usable[: len(common_pages)]
                if not usable or self._gms_directory.mode == "shadow":
                    release_claim()
                    return False

                pages, source_generations = map(list, zip(*usable))
                successor_generations = [
                    (generation + 1) & 0xFFFFFFFF for generation in source_generations
                ]
                staged_records = list(
                    zip(
                        suffix_hashes[: len(usable)],
                        pages,
                        source_generations,
                        successor_generations,
                    )
                )
                self._gms_tp.agree(
                    "adopt:plan", [record[:2] for record in staged_records]
                )
            except GmsTPConsistencyError:
                raise
            except Exception:
                logger.warning(
                    "SGLang GMS HBM lookup failed before directory staging",
                    exc_info=True,
                )
                release_claim()
                return False

            stage_items = [
                {
                    "content_hash": content_hash,
                    "generations": [successor_generation],
                }
                for (
                    content_hash,
                    _page,
                    _source_generation,
                    successor_generation,
                ) in staged_records
            ]
            try:

                def stage_adoption():
                    adopted = self._gms_directory.adopt_claim(claim_token, stage_items)
                    if self._gms_tp.enabled and adopted != len(staged_records):
                        raise RuntimeError("incomplete SGLang TP directory adoption")
                    return adopted

                staged = self._gms_tp.run("adopt:stage", stage_adoption)
            except Exception:
                # The RPC may have committed before its response was lost. No
                # ring generation changed, and promotion will discard ACTIVE
                # staging records; continuing this engine would be unsafe.
                release_claim()
                logger.exception(
                    "SGLang directory adoption outcome is unknown; failing closed"
                )
                raise
            if staged != len(staged_records):
                logger.warning(
                    "SGLang directory staged %d/%d HBM pages",
                    staged,
                    len(staged_records),
                )
                release_claim()
                return False
            claim_token = None

            try:

                def acquire_pages():
                    fresh, leases = adopt_hbm_pages(
                        self.token_to_kv_pool_allocator,
                        pages,
                        source_generations,
                    )
                    expected_leases = [
                        KVLease(page, generation)
                        for page, generation in zip(pages, successor_generations)
                    ]
                    if (
                        fresh is None
                        or len(fresh) != len(staged_records) * self.page_size
                        or leases != expected_leases
                    ):
                        raise RuntimeError("incomplete SGLang HBM page adoption")
                    return fresh, leases

                fresh, leases = self._gms_tp.run("adopt:leases", acquire_pages)
            except GmsTPConsistencyError:
                # Some peers may already own successor leases. Keep them until
                # the failed cohort is fenced; promotion discards ACTIVE state.
                raise
            except Exception:
                logger.warning(
                    "SGLang HBM lease adoption failed after directory staging",
                    exc_info=True,
                )
                try:
                    invalidate_staged()
                except Exception:
                    logger.exception(
                        "failed to invalidate staged SGLang HBM adoption; "
                        "failing closed"
                    )
                    raise
                if leases:
                    rollback_adopted_hbm_pages(self.token_to_kv_pool_allocator, leases)
                return False

            try:

                def prepare_insert():
                    matched = self._match_device_indices(params, result)
                    if matched is None:
                        raise RuntimeError(
                            "SGLang matched prefix has no readable device indices"
                        )
                    value = torch.cat((matched, fresh))
                    prefix_len = matched_len + len(fresh)
                    return InsertParams(
                        key=key[:prefix_len],
                        value=value,
                        prev_prefix_len=matched_len,
                    )

                insert_params = self._gms_tp.run("adopt:prepare-insert", prepare_insert)
            except GmsTPConsistencyError:
                raise
            except Exception:
                logger.warning(
                    "SGLang GMS HBM adoption failed before native insertion",
                    exc_info=True,
                )
                try:
                    invalidate_staged()
                except Exception:
                    logger.exception(
                        "failed to invalidate pre-insert SGLang HBM adoption; "
                        "failing closed"
                    )
                    raise
                rollback_adopted_hbm_pages(self.token_to_kv_pool_allocator, leases)
                return False

            # From the first native insert instruction onward, SGLang may have
            # linked some pages into its radix tree even if it raises or reports
            # failure. There is no native transaction/undo API. Never return the
            # pages to the allocator or lease ring after crossing this boundary.
            try:

                def insert():
                    inserted = self.insert(insert_params)
                    if inserted.last_device_node is None:
                        raise RuntimeError(
                            "SGLang rejected adopted HBM pages after native insertion began"
                        )
                    return inserted

                inserted = self._gms_tp.run("adopt:insert", insert)
            except Exception:
                logger.exception(
                    "SGLang native insertion failed after mutation began; failing closed"
                )
                raise
            if inserted.last_device_node is None:
                raise RuntimeError(
                    "SGLang rejected adopted HBM pages after native insertion began"
                )
            self._gms_recovery_candidates.difference_update(
                suffix_hashes[: len(usable)]
            )
            self._remember_local_pages(suffix_hashes[: len(usable)], pages)
            logger.info("[GMS-KVDirectory] SGLang adopted_hbm_pages=%d", len(leases))
            return True

        def match_prefix(self, params):
            return self._gms_match(params, super().match_prefix)

        def _gms_match(self, params, native_match):
            self._maybe_enter_steady_state()
            result = super().match_prefix(params)
            # Native SGLang cache state is replicated by the scheduler, while
            # GMS page allocation is already agreed by the TP allocator. A
            # collective here would serialize every lookup, including native
            # hits. Directory adoption performs its own fail-closed agreement
            # before any native state is mutated.
            if self._gms_directory.authoritative and self._adopt_directory_suffix(
                params, result
            ):
                result = native_match(params)
            self._attach_request_pages(params, result)
            return result

    return GMSUnifiedRadixCache
