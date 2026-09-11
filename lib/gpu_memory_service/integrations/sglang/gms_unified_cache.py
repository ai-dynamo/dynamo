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
    retain_hbm_indices,
    rollback_adopted_hbm_pages,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult

logger = logging.getLogger(__name__)


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
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.utils import get_hash_str

    class GMSUnifiedRadixCache(UnifiedRadixCache):
        """Add crash-safe HBM discovery without replacing SGLang's tree."""

        def __init__(self, params):
            super().__init__(params)
            self._gms_directory = _make_directory(self.page_size)
            self._gms_engine_id = _engine_id()
            self.token_to_kv_pool_allocator._gms_kv_directory = self._gms_directory

            # TreeCore normally computes these hashes only for HiCache or an
            # external linker. GMS needs the same native hashes, but neither
            # feature. This flag is TreeCore-local and only controls hashing.
            self.tree_core.enable_storage = True
            self.tree_core.backfill_missing_hash_values()
            self._gms_directory.start_async_read()

        @staticmethod
        def _hashes_for_key(key, page_size: int) -> list[bytes]:
            return _directory_hashes(key, page_size, get_hash_str)

        def _publish_finished_prefix(self, key) -> None:
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

            result = super().match_prefix(MatchPrefixParams(key=key))
            indices = result.device_indices
            if len(indices) != len(key):
                raise RuntimeError("completed SGLang prefix is not fully resident")
            page_size = int(self.page_size)
            if page_size <= 0 or len(indices) % page_size:
                raise RuntimeError("completed SGLang prefix is not page-aligned")
            expected = len(indices) // page_size
            hashes = self._hashes_for_key(key, self.page_size)
            if len(hashes) != expected:
                raise RuntimeError(
                    "completed SGLang prefix has incomplete native hashes"
                )
            if len(set(hashes)) != len(hashes):
                raise RuntimeError(
                    "completed SGLang prefix has duplicate native hashes"
                )
            slots = [int(value) for value in indices.detach().cpu().tolist()]
            lease_map = self.token_to_kv_pool_allocator._gms_kv_leases_by_page
            pages = []
            items = []
            for offset, content_hash in enumerate(hashes):
                chunk = slots[offset * page_size : (offset + 1) * page_size]
                page = chunk[0] // page_size
                if chunk != list(range(page * page_size, (page + 1) * page_size)):
                    raise RuntimeError(
                        "completed SGLang prefix does not contain contiguous KV pages"
                    )
                lease = lease_map.get(page)
                if lease is None or int(lease.block_id) != page:
                    raise RuntimeError("completed SGLang HBM page has no lease")
                pages.append(page)
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

            retained = self.token_to_kv_pool_allocator._gms_retained_pages
            retained_before = set(retained)
            newly_retained = set(pages).difference(retained_before)
            try:
                leases = retain_hbm_indices(self.token_to_kv_pool_allocator, indices)
                retained_leases = {
                    int(lease.block_id): int(lease.generation) for lease in leases
                }
                expected_leases = {
                    page: int(lease_map[page].generation) for page in pages
                }
                if len(leases) != expected or retained_leases != expected_leases:
                    raise RuntimeError("could not seal every completed SGLang HBM page")
            except Exception:
                # No directory mutation was attempted. Removing only new flags
                # lets a later native eviction release its still-live lease.
                retained.difference_update(newly_retained)
                raise

            # Even an atomic directory batch can lose its response. Verify
            # invalidation before making a potentially published lease reusable.
            try:
                if self._gms_directory.publish(items) != len(items):
                    raise RuntimeError("incomplete SGLang HBM directory publication")
            except Exception:
                try:
                    _invalidate_and_verify(self._gms_directory, items)
                except Exception as cleanup_error:
                    # An ambiguous directory record is less dangerous while
                    # its exact-generation lease remains sealed. Keep the
                    # retention flags and terminate this rank cohort.
                    raise RuntimeError(
                        "could not verify failed SGLang HBM publication cleanup; "
                        "retaining leases and failing closed"
                    ) from cleanup_error
                retained.difference_update(newly_retained)
                raise

        def cache_finished_req(
            self, req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
        ) -> None:
            from sglang.srt.mem_cache.radix_cache import RadixKey

            token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
            key = RadixKey(
                token_ids,
                getattr(req, "extra_key", None),
                is_bigram=self.tree_core.is_eagle,
                cache_salt=getattr(req, "cache_salt", None),
            ).page_aligned(self.page_size)
            super().cache_finished_req(
                req,
                is_insert=is_insert,
                kv_len_to_handle=kv_len_to_handle,
                **kwargs,
            )
            if is_insert and not self.disable and len(key):
                self._publish_finished_prefix(key)

        def _adopt_directory_suffix(
            self, params: MatchPrefixParams, result: MatchResult
        ) -> bool:
            key = params.key.page_aligned(self.page_size)
            matched_len = len(result.device_indices)
            if matched_len >= len(key):
                return False

            hashes = self._hashes_for_key(key, self.page_size)
            suffix_hashes = hashes[matched_len // self.page_size :]
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

            try:
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
            except Exception:
                logger.warning(
                    "SGLang GMS HBM lookup failed before directory staging",
                    exc_info=True,
                )
                release_claim()
                return False

            try:
                staged = self._gms_directory.adopt_claim(
                    claim_token,
                    [
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
                    ],
                )
            except Exception:
                # No ring generation changed. Promotion can discard an ACTIVE
                # stage, but this engine cannot safely continue after an
                # ambiguous directory response.
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
                fresh, leases = adopt_hbm_pages(
                    self.token_to_kv_pool_allocator, pages, source_generations
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
                value = torch.cat((result.device_indices, fresh))
                prefix_len = matched_len + len(fresh)
                insert_params = InsertParams(
                    key=key[:prefix_len],
                    value=value,
                    prev_prefix_len=matched_len,
                )
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
                inserted = self.insert(insert_params)
                if inserted.last_device_node is None:
                    raise RuntimeError(
                        "SGLang rejected adopted HBM pages after native insertion began"
                    )
            except Exception:
                logger.exception(
                    "SGLang native insertion failed after mutation began; failing closed"
                )
                raise
            logger.info("[GMS-KVDirectory] SGLang adopted_hbm_pages=%d", len(leases))
            return True

        def match_prefix(self, params):
            result = super().match_prefix(params)
            if self._gms_directory.authoritative and self._adopt_directory_suffix(
                params, result
            ):
                return super().match_prefix(params)
            return result

    return GMSUnifiedRadixCache
