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


def _directory_key(native_hash: str) -> bytes:
    return bytes.fromhex(native_hash)


def _engine_id() -> str:
    return str(
        os.environ.get("GMS_SGLANG_ENGINE_ID")
        or os.environ.get("GMS_KVR_ENGINE_ID")
        or "0"
    )


def _standby() -> bool | None:
    if os.environ.get("DYN_GMS_FAILOVER_SHADOW_MODE") is None:
        return None
    return os.environ.get("ENGINE_ID", "0") != "0"


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
        def _hashes_for_key(key, page_size: int) -> list[str]:
            hashes = get_hash_str(key, page_size=page_size)
            assert isinstance(hashes, list)
            return hashes

        def _publish_finished_prefix(self, key) -> None:
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

            result = super().match_prefix(MatchPrefixParams(key=key))
            indices = result.device_indices
            if len(indices) != len(key):
                raise RuntimeError("completed SGLang prefix is not fully resident")
            leases = retain_hbm_indices(self.token_to_kv_pool_allocator, indices)
            expected = len(indices) // int(self.page_size)
            if len(leases) != expected:
                raise RuntimeError("could not seal every completed SGLang HBM page")

            hashes = self._hashes_for_key(key, self.page_size)
            slots = [int(value) for value in indices.detach().cpu().tolist()]
            lease_map = self.token_to_kv_pool_allocator._gms_kv_leases_by_page
            items = []
            for offset, content_hash in enumerate(hashes):
                chunk = slots[offset * self.page_size : (offset + 1) * self.page_size]
                page = chunk[0] // self.page_size
                if len(chunk) != self.page_size or any(
                    slot // self.page_size != page for slot in chunk
                ):
                    raise RuntimeError("completed SGLang prefix crosses a KV page")
                lease = lease_map.get(page)
                if lease is None:
                    raise RuntimeError("completed SGLang HBM page has no lease")
                items.append(
                    {
                        "content_hash": _directory_key(content_hash),
                        "engine_id": self._gms_engine_id,
                        "slot_ids": [page],
                        "generations": [int(lease.generation)],
                        "tier": "hbm",
                        "active": False,
                    }
                )
            publish = getattr(
                self._gms_directory,
                "publish_deferred",
                self._gms_directory.publish,
            )
            if publish(items) != len(items):
                raise RuntimeError("incomplete SGLang HBM directory publication")

        def cache_finished_req(
            self, req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
        ) -> None:
            from sglang.srt.mem_cache.radix_cache import RadixKey

            token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
            key = RadixKey(
                token_ids,
                req.extra_key,
                is_bigram=self.tree_core.is_eagle,
                cache_salt=req.cache_salt,
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
            staged = []
            try:
                entries, claim_token = self._gms_directory.lookup_and_claim(
                    [_directory_key(value) for value in suffix_hashes]
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
                    if claim_token is not None:
                        self._gms_directory.release_claim(claim_token)
                        claim_token = None
                    return False

                pages, source_generations = map(list, zip(*usable))
                successor_generations = [
                    (generation + 1) & 0xFFFFFFFF for generation in source_generations
                ]
                staged = list(
                    zip(
                        suffix_hashes[: len(usable)],
                        pages,
                        source_generations,
                        successor_generations,
                    )
                )

                # Stage the successor generation before advancing the lease ring.
                # A crash from this point leaves an ACTIVE entry, which the next
                # promoted writer drops instead of trusting stale READY metadata.
                adopted = self._gms_directory.adopt_claim(
                    claim_token,
                    [
                        {
                            "content_hash": _directory_key(content_hash),
                            "generations": [successor],
                        }
                        for content_hash, _page, _source, successor in staged
                    ],
                )
                if adopted != len(staged):
                    raise RuntimeError("incomplete SGLang directory adoption")
                claim_token = None
            except Exception:
                # No lease-ring generation has changed. An ambiguous stage must
                # fail the engine closed; continuing could reuse an ACTIVE slot.
                if claim_token is not None:
                    try:
                        self._gms_directory.release_claim(claim_token)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "failed to release staged SGLang directory claim",
                            exc_info=True,
                        )
                    claim_token = None
                logger.exception("SGLang directory adoption staging failed")
                raise

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
                    or len(fresh) != len(staged) * self.page_size
                    or leases != expected_leases
                ):
                    raise RuntimeError("incomplete SGLang HBM page adoption")

                value = torch.cat((result.device_indices, fresh))
                prefix_len = matched_len + len(fresh)
                inserted = self.insert(
                    InsertParams(
                        key=key[:prefix_len],
                        value=value,
                        prev_prefix_len=matched_len,
                    )
                )
                if inserted.last_device_node is None:
                    raise RuntimeError("SGLang rejected adopted HBM pages")
                logger.info(
                    "[GMS-KVDirectory] SGLang adopted_hbm_pages=%d", len(leases)
                )
                return True
            except Exception:  # noqa: BLE001
                logger.warning("SGLang GMS HBM adoption failed", exc_info=True)
                try:
                    self._gms_directory.publish(
                        [
                            {
                                "content_hash": _directory_key(content_hash),
                                "engine_id": self._gms_engine_id,
                                "slot_ids": [page],
                                # The public record still has the claimed source
                                # generation; the successor is only pending.
                                "generations": [source],
                                "tier": "hbm",
                                "sealed": False,
                            }
                            for content_hash, page, source, _successor in staged
                        ]
                    )
                except Exception:
                    # The invalidation outcome is ambiguous. Keep any acquired
                    # leases and terminate rather than expose reusable memory.
                    logger.exception(
                        "failed to invalidate incomplete SGLang HBM adoption"
                    )
                    raise
                if leases:
                    rollback_adopted_hbm_pages(self.token_to_kv_pool_allocator, leases)
                return False
            finally:
                if claim_token is not None:
                    self._gms_directory.release_claim(claim_token)

        def match_prefix(self, params):
            result = super().match_prefix(params)
            if self._gms_directory.authoritative and self._adopt_directory_suffix(
                params, result
            ):
                return super().match_prefix(params)
            return result

    return GMSUnifiedRadixCache
