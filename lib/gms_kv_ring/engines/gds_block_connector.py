# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic block-id-centric GDS-direct connector.

Most KV-cache engines (vLLM, SGLang RadixCache, TRT-LLM
KVCacheManager) expose eviction/restore hooks in terms of opaque
"block ids" or "node ids" — one id per KV-cache slot, each
spanning every attention layer. Our GMSKvRing handle is
layer-centric: each demote_hbm_to_storage call moves one
(layer, offset, size) range. This module bridges them.

Engine-specific re-exports:
  gms_kv_ring.engines.vllm.gds_connector.VllmGdsConnector
  gms_kv_ring.engines.sglang.gds_connector.SGLangGdsConnector

Both are aliases for `BlockGdsConnector` here; what differs is
the integration recipe in each engine module's docstring."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, NamedTuple, Optional

from gms_kv_ring.engines.handle import GMSKvRing

logger = logging.getLogger(__name__)


class EvictResult(NamedTuple):
    """Per-block outcome of evict_blocks_to_storage."""

    succeeded: int
    failed: int
    failed_ids: list[int]


class BlockGdsConnector:
    """Block-id-centric adapter over the layer-centric GMSKvRing.

    `block_layout_fn(block_id) -> list[(layer, offset, size)]`
    returns the (layer, offset, size) tuples for every attention
    layer of a given block. Engine integrators supply this
    function — it's the only engine-version-specific piece.

    Thread-safety: the connector itself holds no mutable state.
    The underlying handle's RPC client is single-threaded; if
    you call from multiple engine threads, serialize externally."""

    def __init__(
        self,
        handle: GMSKvRing,
        block_layout_fn: Callable[[int], list[tuple[int, int, int]]],
    ) -> None:
        self.handle = handle
        self._layout_fn = block_layout_fn

    def is_available(self) -> bool:
        """True iff the daemon's storage backend supports a
        GPU-direct data plane. Engine adapter calls this once at
        startup to decide which eviction path to wire."""
        return self.handle.gpu_direct_storage_available()

    def evict_blocks_to_host(
        self,
        block_ids: Iterable[int],
        *,
        generations: Optional[dict[int, int]] = None,
        timeout_s: float = 5.0,
    ) -> EvictResult:
        """Synchronous host-tier spill fallback.

        This uses the daemon evict ring to mirror HBM into pinned CPU RAM.
        It is intended for planned transition/headroom paths where CPU
        mirroring matters more than GDS overlap. The method waits for the
        daemon ack before returning so callers can safely release the source
        HBM slots on success.
        """
        succeeded = 0
        failed = 0
        failed_ids: list[int] = []
        for block_id in block_ids:
            try:
                gen = 0
                if generations is not None:
                    gen = int(generations.get(int(block_id), 0))
                ranges = [
                    (int(layer), int(size), int(offset))
                    for layer, offset, size in self._layout_fn(int(block_id))
                ]
                result = self.handle.record_evict(
                    int(block_id),
                    ranges,
                    generation=gen,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "BlockGdsConnector.evict_host: block=%d raised: %s",
                    int(block_id),
                    exc,
                )
                result = None
            if result is None:
                failed += 1
                failed_ids.append(int(block_id))
                continue
            slot, target = result
            deadline = time.monotonic() + float(timeout_s)
            while time.monotonic() < deadline:
                value = self.handle.counters.read_slot(slot)
                if value >= target:
                    break
                time.sleep(0.001)
            if self.handle.evict_succeeded(slot, target):
                succeeded += 1
            else:
                failed += 1
                failed_ids.append(int(block_id))
        return EvictResult(succeeded, failed, failed_ids)

    def restore_blocks_remap_from_host(
        self,
        src_engine_id: str,
        src_to_dst: Iterable[tuple],
        *,
        timeout_s: float = 5.0,
    ) -> dict[int, bool]:
        """Synchronous host-tier restore into remapped destination blocks."""
        entries = list(src_to_dst)
        pairs: list[tuple[int, int]] = []
        triples: list[tuple[int, int, int]] = []
        for entry in entries:
            if len(entry) >= 3:
                src, dst, gen = int(entry[0]), int(entry[1]), int(entry[2])
            elif len(entry) >= 2:
                src, dst, gen = int(entry[0]), int(entry[1]), 0
            else:
                continue
            pairs.append((src, dst))
            triples.append((src, dst, gen))
        if not pairs:
            return {}
        try:
            if hasattr(self.handle, "restore_host_blocks_sync"):
                ok = self.handle.restore_host_blocks_sync(
                    str(src_engine_id),
                    triples,
                )
                return {dst: bool(ok) for _src, dst, _gen in triples}
            result = self.handle.record_restore(str(src_engine_id), pairs)
        except Exception:  # noqa: BLE001
            logger.warning(
                "BlockGdsConnector.restore_host: host restore raised",
                exc_info=True,
            )
            return {dst: False for _src, dst in pairs}
        if result is None:
            return {dst: False for _src, dst in pairs}
        slot, target = result
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            value = self.handle.counters.read_slot(slot)
            if value >= target:
                break
            time.sleep(0.001)
        ok = self.handle.restore_succeeded(slot, target)
        return {dst: bool(ok) for _src, dst in pairs}

    def evict_blocks_to_storage(
        self,
        block_ids: Iterable[int],
        *,
        generations: Optional[dict[int, int]] = None,
    ) -> EvictResult:
        """For every (block_id, layer, offset, size) range in the
        list of blocks, call handle.demote_hbm_to_storage. Returns
        the count of fully-evicted blocks vs partial/failed plus
        the ids of the failed blocks so the caller can fall through
        to the host_tier path for them.

        A block is "succeeded" only if ALL of its layer ranges
        demoted successfully. Partial success counts as failed —
        the engine should treat the block as not-spilled and either
        retry or use the standard path.

        `generations`: optional dict[block_id, generation]. When
        present, each per-block demote tags storage with the
        generation so future restores can detect cross-step async
        races (Race #3). When None, generation=0 is sent (legacy
        callers; no race protection on subsequent restores)."""
        succeeded = 0
        failed = 0
        failed_ids: list[int] = []
        for block_id in block_ids:
            gen = 0
            if generations is not None:
                gen = int(generations.get(int(block_id), 0))
            block_ok = True
            for layer, offset, size in self._layout_fn(block_id):
                try:
                    ok = self.handle.demote_hbm_to_storage(
                        layer,
                        offset,
                        size,
                        generation=gen,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "BlockGdsConnector.evict: block=%d layer=%d "
                        "offset=%d size=%d raised: %s",
                        block_id,
                        layer,
                        offset,
                        size,
                        exc,
                    )
                    ok = False
                if not ok:
                    block_ok = False
            if block_ok:
                succeeded += 1
            else:
                failed += 1
                failed_ids.append(block_id)
        return EvictResult(succeeded, failed, failed_ids)

    def restore_blocks_from_storage(
        self,
        block_ids: Iterable[int],
    ) -> dict[int, bool]:
        """Symmetric to evict_blocks_to_storage: per block_id,
        promote every (layer, offset, size) range. Returns a
        per-block dict[block_id, all_layers_ok]. False entries
        indicate the engine must fall to cold compute for that
        block (one or more of its layers wasn't in storage or
        failed CRC verify)."""
        out: dict[int, bool] = {}
        for block_id in block_ids:
            block_ok = True
            for layer, offset, size in self._layout_fn(block_id):
                try:
                    ok = self.handle.promote_storage_to_hbm(
                        layer,
                        offset,
                        size,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "BlockGdsConnector.restore: block=%d "
                        "layer=%d offset=%d size=%d raised: %s",
                        block_id,
                        layer,
                        offset,
                        size,
                        exc,
                    )
                    ok = False
                if not ok:
                    block_ok = False
            out[block_id] = block_ok
        return out

    def restore_blocks_remap(
        self,
        src_to_dst: Iterable[tuple],
    ) -> dict[int, bool]:
        """Restore-on-hit variant for the cache-hit path.

        Engine cache hit hands us tuples (src_block_id,
        dst_block_id) — or, with Race #3 protection enabled,
        (src_block_id, dst_block_id, expected_generation). The
        storage tier still keys by `(engine_id, layer,
        src_block_id * block_bytes)`, but the daemon now checks
        that the slot's recorded generation matches the
        expected_generation before issuing the read; mismatches
        signal failure (the slot was overwritten by a later
        evict; reading would be stale).

        Returns dict keyed by `dst_block_id` so the engine can mark
        successful restores in its block table and fall to cold
        compute for failures."""
        out: dict[int, bool] = {}
        for entry in src_to_dst:
            if len(entry) == 3:
                src, dst, expected_gen = entry
            else:
                src, dst = entry
                expected_gen = 0
            src_ranges = self._layout_fn(src)
            dst_ranges = self._layout_fn(dst)
            if len(src_ranges) != len(dst_ranges):
                raise ValueError(
                    f"BlockGdsConnector.restore_remap: layer count "
                    f"mismatch for src={src} dst={dst}: "
                    f"{len(src_ranges)} vs {len(dst_ranges)}"
                )
            block_ok = True
            for (s_layer, s_off, s_size), (d_layer, d_off, d_size) in zip(
                src_ranges,
                dst_ranges,
            ):
                if s_layer != d_layer or s_size != d_size:
                    raise ValueError(
                        f"BlockGdsConnector.restore_remap: layer "
                        f"layout mismatch src={src} dst={dst}: "
                        f"({s_layer},{s_size}) vs ({d_layer},{d_size})"
                    )
                try:
                    ok = self.handle.promote_storage_to_hbm(
                        s_layer,
                        s_off,
                        s_size,
                        dest_offset=d_off,
                        expected_generation=int(expected_gen),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "BlockGdsConnector.restore_remap: src=%d "
                        "dst=%d layer=%d src_off=%d dst_off=%d "
                        "size=%d raised: %s",
                        src,
                        dst,
                        s_layer,
                        s_off,
                        d_off,
                        s_size,
                        exc,
                    )
                    ok = False
                if not ok:
                    block_ok = False
            out[dst] = block_ok
        return out
