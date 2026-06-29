# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDS-direct connector glue for SGLang's RadixCache / HiCache.

The connector logic is engine-agnostic — see
`gms_kv_ring.engines.gds_block_connector.BlockGdsConnector`. This
module re-exports it under an SGLang-flavored name and documents
the SGLang-specific integration recipe.

SGLang specifics:
  - RadixCache nodes carry token-index slots into the KV pool.
    What vLLM calls "block_id," SGLang calls a per-token slot
    index. The "block" we offload to storage is typically a fixed
    page (e.g., 64 tokens) that aligns with SGLang's `page_size`.
  - For block-granular GDS spill/restore, the caller assembles
    page-aligned slot groups and treats each as a "block id."

Integration recipe:

    from gms_kv_ring.engines.sglang.install_kv_ring import install_for_sglang
    from gms_kv_ring.engines.sglang.gds_connector import SGLangGdsConnector

    handle = install_for_sglang(
        engine_id=..., daemon_socket=..., layers=...,
    )

    def block_layout(page_id):
        # SGLang specifics: layers laid out separately, each layer
        # has shape (num_pages, page_size_bytes). One "block" here
        # = one page across all attention layers.
        return [(L, page_id * page_size_bytes, page_size_bytes)
                for L in range(n_layers)]

    gds = SGLangGdsConnector(handle, block_layout)

    # In RadixCache.evict(num_tokens):
    #   - Pick victim pages (existing LRU/freq logic in SGLang).
    #   - If gds.is_available(), spill them to durable storage
    #     so they survive engine restart and can be promoted on
    #     re-access. Otherwise use the standard host_tier path.
    def evict(self, num_tokens):
        victim_pages = self._select_victims(num_tokens)
        if gds.is_available():
            gds.evict_blocks_to_storage(victim_pages)
        else:
            # standard host_tier path via handle.record_evict per
            # victim, same as the existing on_evict hook.
            ...

    # On match_prefix cache hit, before serving from HBM:
    #   - Identify the hit pages.
    #   - If they're currently in storage (because they were
    #     previously evicted), promote them back to HBM in-place
    #     so the attention path can read from the pool as usual.
    def restore_prefix(self, hit_pages):
        if gds.is_available():
            ok = gds.restore_blocks_from_storage(hit_pages)
            # ok[page_id]=False → engine must fall to cold compute
            # for that page (the slot is "missing" from the cache).
            return ok
        else:
            # standard host_tier promote + restore_ring path
            ...
"""

from __future__ import annotations

from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector, EvictResult

# Engine-flavored alias. Same class; SGLang-specific recipe lives
# in this module's docstring above.
SGLangGdsConnector = BlockGdsConnector

__all__ = ["SGLangGdsConnector", "EvictResult"]
