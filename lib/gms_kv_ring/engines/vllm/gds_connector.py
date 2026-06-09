# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDS-direct connector glue for vLLM's KVCacheConnectorV1.

The connector logic is engine-agnostic — see
`gms_kv_ring.engines.gds_block_connector.BlockGdsConnector`. This
module re-exports it under a vLLM-flavored name and documents the
specific integration recipe for vLLM's KVCacheConnectorV1.

Integration recipe (inside a vLLM KVCacheConnectorV1 implementation):

    from gms_kv_ring.engines.vllm.install_kv_ring import install_for_vllm
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    handle = install_for_vllm(
        engine_id=..., daemon_socket=..., layers=...,
    )

    def block_layout(block_id):
        # vLLM-version-specific. For typical paged-attention with
        # contiguous per-layer regions:
        #   offset = block_id * block_size_bytes
        #   size   = block_size_bytes
        return [(L, block_id * block_size_bytes, block_size_bytes)
                for L in range(n_layers)]

    gds = VllmGdsConnector(handle, block_layout)

    # vLLM's evict_kv_blocks hook:
    def evict_kv_blocks(self, block_ids, ipc_event_handle=b""):
        if gds.is_available():
            gds.evict_blocks_to_storage(block_ids)
        else:
            # standard host_tier path
            for b in block_ids:
                handle.record_evict(b, block_layout(b), ipc_event_handle)

    # vLLM's start_load_kv hook (on cache hit):
    def start_load_kv(self, request):
        block_ids = request.hit_blocks  # vLLM-version-specific
        if gds.is_available():
            ok = gds.restore_blocks_from_storage(block_ids)
            # Caller decides cold-fallback for ok[b]==False entries.
        else:
            # standard host_tier promote + restore_ring path
            ...
"""

from __future__ import annotations

from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector, EvictResult

# Engine-flavored alias. Same class; vLLM-specific recipe lives in
# this module's docstring above.
VllmGdsConnector = BlockGdsConnector

__all__ = ["VllmGdsConnector", "EvictResult"]
