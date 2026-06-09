# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire a GMSKvRing handle into vLLM.

vLLM's KVCacheConnectorV1 interface exposes the two hooks we need:

  • `evict_kv_blocks(block_ids, ipc_event_handle)` — call
    `handle.record_evict(...)` here.
  • `start_load_kv(request)` — translate the request's hit prefix
    into (src_block, dest_block) pairs and call
    `handle.record_restore(...)` then enqueue
    `handle.wait_restore(stream, slot, target)` on the compute stream.

This module deliberately does NOT import vLLM at module load —
unit tests run without vLLM installed. The caller imports vLLM in
its connector and uses our returned handle from inside its hook
methods.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gms_kv_ring.engines.handle import GMSKvRing

logger = logging.getLogger(__name__)


def install_for_vllm(
    *,
    engine_id: str,
    daemon_socket: str,
    layers: list[dict],
    on_evict: Optional[Callable[[GMSKvRing], None]] = None,
    on_hit: Optional[Callable[[GMSKvRing], None]] = None,
    evict_ring_capacity: int = 4096,
    restore_ring_capacity: int = 4096,
    num_counters: int = 512,
) -> GMSKvRing:
    """Build the handle and (optionally) invoke wiring callbacks.

    See `engines/sglang/install_kv_ring.py` for parameter semantics —
    they are identical."""
    handle = GMSKvRing(
        engine_id=engine_id,
        daemon_socket=daemon_socket,
        layers=layers,
        evict_ring_capacity=evict_ring_capacity,
        restore_ring_capacity=restore_ring_capacity,
        num_counters=num_counters,
    )
    if on_evict is not None:
        on_evict(handle)
    if on_hit is not None:
        on_hit(handle)
    logger.info("gms_kv_ring installed for vLLM engine %r", engine_id)
    return handle
