# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install gms_kv_ring into a TRT-LLM worker.

Mirrors `install_for_vllm` and `install_for_sglang`. The TRT-LLM
worker calls this once at `register_kv_caches` time with the
per-(layer × kv_factor) descriptors derived from
`KVCacheBlockPool::primaryPtr`."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gms_kv_ring.engines.handle import GMSKvRing

logger = logging.getLogger(__name__)


def install_for_trtllm(
    *,
    engine_id: str,
    daemon_socket: str,
    layers: "list[dict]",
    on_evict: Optional[Callable[[GMSKvRing], None]] = None,
    on_hit: Optional[Callable[[GMSKvRing], None]] = None,
    evict_ring_capacity: int = 4096,
    restore_ring_capacity: int = 4096,
    num_counters: int = 512,
) -> GMSKvRing:
    """Build a `GMSKvRing` handle bound to this TRT-LLM worker's
    KV pool. Parameter semantics identical to the vLLM/SGLang
    install helpers."""
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
    logger.info("gms_kv_ring installed for TRT-LLM engine %r", engine_id)
    return handle
