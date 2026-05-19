# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-internal accessors for remote-G2 source-side setup.

Helpers shared between the event-driven (alpha) and direct-query (beta)
source-side implementations. They reach into the TRT-LLM engine to
locate the C++ kv_cache_manager wrapper, the secondary pool's base
pointer, and the per-block byte size; both setup flavors need these.
"""

from __future__ import annotations

from typing import Any, Optional


def kv_cache_manager(engine: Any) -> Optional[Any]:
    """Reach the C++ kv_cache_manager wrapper from the dynamo engine.

    Returns None when not reachable (older TRT-LLM, ENCODE worker, etc.);
    caller should treat that as "remote-G2 not enabled here".
    """
    llm = getattr(engine, "llm", None)
    if llm is None:
        return None
    executor = getattr(llm, "_executor", None)
    if executor is None:
        return None
    py_executor = getattr(executor, "engine", None)
    if py_executor is None:
        return None
    return getattr(py_executor, "kv_cache_manager", None)


def secondary_pool_base_ptr(kv: Any) -> int:
    """Return the secondary KV cache pool's base host address, or 0
    when it is not available (no host pool allocated, exposure binding
    missing, etc.).
    """
    try:
        return int(kv.get_secondary_pool_data(0).data_ptr())
    except Exception:
        return 0


def derive_block_size_bytes(kv: Any) -> Optional[int]:
    """Compute per-block byte size by dividing the secondary pool's total
    bytes by the C++-reported secondary block capacity.

    Avoids layout assumptions about the pool tensor's dimension order.
    Uses KvCacheIterationStats.secondary_max_num_blocks (capacity, not
    free/used) which is set at pool allocation and constant for the
    worker's lifetime.
    """
    try:
        pool = kv.get_secondary_pool_data(0)
    except Exception:
        return None
    if pool is None or pool.numel() == 0:
        return None

    try:
        iter_stats = kv.get_iteration_stats()
    except Exception:
        return None
    if not iter_stats:
        return None

    # iter_stats is dict[window_size, KvCacheIterationStats]. The connector
    # operates under the single-window-block-manager constraint, so we take
    # the first (and only) entry.
    stats = next(iter(iter_stats.values()))
    num_secondary = int(getattr(stats, "secondary_max_num_blocks", 0) or 0)
    if num_secondary <= 0:
        return None

    total_bytes = pool.element_size() * pool.numel()
    return total_bytes // num_secondary
