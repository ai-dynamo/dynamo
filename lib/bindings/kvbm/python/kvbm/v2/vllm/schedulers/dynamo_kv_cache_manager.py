# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler factory that replaces the stock vLLM
``KVCacheManager`` with :class:`kvbm.v2.vllm.kv_cache_manager.RustKvCacheManager`
after constructing an otherwise-unmodified ``vllm.v1.core.sched.scheduler.Scheduler``.

This is the narrow swap-in path described in the v2 kv-cache-manager
plan: it does **not** re-enable the much larger disabled Rust
scheduler (``lib/bindings/kvbm/src/v2/mod.rs:7-23``). All scheduling
decisions are still made by vLLM's Python scheduler; only block
allocation / prefix matching is routed through the kvbm-logical-backed
cache manager.

Usage:

.. code-block:: python

    from kvbm.v2.vllm.schedulers.dynamo_kv_cache_manager import (
        make_kvbm_cache_manager_scheduler,
    )
    scheduler = make_kvbm_cache_manager_scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=som,
        include_finished_set=False,
        log_stats=True,
    )

The returned object is the *vLLM* scheduler — callers interact with
it exactly as they would with a stock ``Scheduler``. The only
observable difference is the cache manager swapped in on its
``kv_cache_manager`` attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager


def make_kvbm_cache_manager_scheduler(
    *,
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    structured_output_manager: "StructuredOutputManager",
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> "Scheduler":
    """Build a stock vLLM ``Scheduler`` and swap its ``kv_cache_manager``
    for :class:`RustKvCacheManager`.

    The function intentionally takes keyword-only arguments and names
    them after ``Scheduler.__init__`` so callers can pass whichever
    subset they need without relying on vLLM's positional ordering
    (which has drifted between releases).
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    from kvbm.v2.vllm.kv_cache_manager import RustKvCacheManager

    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
        include_finished_set=include_finished_set,
        log_stats=log_stats,
    )

    # Pull the already-constructed manager's constructor-relevant
    # settings off the vllm config / the stock cache manager vLLM
    # just built — that way we mirror whatever vLLM decided about
    # `enable_caching`, `use_eagle`, etc., rather than rolling our
    # own defaults.
    stock_manager = scheduler.kv_cache_manager
    rust_manager = RustKvCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=vllm_config.model_config.max_model_len,
        enable_caching=getattr(stock_manager, "enable_caching", True),
        use_eagle=getattr(stock_manager, "use_eagle", False),
        log_stats=log_stats,
        enable_kv_cache_events=getattr(
            stock_manager, "enable_kv_cache_events", False
        ),
        dcp_world_size=getattr(stock_manager, "dcp_world_size", 1),
    )
    scheduler.kv_cache_manager = rust_manager
    return scheduler


__all__ = ["make_kvbm_cache_manager_scheduler"]
