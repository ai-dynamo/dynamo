# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
``KvbmCacheManagerScheduler`` — vLLM ``AsyncScheduler`` subclass that
swaps in :class:`kvbm.v2.vllm.kv_cache_manager.RustKvCacheManager` as
``self.kv_cache_manager`` after the stock scheduler finishes
constructing. Intended for ``--scheduler-cls``.

Why ``AsyncScheduler`` and not ``Scheduler``: vLLM's ``--scheduler-cls``
picks one class for both engine modes. ``AsyncScheduler``'s output
placeholders are harmless in sync mode and required for async mode
(see ``components/src/dynamo/vllm/instrumented_scheduler.py`` for the
same rationale).

Connector-leader fallback: on construction we walk ``self.connector``
looking for a kvbm ``SchedulerConnectorLeader``. If present, the
resulting ``RustKvCacheManager`` shares the connector's G1 block
manager + registry (KV-transfer-capable path). If absent, we pass
``connector_leader=None`` and ``RustKvCacheManager`` builds a
standalone, isolated-registry core — which is enough to prove out the
swap without needing ``--kv-transfer-config``.

Inject via::

    vllm serve <model> --scheduler-cls \\
        kvbm.v2.vllm.schedulers.kvbm_cache_manager.KvbmCacheManagerScheduler
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kvbm.v2.vllm.kv_cache_manager import RustKvCacheManager
from kvbm.v2.vllm.schedulers.dynamo_kv_cache_manager import find_kvbm_connector_leader
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager

logger = logging.getLogger(__name__)


class KvbmCacheManagerScheduler(AsyncScheduler):
    """AsyncScheduler with the kvbm v2 RustKvCacheManager swapped in."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            **kwargs,
        )

        stock_manager = self.kv_cache_manager
        connector_leader = find_kvbm_connector_leader(self)

        if connector_leader is not None:
            logger.info(
                "KvbmCacheManagerScheduler: using shared G1 block manager "
                "from kvbm connector leader"
            )
        else:
            logger.info(
                "KvbmCacheManagerScheduler: no kvbm connector leader found; "
                "using isolated G1 block manager (no KV transfer)"
            )

        self.kv_cache_manager = RustKvCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            enable_caching=getattr(stock_manager, "enable_caching", True),
            use_eagle=getattr(stock_manager, "use_eagle", False),
            log_stats=getattr(stock_manager, "log_stats", False),
            enable_kv_cache_events=getattr(
                stock_manager, "enable_kv_cache_events", False
            ),
            dcp_world_size=getattr(stock_manager, "dcp_world_size", 1),
            connector_leader=connector_leader,
        )


__all__ = ["KvbmCacheManagerScheduler"]
