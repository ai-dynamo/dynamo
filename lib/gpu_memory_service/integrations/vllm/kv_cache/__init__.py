# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carrying vLLM's prefix-cache index across a GMS failover.

Enable by naming a scheduler: ``--scheduler-cls
gpu_memory_service.integrations.vllm.kv_cache.scheduler.BlockIndexScheduler``,
with ``GMS_KV_INDEX_PATH`` set. Inert otherwise.
"""

from gpu_memory_service.integrations.vllm.kv_cache.scheduler import (
    BlockIndexScheduler,
    BlockIndexSyncScheduler,
    MirrorsBlockIndex,
)

__all__ = ["BlockIndexScheduler", "BlockIndexSyncScheduler", "MirrorsBlockIndex"]
