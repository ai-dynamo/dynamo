# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named schedulers that carry the prefix-index mirror.

The supported way to turn the feature on: name one of these with
``--scheduler-cls``, the same way GMS is already selected with ``--worker-cls``.

    python -m dynamo.vllm --load-format gms \\
        --worker-cls gpu_memory_service.integrations.vllm.worker.GMSWorker \\
        --scheduler-cls gpu_memory_service.integrations.vllm.schedulers.KvIndexScheduler

Match the base to ``async_scheduling``: naming a sync scheduler while async
scheduling is on leaves two concurrent batches driving a scheduler that never
allocates output placeholders. ``KvIndexScheduler`` is the right choice unless
async scheduling is explicitly disabled.

Both classes are inert unless ``GMS_KV_INDEX_PATH`` is set, so naming one
unconditionally is safe.

This module imports vLLM at import time; ``kv_index`` deliberately does not, so
the mirror stays testable without it.
"""

from __future__ import annotations

from gpu_memory_service.integrations.vllm.kv_index import KvIndexMixin
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

__all__ = ["KvIndexScheduler", "KvIndexSyncScheduler"]


class KvIndexScheduler(KvIndexMixin, AsyncScheduler):
    """Default: async scheduling, which is what vLLM picks when unconfigured."""


class KvIndexSyncScheduler(KvIndexMixin, Scheduler):
    """For deployments that have turned async scheduling off."""
