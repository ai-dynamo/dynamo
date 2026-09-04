# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedulers that mirror their block index into a store.

Both transitions run through ``set_pause_state``: vLLM pauses before clearing
the prefix cache, and unpauses after every rank has re-attached its KV and
before scheduling resumes -- the only moment a handover is both safe and
possible.

Mix ahead of a vLLM scheduler, or name one of the ready-made classes with
``--scheduler-cls``. No plugin, no monkey-patching.

    MirrorsBlockIndex        the four overrides
    BlockIndexScheduler      + AsyncScheduler   (default)
    BlockIndexSyncScheduler  + Scheduler        (async scheduling off)
"""

from __future__ import annotations

import logging

from gpu_memory_service.integrations.vllm.kv_cache import adoption
from gpu_memory_service.kv_cache import store_path
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger(__name__)

__all__ = ["MirrorsBlockIndex", "BlockIndexScheduler", "BlockIndexSyncScheduler"]


class MirrorsBlockIndex:
    """Keeps a scheduler's block index in a store that outlives the engine.

    Four overrides. ``set_pause_state`` brackets a sleep and triggers the
    handover; ``schedule`` and ``update_from_output`` advance the batch counters
    that decide when an entry's KV is known to exist.
    """

    block_index = None
    _paused = False

    def skipping_index_reset(self) -> bool:
        """vLLM clears its index inside a sleep, assuming the KV goes away.

        Under a committed GMS layout it does not, so those entries stay true and
        the store must not mirror that particular reset. Outside a sleep, a
        reset means the operator dropped the cache on purpose.
        """
        return self._paused

    def set_pause_state(self, pause_state) -> None:
        if pause_state != PauseState.UNPAUSED:
            self._paused = True
            super().set_pause_state(pause_state)
            return

        super().set_pause_state(pause_state)
        if not self._paused:
            return  # not a wake from a sleep -- nothing to hand over
        self._paused = False
        if not store_path():
            return
        try:
            adoption.inherit(self)
        except Exception as e:  # persistence must never break serving
            logger.warning("[kv_cache] handover failed (%s); continuing cold", e)
            # Serving cold is fine; leaving behind a store nobody maintains is
            # not -- this engine would overwrite the very blocks it describes.
            adoption.discard(store_path())

    def schedule(self, *args, **kwargs):
        if self.block_index is not None:
            self.block_index.on_schedule()
        return super().schedule(*args, **kwargs)

    def update_from_output(self, *args, **kwargs):
        out = super().update_from_output(*args, **kwargs)
        if self.block_index is not None:
            self.block_index.on_complete()
        return out


class BlockIndexScheduler(MirrorsBlockIndex, AsyncScheduler):
    """Default: async scheduling, which is what vLLM picks when unconfigured."""


class BlockIndexSyncScheduler(MirrorsBlockIndex, Scheduler):
    """For deployments that have turned async scheduling off.

    Naming a sync scheduler while async scheduling is on leaves two concurrent
    batches driving a scheduler that never allocates output placeholders.
    """
