# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM scheduler patches applied in the EngineCore subprocess.

Importing this module monkey-patches ``vllm.v1.core.sched.scheduler.Scheduler``
so that ``reset_prefix_cache(reset_running_requests=True)`` also force-frees
finished-but-delayed requests whose KV blocks are held by a KV connector
(for example ``NixlConnector`` on a prefill worker waiting for the remote
decode worker to pull the KV).

Without this patch, vLLM's sleep path fails on a prefill worker after any
inference that triggered a KV transfer: ``AsyncLLM.pause_generation()`` calls
``pause_scheduler(clear_cache=True)`` which calls
``Scheduler.reset_prefix_cache(reset_running_requests=True)`` which calls
``BlockPool.reset_prefix_cache()``. The block pool rejects the reset while
any block is still referenced, and the delay-freed prefill blocks (pinned
until the remote side pulls the KV) are exactly that. The scheduler then
raises ``RuntimeError("Failed to reset KV cache even when all the running
requests are preempted and moved to the waiting queue. This is likely due
to the presence of running requests waiting for remote KV transfer, which
is not supported yet.")``.

Delivery: set ``vllm_config.scheduler_config.scheduler_cls`` to
``dynamo.vllm.scheduler_patches.PatchedAsyncScheduler`` so the subprocess
imports this module, which applies the monkey-patch on import.
"""

from __future__ import annotations

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger(__name__)


_original_reset_prefix_cache = Scheduler.reset_prefix_cache


def _reset_prefix_cache_with_delay_free(
    self: Scheduler,
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool:
    if reset_running_requests:
        # Force-free any finished request whose blocks are still pinned by a
        # KV connector (NixlConnector prefill side delay-free case). The
        # caller is about to wipe KV state (sleep, weight update, ...), so
        # blocks held for an in-flight remote pull will not complete; free
        # them now so ``BlockPool.reset_prefix_cache()`` sees zero used blocks.
        freed = 0
        for req_id in list(self.requests.keys()):
            request = self.requests[req_id]
            if request.is_finished():
                self._free_blocks(request)
                freed += 1
        if freed:
            logger.info(
                "reset_prefix_cache: force-freed %d finished-but-delayed "
                "requests (likely pending remote KV transfer).",
                freed,
            )
        # Clear scheduler-level KV-connector tracking sets so stale IDs do
        # not resurface on the next step.
        self.finished_recving_kv_req_ids.clear()
        self.failed_recving_kv_req_ids.clear()

    return _original_reset_prefix_cache(self, reset_running_requests, reset_connector)


Scheduler.reset_prefix_cache = _reset_prefix_cache_with_delay_free


class PatchedAsyncScheduler(AsyncScheduler):
    """Default Dynamo scheduler: plain ``AsyncScheduler`` whose import path
    applies the Dynamo reset-prefix-cache patch above.

    Used as ``scheduler_cls`` whenever a custom scheduler is not configured,
    so the EngineCore subprocess imports this module and the patch takes
    effect across disaggregated deployments.
    """
