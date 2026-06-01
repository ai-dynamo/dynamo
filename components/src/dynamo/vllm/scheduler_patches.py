# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary vLLM scheduler patches for NIXL disaggregated sleep.

The upstream fix is being tracked in vllm-project/vllm#41180. Until it lands in
the runtime images Dynamo uses, this module makes ``/engine/sleep`` robust after
prefill-side NIXL KV transfer by:

* freeing requests whose blocks are still pinned by connector delayed-free state
  before ``Scheduler.reset_prefix_cache(reset_running_requests=True)`` delegates
  to upstream vLLM; and
* ignoring late ``finished_sending`` / ``finished_recving`` notifications for
  request ids that were already freed during that reset.

``dynamo.vllm.args`` selects :class:`PatchedAsyncScheduler` only for NIXL configs
with no user-specified scheduler class. vLLM resolves that class in the
EngineCore subprocess, so the monkey patches below are installed in the process
that owns the scheduler.
"""

from __future__ import annotations

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

logger = logging.getLogger(__name__)

# Scheduler attributes used by the patch.
_DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR = "_dynamo_original_reset_prefix_cache"
_IGNORE_LATE_KV_XFER_ATTR = "_dynamo_ignored_kv_xfer_req_ids"


def _get_ignore_set(scheduler: Scheduler) -> set[str]:
    s = getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, None)
    if s is None:
        s = set()
        setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, s)
    return s


def _remember_for_late_kv_xfer_drop(scheduler: Scheduler, req_id: str) -> None:
    """Record that a request was force-freed during cache reset."""

    _get_ignore_set(scheduler).add(req_id)


def _take_ignored_req_id(scheduler: Scheduler, req_id: str) -> bool:
    """Return True if a late connector notification should be dropped."""

    ignore = _get_ignore_set(scheduler)
    if req_id not in ignore:
        return False
    ignore.discard(req_id)
    return True


def _reset_prefix_cache_with_delay_free(
    self: Scheduler,
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool:
    if reset_running_requests:
        freed_finished = 0
        freed_waiting_recv = 0
        waiting_recv_requests = []
        for req_id in list(self.requests.keys()):
            request = self.requests[req_id]
            if request.is_finished():
                # Prefill finished-but-delayed (NixlConnector
                # delay_free_blocks=True).
                self._free_blocks(request)
                _remember_for_late_kv_xfer_drop(self, req_id)
                freed_finished += 1
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Decode-side: blocks allocated at schedule time, normally
                # freed only when the remote send signals completion.
                # Caller is about to wipe KV memory anyway.
                request.status = RequestStatus.FINISHED_ABORTED
                waiting_recv_requests.append(request)
                self._free_blocks(request)
                _remember_for_late_kv_xfer_drop(self, req_id)
                freed_waiting_recv += 1
        if waiting_recv_requests:
            self.waiting.remove_requests(waiting_recv_requests)
            self.skipped_waiting.remove_requests(waiting_recv_requests)
        if freed_finished or freed_waiting_recv:
            logger.info(
                "reset_prefix_cache: force-freed %d finished-but-delayed "
                "and %d WAITING_FOR_REMOTE_KVS requests before cache "
                "reset (ignore-set size=%d).",
                freed_finished,
                freed_waiting_recv,
                len(_get_ignore_set(self)),
            )
        self.finished_recving_kv_req_ids.clear()
        self.failed_recving_kv_req_ids.clear()

    original = getattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)
    return original(self, reset_running_requests, reset_connector)


def _update_from_kv_xfer_finished_safe(self: Scheduler, kv_connector_output) -> None:
    """Drop stale connector completions before applying upstream's state update.

    Upstream asserts every completion still has a live ``self.requests`` entry.
    That is not true after sleep/reset force-frees NIXL-delayed blocks.
    """
    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    for req_id in kv_connector_output.finished_recving or ():
        if _take_ignored_req_id(self, req_id) or req_id not in self.requests:
            logger.debug("Dropping stale finished_recving for request %s", req_id)
            continue
        req = self.requests[req_id]
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(self.requests[req_id])

    for req_id in kv_connector_output.finished_sending or ():
        if _take_ignored_req_id(self, req_id) or req_id not in self.requests:
            logger.debug("Dropping stale finished_sending for request %s", req_id)
            continue
        self._free_blocks(self.requests[req_id])


def _install_scheduler_patches() -> None:
    """Install the monkey patches, preserving the true upstream methods."""

    original_reset = getattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR, None)
    if original_reset is None:
        original_reset = Scheduler.reset_prefix_cache

    setattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR, original_reset)

    Scheduler.reset_prefix_cache = _reset_prefix_cache_with_delay_free
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished_safe


_install_scheduler_patches()


class PatchedAsyncScheduler(AsyncScheduler):
    """Import target used as ``--scheduler-cls`` for NIXL workers."""
