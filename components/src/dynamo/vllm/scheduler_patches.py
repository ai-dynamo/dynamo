# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patches vLLM Scheduler for Dynamo's /engine/sleep path.

Two methods are replaced to handle in-flight NIXL KV transfers:

- ``reset_prefix_cache`` — force-frees finished-but-delayed (NixlConnector
  ``delay_free_blocks=True``) and ``WAITING_FOR_REMOTE_KVS`` requests whose
  KV blocks would otherwise block the BlockPool reset.
- ``_update_from_kv_xfer_finished`` — drops late ``finished_sending`` /
  ``finished_recving`` signals for req_ids the first patch already
  force-freed, avoiding the fatal ``assert req_id in self.requests`` when
  the NIXL orphan timer fires ~480 s post-sleep.

Full writeup: ``.agents/skills/vllm/references/prefill-sleep-nixl.md``.
"""

from __future__ import annotations

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

logger = logging.getLogger(__name__)

# Guards against editable-install double-import. The original method is stored
# on Scheduler so importlib.reload can rebind this module's wrapper safely.
_DYNAMO_PATCHED_MARKER = "_dynamo_scheduler_patched"
_DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR = "_dynamo_original_reset_prefix_cache"

# Per-Scheduler set of req_ids force-freed by reset_prefix_cache; consumed
# by _update_from_kv_xfer_finished to drop the corresponding late signals.
_IGNORE_LATE_KV_XFER_ATTR = "_dynamo_ignored_kv_xfer_req_ids"


def _get_ignore_set(scheduler: Scheduler) -> set[str]:
    s = getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, None)
    if s is None:
        s = set()
        setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, s)
    return s


def _reset_prefix_cache_with_delay_free(
    self: Scheduler,
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool:
    if reset_running_requests:
        ignore = _get_ignore_set(self)
        freed_finished = 0
        freed_waiting_recv = 0
        for req_id in list(self.requests.keys()):
            request = self.requests[req_id]
            if request.is_finished():
                self._free_blocks(request)
                ignore.add(req_id)
                freed_finished += 1
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                request.status = RequestStatus.FINISHED_ABORTED
                self._free_blocks(request)
                ignore.add(req_id)
                freed_waiting_recv += 1
        if freed_finished or freed_waiting_recv:
            logger.info(
                "reset_prefix_cache: force-freed %d finished-but-delayed "
                "and %d WAITING_FOR_REMOTE_KVS requests (ignore-set size=%d).",
                freed_finished,
                freed_waiting_recv,
                len(ignore),
            )
        self.finished_recving_kv_req_ids.clear()
        self.failed_recving_kv_req_ids.clear()

    original = getattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)
    return original(self, reset_running_requests, reset_connector)


def _update_from_kv_xfer_finished_safe(self: Scheduler, kv_connector_output) -> None:
    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    ignore = _get_ignore_set(self)

    for req_id in kv_connector_output.finished_recving or ():
        if req_id in ignore:
            ignore.discard(req_id)
            continue
        if req_id not in self.requests:
            continue
        req = self.requests[req_id]
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(req)

    for req_id in kv_connector_output.finished_sending or ():
        if req_id in ignore:
            ignore.discard(req_id)
            continue
        if req_id not in self.requests:
            continue
        self._free_blocks(self.requests[req_id])


def _install_scheduler_patches() -> None:
    original = getattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR, None)
    if original is None:
        current_reset_prefix_cache = Scheduler.reset_prefix_cache
        original = getattr(current_reset_prefix_cache, "__globals__", {}).get(
            "_original_reset_prefix_cache", current_reset_prefix_cache
        )
        setattr(Scheduler, _DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR, original)

    Scheduler.reset_prefix_cache = _reset_prefix_cache_with_delay_free
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished_safe
    setattr(Scheduler, _DYNAMO_PATCHED_MARKER, True)


_install_scheduler_patches()


class PatchedAsyncScheduler(AsyncScheduler):
    """Trivial subclass whose import triggers the monkey-patches above.

    Used as the default ``scheduler_cls`` so EngineCore re-applies the
    patch in ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` mode, where import
    side effects from the parent are not inherited.
    """
