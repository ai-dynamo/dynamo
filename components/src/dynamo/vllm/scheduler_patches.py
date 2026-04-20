# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM scheduler patches applied in the EngineCore subprocess.

Imports here monkey-patch two ``vllm.v1.core.sched.scheduler.Scheduler``
methods so that ``/engine/sleep`` (and any other code path that calls
``reset_prefix_cache(reset_running_requests=True)``) succeeds on a prefill
or decode worker with in-flight NIXL KV transfers.

Without this patch, the Dynamo vLLM sleep path fails in two distinct places:

    (1) Immediate failure — the reported bug
    ==========================================
    AsyncLLM.pause_generation(clear_cache=True)
      └── EngineCore.pause_scheduler(clear_cache=True)
           └── EngineCore._reset_caches(reset_running_requests=True)
                └── Scheduler.reset_prefix_cache(reset_running_requests=True)
                     │
                     ├── only frees self.running
                     └── RuntimeError: "Failed to reset KV cache even when
                         all the running requests are preempted... running
                         requests waiting for remote KV transfer, which is
                         not supported yet."

    Root cause: NixlConnector marks finished prefill requests with
    ``delay_free_blocks=True`` so their blocks remain pinned while the
    remote decode worker pulls the KV. Those requests live in
    ``self.requests`` but NOT in ``self.running`` / ``self.waiting``, so
    vLLM's reset_prefix_cache never frees them, BlockPool reports non-zero
    ref counts, and raises. The decode-side equivalent fires when a
    request is stuck in ``WAITING_FOR_REMOTE_KVS`` at sleep time.

    Fix: iterate ``self.requests`` during ``reset_prefix_cache`` and
    force-free every request still holding KV blocks via ``_free_blocks``.

    (2) Latent failure — the NIXL orphan race
    ==========================================
    After a force-free, the NixlConnector worker still has the req_id in
    its outstanding-send bookkeeping. ``VLLM_NIXL_ABORT_REQUEST_TIMEOUT``
    seconds later (default 480 s), the worker's ``get_finished()`` fires
    the orphan cleanup ("Releasing expired KV blocks... retrieved by 0
    decode worker(s)") and signals ``finished_sending`` back to the
    scheduler for a req_id that was force-freed minutes earlier.
    ``Scheduler._update_from_kv_xfer_finished`` then hits
    ``assert req_id in self.requests``, EngineCore dies, pod restarts.

    Fix: every req_id force-freed by the patch above is recorded on the
    scheduler. A second patch on ``_update_from_kv_xfer_finished`` drops
    any late ``finished_sending`` / ``finished_recving`` entry whose
    req_id is either in that ignore set or no longer in ``self.requests``.
    Defensive-by-default — any late signal for a vanished request is a
    no-op instead of a fatal assertion.

Delivery: ``dynamo.vllm.args`` imports this module unconditionally at
engine-config build time. Under the default ``fork`` multiprocessing
method, the parent applies the patch and the EngineCore subprocess
inherits it. Under ``spawn``, the default ``scheduler_cls`` is set to
``dynamo.vllm.scheduler_patches.PatchedAsyncScheduler`` so the subprocess
re-imports this module and re-applies the patch on its own copy of
``Scheduler``.
"""

from __future__ import annotations

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

logger = logging.getLogger(__name__)

# Idempotency marker: applying both monkey-patches twice would capture the
# already-patched originals and infinite-recurse. Python module caching
# normally prevents this, but editable installs with duplicate sys.path
# entries have hit double-import in the past.
_DYNAMO_PATCHED_MARKER = "_dynamo_scheduler_patched"

# Per-Scheduler attribute name for the "ignore any late KV-xfer signal for
# these request ids" set. Populated by the reset_prefix_cache patch,
# consumed by the _update_from_kv_xfer_finished patch. Created lazily on
# first use.
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
                # Prefill finished-but-delayed (NixlConnector
                # delay_free_blocks=True) or aborted-while-waiting-for-
                # remote-kvs requests.
                self._free_blocks(request)
                ignore.add(req_id)
                freed_finished += 1
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Decode-side: blocks allocated at schedule time, normally
                # freed only when the remote send signals completion.
                # Caller is about to wipe KV memory anyway.
                request.status = RequestStatus.FINISHED_ABORTED
                self._free_blocks(request)
                ignore.add(req_id)
                freed_waiting_recv += 1
        if freed_finished or freed_waiting_recv:
            logger.info(
                "reset_prefix_cache: force-freed %d finished-but-delayed "
                "and %d WAITING_FOR_REMOTE_KVS requests before cache "
                "reset (ignore-set size=%d).",
                freed_finished,
                freed_waiting_recv,
                len(ignore),
            )
        self.finished_recving_kv_req_ids.clear()
        self.failed_recving_kv_req_ids.clear()

    return _original_reset_prefix_cache(self, reset_running_requests, reset_connector)


def _update_from_kv_xfer_finished_safe(self: Scheduler, kv_connector_output) -> None:
    """Defensive replacement for ``Scheduler._update_from_kv_xfer_finished``.

    Upstream asserts ``req_id in self.requests`` for every
    ``finished_sending`` / ``finished_recving`` entry. That assertion is
    fatal when the request was force-freed earlier by our
    ``reset_prefix_cache`` patch (e.g. during ``/engine/sleep``). Drop
    those entries instead of killing the engine.
    """
    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    ignore = _get_ignore_set(self)

    for req_id in kv_connector_output.finished_recving or ():
        if req_id in ignore or req_id not in self.requests:
            ignore.discard(req_id)
            logger.debug(
                "Dropping late finished_recving for stale request %s "
                "(force-freed by /engine/sleep or similar).",
                req_id,
            )
            continue
        req = self.requests[req_id]
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        else:
            assert RequestStatus.is_finished(req.status)
            self._free_blocks(self.requests[req_id])

    for req_id in kv_connector_output.finished_sending or ():
        if req_id in ignore or req_id not in self.requests:
            ignore.discard(req_id)
            logger.debug(
                "Dropping late finished_sending for stale request %s "
                "(force-freed by /engine/sleep or similar).",
                req_id,
            )
            continue
        self._free_blocks(self.requests[req_id])


if not getattr(Scheduler, _DYNAMO_PATCHED_MARKER, False):
    _original_reset_prefix_cache = Scheduler.reset_prefix_cache
    _original_update_from_kv_xfer_finished = Scheduler._update_from_kv_xfer_finished
    Scheduler.reset_prefix_cache = _reset_prefix_cache_with_delay_free
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished_safe
    setattr(Scheduler, _DYNAMO_PATCHED_MARKER, True)


class PatchedAsyncScheduler(AsyncScheduler):
    """``AsyncScheduler`` subclass whose only purpose is to cause
    ``dynamo.vllm.scheduler_patches`` to be imported in the EngineCore
    subprocess when specified as ``--scheduler-cls``.

    The monkey-patches above are applied at module import time, so the
    choice of scheduler class does not matter for correctness — any
    import path into this module is sufficient. ``dynamo.vllm.args``
    imports this module directly and unconditionally, so this class is
    mainly needed under ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` where the
    EngineCore subprocess does not inherit the parent's import side
    effects.
    """
