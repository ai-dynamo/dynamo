# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Dynamo vLLM Scheduler patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def scheduler():
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp  # noqa: F401  # applies patches

    s = MagicMock(spec=Scheduler)
    s.reset_prefix_cache = sp._reset_prefix_cache_with_delay_free.__get__(s, Scheduler)
    s._update_from_kv_xfer_finished = sp._update_from_kv_xfer_finished_safe.__get__(
        s, Scheduler
    )
    s.finished_recving_kv_req_ids = set()
    s.failed_recving_kv_req_ids = set()
    s.connector = None
    s.requests = {}
    s._free_blocks = MagicMock()
    return s


def _req(req_id: str, *, finished: bool = False, status=None):
    r = SimpleNamespace(request_id=req_id, status=status)
    r.is_finished = lambda: finished
    return r


@pytest.fixture
def no_op_original(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    monkeypatch.setattr(
        Scheduler,
        sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR,
        lambda *a, **kw: True,
    )


def test_reset_force_frees_finished_and_records_ignore(scheduler, no_op_original):
    # Mode 1 core: finished-but-delayed requests get _free_blocks'd and
    # tracked in the ignore set; unrelated active requests are untouched.
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    finished = _req("delayed", finished=True)
    active = _req("active", finished=False)
    scheduler.requests = {"delayed": finished, "active": active}

    scheduler.reset_prefix_cache(reset_running_requests=True)

    scheduler._free_blocks.assert_called_once_with(finished)
    assert getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR) == {"delayed"}


def test_reset_force_frees_waiting_for_remote_kvs(scheduler, no_op_original):
    # Decode-side case: WAITING_FOR_REMOTE_KVS requests are aborted and
    # freed so the BlockPool reset can proceed.
    from vllm.v1.request import RequestStatus

    waiting = _req("recv", status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler.requests = {"recv": waiting}

    scheduler.reset_prefix_cache(reset_running_requests=True)

    scheduler._free_blocks.assert_called_once_with(waiting)
    assert waiting.status == RequestStatus.FINISHED_ABORTED


def test_update_drops_ignored_signals_for_both_directions(scheduler):
    # Mode 2 core: late finished_sending/finished_recving for a
    # previously force-freed req_id must be dropped (no assert, no
    # _free_blocks). Ignore-set entries are consumed on drop.
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, {"stale-send", "stale-recv"})

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(
            finished_recving=["stale-recv"],
            finished_sending=["stale-send"],
        )
    )

    scheduler._free_blocks.assert_not_called()
    assert getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR) == set()


def test_update_drops_unknown_req_id_without_consuming_ignore(scheduler):
    # Defensive branch: any req_id missing from self.requests (even
    # without an ignore-set entry) is dropped instead of asserting. Unknown
    # req_ids must not consume unrelated ignore markers.
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, {"still-late"})
    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(finished_recving=[], finished_sending=["never-seen"])
    )
    scheduler._free_blocks.assert_not_called()
    assert getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR) == {"still-late"}


def test_update_preserves_live_finished_sending(scheduler):
    # Happy path: live req in self.requests still gets _free_blocks.
    live = _req("live", finished=True)
    scheduler.requests = {"live": live}

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(finished_recving=[], finished_sending=["live"])
    )

    scheduler._free_blocks.assert_called_once_with(live)


def test_module_patches_are_installed():
    # Verifies both monkey-patches took effect and PatchedAsyncScheduler
    # is usable as a scheduler_cls.
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert (
        Scheduler._update_from_kv_xfer_finished is sp._update_from_kv_xfer_finished_safe
    )
    assert issubclass(sp.PatchedAsyncScheduler, Scheduler)
    assert getattr(Scheduler, sp._DYNAMO_PATCHED_MARKER, False) is True


def test_patch_is_idempotent_on_reimport():
    # Reimport must keep the true original stored on Scheduler while rebinding
    # Scheduler methods to the current module's wrappers.
    import importlib

    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    original_reset = getattr(Scheduler, sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)
    importlib.reload(sp)
    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert (
        Scheduler._update_from_kv_xfer_finished is sp._update_from_kv_xfer_finished_safe
    )
    assert (
        getattr(Scheduler, sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)
        is original_reset
    )


def test_reloaded_wrapper_uses_class_stored_original(monkeypatch):
    import importlib

    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    monkeypatch.setattr(
        Scheduler,
        sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR,
        lambda *a, **kw: True,
    )

    importlib.reload(sp)

    scheduler = MagicMock(spec=Scheduler)
    assert sp._reset_prefix_cache_with_delay_free(scheduler) is True
