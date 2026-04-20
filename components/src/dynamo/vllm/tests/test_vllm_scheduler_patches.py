# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Dynamo scheduler patches.

Covers:
- ``reset_prefix_cache`` force-freeing KV-pinned requests (prefill
  finished-but-delayed and decode WAITING_FOR_REMOTE_KVS).
- ``_update_from_kv_xfer_finished`` safely dropping late signals for
  already-force-freed req_ids (NIXL orphan-timer race).
- Module idempotency on double-import.
"""

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
    """A mock Scheduler with our two patches bound to it."""
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
    return s


def _make_finished_request(req_id: str):
    req = SimpleNamespace(request_id=req_id, status=None)
    req.is_finished = lambda: True
    return req


def _make_active_request(req_id: str):
    req = SimpleNamespace(request_id=req_id, status=None)
    req.is_finished = lambda: False
    return req


def _make_waiting_recv_request(req_id: str):
    from vllm.v1.request import RequestStatus

    req = SimpleNamespace(
        request_id=req_id, status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    req.is_finished = lambda: False
    return req


def test_reset_force_frees_finished_requests(scheduler):
    finished_req = _make_finished_request("req-delayed-1")
    active_req = _make_active_request("req-active")
    scheduler.requests = {
        "req-delayed-1": finished_req,
        "req-active": active_req,
    }
    scheduler._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        result = scheduler.reset_prefix_cache(reset_running_requests=True)

    assert result is True
    scheduler._free_blocks.assert_called_once_with(finished_req)


def test_reset_force_frees_waiting_for_remote_kvs(scheduler):
    from vllm.v1.request import RequestStatus

    waiting_req = _make_waiting_recv_request("req-recv-1")
    scheduler.requests = {"req-recv-1": waiting_req}
    scheduler._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        scheduler.reset_prefix_cache(reset_running_requests=True)

    scheduler._free_blocks.assert_called_once_with(waiting_req)
    assert waiting_req.status == RequestStatus.FINISHED_ABORTED


def test_reset_is_noop_when_reset_running_requests_false(scheduler):
    finished_req = _make_finished_request("req-delayed-1")
    scheduler.requests = {"req-delayed-1": finished_req}
    scheduler._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: False)
        result = scheduler.reset_prefix_cache(reset_running_requests=False)

    assert result is False
    scheduler._free_blocks.assert_not_called()


def test_reset_clears_kv_connector_tracking_sets(scheduler):
    scheduler.finished_recving_kv_req_ids = {"a", "b"}
    scheduler.failed_recving_kv_req_ids = {"c"}

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        scheduler.reset_prefix_cache(reset_running_requests=True)

    assert scheduler.finished_recving_kv_req_ids == set()
    assert scheduler.failed_recving_kv_req_ids == set()


def test_reset_populates_ignore_set_with_force_freed_ids(scheduler):
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    finished_req = _make_finished_request("req-finished")
    waiting_req = _make_waiting_recv_request("req-waiting")
    scheduler.requests = {
        "req-finished": finished_req,
        "req-waiting": waiting_req,
    }
    scheduler._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        scheduler.reset_prefix_cache(reset_running_requests=True)

    ignored = getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR)
    assert ignored == {"req-finished", "req-waiting"}


def test_update_from_kv_xfer_finished_drops_ignored_finished_sending(scheduler):
    """After reset force-freed a req_id, a late finished_sending for it
    must NOT assert — it must be dropped silently.
    """
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, {"stale-req-id"})
    scheduler.requests = {}
    scheduler._free_blocks = MagicMock()

    output = SimpleNamespace(
        finished_recving=[],
        finished_sending=["stale-req-id"],
    )
    scheduler._update_from_kv_xfer_finished(output)

    scheduler._free_blocks.assert_not_called()
    assert getattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR) == set()


def test_update_from_kv_xfer_finished_drops_unknown_req_id(scheduler):
    """A finished_sending for a req_id that isn't in self.requests and
    isn't in the ignore set must still be dropped rather than asserting.
    """
    scheduler.requests = {}
    scheduler._free_blocks = MagicMock()

    output = SimpleNamespace(
        finished_recving=[],
        finished_sending=["never-seen-req-id"],
    )
    scheduler._update_from_kv_xfer_finished(output)
    scheduler._free_blocks.assert_not_called()


def test_update_from_kv_xfer_finished_preserves_live_finished_sending(scheduler):
    """Live requests' finished_sending must still call _free_blocks."""
    live_req = _make_finished_request("live-req")
    scheduler.requests = {"live-req": live_req}
    scheduler._free_blocks = MagicMock()

    output = SimpleNamespace(
        finished_recving=[],
        finished_sending=["live-req"],
    )
    scheduler._update_from_kv_xfer_finished(output)
    scheduler._free_blocks.assert_called_once_with(live_req)


def test_update_from_kv_xfer_finished_drops_ignored_finished_recving(scheduler):
    from dynamo.vllm.scheduler_patches import _IGNORE_LATE_KV_XFER_ATTR

    setattr(scheduler, _IGNORE_LATE_KV_XFER_ATTR, {"stale"})
    scheduler.requests = {}
    scheduler._free_blocks = MagicMock()

    output = SimpleNamespace(
        finished_recving=["stale"],
        finished_sending=[],
    )
    scheduler._update_from_kv_xfer_finished(output)
    scheduler._free_blocks.assert_not_called()


def test_module_patches_applied_and_class_is_subclass():
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert (
        Scheduler._update_from_kv_xfer_finished is sp._update_from_kv_xfer_finished_safe
    )
    assert issubclass(sp.PatchedAsyncScheduler, Scheduler)
    assert getattr(Scheduler, sp._DYNAMO_PATCHED_MARKER, False) is True


def test_patch_is_idempotent_on_reimport():
    """Reimporting scheduler_patches must not re-wrap already-patched
    methods. Without the idempotency marker, the second import would
    capture the already-patched method as the original and every call
    would infinite-recurse.
    """
    import importlib

    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    patched_reset = Scheduler.reset_prefix_cache
    patched_update = Scheduler._update_from_kv_xfer_finished
    importlib.reload(sp)
    assert Scheduler.reset_prefix_cache is patched_reset
    assert Scheduler._update_from_kv_xfer_finished is patched_update
