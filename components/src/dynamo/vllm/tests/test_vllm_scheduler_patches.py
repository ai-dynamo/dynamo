# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the temporary NIXL scheduler patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def scheduler(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    monkeypatch.setattr(
        Scheduler,
        sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR,
        lambda *args, **kwargs: True,
    )

    s = MagicMock(spec=Scheduler)
    s.reset_prefix_cache = sp._reset_prefix_cache_with_delay_free.__get__(s, Scheduler)
    s._update_from_kv_xfer_finished = sp._update_from_kv_xfer_finished_safe.__get__(
        s, Scheduler
    )
    s.finished_recving_kv_req_ids = set()
    s.failed_recving_kv_req_ids = set()
    s.connector = None
    s.requests = {}
    s.waiting = MagicMock()
    s.skipped_waiting = MagicMock()
    s._free_blocks = MagicMock()
    return s


def _req(req_id: str, *, finished: bool = False, status=None):
    request = SimpleNamespace(request_id=req_id, status=status)
    request.is_finished = lambda: finished
    return request


def test_reset_frees_finished_delay_free_request(scheduler):
    """Finished NIXL-prefill requests may still pin KV blocks."""

    import dynamo.vllm.scheduler_patches as sp

    delayed = _req("delayed", finished=True)
    active = _req("active")
    scheduler.requests = {"delayed": delayed, "active": active}

    assert scheduler.reset_prefix_cache(reset_running_requests=True) is True

    scheduler._free_blocks.assert_called_once_with(delayed)
    assert getattr(scheduler, sp._IGNORE_LATE_KV_XFER_ATTR) == {"delayed"}


def test_reset_frees_waiting_for_remote_kvs_request(scheduler):
    """Decode-side WAITING_FOR_REMOTE_KVS requests must not block sleep."""

    from vllm.v1.request import RequestStatus

    waiting = _req("recv", status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler.requests = {"recv": waiting}

    assert scheduler.reset_prefix_cache(reset_running_requests=True) is True

    scheduler._free_blocks.assert_called_once_with(waiting)
    scheduler.waiting.remove_requests.assert_called_once_with([waiting])
    scheduler.skipped_waiting.remove_requests.assert_called_once_with([waiting])
    assert waiting.status == RequestStatus.FINISHED_ABORTED


def test_late_finished_signals_for_force_freed_requests_are_dropped(scheduler):
    """Late NIXL completion events for force-freed req_ids are harmless."""

    import dynamo.vllm.scheduler_patches as sp

    setattr(scheduler, sp._IGNORE_LATE_KV_XFER_ATTR, {"stale-send", "stale-recv"})

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(
            finished_recving={"stale-recv"},
            finished_sending={"stale-send"},
        )
    )

    scheduler._free_blocks.assert_not_called()
    assert getattr(scheduler, sp._IGNORE_LATE_KV_XFER_ATTR) == set()


def test_unknown_finished_signal_is_dropped(scheduler):
    """Do not assert if vLLM reports a completion for a vanished request."""

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(finished_recving=set(), finished_sending={"unknown"})
    )

    scheduler._free_blocks.assert_not_called()


def test_live_finished_signal_keeps_upstream_behavior(scheduler):
    """Live requests still use the normal _free_blocks path."""

    live = _req("live", finished=True)
    scheduler.requests = {"live": live}

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(finished_recving=set(), finished_sending={"live"})
    )

    scheduler._free_blocks.assert_called_once_with(live)


def test_patch_installs_importable_scheduler_cls():
    """Importing the scheduler class installs both monkey patches."""

    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert (
        Scheduler._update_from_kv_xfer_finished is sp._update_from_kv_xfer_finished_safe
    )
    assert issubclass(sp.PatchedAsyncScheduler, Scheduler)


def test_reimport_preserves_original_reset_method():
    """Reloading should not wrap the wrapper as the upstream original."""

    import importlib

    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    original_reset = getattr(Scheduler, sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)

    importlib.reload(sp)

    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert (
        getattr(Scheduler, sp._DYNAMO_ORIGINAL_RESET_PREFIX_CACHE_ATTR)
        is original_reset
    )
