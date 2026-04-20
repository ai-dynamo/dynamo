# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Dynamo scheduler patch that force-frees finished-but-delayed
requests during reset_prefix_cache.

Context: on a prefill worker with NixlConnector, finished requests are left in
``scheduler.requests`` with blocks still allocated, waiting for the remote
decode worker to pull the KV. Sleep calls
``Scheduler.reset_prefix_cache(reset_running_requests=True)`` which must succeed
even in that state.
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
def scheduler_with_patch():
    """Apply the Dynamo patch and return a mock scheduler instance."""
    # Import lazily so the patch is applied inside the test scope.
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches  # noqa: F401  # applies monkey-patch

    scheduler = MagicMock(spec=Scheduler)
    scheduler.reset_prefix_cache = (
        dynamo.vllm.scheduler_patches._reset_prefix_cache_with_delay_free.__get__(
            scheduler, Scheduler
        )
    )
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    return scheduler


def _make_request(req_id: str, finished: bool):
    req = SimpleNamespace(request_id=req_id)
    req.is_finished = lambda finished=finished: finished
    return req


def test_patch_force_frees_finished_requests(scheduler_with_patch):
    finished_req = _make_request("req-delayed-1", finished=True)
    active_req = _make_request("req-active", finished=False)
    scheduler_with_patch.requests = {
        "req-delayed-1": finished_req,
        "req-active": active_req,
    }
    # Original method is bound on the class — stub out so the patch delegates
    # to a known result.
    scheduler_with_patch._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        result = scheduler_with_patch.reset_prefix_cache(reset_running_requests=True)

    assert result is True
    scheduler_with_patch._free_blocks.assert_called_once_with(finished_req)


def test_patch_is_noop_when_reset_running_requests_false(scheduler_with_patch):
    finished_req = _make_request("req-delayed-1", finished=True)
    scheduler_with_patch.requests = {"req-delayed-1": finished_req}
    scheduler_with_patch._free_blocks = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: False)
        result = scheduler_with_patch.reset_prefix_cache(reset_running_requests=False)

    assert result is False
    scheduler_with_patch._free_blocks.assert_not_called()


def test_patch_clears_kv_connector_tracking_sets(scheduler_with_patch):
    scheduler_with_patch.requests = {}
    scheduler_with_patch.finished_recving_kv_req_ids = {"a", "b"}
    scheduler_with_patch.failed_recving_kv_req_ids = {"c"}

    with pytest.MonkeyPatch.context() as mp:
        import dynamo.vllm.scheduler_patches as sp

        mp.setattr(sp, "_original_reset_prefix_cache", lambda *a, **kw: True)
        scheduler_with_patch.reset_prefix_cache(reset_running_requests=True)

    assert scheduler_with_patch.finished_recving_kv_req_ids == set()
    assert scheduler_with_patch.failed_recving_kv_req_ids == set()


def test_patched_async_scheduler_module_importable():
    """Importing the patch module should apply the monkey-patch on Scheduler."""
    from vllm.v1.core.sched.scheduler import Scheduler

    import dynamo.vllm.scheduler_patches as sp

    assert Scheduler.reset_prefix_cache is sp._reset_prefix_cache_with_delay_free
    assert issubclass(sp.PatchedAsyncScheduler, Scheduler)
