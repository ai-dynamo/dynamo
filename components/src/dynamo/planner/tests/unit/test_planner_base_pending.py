# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pending-desired worker count tracking in NativePlannerBase.

`GlobalPlannerConnector` cannot observe DGD status and always reports
``is_stable=True``. Without an extra tracking mechanism, ``_scaling_in_progress``
would treat the system as stable during a rollout and fire new scaling
decisions on top of pending ones. ``NativePlannerBase`` therefore records the
last desired prefill/decode targets itself and reports them as ``expected_*``
until discovery catches up.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(scope="module", autouse=True)
def _stub_heavy_deps():
    # Stub optional Rust/IO-heavy dynamo modules when absent so planner.core.base
    # imports cleanly. Scoped to this module and torn down after so sibling test
    # modules see the real modules.
    import sys
    import types

    stubs = {
        "dynamo._core": {
            "Client": MagicMock,
            "DistributedRuntime": MagicMock,
            "VirtualConnectorCoordinator": MagicMock,
        },
        "dynamo.runtime": {
            "DistributedRuntime": MagicMock,
            "dynamo_worker": lambda: lambda f: f,
        },
        "dynamo.runtime.logging": {
            "configure_dynamo_logging": lambda: None,
        },
        "dynamo.llm": {
            "FpmEventSubscriber": MagicMock,
            "FpmEventRelay": MagicMock,
        },
        "dynamo.common.forward_pass_metrics": {
            "ForwardPassMetrics": MagicMock,
            "ScheduledRequestMetrics": MagicMock,
        },
    }
    mp = pytest.MonkeyPatch()
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        mp.setitem(sys.modules, name, mod)
    yield
    mp.undo()


@pytest.fixture
def planner():
    """Bare NativePlannerBase instance with only the state the methods under test touch."""
    from dynamo.planner.core.base import NativePlannerBase

    inst = NativePlannerBase.__new__(NativePlannerBase)
    inst._pending_desired_prefill = None
    inst._pending_desired_decode = None
    inst._pending_desired_prefill_ts = None
    inst._pending_desired_decode_ts = None
    inst.require_prefill = True
    inst.require_decode = True
    inst.connector = AsyncMock()
    cfg = MagicMock()
    cfg.advisory = False
    cfg.throughput_adjustment_interval = 180
    inst.config = cfg
    # Stub the raw worker-count source; individual tests set the return value.
    inst._get_worker_counts_raw = AsyncMock(return_value=(0, 0, True))
    return inst


# ---------------------------------------------------------------------------
# _collect_worker_counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_worker_counts_no_pending_stable(planner):
    """Baseline: no pending target, connector reports stable → expected == ready."""
    planner._get_worker_counts_raw.return_value = (2, 3, True)
    counts = await planner._collect_worker_counts()

    assert counts.ready_num_prefill == 2
    assert counts.ready_num_decode == 3
    assert counts.expected_num_prefill == 2
    assert counts.expected_num_decode == 3


@pytest.mark.asyncio
async def test_collect_worker_counts_no_pending_unstable(planner):
    """Unstable connector without a pending target → expected is None."""
    planner._get_worker_counts_raw.return_value = (2, 3, False)
    counts = await planner._collect_worker_counts()

    assert counts.ready_num_prefill == 2
    assert counts.ready_num_decode == 3
    assert counts.expected_num_prefill is None
    assert counts.expected_num_decode is None


@pytest.mark.asyncio
async def test_collect_worker_counts_pending_matches_clears_pending(planner):
    """Pending target that matches discovery is cleared and expected == ready."""
    import time as _t

    now = _t.time()
    planner._pending_desired_prefill = 4
    planner._pending_desired_decode = 6
    planner._pending_desired_prefill_ts = now
    planner._pending_desired_decode_ts = now
    planner._get_worker_counts_raw.return_value = (4, 6, True)

    counts = await planner._collect_worker_counts()

    assert planner._pending_desired_prefill is None
    assert planner._pending_desired_decode is None
    assert planner._pending_desired_prefill_ts is None
    assert planner._pending_desired_decode_ts is None
    assert counts.expected_num_prefill == 4
    assert counts.expected_num_decode == 6


@pytest.mark.asyncio
async def test_collect_worker_counts_pending_mismatch_reports_pending(planner):
    """Pending target that does not match discovery yet → expected == pending."""
    import time as _t

    now = _t.time()
    planner._pending_desired_prefill = 5
    planner._pending_desired_decode = 7
    planner._pending_desired_prefill_ts = now
    planner._pending_desired_decode_ts = now
    planner._get_worker_counts_raw.return_value = (2, 3, True)

    counts = await planner._collect_worker_counts()

    # Pending is retained until discovery catches up.
    assert planner._pending_desired_prefill == 5
    assert planner._pending_desired_decode == 7
    assert counts.ready_num_prefill == 2
    assert counts.ready_num_decode == 3
    # expected_* reports the *pending* target so _scaling_in_progress sees
    # expected != actual and blocks new decisions.
    assert counts.expected_num_prefill == 5
    assert counts.expected_num_decode == 7


@pytest.mark.asyncio
async def test_collect_worker_counts_independent_axes(planner):
    """Prefill and decode pending tracking are independent."""
    import time as _t

    now = _t.time()
    planner._pending_desired_prefill = 5  # mismatch
    planner._pending_desired_decode = 3  # will match
    planner._pending_desired_prefill_ts = now
    planner._pending_desired_decode_ts = now
    planner._get_worker_counts_raw.return_value = (2, 3, True)

    counts = await planner._collect_worker_counts()

    assert planner._pending_desired_prefill == 5  # still pending
    assert planner._pending_desired_decode is None  # cleared
    assert counts.expected_num_prefill == 5
    assert counts.expected_num_decode == 3


@pytest.mark.asyncio
async def test_collect_worker_counts_pending_expires_when_stale(planner):
    """A pending target older than the expiry window is cleared with a warning.

    Without expiry, a rejected/aborted GlobalPlanner request would permanently
    silence local scaling because ``expected != actual`` would never resolve.
    """
    import time as _t

    # _PENDING_EXPIRY_INTERVALS (5) * throughput_adjustment_interval (180) = 900s.
    stale = _t.time() - 1000
    planner._pending_desired_prefill = 5
    planner._pending_desired_prefill_ts = stale
    planner._get_worker_counts_raw.return_value = (2, 3, True)

    counts = await planner._collect_worker_counts()

    assert planner._pending_desired_prefill is None
    assert planner._pending_desired_prefill_ts is None
    # Falls back to the baseline (num_p when stable).
    assert counts.expected_num_prefill == 2


@pytest.mark.asyncio
async def test_collect_worker_counts_pending_recent_does_not_expire(planner):
    """A pending target inside the expiry window is retained."""
    import time as _t

    # Well within the 900s window.
    planner._pending_desired_prefill = 5
    planner._pending_desired_prefill_ts = _t.time() - 60
    planner._get_worker_counts_raw.return_value = (2, 3, True)

    counts = await planner._collect_worker_counts()

    assert planner._pending_desired_prefill == 5
    assert counts.expected_num_prefill == 5


@pytest.mark.asyncio
async def test_collect_worker_counts_require_flags_gate_output(planner):
    """require_prefill / require_decode set to False nulls out that axis."""
    planner.require_prefill = False
    planner.require_decode = True
    planner._get_worker_counts_raw.return_value = (2, 3, True)

    counts = await planner._collect_worker_counts()

    assert counts.ready_num_prefill is None
    assert counts.expected_num_prefill is None
    assert counts.ready_num_decode == 3
    assert counts.expected_num_decode == 3


# ---------------------------------------------------------------------------
# _apply_scaling_targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_scaling_targets_records_pending_prefill(planner):
    import time as _t

    from dynamo.planner.config.defaults import SubComponentType, TargetReplica

    before = _t.time()
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=4)
    ]
    await planner._apply_scaling_targets(targets, blocking=False)

    planner.connector.set_component_replicas.assert_awaited_once_with(
        targets, blocking=False
    )
    assert planner._pending_desired_prefill == 4
    assert planner._pending_desired_decode is None
    assert planner._pending_desired_prefill_ts is not None
    assert planner._pending_desired_prefill_ts >= before
    assert planner._pending_desired_decode_ts is None


@pytest.mark.asyncio
async def test_apply_scaling_targets_records_pending_decode(planner):
    import time as _t

    from dynamo.planner.config.defaults import SubComponentType, TargetReplica

    before = _t.time()
    targets = [
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=6)
    ]
    await planner._apply_scaling_targets(targets, blocking=True)

    planner.connector.set_component_replicas.assert_awaited_once_with(
        targets, blocking=True
    )
    assert planner._pending_desired_prefill is None
    assert planner._pending_desired_decode == 6
    assert planner._pending_desired_prefill_ts is None
    assert planner._pending_desired_decode_ts is not None
    assert planner._pending_desired_decode_ts >= before


@pytest.mark.asyncio
async def test_apply_scaling_targets_records_both(planner):
    from dynamo.planner.config.defaults import SubComponentType, TargetReplica

    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=2),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=5),
    ]
    await planner._apply_scaling_targets(targets)

    assert planner._pending_desired_prefill == 2
    assert planner._pending_desired_decode == 5
    assert planner._pending_desired_prefill_ts is not None
    assert planner._pending_desired_decode_ts is not None


@pytest.mark.asyncio
async def test_apply_scaling_targets_advisory_does_not_track_or_send(planner):
    """Advisory mode must not push to the connector and must not mark pending."""
    from dynamo.planner.config.defaults import SubComponentType, TargetReplica

    planner.config.advisory = True
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    await planner._apply_scaling_targets(targets)

    planner.connector.set_component_replicas.assert_not_awaited()
    assert planner._pending_desired_prefill is None
    assert planner._pending_desired_decode is None
    assert planner._pending_desired_prefill_ts is None
    assert planner._pending_desired_decode_ts is None


@pytest.mark.asyncio
async def test_apply_scaling_targets_empty_is_noop(planner):
    await planner._apply_scaling_targets([])

    planner.connector.set_component_replicas.assert_not_awaited()
    assert planner._pending_desired_prefill is None
    assert planner._pending_desired_decode is None
    assert planner._pending_desired_prefill_ts is None
    assert planner._pending_desired_decode_ts is None
