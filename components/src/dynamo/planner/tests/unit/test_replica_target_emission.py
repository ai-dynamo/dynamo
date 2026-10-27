# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replica-target emission from each native planner mode's ``_apply_effects``."""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.adapters import (
    AggPlanner,
    DecodePlanner,
    DisaggPlanner,
    PrefillPlanner,
)
from dynamo.planner.core.types import PlannerEffects, ScalingDecision
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PREFILL = SubComponentType.PREFILL
DECODE = SubComponentType.DECODE
GAUGES = {
    PREFILL: "predicted_num_prefill_replicas",
    DECODE: "predicted_num_decode_replicas",
}
FIELDS = {PREFILL: "num_prefill", DECODE: "num_decode"}
WORKER_NAMES = {PREFILL: "prefill-worker", DECODE: "decode-worker"}

# (planner class, config mode, roles the mode owns)
MODES = [
    pytest.param(PrefillPlanner, "prefill", (PREFILL,), id="prefill"),
    pytest.param(DecodePlanner, "decode", (DECODE,), id="decode"),
    pytest.param(AggPlanner, "agg", (DECODE,), id="agg"),
    pytest.param(DisaggPlanner, "disagg", (PREFILL, DECODE), id="disagg"),
]
SINGLE_ROLE_MODES = [
    pytest.param(PrefillPlanner, "prefill", PREFILL, DECODE, id="prefill"),
    pytest.param(DecodePlanner, "decode", DECODE, PREFILL, id="decode"),
    pytest.param(AggPlanner, "agg", DECODE, PREFILL, id="agg"),
]


@pytest.fixture
def prometheus_port():
    """Choose a nonzero port for metrics while the HTTP server is mocked."""
    # start_http_server is patched, so the port is never bound; it only has to
    # be nonzero to enable gauge publication.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_planner(cls, mode, *, roles_with_info, port, advisory=False):
    """Create a planner with mocked metrics and the specified worker information."""
    state = DeploymentState()
    if PREFILL in roles_with_info:
        state.prefill.info = WorkerInfo(k8s_name=WORKER_NAMES[PREFILL])
    if DECODE in roles_with_info:
        state.decode.info = WorkerInfo(k8s_name=WORKER_NAMES[DECODE])
    environment = MagicMock()
    environment.deployment_state.return_value = state
    environment.apply_scaling = AsyncMock()

    config = PlannerConfig(
        mode=mode,
        advisory=advisory,
        namespace="test-namespace",
        metric_reporting_prometheus_port=port,
        live_dashboard_port=0,
        report_interval_hours=None,
    )
    with (
        patch(
            "dynamo.planner.core.base.PlannerPrometheusMetrics",
            return_value=MagicMock(),
        ),
        patch("dynamo.planner.core.base.start_http_server"),
    ):
        planner = cls(None, config, environment)
    return planner, environment


def _gauge(planner, role):
    """Return the mock that records updates to the selected role's gauge."""
    return getattr(planner.prometheus_metrics, GAUGES[role]).set


def _applied(environment):
    """Targets of every apply_scaling call as (role, name, replicas) tuples."""
    return [
        [
            (t.sub_component_type, t.component_name, t.desired_replicas)
            for t in c.args[0]
        ]
        for c in environment.apply_scaling.await_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("desired", [3, 0])
@pytest.mark.parametrize("cls,mode,roles", MODES)
async def test_own_roles_publish_gauge_and_submit_one_batch(
    cls, mode, roles, desired, prometheus_port
):
    """Check that each owned role publishes its target in one ordered batch."""
    # 0 is a real target: it must be published and submitted like any other.
    planner, environment = _make_planner(
        cls, mode, roles_with_info=roles, port=prometheus_port
    )
    decision = ScalingDecision(**{FIELDS[r]: desired for r in roles})

    await planner._apply_effects(PlannerEffects(scale_to=decision))

    for role in roles:
        _gauge(planner, role).assert_called_once_with(desired)
    assert _applied(environment) == [
        [(role, WORKER_NAMES[role], desired) for role in roles]
    ]
    for call_ in environment.apply_scaling.await_args_list:
        assert call_.kwargs == {"blocking": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,mode,roles", MODES)
async def test_absent_targets_publish_and_submit_nothing(
    cls, mode, roles, prometheus_port
):
    """Check that missing targets leave gauges and scaling unchanged."""
    planner, environment = _make_planner(
        cls, mode, roles_with_info=roles, port=prometheus_port
    )

    await planner._apply_effects(PlannerEffects(scale_to=ScalingDecision()))
    await planner._apply_effects(PlannerEffects(scale_to=None))

    for role in (PREFILL, DECODE):
        _gauge(planner, role).assert_not_called()
    environment.apply_scaling.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,mode,own,other", SINGLE_ROLE_MODES)
async def test_single_role_modes_ignore_other_role(
    cls, mode, own, other, prometheus_port
):
    """Check that a mode never publishes or resolves the role it does not own."""
    # The other role has no worker info; resolving it would raise.
    planner, environment = _make_planner(
        cls, mode, roles_with_info=(own,), port=prometheus_port
    )

    await planner._apply_effects(
        PlannerEffects(scale_to=ScalingDecision(**{FIELDS[other]: 5}))
    )
    _gauge(planner, other).assert_not_called()
    environment.apply_scaling.assert_not_called()

    await planner._apply_effects(
        PlannerEffects(scale_to=ScalingDecision(**{FIELDS[own]: 2, FIELDS[other]: 5}))
    )
    _gauge(planner, own).assert_called_once_with(2)
    _gauge(planner, other).assert_not_called()
    assert _applied(environment) == [[(own, WORKER_NAMES[own], 2)]]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,mode,roles", MODES)
async def test_advisory_publishes_gauges_without_scaling(
    cls, mode, roles, prometheus_port
):
    """Check that advisory mode publishes targets without applying scaling."""
    planner, environment = _make_planner(
        cls, mode, roles_with_info=roles, port=prometheus_port, advisory=True
    )
    decision = ScalingDecision(**{FIELDS[r]: 4 for r in roles})

    await planner._apply_effects(PlannerEffects(scale_to=decision))

    for role in roles:
        _gauge(planner, role).assert_called_once_with(4)
    environment.apply_scaling.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [PREFILL, DECODE])
async def test_disagg_with_one_role_set_submits_only_that_role(role, prometheus_port):
    """Check that a disaggregated target needs only its own worker information."""
    planner, environment = _make_planner(
        DisaggPlanner, "disagg", roles_with_info=(role,), port=prometheus_port
    )

    await planner._apply_effects(
        PlannerEffects(scale_to=ScalingDecision(**{FIELDS[role]: 1}))
    )

    other = DECODE if role == PREFILL else PREFILL
    _gauge(planner, role).assert_called_once_with(1)
    _gauge(planner, other).assert_not_called()
    assert _applied(environment) == [[(role, WORKER_NAMES[role], 1)]]


@pytest.mark.asyncio
@pytest.mark.parametrize("roles_with_info", [(), (PREFILL,), (DECODE,)])
async def test_disagg_publishes_both_gauges_before_resolving_workers(
    roles_with_info, prometheus_port
):
    """Check that both gauges update even when required worker information is missing."""
    planner, environment = _make_planner(
        DisaggPlanner, "disagg", roles_with_info=roles_with_info, port=prometheus_port
    )

    with pytest.raises(RuntimeError, match="Missing worker info"):
        await planner._apply_effects(
            PlannerEffects(scale_to=ScalingDecision(num_prefill=2, num_decode=3))
        )

    _gauge(planner, PREFILL).assert_called_once_with(2)
    _gauge(planner, DECODE).assert_called_once_with(3)
    environment.apply_scaling.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,mode,roles", MODES)
async def test_prometheus_port_zero_skips_gauges_but_still_scales(cls, mode, roles):
    """Check that disabling metrics leaves scaling targets unchanged."""
    planner, environment = _make_planner(cls, mode, roles_with_info=roles, port=0)
    decision = ScalingDecision(**{FIELDS[r]: 2 for r in roles})

    await planner._apply_effects(PlannerEffects(scale_to=decision))

    for role in (PREFILL, DECODE):
        _gauge(planner, role).assert_not_called()
    assert _applied(environment) == [[(role, WORKER_NAMES[role], 2) for role in roles]]
