# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BuiltinReconcile (PR 6 6-5)."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.builtins.reconcile import BuiltinReconcile
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    ComponentTarget,
    OverrideType,
    PipelineContext,
    ReconcileStageRequest,
    ResetRequest,
    ScalingProposal,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeOrchestrator:
    def __init__(self):
        self._store = {}

    def get_regression(self, kind):
        return self._store.get(kind)

    def update_regression(self, kind, model):
        self._store[kind] = model


class _FakeConfig:
    pass


def _make():
    return BuiltinReconcile(_FakeOrchestrator(), _FakeConfig())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_no_proposal_returns_accept():
    plugin = _make()
    req = ReconcileStageRequest(context=PipelineContext())
    resp = await plugin.Reconcile(req)
    assert resp.result_kind == "accept"
    assert resp.accept is not None


@pytest.mark.asyncio
async def test_empty_targets_returns_accept():
    plugin = _make()
    req = ReconcileStageRequest(
        context=PipelineContext(proposal=ScalingProposal(targets=[]))
    )
    resp = await plugin.Reconcile(req)
    assert resp.result_kind == "accept"


@pytest.mark.asyncio
async def test_proposal_with_set_targets_reemits_as_set_override():
    plugin = _make()
    req = ReconcileStageRequest(
        context=PipelineContext(
            proposal=ScalingProposal(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=5,
                        type=OverrideType.SET,
                    ),
                    ComponentTarget(
                        sub_component_type="decode",
                        replicas=3,
                        type=OverrideType.SET,
                    ),
                ],
                source="upstream",
            )
        )
    )
    resp = await plugin.Reconcile(req)
    assert resp.result_kind == "override"
    assert resp.override is not None
    assert resp.override.reason == "builtin_reconcile"
    pairs = [(t.sub_component_type, t.replicas, t.type) for t in resp.override.targets]
    assert pairs == [
        ("prefill", 5, OverrideType.SET),
        ("decode", 3, OverrideType.SET),
    ]


@pytest.mark.asyncio
async def test_proposal_with_none_replicas_skipped():
    plugin = _make()
    req = ReconcileStageRequest(
        context=PipelineContext(
            proposal=ScalingProposal(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=None,
                        type=OverrideType.SET,
                    ),
                    ComponentTarget(
                        sub_component_type="decode",
                        replicas=2,
                        type=OverrideType.SET,
                    ),
                ]
            )
        )
    )
    resp = await plugin.Reconcile(req)
    assert resp.result_kind == "override"
    pairs = [(t.sub_component_type, t.replicas) for t in resp.override.targets]
    assert pairs == [("decode", 2)]


@pytest.mark.asyncio
async def test_all_none_replicas_returns_accept():
    plugin = _make()
    req = ReconcileStageRequest(
        context=PipelineContext(
            proposal=ScalingProposal(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=None,
                        type=OverrideType.SET,
                    ),
                ]
            )
        )
    )
    resp = await plugin.Reconcile(req)
    assert resp.result_kind == "accept"


@pytest.mark.asyncio
async def test_bootstrap_and_reset_default_noops():
    plugin = _make()
    b = await plugin.Bootstrap(BootstrapRequest())
    assert b.ok is True
    r = await plugin.Reset(ResetRequest())
    assert r.ok is True


@pytest.mark.asyncio
async def test_preserves_component_name_on_reemit():
    plugin = _make()
    req = ReconcileStageRequest(
        context=PipelineContext(
            proposal=ScalingProposal(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="pool-A",
                        replicas=4,
                        type=OverrideType.SET,
                    ),
                ]
            )
        )
    )
    resp = await plugin.Reconcile(req)
    (t,) = resp.override.targets
    assert t.component_name == "pool-A"
    assert t.sub_component_type == "prefill"
    assert t.replicas == 4
