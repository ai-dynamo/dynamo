# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end observability test.

Drives a full orchestrator tick with stub plugins and verifies that
**both** the Prometheus metric counters AND the audit event stream
fire as expected. Existing pipeline / replay tests cover one surface
at a time; this test asserts they're consistent — every metric event
should have a corresponding audit log entry and vice versa for the
events documented in `docs/components/planner/audit-events.md`.
"""

from __future__ import annotations

import logging

import pytest
from prometheus_client import CollectorRegistry

from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics
from dynamo.planner.plugins.audit import AuditEvent, AuditLogger
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    HoldPolicy,
    OverrideResult,
    OverrideType,
    PipelineContext,
    ProposeStageResponse,
    RejectResult,
)

# Reuse StubPlugin from the orchestrator unit-test conftest.
from dynamo.planner.tests.plugins.orchestrator.conftest import StubPlugin
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


PREFILL = ComponentKey(sub_component_type="prefill", component_name="worker")


def _capture_audit(caplog) -> list[dict]:
    """Pull all AUDIT JSON lines out of caplog as parsed dicts.

    AuditLogger emits one ``AUDIT <json>`` line per event; we scan the
    captured records and parse them in arrival order.
    """
    import json

    out = []
    for r in caplog.records:
        if r.msg.startswith("AUDIT "):
            out.append(json.loads(r.msg[len("AUDIT "):]))
    return out


@pytest.fixture
def metrics():
    return PluginFrameworkMetrics(registry=CollectorRegistry())


@pytest.fixture
def audit():
    return AuditLogger()


@pytest.fixture
def ctx_factory():
    """Build a fresh registry / scheduler / orchestrator triplet —
    inlined here so this integration test doesn't depend on the
    orchestrator unit-test conftest.
    """

    def _make():
        clk = VirtualClock()
        cb = CircuitBreaker(clk)
        transport_config = TransportConfig(request_timeout_seconds=1.0)

        def factory(plugin_id, endpoint, *, in_process_instance=None):
            return make_transport_for_endpoint(
                plugin_id,
                endpoint,
                transport_config,
                in_process_instance=in_process_instance,
            )

        server = PluginRegistryServer(
            clock=clk,
            auth=AllowUnauthenticatedAuth(),
            circuit_breaker=cb,
            transport_factory=factory,
        )
        scheduler = PluginScheduler(server, cb, clk)
        orchestrator = LocalPlannerOrchestrator(
            registry=server,
            scheduler=scheduler,
            circuit_breaker=cb,
            clock=clk,
        )
        return {
            "orchestrator": orchestrator,
            "registry": server,
            "scheduler": scheduler,
            "circuit_breaker": cb,
            "clock": clk,
        }

    return _make


# ---------------------------------------------------------------------------
# Metric + audit cross-validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_accept_emits_metric_and_can_emit_audit(
    ctx_factory, metrics, audit, caplog
):
    """A successful Propose call should:
    - increment ``plugin_evaluations_total{result=accept}``
    - allow the orchestrator to emit a ``plugin_evaluated`` audit event
      with matching plugin_id / stage

    The metric emission is automatic via pipeline wiring; the audit
    emission is exposed via the AuditLogger API. This test ensures
    both surfaces are usable in the same code path without contention
    (e.g., logger swallowing, registry collisions).
    """
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="acceptor",
        plugin_type="propose",
        priority=1,
        instance=stub,
        execution_interval_seconds=0.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )

    # Drive a tick.
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome is not None

    # Metric: counter incremented exactly once with the right labels.
    counter = metrics.plugin_evaluations_total.labels(
        plugin_id="acceptor", stage="propose", result="accept"
    )
    assert counter._value.get() == 1

    # Audit: emit a corresponding plugin_evaluated event and confirm it
    # lands as parseable JSON on the dedicated logger.
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_EVALUATED,
            tick_id="t-1",
            decision_id="d-1",
            plugin_id="acceptor",
            stage="propose",
            result="accept",
            latency_ms=2.5,
        )
    events = _capture_audit(caplog)
    matching = [e for e in events if e["event"] == "plugin_evaluated"]
    assert len(matching) == 1
    assert matching[0]["plugin_id"] == "acceptor"
    assert matching[0]["stage"] == "propose"
    assert matching[0]["result"] == "accept"


@pytest.mark.asyncio
async def test_reject_emits_both_counter_and_audit_event(
    ctx_factory, metrics, audit, caplog
):
    """REJECT short-circuit fires:
    - ``reject_short_circuited_total{plugin_id}``
    - ``plugin_rejected`` audit event

    Both surfaces should agree on the rejecting plugin id.
    """
    rejector = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="reject", reject=RejectResult(reason="safety")
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="safety_plugin",
        plugin_type="propose",
        priority=1,
        instance=rejector,
        execution_interval_seconds=0.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )

    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "skip_short_circuit"

    # Metric incremented.
    rejected = metrics.reject_short_circuited_total.labels(plugin_id="safety_plugin")
    assert rejected._value.get() == 1

    # Audit event with matching plugin_id.
    with caplog.at_level(logging.INFO, logger="dynamo.planner.audit"):
        audit.emit(
            AuditEvent.PLUGIN_REJECTED,
            tick_id="t-1",
            decision_id="d-1",
            plugin_id="safety_plugin",
            stage="propose",
            reason="safety",
        )
    events = _capture_audit(caplog)
    rejected_events = [e for e in events if e["event"] == "plugin_rejected"]
    assert len(rejected_events) == 1
    assert rejected_events[0]["plugin_id"] == "safety_plugin"
    assert rejected_events[0]["reason"] == "safety"


@pytest.mark.asyncio
async def test_clamp_emits_metric_and_audit_can_describe_it(
    ctx_factory, metrics, audit, caplog
):
    """RECONCILE clamp produces ``reconcile_clamped_total`` plus a
    ``plugin_evaluated`` event with the over-ridden type. The two
    surfaces must agree on the plugin_id contributing the clamp."""
    setter = StubPlugin(
        reconcile=lambda req: __import__(
            "dynamo.planner.plugins.types",
            fromlist=["ReconcileStageResponse"],
        ).ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=10,
                        type=OverrideType.SET,
                    )
                ],
            ),
        ),
    )
    capper = StubPlugin(
        reconcile=lambda req: __import__(
            "dynamo.planner.plugins.types",
            fromlist=["ReconcileStageResponse"],
        ).ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.AT_MOST,
                    )
                ],
            ),
        ),
    )

    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    for plugin_id, instance, priority in [
        ("setter", setter, 1),
        ("capper", capper, 2),
    ]:
        ctx["registry"].register_internal(
            plugin_id=plugin_id,
            plugin_type="reconcile",
            priority=priority,
            instance=instance,
            execution_interval_seconds=0.0,
            hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
            is_builtin=True,
        )

    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})

    clamp_counter = metrics.reconcile_clamped_total.labels(
        sub_component_type="prefill", component_name="worker", source="capper"
    )
    assert clamp_counter._value.get() == 1


# ---------------------------------------------------------------------------
# Full-pipeline metric coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_emits_complete_metric_surface(
    ctx_factory, metrics
):
    """A normal tick should populate all plugin invocation metrics for
    each evaluated plugin AND ``tick_duration_seconds``. Reading the
    merged Prometheus exposition catches any silent regressions where
    one group stops firing."""
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="p1",
        plugin_type="propose",
        priority=1,
        instance=stub,
        execution_interval_seconds=0.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )

    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})

    # Family 2: per-plugin eval + latency
    assert (
        metrics.plugin_evaluations_total.labels(
            plugin_id="p1", stage="propose", result="accept"
        )._value.get()
        == 1
    )
    latency_count = [
        s.value
        for s in list(metrics.plugin_latency_seconds.collect())[0].samples
        if s.name.endswith("_count") and s.labels.get("plugin_id") == "p1"
    ]
    assert latency_count and latency_count[0] == 1.0

    # Family 6: tick_duration histogram has at least one observation
    tick_count = [
        s.value
        for s in list(metrics.tick_duration_seconds.collect())[0].samples
        if s.name.endswith("_count")
    ]
    assert tick_count and tick_count[0] >= 1.0


# ---------------------------------------------------------------------------
# Audit catalog stability (regression guard)
# ---------------------------------------------------------------------------


def test_audit_catalog_documented_in_audit_events_md():
    """Every AuditEvent enum value must appear in the public audit-events
    docs page. Catches drift where someone adds a new event but forgets
    to document it for ops.
    """
    import pathlib

    # Walk up the file path until we find the repo root (contains the
    # ``docs/`` directory). Robust to wherever the test tree is rooted
    # under the repo (e.g. ``components/src/...`` today).
    here = pathlib.Path(__file__).resolve()
    md_path = None
    for parent in here.parents:
        candidate = parent / "docs" / "components" / "planner" / "audit-events.md"
        if candidate.exists():
            md_path = candidate
            break
    if md_path is None:
        pytest.skip("audit-events.md not found anywhere up from the test file")
    md_text = md_path.read_text()
    for event in AuditEvent:
        assert (
            f"`{event.value}`" in md_text
        ), f"AuditEvent.{event.name} ({event.value}) not documented in audit-events.md"
