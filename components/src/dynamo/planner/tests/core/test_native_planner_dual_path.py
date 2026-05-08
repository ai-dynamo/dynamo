# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``NativePlannerBase._ensure_engine`` dual-path wiring
(PR 7 sub-task 7-3 finish).

``NativePlannerBase.__init__`` has substantial external wiring
(connectors, Prometheus, runtime client caches) that makes full
instantiation heavy for unit tests. These tests exercise
``_ensure_engine`` directly by constructing a minimal test harness
providing only the attributes the method reads: ``self.config``,
``self.prefill_worker_info``, ``self.decode_worker_info``, and the
cached engine / PSM fields.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.engine_protocol import EngineProtocol, _PSMEngineAdapter
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _MinimalPlanner(NativePlannerBase):
    """Subclass that short-circuits ``__init__`` so the engine-wiring
    logic can be tested without spinning up connectors / Prometheus /
    event subscribers."""

    def __init__(self, config: PlannerConfig):
        # Deliberately skip the parent __init__ — only set attributes
        # ``_ensure_engine`` (and the code it calls) actually touch.
        # ``PlannerPrometheusMetrics`` registers module-global metrics;
        # instantiating it per-test raises "Duplicated timeseries" on
        # repeated runs, so we leave it unset. ``_log_decision_summary``
        # doesn't access it; only ``_report_diagnostics`` does, which we
        # guard via ``prometheus_port == 0``.
        self.config = config
        self.runtime = None
        self.namespace = config.namespace
        self.model_name = None
        self.prefill_worker_info = _FakeWorkerInfo()
        self.decode_worker_info = _FakeWorkerInfo()
        self._state_machine = None
        self._engine = None
        self._last_worker_counts = None
        self.prometheus_port = 0
        self.prometheus_metrics = None  # not needed for _ensure_engine path


class _FakeWorkerInfo:
    """Duck-typed stand-in for planner.core.base.WorkerInfo — the
    ``build_worker_capabilities`` helper reads ``num_gpu`` /
    ``max_num_batched_tokens`` / etc. but tolerates ``None``."""

    def __init__(self):
        self.num_gpu = None
        self.max_num_batched_tokens = None
        self.max_num_seqs = None
        self.context_length = None
        self.max_kv_tokens = None


def _config(use_orchestrator: bool) -> PlannerConfig:
    return PlannerConfig(
        environment="kubernetes",
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        optimization_target="throughput",  # easy mode — no regression bootstrap
        scheduling={"use_orchestrator": use_orchestrator},
    )


# ---------------------------------------------------------------------------
# Default path: PSM (use_orchestrator=False)
# ---------------------------------------------------------------------------


def test_default_path_uses_psm_engine_adapter():
    planner = _MinimalPlanner(_config(use_orchestrator=False))
    engine = planner._ensure_engine()
    assert isinstance(engine, _PSMEngineAdapter)
    # Backwards-compat: state_machine property resolves to the PSM that
    # the adapter wraps.
    assert planner._state_machine is not None
    assert isinstance(planner._state_machine, PlannerStateMachine)
    assert planner.state_machine is planner._state_machine


def test_default_path_caches_engine():
    planner = _MinimalPlanner(_config(use_orchestrator=False))
    engine1 = planner._ensure_engine()
    engine2 = planner._ensure_engine()
    assert engine1 is engine2


def test_default_path_engine_satisfies_protocol():
    planner = _MinimalPlanner(_config(use_orchestrator=False))
    assert isinstance(planner._ensure_engine(), EngineProtocol)


# ---------------------------------------------------------------------------
# Orchestrator path (use_orchestrator=True)
# ---------------------------------------------------------------------------


def test_orchestrator_path_uses_orchestrator_engine_adapter():
    planner = _MinimalPlanner(_config(use_orchestrator=True))
    engine = planner._ensure_engine()
    assert isinstance(engine, OrchestratorEngineAdapter)
    # PSM is NOT constructed in the orchestrator path.
    assert planner._state_machine is None


def test_orchestrator_path_caches_engine():
    planner = _MinimalPlanner(_config(use_orchestrator=True))
    engine1 = planner._ensure_engine()
    engine2 = planner._ensure_engine()
    assert engine1 is engine2


def test_orchestrator_path_engine_satisfies_protocol():
    planner = _MinimalPlanner(_config(use_orchestrator=True))
    assert isinstance(planner._ensure_engine(), EngineProtocol)


# ---------------------------------------------------------------------------
# last_worker_counts cache (feeds _log_decision_summary in orchestrator path)
# ---------------------------------------------------------------------------


def test_last_worker_counts_initially_none():
    planner = _MinimalPlanner(_config(use_orchestrator=False))
    assert planner._last_worker_counts is None


def test_log_decision_summary_reads_last_worker_counts_if_set():
    from dynamo.planner.core.types import (
        PlannerEffects,
        ScalingDecision,
        TickDiagnostics,
        WorkerCounts,
    )

    planner = _MinimalPlanner(_config(use_orchestrator=True))
    planner._last_worker_counts = WorkerCounts(
        ready_num_prefill=3, ready_num_decode=2
    )
    effects = PlannerEffects(
        scale_to=ScalingDecision(num_prefill=5, num_decode=2),
        next_tick=None,
        diagnostics=TickDiagnostics(),
    )
    # Just verify no AttributeError — in orchestrator path PSM is None,
    # so the code must fall back to _last_worker_counts.
    planner._log_decision_summary(effects)


def test_log_decision_summary_falls_back_gracefully_without_any_source():
    from dynamo.planner.core.types import PlannerEffects, TickDiagnostics

    planner = _MinimalPlanner(_config(use_orchestrator=True))
    # Neither _state_machine nor _last_worker_counts set — must not crash.
    planner._log_decision_summary(
        PlannerEffects(scale_to=None, next_tick=None, diagnostics=TickDiagnostics())
    )
