# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``OrchestratorEngineAdapter._project_load_diagnostics``
(PR 8 sub-task 8-9).

Validates the end-to-end chain: ``BuiltinLoadPropose`` writes to
``_last_load_diagnostics`` during ``Propose``, the adapter reads it
after ``orchestrator.tick``, and projects the right fields onto
``TickDiagnostics`` depending on planner mode.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(mode="agg", **overrides) -> PlannerConfig:
    defaults = dict(
        environment="kubernetes",
        mode=mode,
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        optimization_target="throughput",
    )
    defaults.update(overrides)
    return PlannerConfig(**defaults)


def _caps(mode) -> WorkerCapabilities:
    eng = EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384)
    if mode in ("prefill",):
        return WorkerCapabilities(prefill=eng)
    if mode == "disagg":
        return WorkerCapabilities(prefill=eng, decode=eng)
    # agg + decode use the decode slot.
    return WorkerCapabilities(decode=eng)


# ---------------------------------------------------------------------------
# Aggregate projection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agg_reason_projects_onto_load_decision_reason():
    """Agg mode: plugin's ``agg`` slot lands in
    ``diagnostics.load_decision_reason``."""
    config = _config(mode="agg")
    adapter = OrchestratorEngineAdapter(config, _caps("agg"))
    await adapter.bootstrap_plugins()
    adapter.initial_tick(0.0)

    # Drive a tick that will hit "no_fpm_data" branch.
    ti = TickInput(
        now_s=5.0,
        fpm_observations=FpmObservations(),  # empty
        worker_counts=WorkerCounts(ready_num_decode=1),
    )
    scheduled = ScheduledTick(
        at_s=5.0, run_load_scaling=True, run_throughput_scaling=False
    )
    effects = await adapter.tick(scheduled, ti)

    # Plugin's internal state reflects the branch.
    assert adapter._builtins["load_propose"]._last_load_diagnostics["agg"] == "no_fpm_data"
    # And the adapter projected it onto the effects' diagnostics.
    assert effects.diagnostics.load_decision_reason == "no_fpm_data"
    # Per-component fields remain None (agg mode has no per-component).
    assert effects.diagnostics.load_decision_reason_prefill is None
    assert effects.diagnostics.load_decision_reason_decode is None

    await adapter.shutdown()


# ---------------------------------------------------------------------------
# Disagg projection: per-component + aggregate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disagg_reason_projects_per_component_and_aggregate():
    config = _config(mode="disagg", optimization_target="latency")
    adapter = OrchestratorEngineAdapter(config, _caps("disagg"))
    await adapter.bootstrap_plugins()
    adapter.initial_tick(0.0)

    ti = TickInput(
        now_s=5.0,
        fpm_observations=FpmObservations(),  # empty → no_fpm_data both sides
        worker_counts=WorkerCounts(ready_num_prefill=1, ready_num_decode=1),
    )
    effects = await adapter.tick(
        ScheduledTick(at_s=5.0, run_load_scaling=True, run_throughput_scaling=False),
        ti,
    )

    assert effects.diagnostics.load_decision_reason_prefill == "no_fpm_data"
    assert effects.diagnostics.load_decision_reason_decode == "no_fpm_data"
    # Aggregate prefers scale_up > scale_down > no_change > ...
    # With both == "no_fpm_data", aggregate = "no_fpm_data".
    assert effects.diagnostics.load_decision_reason == "no_fpm_data"

    await adapter.shutdown()


# ---------------------------------------------------------------------------
# Aggregate precedence (unit test on the static helper)
# ---------------------------------------------------------------------------


def test_aggregate_reason_prioritises_scale_up_over_no_change():
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )

    agg = OrchestratorEngineAdapter._aggregate_disagg_reason
    assert agg("scale_up", "no_change") == "scale_up"
    assert agg("no_change", "scale_up") == "scale_up"
    assert agg("scale_down", "no_change") == "scale_down"
    assert agg("no_change", "scale_down") == "scale_down"


def test_aggregate_reason_handles_nones():
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )

    agg = OrchestratorEngineAdapter._aggregate_disagg_reason
    assert agg(None, "no_change") == "no_change"
    assert agg("scale_up", None) == "scale_up"
    assert agg(None, None) is None


def test_aggregate_reason_prefers_real_decision_over_skip_reasons():
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )

    agg = OrchestratorEngineAdapter._aggregate_disagg_reason
    # One side produced a real decision, the other lacked data —
    # the decision wins.
    assert agg("scale_up", "insufficient_data") == "scale_up"
    assert agg("insufficient_data", "no_change") == "no_change"


# ---------------------------------------------------------------------------
# A2: throughput-decision projection (mirrors the load tests above)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_throughput_disagg_reason_projects_per_component_and_aggregate():
    """Disagg + throughput-only: when neither regression nor caps exist
    the plugin stamps ``model_not_ready`` on both sides; the adapter
    must project to per-component fields AND the aggregate."""
    config = _config(
        mode="disagg",
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
    )
    adapter = OrchestratorEngineAdapter(config, _caps("disagg"))
    await adapter.bootstrap_plugins()
    adapter.initial_tick(0.0)

    ti = TickInput(
        now_s=5.0,
        fpm_observations=FpmObservations(),
        worker_counts=WorkerCounts(ready_num_prefill=1, ready_num_decode=1),
        traffic=type(
            "T", (), {"duration_s": 10.0, "num_req": 50, "isl": 500, "osl": 100}
        )(),
    )
    effects = await adapter.tick(
        ScheduledTick(
            at_s=5.0, run_load_scaling=False, run_throughput_scaling=True
        ),
        ti,
    )

    plugin = adapter._builtins["throughput_propose"]
    d = plugin._last_throughput_diagnostics
    # No regressions seeded → both _compute_* land on ``model_not_ready``.
    assert d["prefill"] == "model_not_ready"
    assert d["decode"] == "model_not_ready"

    assert effects.diagnostics.throughput_decision_reason_prefill == "model_not_ready"
    assert effects.diagnostics.throughput_decision_reason_decode == "model_not_ready"
    assert effects.diagnostics.throughput_decision_reason == "model_not_ready"

    await adapter.shutdown()


@pytest.mark.asyncio
async def test_throughput_agg_reason_projects_onto_aggregate_field():
    """Agg mode: plugin stamps ``agg`` slot only; adapter projects to
    aggregate ``throughput_decision_reason`` and leaves per-component
    fields as None."""
    config = _config(
        mode="agg",
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
    )
    adapter = OrchestratorEngineAdapter(config, _caps("agg"))
    await adapter.bootstrap_plugins()
    adapter.initial_tick(0.0)

    ti = TickInput(
        now_s=5.0,
        fpm_observations=FpmObservations(),
        worker_counts=WorkerCounts(ready_num_decode=1),
        traffic=type(
            "T", (), {"duration_s": 10.0, "num_req": 50, "isl": 500, "osl": 100}
        )(),
    )
    effects = await adapter.tick(
        ScheduledTick(
            at_s=5.0, run_load_scaling=False, run_throughput_scaling=True
        ),
        ti,
    )
    assert effects.diagnostics.throughput_decision_reason == "model_not_ready"
    assert effects.diagnostics.throughput_decision_reason_prefill is None
    assert effects.diagnostics.throughput_decision_reason_decode is None

    await adapter.shutdown()


def test_aggregate_throughput_reason_priority():
    """``scale`` beats ``set_lower_bound`` beats skip reasons."""
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )

    agg = OrchestratorEngineAdapter._aggregate_throughput_reason
    assert agg("scale", "set_lower_bound") == "scale"
    assert agg("set_lower_bound", "scale") == "scale"
    assert agg("set_lower_bound", "model_not_ready") == "set_lower_bound"
    assert agg("model_not_ready", "no_traffic_data") == "model_not_ready"
    assert agg(None, "scale") == "scale"
    assert agg(None, None) is None
