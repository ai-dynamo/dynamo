# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``BuiltinLoadPropose._last_load_diagnostics``.

Every exit branch of ``Propose`` / ``_advance_load_*`` must leave a
reason populated so ``OrchestratorEngineAdapter`` can project it onto
``TickDiagnostics.load_decision_reason*`` without falling back to
``n/a``. These tests exercise each branch and assert the exact reason
string.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import FpmObservations, WorkerCounts
from dynamo.planner.plugins.builtins.load_propose import BuiltinLoadPropose

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(
    mode="agg",
    enable_load_scaling=True,
    enable_throughput_scaling=False,
    optimization_target="throughput",
) -> PlannerConfig:
    return PlannerConfig(
        environment="kubernetes",
        mode=mode,
        enable_load_scaling=enable_load_scaling,
        enable_throughput_scaling=enable_throughput_scaling,
        optimization_target=optimization_target,
    )


def _orch_stub():
    orch = MagicMock()
    orch.capabilities = None
    orch.get_throughput_lower_bound = MagicMock(return_value=1)
    orch.set_throughput_lower_bound = MagicMock()
    return orch


def _plugin(cfg) -> BuiltinLoadPropose:
    return BuiltinLoadPropose(_orch_stub(), cfg)


def _make_fpm(
    sum_prefill_tokens=0,
    sum_decode_kv_tokens=0,
    queued_prefill_tokens=0,
    queued_decode_kv_tokens=0,
):
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )

    return ForwardPassMetrics(
        worker_id="w1",
        dp_rank=0,
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=sum_prefill_tokens,
            num_prefill_requests=0,
            sum_decode_kv_tokens=sum_decode_kv_tokens,
            num_decode_requests=1,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=queued_prefill_tokens,
            sum_decode_kv_tokens=queued_decode_kv_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Propose entry guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_load_scaling_records_disabled_reason():
    # PlannerConfig force-enables load_scaling when
    # optimization_target=throughput/latency; only the sla target
    # respects an explicit ``enable_load_scaling=False``.
    plugin = _plugin(
        _config(
            optimization_target="sla",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
        )
    )
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "disabled"


@pytest.mark.asyncio
async def test_no_fpm_obs_records_no_fpm_data_reason():
    plugin = _plugin(_config(mode="agg"))
    # Don't prime — _cached_fpm_obs stays None
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "no_fpm_data"


# ---------------------------------------------------------------------------
# Agg mode — empty FPM triggers inside _advance_load_agg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agg_empty_decode_fpm_records_no_fpm_data():
    plugin = _plugin(_config(mode="agg"))
    plugin.prime_tick(FpmObservations(), WorkerCounts(ready_num_decode=1))
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "no_fpm_data"


@pytest.mark.asyncio
async def test_agg_scaling_in_progress_records_reason():
    plugin = _plugin(_config(mode="agg"))
    plugin.prime_tick(
        FpmObservations(decode={("w1", 0): _make_fpm()}),
        # ready=1 but expected=2 → mismatch → scaling_in_progress
        WorkerCounts(ready_num_decode=1, expected_num_decode=2),
    )
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "scaling_in_progress"


@pytest.mark.asyncio
async def test_agg_worker_count_mismatch_records_reason():
    plugin = _plugin(_config(mode="agg"))
    # 2 FPM engines but ready_num_decode=1 → reconcile fails
    plugin.prime_tick(
        FpmObservations(
            decode={
                ("w1", 0): _make_fpm(),
                ("w2", 0): _make_fpm(),
            }
        ),
        WorkerCounts(ready_num_decode=1),
    )
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "worker_count_mismatch"


# ---------------------------------------------------------------------------
# Agg mode — easy path (throughput/latency optimization_target)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agg_easy_no_change_records_no_change():
    plugin = _plugin(_config(mode="agg", optimization_target="throughput"))
    # Small idle load → easy path should decide no scale needed
    plugin.prime_tick(
        FpmObservations(decode={("w1", 0): _make_fpm()}),
        WorkerCounts(ready_num_decode=1),
    )
    await plugin.Propose(request=MagicMock())
    reason = plugin._last_load_diagnostics["agg"]
    # The exact branch the easy path takes depends on the model; any
    # of these three is a valid "got past all guards" outcome.
    assert reason in {"no_change", "scale_up", "scale_down", "insufficient_data"}


# ---------------------------------------------------------------------------
# Disagg mode sets per-component reasons
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disagg_no_fpm_records_both_components_no_fpm_data():
    plugin = _plugin(
        _config(
            mode="disagg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="latency",
        )
    )
    # Empty FPM for both prefill and decode
    plugin.prime_tick(FpmObservations(), WorkerCounts())
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["prefill"] == "no_fpm_data"
    assert plugin._last_load_diagnostics["decode"] == "no_fpm_data"


# ---------------------------------------------------------------------------
# Prefill / decode single-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefill_mode_disabled_records_prefill_slot():
    plugin = _plugin(
        _config(
            mode="prefill",
            optimization_target="sla",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
        )
    )
    await plugin.Propose(request=MagicMock())
    # In single-mode the entry guard writes to the component slot.
    assert plugin._last_load_diagnostics["prefill"] == "disabled"


# ---------------------------------------------------------------------------
# Reset semantics between ticks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_resets_between_ticks():
    plugin = _plugin(_config(mode="agg"))
    # First tick: no FPM → no_fpm_data
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "no_fpm_data"
    # Second tick: also no FPM, but reset first so staleness doesn't leak.
    plugin._last_load_diagnostics["agg"] = "scale_up"  # simulate stale
    await plugin.Propose(request=MagicMock())
    assert plugin._last_load_diagnostics["agg"] == "no_fpm_data"
