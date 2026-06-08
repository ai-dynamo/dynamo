# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the replay-path FPM feed.

``ReplayPlannerAdapter._feed_extra_fpm_to_regression`` feeds accumulated
intra-tick FPM snapshots into the regression model. The regression slots
hold ``PlannerEnginePerfModel`` (built by the PSM in SLA mode), which
exposes only ``add_observations(dict)`` — the pre-fix singular
``add_observation(fpm)`` raised ``AttributeError`` and crashed the
*default* (``use_orchestrator=False``) SLA-mode replay on any tick that
carried more than one FPM snapshot per worker. No test covered this
method, so the crash shipped silently.

This test drives the method against a real PSM-built regression and
asserts it does not raise.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    _build_fpm_from_dict,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _agg_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


def _agg_config_sla() -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
    )


def _snap(worker_id: str, wall_time: float) -> dict:
    """A bridge FPM snapshot dict with every key ``_build_fpm_from_dict`` reads."""
    return {
        "worker_id": worker_id,
        "wall_time": wall_time,
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "var_prefill_length": 0.0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 1,
        "sum_decode_kv_tokens": 100,
        "var_decode_kv_tokens": 0.0,
        "num_queued_prefill": 0,
        "sum_queued_prefill_tokens": 0,
        "var_queued_prefill_length": 0.0,
        "num_queued_decode": 0,
        "sum_queued_decode_kv_tokens": 0,
        "var_queued_decode_kv_tokens": 0.0,
    }


def _adapter_for_psm(
    cfg: PlannerConfig, caps: WorkerCapabilities
) -> ReplayPlannerAdapter:
    """Build just enough of a ReplayPlannerAdapter to exercise
    ``_feed_extra_fpm_to_regression`` without a full replay harness.

    The method only touches ``_config`` (mode / optimization_target via
    ``_is_easy_mode``) and ``_get_regression`` (which on the PSM path reads
    ``_use_orchestrator`` + ``_sm``)."""
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._use_orchestrator = False
    adapter._sm = PlannerStateMachine(cfg, caps)
    return adapter


def test_feed_extra_fpm_to_regression_does_not_crash_psm_sla():
    """Two decode snapshots for the same worker → one is non-excluded and
    flows into the regression. Pre-fix this raised AttributeError on the
    PlannerEnginePerfModel slot."""
    cfg = _agg_config_sla()
    adapter = _adapter_for_psm(cfg, _agg_caps())

    decode_snaps = [
        _snap("w1", wall_time=1.0),
        _snap("w1", wall_time=2.0),  # last-per-worker → excluded; the first feeds
    ]
    # Pre-fix:
    #   AttributeError: 'PlannerEnginePerfModel' object has no attribute
    #   'add_observation'
    adapter._feed_extra_fpm_to_regression(decode_snaps=decode_snaps, prefill_snaps=[])


def _orch_agg_config_sla() -> PlannerConfig:
    cfg = _agg_config_sla()
    cfg.scheduling.use_orchestrator = True
    return cfg


def test_install_benchmark_fpms_installs_regression_on_orchestrator_path():
    """Review #3: the orchestrator replay path must actually install
    regressions. ``ReplayPlannerAdapter.install_benchmark_fpms`` routes to
    ``OrchestratorEngineAdapter.install_regressions_from_fpms`` so
    ``get_regression`` is non-None afterwards. Pre-fix, replay/main.py only
    fed ``adapter._sm`` (None under use_orchestrator), so the orchestrator
    regression stayed empty and replay diverged from PSM."""
    cfg = _orch_agg_config_sla()
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._use_orchestrator = True
    adapter._sm = None
    adapter._engine = OrchestratorEngineAdapter(cfg, _agg_caps())

    # Before: no regression installed on the orchestrator path.
    assert adapter._engine._orchestrator.get_regression("agg") is None

    adapter.install_benchmark_fpms(agg_fpms=[_build_fpm_from_dict(_snap("w1", 1.0))])

    # After: the agg regression is installed (non-None).
    assert adapter._engine._orchestrator.get_regression("agg") is not None
