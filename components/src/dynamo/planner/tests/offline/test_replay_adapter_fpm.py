# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for planner replay FPM handling."""

from __future__ import annotations

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    ScheduledTick,
    WorkerCapabilities,
)
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    _build_fpm_from_dict,
    _merge_traffic,
    _update_fpm_cache,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.replay.main import _engine_caps

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


def _snap(worker_id: str, wall_time: float, dp_rank: int = 0) -> dict:
    """A bridge FPM snapshot dict with every key ``_build_fpm_from_dict`` reads."""
    return {
        "worker_id": worker_id,
        "dp_rank": dp_rank,
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


def test_fpm_cache_keeps_all_ranks_for_each_active_worker():
    cache = {}
    snapshots = [
        _snap("0", wall_time=1.0, dp_rank=0),
        _snap("0", wall_time=1.0, dp_rank=1),
        _snap("1", wall_time=1.0, dp_rank=0),
        _snap("1", wall_time=1.0, dp_rank=1),
    ]

    _update_fpm_cache(cache, snapshots, active_count=2)

    assert set(cache) == {("0", 0), ("0", 1), ("1", 0), ("1", 1)}

    _update_fpm_cache(cache, [], active_count=1)

    assert set(cache) == {("0", 0), ("0", 1)}


def _orch_agg_config_sla() -> PlannerConfig:
    return _agg_config_sla()


def test_install_benchmark_fpms_installs_regression_on_orchestrator_path():
    """Review #3: the orchestrator replay path must actually install
    regressions. ``ReplayPlannerAdapter.install_benchmark_fpms`` routes to
    ``OrchestratorEngineAdapter.install_regressions_from_fpms`` so
    ``get_regression`` is non-None afterwards. Pre-fix, replay/main.py only
    bypassed the orchestrator engine, so the orchestrator regression stayed empty and
    replay diverged from live planner behavior."""
    cfg = _orch_agg_config_sla()
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._engine = OrchestratorEngineAdapter(cfg, _agg_caps())

    # Before: no regression installed on the orchestrator path.
    assert adapter._engine._orchestrator.get_regression("agg") is None

    adapter.install_benchmark_fpms(agg_fpms=[_build_fpm_from_dict(_snap("w1", 1.0))])

    # After: the agg regression is installed (non-None).
    assert adapter._engine._orchestrator.get_regression("agg") is not None


def test_build_tick_input_maps_replay_accept_length():
    # The Rust simulation drains the per-tick traffic window into
    # ``result["traffic"]``; a need_traffic_metrics tick maps it onto
    # ``TickInput.traffic`` (accept_length, isl/osl, kv-hit, latency).
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._prefill_fpm_cache = {}
    adapter._decode_fpm_cache = {}

    tick = ScheduledTick(at_s=60.0, need_traffic_metrics=True)
    result = {
        "now_ms": 1_000.0,
        "active_prefill_count": 0,
        "active_decode_count": 0,
        "traffic": {
            "duration_s": 60.0,
            "num_req": 4,
            "avg_isl": 512.0,
            "avg_osl": 128.0,
            "avg_kv_hit_rate": 0.25,
            "avg_accept_length": 2.5,
            "avg_ttft_ms": 10.0,
            "avg_itl_ms": 5.0,
        },
    }
    ti = adapter._build_tick_input(tick, result)

    assert ti.now_s == 60.0
    assert ti.traffic is not None
    assert ti.traffic.accept_length == 2.5
    assert adapter._last_traffic.accept_length == 2.5


def test_build_tick_input_keeps_only_latest_fpm_until_fpm_tick():
    cfg = PlannerConfig(mode="agg", optimization_target="throughput")
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._is_disagg = False
    adapter._prefill_fpm_cache = {}
    adapter._decode_fpm_cache = {}
    adapter._scaling_target_prefill = None
    adapter._scaling_target_decode = None

    no_fpm_tick = ScheduledTick(
        at_s=1.0,
        need_worker_states=True,
        need_worker_fpm=False,
    )
    first = adapter._build_tick_input(
        no_fpm_tick,
        {
            "now_ms": 1_000.0,
            "active_prefill_count": 0,
            "active_decode_count": 1,
            "decode_fpm_snapshots": [
                _snap("0", wall_time=1.0, dp_rank=0),
                _snap("0", wall_time=1.0, dp_rank=1),
                _snap("0", wall_time=2.0, dp_rank=0),
                _snap("0", wall_time=2.0, dp_rank=1),
            ],
            "prefill_fpm_snapshots": [],
        },
    )
    assert first.fpm_observations is None
    assert set(adapter._decode_fpm_cache) == {("0", 0), ("0", 1)}
    assert adapter._decode_fpm_cache[("0", 0)].wall_time == 2.0
    assert adapter._decode_fpm_cache[("0", 1)].wall_time == 2.0

    fpm_tick = ScheduledTick(
        at_s=7.0,
        need_worker_states=True,
        need_worker_fpm=True,
    )
    second = adapter._build_tick_input(
        fpm_tick,
        {
            "now_ms": 7_000.0,
            "active_prefill_count": 0,
            "active_decode_count": 1,
            "decode_fpm_snapshots": [],
            "prefill_fpm_snapshots": [],
        },
    )

    assert second.fpm_observations is not None
    assert set(second.fpm_observations.decode) == {("0", 0), ("0", 1)}
    assert second.fpm_observations.decode[("0", 0)].wall_time == 2.0
    assert second.fpm_observations.decode[("0", 1)].wall_time == 2.0


def test_replay_engine_caps_exposes_aic_nextn():
    caps = _engine_caps(MockEngineArgs(aic_nextn=2))

    assert caps.speculative_nextn == 2


def test_replay_engine_caps_aggregates_attention_dp_capacity_and_gpu_width():
    caps = _engine_caps(
        MockEngineArgs(
            num_gpu_blocks=100,
            block_size=16,
            dp_size=4,
            aic_tp_size=2,
        )
    )

    assert caps.max_kv_tokens == 100 * 16 * 4
    assert caps.num_gpu == 2 * 4


def test_replay_engine_caps_keeps_single_rank_defaults():
    caps = _engine_caps(MockEngineArgs(num_gpu_blocks=100, block_size=16))

    assert caps.max_kv_tokens == 100 * 16
    assert caps.num_gpu == 1


def test_merge_traffic_weights_ratio_fields_by_native_counts():
    # kv_hit_rate and accept_length must merge by their true denominators
    # (hit_rate_count / accept_length_forward_count), not num_req, so a window
    # whose ratio-sample count is disproportionate to its request count still
    # contributes its exact share. Here num_req-weighting would give the wrong
    # answer (0.9 and 1.2); count-weighting reconstructs the exact mean.
    a = {
        "num_req": 1,
        "duration_s": 1.0,
        "avg_isl": 100.0,
        "avg_osl": 50.0,
        "avg_kv_hit_rate": 0.0,
        "hit_rate_count": 90,
        "avg_accept_length": 3.0,
        "accept_length_forward_count": 90,
    }
    b = {
        "num_req": 9,
        "duration_s": 1.0,
        "avg_isl": 100.0,
        "avg_osl": 50.0,
        "avg_kv_hit_rate": 1.0,
        "hit_rate_count": 10,
        "avg_accept_length": 1.0,
        "accept_length_forward_count": 10,
    }
    merged = _merge_traffic(a, b)
    assert merged["avg_kv_hit_rate"] == pytest.approx(
        (0.0 * 90 + 1.0 * 10) / 100
    )  # 0.1
    assert merged["avg_accept_length"] == pytest.approx(
        (3.0 * 90 + 1.0 * 10) / 100
    )  # 2.8
    assert merged["num_req"] == 10
    assert merged["hit_rate_count"] == 100
    assert merged["accept_length_forward_count"] == 100
    assert merged["avg_isl"] == pytest.approx(100.0)
