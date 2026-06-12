# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the replay-path FPM feed.

``ReplayPlannerAdapter._feed_extra_fpm_to_regression`` feeds accumulated
intra-tick FPM snapshots into the regression model. The regression slots
hold ``PlannerEnginePerfModel``, which exposes only
``add_observations(dict)``. The pre-fix singular ``add_observation(fpm)``
raised ``AttributeError`` on replay ticks that carried more than one FPM
snapshot per worker.

This test drives the method against a real orchestrator-owned regression and
asserts it does not raise.
"""

from __future__ import annotations

import json

import pytest

from dynamo.mocker import MockEngineArgs, PlannerReplayBridge
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    ScheduledTick,
    WorkerCapabilities,
)
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    _build_fpm_from_dict,
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


def test_get_regression_uses_orchestrator_scaling_state_without_aic_install():
    """Replay without AIC benchmark FPMs still needs the live regression slot.

    The orchestrator's public regression store is populated during benchmark
    bootstrap for external-plugin access. No-AIC replay instead starts from
    the adapter's ``PlannerScalingState`` regression and feeds intra-tick FPMs
    into it.
    """
    cfg = _orch_agg_config_sla()
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._engine = OrchestratorEngineAdapter(cfg, _agg_caps())

    assert adapter._engine._orchestrator.get_regression("agg") is None
    assert adapter._get_regression("agg") is not None

    decode_snaps = [
        _snap("1", wall_time=1.0),
        _snap("1", wall_time=2.0),  # last-per-worker -> excluded
    ]
    adapter._feed_extra_fpm_to_regression(
        decode_snaps=decode_snaps,
        prefill_snaps=[],
    )


class _TrafficBridge:
    def drain_traffic(self):
        return {
            "duration_s": 60.0,
            "num_req": 4,
            "avg_isl": 512.0,
            "avg_osl": 128.0,
            "avg_kv_hit_rate": 0.25,
            "avg_accept_length": 2.5,
            "avg_ttft_ms": 10.0,
            "avg_itl_ms": 5.0,
        }


def test_build_tick_input_maps_replay_accept_length():
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._bridge = _TrafficBridge()

    tick = ScheduledTick(at_s=60.0, need_traffic_metrics=True)
    ti = adapter._build_tick_input(tick, {"now_ms": 1_000.0})

    assert ti.now_s == 60.0
    assert ti.traffic is not None
    assert ti.traffic.accept_length == 2.5
    assert adapter._last_traffic.accept_length == 2.5


def test_build_tick_input_buffers_fpm_until_fpm_tick():
    cfg = PlannerConfig(mode="agg", optimization_target="throughput")
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = cfg
    adapter._is_disagg = False
    adapter._bridge = _TrafficBridge()
    adapter._prefill_fpm_cache = {}
    adapter._decode_fpm_cache = {}
    adapter._pending_prefill_fpm_snaps = []
    adapter._pending_decode_fpm_snaps = []
    adapter._scaling_target_prefill = None
    adapter._scaling_target_decode = None

    no_fpm_tick = ScheduledTick(
        at_s=1.0,
        need_worker_states=True,
        need_worker_fpm=False,
    )
    snap = _snap("1", wall_time=1.0)
    first = adapter._build_tick_input(
        no_fpm_tick,
        {
            "now_ms": 1_000.0,
            "active_prefill_count": 0,
            "active_decode_count": 1,
            "decode_fpm_snapshots": [snap],
            "prefill_fpm_snapshots": [],
        },
    )
    assert first.fpm_observations is None

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
    assert ("1", 0) in second.fpm_observations.decode


def test_planner_bridge_drains_mtp_accept_length(tmp_path):
    trace_path = tmp_path / "mtp_trace.jsonl"
    records = [
        {
            "timestamp": 0.0,
            "session_id": f"req-{i}",
            "input_length": 128,
            "output_length": 12,
            "hash_ids": [100 + i * 2, 101 + i * 2],
        }
        for i in range(2)
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    agg_args = MockEngineArgs(
        block_size=64,
        num_gpu_blocks=512,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        speedup_ratio=1000.0,
        aic_nextn=2,
        aic_nextn_accept_rates="1,1",
    )

    bridge = PlannerReplayBridge(
        trace_file=trace_path,
        extra_engine_args=agg_args,
        num_workers=1,
        trace_block_size=64,
    )
    bridge.advance_to(1000.0)

    traffic = bridge.drain_traffic()
    assert traffic["avg_osl"] == 12.0
    assert traffic["avg_accept_length"] == pytest.approx(3.0)

    prefill_args = MockEngineArgs(
        block_size=64,
        num_gpu_blocks=512,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        speedup_ratio=1000.0,
        worker_type="prefill",
    )
    decode_args = MockEngineArgs(
        block_size=64,
        num_gpu_blocks=512,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        speedup_ratio=1000.0,
        worker_type="decode",
        aic_nextn=2,
        aic_nextn_accept_rates="1,1",
    )
    bridge = PlannerReplayBridge.create_disagg(
        trace_file=trace_path,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=1,
        num_decode_workers=1,
        trace_block_size=64,
    )
    bridge.advance_to(1000.0)

    traffic = bridge.drain_traffic()
    assert traffic["avg_osl"] == 12.0
    assert traffic["avg_accept_length"] == pytest.approx(3.0)


def test_replay_engine_caps_exposes_aic_nextn():
    caps = _engine_caps(MockEngineArgs(aic_nextn=2))

    assert caps.speculative_nextn == 2
