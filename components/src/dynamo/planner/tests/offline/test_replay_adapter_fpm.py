# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before replay CLI imports.

"""Regression tests for planner replay FPM handling."""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

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
from dynamo.replay import planner as replay_planner
from dynamo.replay.planner import _engine_caps

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
    """A replay FPM snapshot dict with every key ``_build_fpm_from_dict`` reads."""
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

    _update_fpm_cache(cache, snapshots, active_worker_ids=[0, 1])

    assert set(cache) == {("0", 0), ("0", 1), ("1", 0), ("1", 1)}

    _update_fpm_cache(cache, [], active_worker_ids=[0])

    assert set(cache) == {("0", 0), ("0", 1)}


def test_fpm_cache_prunes_by_active_identity_after_worker_replacement():
    cache = {}
    _update_fpm_cache(
        cache,
        [_snap("0", wall_time=1.0), _snap("1", wall_time=1.0)],
        active_worker_ids=[0, 1],
    )

    _update_fpm_cache(
        cache,
        [_snap("2", wall_time=2.0)],
        active_worker_ids=[0, 2],
    )

    assert set(cache) == {("0", 0), ("2", 0)}


def _orch_agg_config_sla() -> PlannerConfig:
    return _agg_config_sla()


def test_replay_adapter_uses_injected_engine_protocol_and_owns_cleanup():
    class _Engine:
        def __init__(self):
            self.closed = False

        def initial_tick(self, start_s):
            return ScheduledTick(at_s=start_s + 5.0)

        async def tick(self, scheduled_tick, tick_input):
            raise AssertionError("tick is not needed by this ownership test")

        async def shutdown(self):
            self.closed = True

    engine = _Engine()
    adapter = ReplayPlannerAdapter(
        PlannerConfig(mode="agg"),
        capabilities=_agg_caps(),
        engine=engine,
    )

    with adapter:
        assert adapter.initial_tick_ms() == 5_000.0

    assert engine.closed


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


def test_summary_only_planner_details_preserve_tick_count():
    class _Recorder:
        def finalize(self):
            raise AssertionError("summary-only replay must not finalize diagnostics")

    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._capture_details = False
    adapter._recorder = _Recorder()
    adapter._config = PlannerConfig(mode="agg")
    adapter._benchmark_granularity = 8
    adapter._bootstrap_metadata = {"status": "not_required"}
    adapter._ticks = [{"large": "tick payload"}]
    adapter._scaling_events = []
    adapter._total_ticks = 4

    details = adapter.finalize([{"large": "lifecycle payload"}])

    assert details.total_ticks == 4
    assert details.ticks == []
    assert details.lifecycle_operations == []
    assert details.metadata["details_captured"] is False


def test_planner_metadata_identifies_custom_plugins_without_secrets():
    config = PlannerConfig(
        mode="agg",
        plugin_registration={
            "in_process_plugins": [
                {
                    "module": "custom.plugins",
                    "class": "Predictor",
                    "plugin_id": "custom_predict",
                    "plugin_type": "predict",
                    "priority": 5,
                    "kwargs": {"window": 4},
                }
            ]
        },
        scheduling={
            "external_plugins": [
                {
                    "plugin_id": "external_propose",
                    "plugin_type": "propose",
                    "priority": 10,
                    "endpoint": "grpc://planner-plugin:9000",
                    "auth_token": "secret",
                    "version": "v2",
                }
            ]
        },
    )
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = config
    adapter._benchmark_granularity = 8
    adapter._bootstrap_metadata = {"status": "not_required"}
    adapter._capture_details = True

    metadata = adapter._planner_metadata()
    identities = metadata["configured_plugin_identities"]

    assert [identity["plugin_id"] for identity in identities] == [
        "custom_predict",
        "external_propose",
    ]
    serialized = str(identities)
    assert "secret" not in serialized
    assert "grpc://planner-plugin:9000" not in serialized

    changed_port_config = config.model_copy(
        update={"control_api_port": config.control_api_port + 1}
    )
    adapter._config = changed_port_config
    assert (
        adapter._planner_metadata()["planner_config_digest"]
        == metadata["planner_config_digest"]
    )

    changed_config = config.model_copy(deep=True)
    changed_config.plugin_registration.in_process_plugins[0].kwargs["window"] = 8
    adapter._config = changed_config
    assert (
        adapter._planner_metadata()["planner_config_digest"]
        != metadata["planner_config_digest"]
    )


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
        "active_prefill_ids": [],
        "active_decode_ids": [],
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
            "active_prefill_ids": [],
            "active_decode_ids": [0],
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
            "active_prefill_ids": [],
            "active_decode_ids": [0],
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


def test_disagg_bootstrap_uses_role_specific_performance_model_identities(
    monkeypatch,
):
    class _Session:
        def __init__(self, tp_size):
            self.tp_size = tp_size

        def predict_prefill(self, batch_size, isl, prefix):
            del batch_size, isl, prefix
            return float(self.tp_size)

        def predict_decode(self, batch_size, isl, osl):
            del batch_size, isl, osl
            return float(self.tp_size)

    class _Adapter:
        def __init__(self):
            self.bootstrap_metadata = None
            self.prefill_fpms = None
            self.decode_fpms = None

        def set_bootstrap_metadata(self, metadata):
            self.bootstrap_metadata = metadata

        def _is_easy_mode(self):
            return False

        def install_benchmark_fpms(
            self, *, agg_fpms=None, prefill_fpms=None, decode_fpms=None
        ):
            assert agg_fpms is None
            self.prefill_fpms = prefill_fpms
            self.decode_fpms = decode_fpms

    adapter = _Adapter()
    session_requests = []

    def create_session(**kwargs):
        session_requests.append(kwargs)
        return _Session(kwargs["tp_size"])

    monkeypatch.setattr(replay_planner, "create_session", create_session)
    monkeypatch.setattr(
        "dynamo.planner.offline.replay_adapter.create_replay_planner_adapter",
        lambda **kwargs: adapter,
    )
    prefill_args = MockEngineArgs(
        max_num_batched_tokens=128,
        max_num_seqs=1,
        num_gpu_blocks=64,
        block_size=16,
    )
    decode_args = MockEngineArgs(
        max_num_batched_tokens=128,
        max_num_seqs=2,
        num_gpu_blocks=64,
        block_size=16,
    )
    metadata = {
        "prefill": {
            "provider": "aic",
            "config": {
                "backend": "vllm",
                "system": "h200_sxm",
                "model_path": "example/model",
                "tp_size": 2,
                "attention_dp_size": 1,
            },
        },
        "decode": {
            "provider": "aic",
            "config": {
                "backend": "vllm",
                "system": "h200_sxm",
                "model_path": "example/model",
                "tp_size": 1,
                "attention_dp_size": 1,
            },
        },
    }

    result = replay_planner.prepare_planner_replay(
        extra_engine_args=None,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        planner_config_arg=json.dumps(
            {
                "mode": "disagg",
                "optimization_target": "sla",
                "enable_throughput_scaling": True,
                "enable_load_scaling": False,
            }
        ),
        benchmark_granularity=1,
        performance_model_metadata=metadata,
    )

    assert result is adapter
    assert [request["tp_size"] for request in session_requests] == [2, 1]
    assert adapter.prefill_fpms
    assert adapter.decode_fpms
    assert adapter.prefill_fpms[0].wall_time == pytest.approx(0.002)
    assert adapter.decode_fpms[0].wall_time == pytest.approx(0.001)


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


def test_merge_traffic_keeps_offered_count_separate_from_completion_samples():
    a = {
        "num_req": 100,
        "duration_s": 1.0,
        "avg_isl": 10.0,
        "avg_osl": 20.0,
        "shape_count": 1,
        "avg_ttft_ms": 1_000.0,
        "ttft_count": 1,
        "avg_itl_ms": 10.0,
        "itl_count": 1,
    }
    b = {
        "num_req": 1,
        "duration_s": 1.0,
        "avg_isl": 100.0,
        "avg_osl": 200.0,
        "shape_count": 9,
        "avg_ttft_ms": 2_000.0,
        "ttft_count": 9,
        "avg_itl_ms": 20.0,
        "itl_count": 9,
    }

    merged = _merge_traffic(a, b)

    assert merged["num_req"] == 101
    assert merged["shape_count"] == 10
    assert merged["avg_isl"] == pytest.approx(91.0)
    assert merged["avg_osl"] == pytest.approx(182.0)
    assert merged["avg_ttft_ms"] == pytest.approx(1_900.0)
    assert merged["avg_itl_ms"] == pytest.approx(19.0)
