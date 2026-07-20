# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReplayEvaluator dispatch across 3 load shapes x {static, planner} (dynamo stubbed)."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("dynamo.mocker")

import dynamo.mocker
import dynamo.replay.api

from dynamo.profiler.spica.config import (
    OptimizationGoal,
    OptimizationTarget,
    SLATarget,
    Workload,
)
from dynamo.profiler.spica.deploy import DeploymentPlan
from dynamo.profiler.spica.evaluator import ReplayEvaluator


class _FakeArgs:
    @classmethod
    def from_json(cls, s):
        return ("ARGS", json.loads(s))


def _wl():
    return Workload(trace_path="/tmp/t.jsonl")


def _agg_plan(static):
    return DeploymentPlan(
        deployment_mode="agg",
        is_static=static,
        agg_engine_args={"aic_tp_size": 4, "max_num_seqs": 512},
        prefill_engine_args=None,
        decode_engine_args=None,
        num_workers=2,
        num_prefill_workers=0,
        num_decode_workers=0,
        router_mode="round_robin",
        router_config=None,
        planner_config=None
        if static
        else {"mode": "agg", "optimization_target": "sla"},
    )


def _disagg_plan(static):
    return DeploymentPlan(
        deployment_mode="disagg",
        is_static=static,
        agg_engine_args=None,
        prefill_engine_args={"aic_tp_size": 2, "max_num_seqs": 256},
        decode_engine_args={"aic_tp_size": 4, "max_num_seqs": 512},
        num_workers=0,
        num_prefill_workers=3,
        num_decode_workers=5,
        router_mode="round_robin",
        router_config=None,
        planner_config=None
        if static
        else {"mode": "disagg", "optimization_target": "sla"},
    )


def test_static_agg_uses_plain_path(monkeypatch):
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 42.0},
    )
    ev = ReplayEvaluator(_wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT))
    report = ev.evaluate(_agg_plan(static=True))
    assert report["output_throughput_tok_s"] == 42.0
    assert (
        rec["num_workers"] == 2
        and rec["trace_files"] == "/tmp/t.jsonl"
        and rec["router_mode"] == "round_robin"
    )
    assert "sla_ttft_ms" not in rec  # no SLA on a throughput goal -> none threaded


def test_static_path_threads_goodput_sla(monkeypatch):
    # the mocker is SLA-aware on the plain path too, so a static/disabled candidate
    # still receives the goodput SLA and can emit goodput.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw)
        or {"goodput_output_throughput_tok_s": 100.0, "gpu_hours": 1.0},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    report = ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=True))
    assert report["goodput_output_throughput_tok_s"] == 100.0
    assert (
        rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0
    )  # SLA threaded to the plain path


def test_current_trace_api_filters_legacy_kwargs_and_fails_closed_without_goodput(
    monkeypatch,
):
    # Current Dynamo replay wrappers do not accept planner_config, benchmark_granularity,
    # or SLA kwargs on static replay. Aggregate means cannot reconstruct per-request
    # goodput, so an incompatible wrapper must fail instead of fabricating a score.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}

    def run_trace_replay(
        trace_file,
        *,
        extra_engine_args=None,
        num_workers=1,
        replay_concurrency=None,
        router_mode=None,
    ):
        rec.update(
            trace_file=trace_file,
            extra_engine_args=extra_engine_args,
            num_workers=num_workers,
            replay_concurrency=replay_concurrency,
            router_mode=router_mode,
        )
        return {
            "output_throughput_tok_s": 77.0,
            "mean_ttft_ms": 10.0,
            "mean_tpot_ms": 2.0,
            "mean_e2e_latency_ms": 42.0,
        }

    monkeypatch.setattr(dynamo.replay.api, "run_trace_replay", run_trace_replay)
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    with pytest.raises(RuntimeError, match="per-request SLA accounting"):
        ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=True))
    assert rec["trace_file"] == "/tmp/t.jsonl"
    assert set(rec) == {
        "trace_file",
        "extra_engine_args",
        "num_workers",
        "replay_concurrency",
        "router_mode",
    }


def test_current_trace_api_rejects_planner_config(monkeypatch):
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)

    def run_trace_replay(trace_file, *, extra_engine_args=None, num_workers=1):
        return {"output_throughput_tok_s": 1.0}

    monkeypatch.setattr(dynamo.replay.api, "run_trace_replay", run_trace_replay)
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    with pytest.raises(RuntimeError, match="does not accept planner_config"):
        ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=False))


def test_scaling_agg_threads_planner_config_when_supported(monkeypatch):
    # Older Dynamo replay wrappers accept planner_config directly; keep threading it
    # when the installed API advertises that kwarg.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw)
        or {"gpu_hours": 2.0, "goodput_output_throughput_tok_s": 100.0},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    report = ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=False))
    assert report["gpu_hours"] == 2.0
    # goodput SLA threaded to the planner path; planner config carried as a dict
    assert rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0
    assert rec["planner_config"]["optimization_target"] == "sla"
    assert rec["num_workers"] == 2


def test_scaling_report_preserves_planner_tick_count(monkeypatch):
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: SimpleNamespace(
            trace_report={"output_throughput_tok_s": 42.0},
            total_ticks=3,
        ),
    )
    goal = OptimizationGoal(target=OptimizationTarget.THROUGHPUT)
    report = ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=False))
    assert report["output_throughput_tok_s"] == 42.0
    assert report["planner_total_ticks"] == 3.0


def test_kv_router_config_is_built_and_passed(monkeypatch):
    # the searched kv-router weights must reach the replay as a real KvRouterConfig
    # (round_robin -> None). Regression for the "router_config=None" stub.
    from dynamo._core import KvRouterConfig

    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 1.0},
    )
    ev = ReplayEvaluator(_wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT))

    # round_robin -> no router config
    ev.evaluate(_agg_plan(static=True))
    assert rec["router_mode"] == "round_robin" and rec["router_config"] is None

    # kv_router with weights -> a KvRouterConfig carrying them
    plan = dataclasses.replace(
        _agg_plan(static=True),
        router_mode="kv_router",
        router_config={"overlap_score_credit": 0.5, "router_temperature": 0.2},
    )
    ev.evaluate(plan)
    assert rec["router_mode"] == "kv_router"
    assert isinstance(rec["router_config"], KvRouterConfig)


def test_static_trace_threads_replay_concurrency(monkeypatch):
    # a closed-loop concurrency cap on a trace reaches run_trace_replay as
    # replay_concurrency (run_trace_replay defaults replay_mode='offline').
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 1.0},
    )
    wl = Workload(trace_path="/tmp/t.jsonl", replay_concurrency=32)
    ReplayEvaluator(
        wl, OptimizationGoal(target=OptimizationTarget.THROUGHPUT)
    ).evaluate(_agg_plan(static=True))
    assert rec["replay_concurrency"] == 32


def test_scaling_trace_threads_replay_concurrency_when_supported(monkeypatch):
    # closed-loop concurrency over a trace + planner works when the replay wrapper
    # accepts planner_config.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw)
        or {"gpu_hours": 1.0, "goodput_output_throughput_tok_s": 1.0},
    )
    wl = Workload(trace_path="/tmp/t.jsonl", replay_concurrency=32)
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    ReplayEvaluator(wl, goal).evaluate(_agg_plan(static=False))
    assert rec["replay_concurrency"] == 32 and rec["trace_files"] == "/tmp/t.jsonl"


def _syn_wl(**kw):
    # num_request_ratio=25 -> resolved request_count = 25 * load (so concurrency=4 -> 100,
    # exercising the ratio math through the evaluator).
    base = dict(isl=128, osl=64, num_request_ratio=25)
    base.update(kw)
    return Workload(**base)


def test_synthetic_static_uses_run_synthetic_trace_replay(monkeypatch):
    # synthetic + static -> a single run_synthetic_trace_replay call with planner_config=None
    # (fixed-fleet replay, no scaling); goodput SLA threaded; in-flight cap = concurrency.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_synthetic_trace_replay",
        lambda **kw: rec.update(kw)
        or {"goodput_output_throughput_tok_s": 50.0, "gpu_hours": 0.5},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    report = ReplayEvaluator(_syn_wl(concurrency=4.0), goal).evaluate(
        _agg_plan(static=True)
    )
    assert report["goodput_output_throughput_tok_s"] == 50.0
    assert (
        rec["input_tokens"] == 128
        and rec["output_tokens"] == 64
        and rec["request_count"] == 100
    )
    assert rec["replay_concurrency"] == 4  # closed-loop cap from concurrency
    assert rec["sla_ttft_ms"] == 2000.0
    assert rec["planner_config"] is None  # static -> no planner in the loop
    assert rec["num_workers"] == 2


def test_concurrency_override_drives_cap_and_request_count(monkeypatch):
    # KV-load mode derives an absolute concurrency before evaluation; that override sets
    # BOTH the closed-loop in-flight cap and the num_request_ratio-scaled request count.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_synthetic_trace_replay",
        lambda **kw: rec.update(kw)
        or {"output_throughput_tok_s": 50.0, "gpu_hours": 0.5},
    )
    wl = Workload(isl=128, osl=64, kv_load_ratio=[0.0, 1.0], num_request_ratio=25)
    goal = OptimizationGoal(target=OptimizationTarget.PARETO)  # no SLA needed
    ReplayEvaluator(wl, goal).evaluate(_agg_plan(static=True), concurrency_override=8)
    assert rec["replay_concurrency"] == 8  # cap = the candidate-derived concurrency
    assert rec["request_count"] == 200  # num_request_ratio(25) * 8
    assert rec["planner_config"] is None


def test_synthetic_planner_threads_planner_config_when_supported(monkeypatch):
    # synthetic + planner -> run_synthetic_trace_replay with planner_config when supported.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_synthetic_trace_replay",
        lambda **kw: rec.update(kw)
        or {"gpu_hours": 2.0, "goodput_output_throughput_tok_s": 1.0},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=1500.0, itl_ms=50.0),
    )
    # request-rate workload -> open-loop (no cap); arrival_interval derived from the rate
    ReplayEvaluator(_syn_wl(request_rate=20.0), goal).evaluate(_agg_plan(static=False))
    assert rec["replay_concurrency"] is None
    assert rec["input_tokens"] == 128 and rec["output_tokens"] == 64
    assert rec["arrival_interval_ms"] == 50.0  # 1000 / 20
    assert rec["planner_config"]["mode"] == "agg"
    assert rec["sla_ttft_ms"] == 1500.0


def test_static_trace_disagg_uses_plain_path(monkeypatch):
    # disagg + trace + static -> run_trace_replay gets prefill/decode engine args
    # and the per-role worker counts (no agg num_workers/extra_engine_args).
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 7.0},
    )
    report = ReplayEvaluator(
        _wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT)
    ).evaluate(_disagg_plan(static=True))
    assert report["output_throughput_tok_s"] == 7.0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert "num_workers" not in rec and "extra_engine_args" not in rec


def test_scaling_trace_disagg_threads_planner_config_when_supported(monkeypatch):
    # disagg + trace + planner -> run_trace_replay with per-role workers and planner_config
    # when supported.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw)
        or {"gpu_hours": 3.0, "goodput_output_throughput_tok_s": 90.0},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    report = ReplayEvaluator(_wl(), goal).evaluate(_disagg_plan(static=False))
    assert report["gpu_hours"] == 3.0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0
    assert rec["planner_config"]["mode"] == "disagg"


def test_synthetic_static_disagg_uses_run_synthetic_trace_replay(monkeypatch):
    # disagg + synthetic + static -> run_synthetic_trace_replay with prefill/decode args and
    # planner_config=None; goodput SLA threaded; closed-loop in-flight cap = concurrency.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_synthetic_trace_replay",
        lambda **kw: rec.update(kw)
        or {"goodput_output_throughput_tok_s": 60.0, "gpu_hours": 0.75},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0),
    )
    report = ReplayEvaluator(_syn_wl(concurrency=4.0), goal).evaluate(
        _disagg_plan(static=True)
    )
    assert report["goodput_output_throughput_tok_s"] == 60.0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert (
        rec["input_tokens"] == 128
        and rec["output_tokens"] == 64
        and rec["request_count"] == 100
    )
    assert rec["replay_concurrency"] == 4  # closed-loop cap from concurrency
    assert rec["sla_ttft_ms"] == 2000.0
    assert rec["planner_config"] is None
    assert "num_workers" not in rec and "extra_engine_args" not in rec


def test_synthetic_planner_disagg_threads_planner_config_when_supported(monkeypatch):
    # disagg + synthetic + planner -> run_synthetic_trace_replay with per-role workers and
    # planner_config when supported.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs, raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_synthetic_trace_replay",
        lambda **kw: rec.update(kw)
        or {"gpu_hours": 2.0, "goodput_output_throughput_tok_s": 1.0},
    )
    goal = OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(ttft_ms=1500.0, itl_ms=50.0),
    )
    # request-rate workload -> open-loop (no cap); arrival_interval derived from the rate
    ReplayEvaluator(_syn_wl(request_rate=20.0), goal).evaluate(
        _disagg_plan(static=False)
    )
    assert rec["replay_concurrency"] is None
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert rec["arrival_interval_ms"] == 50.0  # 1000 / 20
    assert rec["planner_config"]["mode"] == "disagg"
    assert rec["sla_ttft_ms"] == 1500.0


def test_workload_validation():
    with pytest.raises(ValueError, match="positive integer"):  # trace + bad cap
        Workload(trace_path="/tmp/t.jsonl", replay_concurrency=0)
    with pytest.raises(
        ValueError, match="for trace workloads"
    ):  # synthetic can't use replay_concurrency
        _syn_wl(concurrency=1.0, replay_concurrency=8)
    with pytest.raises(
        ValueError, match="exactly one of request_rate, concurrency, or kv_load_ratio"
    ):
        _syn_wl()  # synthetic needs a load shape
    with pytest.raises(
        ValueError, match="exactly one of request_rate, concurrency, or kv_load_ratio"
    ):
        _syn_wl(request_rate=10.0, concurrency=4.0)  # not both
    with pytest.raises(ValueError, match="must not set synthetic fields"):
        Workload(trace_path="/tmp/t.jsonl", isl=128)
