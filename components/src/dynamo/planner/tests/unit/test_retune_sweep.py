# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the sweep module imports it.

"""Unit tests for the retune sweep: scoring, worker isolation and the pre-sweep.

Sweeps start their workers with ``spawn``, so the fake evaluators below are
module-level functions that a worker imports by reference.
"""

import dataclasses
import json
import math
import multiprocessing
import os
import pickle
import random
import threading
import time
import types

import psutil
import pytest

aisimulate_sweeper = pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.config.planner_config import RetuneEngineSpec
from dynamo.planner.core.types import RetuneParams, TrafficObservation
from dynamo.planner.retune import sweep as sweep_module
from dynamo.planner.retune.sweep import (
    GOODPUT_KEY,
    SweepRequest,
    evaluate_candidate,
    predictor_presweep,
    run_sweep,
    score_summary,
    with_predictor_fields,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

SCHEMA = "dynamo.request.trace.v1"
ORIGIN_MS = 1_700_000_000_000
# Router values that tell the fake evaluator how to behave.
ABORT, HANG, RAISE = 9.0, 8.0, 7.0


def _rows(arrivals_ms, *, isl=256, osl=16, total_ms=50.0, prefix="r"):
    return [
        {
            "schema": SCHEMA,
            "event_type": "request_end",
            "event_time_unix_ms": int(arrival + total_ms),
            "request": {
                "request_id": f"{prefix}{index}",
                "request_received_ms": int(arrival),
                "total_time_ms": total_ms,
                "output_tokens": osl,
                "replay": {
                    "trace_block_size": 64,
                    "input_length": isl,
                    "input_sequence_hashes": [index % 3, 100 + index, 200 + index, 7],
                },
            },
        }
        for index, arrival in enumerate(arrivals_ms)
    ]


def _write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _engine_args(**engine):
    return normalize_mocker_config(
        {
            "engine": {
                "block_size": 64,
                "num_gpu_blocks": 4096,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 8192,
                "speedup_ratio": 1000.0,
                **engine,
            }
        }
    )


def _kv_engine(disagg=False):
    router_config = json.loads(KvRouterConfig().to_json())
    if disagg:
        return RetuneEngineSpec(
            prefill_engine_args=_engine_args(worker_type="prefill"),
            decode_engine_args=_engine_args(worker_type="decode"),
            router_config=router_config,
        )
    return RetuneEngineSpec(
        extra_engine_args=_engine_args(), router_config=router_config
    )


def _params(credit=1.0, temperature=0.0, **planner):
    return RetuneParams(
        router={"overlap_score_credit": credit, "router_temperature": temperature},
        planner=planner,
    )


def _request(trace_paths, candidates, baseline, **overrides):
    values = {
        "window_index": 0,
        "window_start_ms": ORIGIN_MS,
        "window_end_ms": ORIGIN_MS + 10_000,
        "history_start_ms": ORIGIN_MS,
        "as_of_ms": ORIGIN_MS + 20_000,
        "trace_paths": tuple(trace_paths),
        "prune_by_mtime": False,
        "arrival_speedup_ratio": 1.0,
        "engine": RetuneEngineSpec(),
        "candidates": tuple(sorted(candidates, key=RetuneParams.key)),
        "baseline": baseline,
        "replicas": {"aggregated": 2},
        "base_planner": None,
        "predictor_presets": None,
        "sla_ttft_ms": 500.0,
        "sla_itl_ms": 30.0,
        "max_parallel": 2,
        "timeout_s": 120.0,
    }
    values.update(overrides)
    return SweepRequest(**values)


def _fake_evaluator(
    window_path,
    engine,
    arrival_speedup_ratio,
    params,
    replicas,
    base_planner,
    sla_ttft_ms,
    sla_itl_ms,
    window_s,
):
    action = params.router.get("router_temperature")
    if action == ABORT:
        os.abort()
    if action == HANG:
        time.sleep(600)
    if action == RAISE:
        raise RuntimeError("replay exploded")
    with open(window_path, encoding="utf-8") as source:
        rows = sum(1 for _ in source)
    warmup_trace = (
        None
        if base_planner is None
        else base_planner.get("load_predictor_warmup_trace")
    )
    history_rows = None
    if warmup_trace is not None:
        with open(warmup_trace, encoding="utf-8") as source:
            history_rows = sum(1 for _ in source)
    return 1.0 + params.router.get("overlap_score_credit", 0.0), {
        "rows": rows,
        "planner": dict(params.planner),
        "replicas": dict(replicas),
        "window_s": window_s,
        "warmup_trace": warmup_trace,
        "history_rows": history_rows,
        "retune": None if base_planner is None else base_planner.get("retune", "?"),
        "reports": (
            None
            if base_planner is None
            else (
                base_planner.get("report_interval_hours", "?"),
                base_planner.get("live_dashboard_port", "?"),
            )
        ),
        "pid": os.getpid(),
    }


def _evaluate_and_count_children(*args):
    score, metrics = evaluate_candidate(*args)
    return score, {**metrics, "children": len(psutil.Process().children())}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_objective_value_formula_has_not_drifted():
    # GOODPUT_PER_GPU = goodput / (gpu_hours / duration_hours)
    summary = {GOODPUT_KEY: 120.0, "gpu_hours": 2.0, "duration_ms": 1_800_000.0}
    objective = aisimulate_sweeper.objective_value(
        summary, aisimulate_sweeper.OptimizationTarget.GOODPUT_PER_GPU
    )
    assert objective == pytest.approx(120.0 / (2.0 / 0.5))


def test_score_uses_the_fixed_window_denominator():
    summary = {GOODPUT_KEY: 120.0, "gpu_hours": 2.0, "duration_ms": 1_800_000.0}

    score, metrics = score_summary(summary, window_s=1_500.0)

    # goodput tokens over the run / (window seconds x average GPUs)
    assert score == pytest.approx(120.0 * 1_800.0 / (1_500.0 * 4.0))
    assert metrics["objective_goodput_per_gpu"] == pytest.approx(30.0)
    assert metrics["duration_ms"] == 1_800_000.0


@pytest.mark.parametrize(
    "summary",
    [
        {"gpu_hours": 1.0, "duration_ms": 1.0},
        {GOODPUT_KEY: None, "gpu_hours": 1.0, "duration_ms": 1.0},
        {GOODPUT_KEY: 1.0, "gpu_hours": 0.0, "duration_ms": 1.0},
        {GOODPUT_KEY: 1.0, "gpu_hours": 1.0, "duration_ms": 0.0},
        {GOODPUT_KEY: 1.0, "duration_ms": 1.0},
        {GOODPUT_KEY: math.nan, "gpu_hours": 1.0, "duration_ms": 1.0},
    ],
)
def test_score_fails_closed_without_score_inputs(summary):
    with pytest.raises(ValueError):
        score_summary(summary, window_s=10.0)


def test_with_predictor_fields_keeps_the_baseline_predictor():
    baseline = _params(1.0, load_predictor="arima")
    other = _params(0.0, load_scaling_down_sensitivity=70)
    fields = {"load_predictor": "kalman", "kalman_r": 5.0}

    assert with_predictor_fields(baseline, baseline, fields) is baseline
    unchanged = _params(0.0, load_scaling_down_sensitivity=70, load_predictor="arima")
    assert with_predictor_fields(unchanged, baseline, None) is unchanged
    assert with_predictor_fields(other, baseline, None).planner == {
        "load_scaling_down_sensitivity": 70,
        "load_predictor": "arima",
    }
    assert with_predictor_fields(other, baseline, fields).planner == {
        "load_scaling_down_sensitivity": 70,
        "load_predictor": "kalman",
        "kalman_r": 5.0,
    }


def test_a_grid_point_equivalent_to_the_baseline_gets_its_key():
    # With planner: default the current point carries every predictor field,
    # a grid point only the searched ones; a winner equal to the current
    # predictor must not make the same inner planner replay twice.
    current = {
        "load_predictor": "prophet",
        "load_predictor_log1p": False,
        "prophet_window_size": 50,
        "kalman_r": 10.0,
        "load_scaling_down_sensitivity": 90,
    }
    baseline = RetuneParams(router={"overlap_score_credit": 2.0}, planner=current)
    grid_point = RetuneParams(
        router={"overlap_score_credit": 2.0},
        planner={"load_scaling_down_sensitivity": 90},
    )
    other = RetuneParams(
        router={"overlap_score_credit": 1.0},
        planner={"load_scaling_down_sensitivity": 70},
    )
    winner = {
        "load_predictor": "prophet",
        "load_predictor_log1p": False,
        "prophet_window_size": 50,
    }

    assert with_predictor_fields(grid_point, baseline, winner) == baseline
    assert with_predictor_fields(other, baseline, winner).planner == {
        **current,
        "load_scaling_down_sensitivity": 70,
    }


def test_request_and_engine_survive_pickling(tmp_path):
    baseline = _params(1.0)
    request = _request(
        ["trace.jsonl"], [baseline, _params(0.0)], baseline, engine=_kv_engine()
    )

    restored = pickle.loads(pickle.dumps(request))

    assert restored.candidates == request.candidates
    assert restored.engine == request.engine
    assert restored.window_s == 10.0


def test_request_rejects_unsorted_or_foreign_baseline():
    a, b = sorted([_params(0.0), _params(1.0)], key=RetuneParams.key)
    valid = _request(["t"], [a, b], a)
    with pytest.raises(ValueError, match="sorted"):
        dataclasses.replace(valid, candidates=(b, a))
    with pytest.raises(ValueError, match="sorted"):
        dataclasses.replace(valid, candidates=(a, a))
    with pytest.raises(ValueError, match="baseline"):
        _request(["t"], [a], b)
    with pytest.raises(ValueError, match="presets"):
        _request(["t"], [a], a, predictor_presets=("nope",))


# ---------------------------------------------------------------------------
# Real replays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disagg", [False, True], ids=["agg", "disagg"])
def test_evaluate_candidate_scores_a_tiny_window(tmp_path, disagg):
    window = _write_trace(
        tmp_path / "window.jsonl", _rows([ORIGIN_MS + 250 * i for i in range(30)])
    )
    replicas = {"prefill": 1, "decode": 2} if disagg else {"aggregated": 2}

    score, metrics = evaluate_candidate(
        window,
        _kv_engine(disagg),
        1.0,
        _params(0.5),
        replicas,
        None,
        500.0,
        30.0,
        10.0,
    )

    assert score > 0
    assert metrics["completed_requests"] == 30
    assert metrics[GOODPUT_KEY] > 0
    assert metrics["gpu_hours"] > 0 and metrics["duration_ms"] > 0


def test_evaluate_candidate_runs_an_inner_load_planner(tmp_path):
    pytest.importorskip("aisimulate.replay")
    window = _write_trace(
        tmp_path / "window.jsonl", _rows([ORIGIN_MS + 250 * i for i in range(30)])
    )
    base_planner = {
        "mode": "agg",
        "optimization_target": "throughput",
        "report_interval_hours": None,
        "live_dashboard_port": 0,
        "retune": None,
    }

    score, metrics = evaluate_candidate(
        window,
        _kv_engine(),
        1.0,
        RetuneParams(planner={"load_scaling_down_sensitivity": 70}),
        {"aggregated": 1},
        base_planner,
        500.0,
        30.0,
        10.0,
    )

    assert score > 0
    assert metrics["completed_requests"] == 30


def test_evaluate_candidate_rejects_planner_params_without_planner(tmp_path):
    with pytest.raises(ValueError, match="inner planner"):
        evaluate_candidate(
            "unused.jsonl",
            _kv_engine(),
            1.0,
            RetuneParams(planner={"load_scaling_down_sensitivity": 70}),
            {"aggregated": 1},
            None,
            500.0,
            30.0,
            10.0,
        )


@pytest.mark.timeout(240)
def test_run_sweep_replays_each_candidate_in_spawned_workers(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        _rows([ORIGIN_MS + 250 * i for i in range(30)] + [ORIGIN_MS + 50_000]),
    )
    baseline = RetuneParams(router={"overlap_score_credit": 1.0})
    other = RetuneParams(router={"overlap_score_credit": 0.0})
    request = _request(
        [trace],
        [baseline, other],
        baseline,
        engine=_kv_engine(disagg=True),
        replicas={"prefill": 1, "decode": 1},
    )

    result = run_sweep(request, evaluator=_evaluate_and_count_children)

    assert result.status == "ok", result.error
    assert result.num_requests == 30
    assert result.row_counts["outside_window"] == 1
    assert set(result.scores) == {baseline.key(), other.key()}
    assert all(score is not None and score > 0 for score in result.scores.values())
    for key, metrics in result.metrics.items():
        # Inner replays run in-process in the worker: no grandchildren.
        assert metrics["children"] == 0
        assert RetuneParams(**metrics["params"]).key() == key
    assert result.warmup_observations is None
    assert result.predictor_presweep is None
    assert multiprocessing.active_children() == []


# ---------------------------------------------------------------------------
# Worker isolation, deadline and cancellation (fake evaluator)
# ---------------------------------------------------------------------------


def _fake_trace(tmp_path):
    return _write_trace(
        tmp_path / "trace.jsonl", _rows([ORIGIN_MS + 1_000 * i for i in range(5)])
    )


@pytest.mark.timeout(240)
def test_a_dead_worker_fails_only_its_candidate(tmp_path):
    ok_a, ok_b = _params(0.0), _params(2.0)
    aborts, raises = _params(1.0, ABORT), _params(1.0, RAISE)
    request = _request(
        [_fake_trace(tmp_path)], [ok_a, ok_b, aborts, raises], ok_a, max_parallel=2
    )

    result = run_sweep(request, evaluator=_fake_evaluator)

    assert result.status == "ok", result.error
    assert result.scores[ok_a.key()] == 1.0
    assert result.scores[ok_b.key()] == 3.0
    assert result.scores[aborts.key()] is None
    assert result.scores[raises.key()] is None
    assert "exited with code" in result.metrics[aborts.key()]["error"]
    assert "replay exploded" in result.metrics[raises.key()]["error"]
    assert result.metrics[ok_a.key()]["rows"] == 5
    assert result.metrics[ok_a.key()]["window_s"] == 10.0
    assert multiprocessing.active_children() == []


@pytest.mark.timeout(240)
def test_deadline_kills_workers_and_reports_timeout(tmp_path):
    hangs = _params(1.0, HANG)
    # The deadline passes long before the worker has imported; nothing waits
    # for the import.
    request = _request([_fake_trace(tmp_path)], [hangs], hangs, timeout_s=2.0)

    started = time.monotonic()
    result = run_sweep(request, evaluator=_fake_evaluator)

    assert result.status == "timeout"
    assert result.scores == {}
    assert time.monotonic() - started < 60
    assert multiprocessing.active_children() == []


@pytest.mark.timeout(240)
def test_cancel_kills_workers(tmp_path):
    hangs = _params(1.0, HANG)
    request = _request([_fake_trace(tmp_path)], [hangs], hangs, timeout_s=200.0)
    cancel = threading.Event()
    timer = threading.Timer(1.0, cancel.set)
    timer.start()
    try:
        result = run_sweep(request, cancel=cancel, evaluator=_fake_evaluator)
    finally:
        timer.cancel()

    assert result.status == "failed"
    assert result.error == "sweep cancelled"
    assert multiprocessing.active_children() == []


def test_a_task_whose_workers_die_before_accepting_it_fails():
    pool = sweep_module._WorkerPool(1, time.monotonic() + 60, threading.Event())
    spawned = []

    def dead_worker():
        process = types.SimpleNamespace(
            is_alive=lambda: False, join=lambda *_: None, kill=lambda: None
        )
        worker = sweep_module._Worker(
            process=process, conn=types.SimpleNamespace(close=lambda: None)
        )
        pool._workers.append(worker)
        spawned.append(worker)
        return worker

    pool._spawn = dead_worker

    results = pool.run([("task", print, ())])

    ok, error = results["task"]
    assert not ok and "before accepting the task 3 times" in error
    assert len(spawned) == 3 and pool._workers == []


def test_empty_window_evaluates_nothing(tmp_path):
    trace = _write_trace(tmp_path / "trace.jsonl", _rows([ORIGIN_MS - 5_000]))
    baseline = _params(1.0)

    result = run_sweep(
        _request([trace], [baseline], baseline), evaluator=_fake_evaluator
    )

    assert result.status == "empty"
    assert result.num_requests == 0
    assert result.row_counts["outside_window"] == 1
    assert result.scores == {}


def test_read_failure_is_a_failed_result(tmp_path):
    baseline = _params(1.0)

    result = run_sweep(
        _request([str(tmp_path / "missing")], [baseline], baseline),
        evaluator=_fake_evaluator,
    )

    assert result.status == "failed"
    assert "FileNotFoundError" in result.error


# ---------------------------------------------------------------------------
# Predictor pre-sweep
# ---------------------------------------------------------------------------


def _ramp(count, *, start=10.0, step=4.0):
    return [
        TrafficObservation(duration_s=10.0, num_req=start + step * i, isl=100, osl=10)
        for i in range(count)
    ]


def _wavy(count, *, seed=7):
    """A noisy periodic series, so the predictor families rank apart."""

    rng = random.Random(seed)
    return [
        TrafficObservation(
            duration_s=180.0,
            num_req=50 + 30 * math.sin(i / 6) + rng.uniform(-5, 5),
            isl=1000 + rng.uniform(-100, 100),
            osl=200 + 20 * math.cos(i / 5),
        )
        for i in range(count)
    ]


def _predictors_module():
    # The predictor module imports all three forecasting libraries.
    for name in ("filterpy", "pmdarima", "prophet"):
        pytest.importorskip(name)
    from dynamo.planner.core.load import predictors

    return predictors


def _spy_on_forecasts(monkeypatch, predictor_class):
    """Record the history size each ``predict_next`` call sees."""

    seen: list[int] = []
    original = predictor_class.predict_next

    def predict_next(self):
        seen.append(len(self.data_buffer))
        return original(self)

    monkeypatch.setattr(predictor_class, "predict_next", predict_next)
    return seen


def test_presweep_scores_only_the_window_and_picks_the_lowest_loss():
    pytest.importorskip("filterpy")
    observations = _ramp(30)

    result = predictor_presweep(
        observations, ["constant_last", "kalman_default_raw"], 10, eval_steps=10
    )

    assert result["eval_steps"] == 10
    assert result["warmup_steps"] == 20
    assert set(result["losses"]) == {"constant_last", "kalman_default_raw"}
    assert all(loss is not None for loss in result["losses"].values())
    # A linear ramp favours the trend-following Kalman filter.
    assert result["winner"] == "kalman_default_raw"
    assert result["fields"]["load_predictor"] == "kalman"
    assert result["losses"]["kalman_default_raw"] < result["losses"]["constant_last"]
    assert result["wall_s"] >= 0

    nothing = predictor_presweep(observations, ["constant_last"], 10, eval_steps=0)
    assert nothing["winner"] is None
    assert nothing["fields"] == {}
    assert nothing["losses"] == {"constant_last": None}


def test_presweep_warms_history_without_forecasting(monkeypatch):
    predictors = _predictors_module()
    constant = _spy_on_forecasts(monkeypatch, predictors.ConstantPredictor)
    kalman = _spy_on_forecasts(monkeypatch, predictors.KalmanPredictor)
    history, window = 40, 10

    result = predictor_presweep(
        _ramp(history + window),
        ["constant_last", "kalman_default_raw"],
        10,
        eval_steps=window,
    )

    assert (result["warmup_steps"], result["eval_steps"]) == (history, window)
    assert all(loss is not None for loss in result["losses"].values())
    # Three series x window buckets forecasts per preset, none for history:
    # the first forecast already sees every history bucket.
    for seen in (constant, kalman):
        assert len(seen) == 3 * window
        assert seen[:3] == [history] * 3
        assert seen[-3:] == [history + window - 1] * 3


# 17-26 s measured on a loaded 24-core host, mostly Prophet fits; contended
# CI runners (arm64) fit Prophet slower.
@pytest.mark.timeout(120)
# Prophet/pandas emit this on newer NumPy; CI turns warnings into errors.
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
)
def test_presweep_ranking_matches_evaluate_preset():
    _predictors_module()
    from dynamo.planner.simulation.load_predictor import (
        LOAD_PREDICTOR_PRESETS,
        Window,
        evaluate_preset,
    )

    def evaluate_preset_losses(observations, names, eval_steps):
        windows = [Window(obs.num_req, obs.isl, obs.osl) for obs in observations]
        warmup = len(windows) - eval_steps
        return {
            name: evaluate_preset(windows, LOAD_PREDICTOR_PRESETS[name], 180, warmup)
            for name in names
        }

    def ranking(losses):
        return sorted(losses, key=lambda name: (losses[name], list(losses).index(name)))

    # Short history (fewer than the 5 points ARIMA and Prophet need to fit):
    # every family matches evaluate_preset exactly, so do the rankings.
    names = [
        "constant_last",
        "arima_raw",
        "prophet_w20_log1p",
        "kalman_default_raw",
        "kalman_reactive_raw",
    ]
    short = _wavy(3 + 8)
    result = predictor_presweep(short, names, 180, eval_steps=8)
    expected = evaluate_preset_losses(short, names, 8)
    assert result["losses"] == pytest.approx(expected, rel=1e-9)
    assert ranking(result["losses"]) == ranking(expected)
    assert result["winner"] == ranking(expected)[0]

    # Long history: a forecast does not change the state constant and Kalman
    # predictors carry into the next bucket, so skipping the warmup forecasts
    # changes nothing. (Prophet neither, but it is slow; ARIMA instead fits
    # once on the whole history.)
    names = ["constant_last", "kalman_default_raw", "kalman_reactive_log1p"]
    long = _wavy(300 + 20)
    result = predictor_presweep(long, names, 180, eval_steps=20)
    expected = evaluate_preset_losses(long, names, 20)
    assert result["losses"] == pytest.approx(expected, rel=1e-12)
    assert ranking(result["losses"]) == ranking(expected)


def test_presweep_cost_is_bounded_by_the_window(monkeypatch):
    """2,000 history + 120 window buckets cost 3 x 120 forecasts per preset.

    Kalman is cheap, so each of its forecasts is also charged 5 ms of virtual
    time on the clock ``wall_s`` is measured with: 360 forecasts add 1.8 s,
    while forecasting the history as well (6,360) would add 31.8 s.
    """

    predictors = _predictors_module()
    kalman = _spy_on_forecasts(monkeypatch, predictors.KalmanPredictor)
    monotonic = time.monotonic
    monkeypatch.setattr(
        sweep_module,
        "time",
        types.SimpleNamespace(monotonic=lambda: monotonic() + 0.005 * len(kalman)),
    )

    result = predictor_presweep(
        _wavy(2_000 + 120),
        ["kalman_default_raw", "constant_last"],
        180,
        eval_steps=120,
    )

    assert (result["warmup_steps"], result["eval_steps"]) == (2_000, 120)
    assert len(kalman) == 3 * 120
    assert all(loss is not None for loss in result["losses"].values())
    # 1.8 s charged plus the real cost, which is well under a second here.
    assert 1.8 <= result["wall_s"] < 1.8 + 5.0


@pytest.mark.timeout(240)
def test_run_sweep_presweep_fixes_the_winner_into_non_baseline_candidates(tmp_path):
    pytest.importorskip("filterpy")
    interval_s = 10
    # History ramps up over 20 intervals before the window; the window keeps
    # ramping. One request per 1/n of an interval approximates the ramp. The
    # first 10 intervals come from a separate warmup trace (history_paths).
    arrivals = []
    for bucket in range(30):
        count = 2 + bucket
        for index in range(count):
            arrivals.append(
                ORIGIN_MS - 200_000 + bucket * 10_000 + index * (10_000 // count)
            )
    split = ORIGIN_MS - 100_000
    warmup_trace = _write_trace(
        tmp_path / "warmup.jsonl",
        # The stray in-window row shows windows never read history_paths.
        _rows([a for a in arrivals if a < split] + [ORIGIN_MS + 5], prefix="h"),
    )
    trace = _write_trace(
        tmp_path / "trace.jsonl", _rows([a for a in arrivals if a >= split])
    )
    window_start = ORIGIN_MS
    baseline = _params(1.0, load_predictor="arima", load_predictor_log1p=False)
    candidate = _params(0.0)
    base_planner = {
        "mode": "agg",
        "enable_throughput_scaling": True,
        "throughput_adjustment_interval_seconds": interval_s,
        "load_predictor_warmup_trace": "/outer/warmup.jsonl",
        "retune": {"interval_seconds": 100},
    }
    request = _request(
        [trace],
        [baseline, candidate],
        baseline,
        window_start_ms=window_start,
        window_end_ms=window_start + 100_000,
        history_start_ms=window_start - 200_000,
        as_of_ms=window_start + 200_000,
        base_planner=base_planner,
        predictor_presets=("constant_last", "kalman_default_raw"),
        max_parallel=1,
        history_paths=(warmup_trace,),
    )

    result = run_sweep(request, evaluator=_fake_evaluator)

    assert result.status == "ok", result.error
    window_rows = sum(1 for a in arrivals if a >= ORIGIN_MS)
    assert result.num_requests == window_rows
    assert result.row_counts["history_requests"] == len(arrivals) - window_rows
    presweep = result.predictor_presweep
    assert presweep["eval_steps"] == 10
    assert presweep["warmup_steps"] == 20
    assert set(presweep["losses"]) == {"constant_last", "kalman_default_raw"}
    assert presweep["winner"] == "kalman_default_raw"
    # The baseline keeps the current predictor; candidates carry the winner's.
    assert result.metrics[baseline.key()]["planner"]["load_predictor"] == "arima"
    assert result.metrics[candidate.key()]["planner"] == presweep["fields"]
    # Inner planners are warmed from the history file, never the outer trace.
    metrics = result.metrics[candidate.key()]
    assert metrics["warmup_trace"] != "/outer/warmup.jsonl"
    assert metrics["history_rows"] == result.row_counts["history_requests"] > 0
    assert metrics["retune"] is None
    assert metrics["reports"] == (None, 0)
    # Warmup covers history and window at the throughput interval.
    observations = result.warmup_observations
    assert len(observations) == 30
    assert all(obs.duration_s == interval_s for obs in observations)
    assert sum(obs.num_req for obs in observations) == len(arrivals)
