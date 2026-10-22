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
import random
import threading
import time
import types

import psutil
import pytest

pytest.importorskip(
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
# Router temperatures that tell the test evaluators how to behave.
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


def _disagg_engine():
    return RetuneEngineSpec(
        prefill_engine_args=_engine_args(worker_type="prefill"),
        decode_engine_args=_engine_args(worker_type="decode"),
        router_config=json.loads(KvRouterConfig().to_json()),
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
    if params.router.get("router_temperature") == HANG:
        time.sleep(600)
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
        "planner": dict(params.planner),
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
    }


def _abort_raise_or_evaluate(window_path, engine, arrival_speedup_ratio, params, *rest):
    """Kill or fail the worker on a marker temperature, else replay for real."""

    action = params.router.get("router_temperature")
    if action == ABORT:
        os.abort()
    if action == RAISE:
        raise RuntimeError("replay exploded")
    score, metrics = evaluate_candidate(
        window_path, engine, arrival_speedup_ratio, params, *rest
    )
    return score, {**metrics, "children": len(psutil.Process().children())}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_uses_the_fixed_window_denominator():
    summary = {GOODPUT_KEY: 120.0, "gpu_hours": 2.0, "duration_ms": 1_800_000.0}

    score, metrics = score_summary(summary, window_s=1_500.0)

    # goodput tokens over the run / (window seconds x average GPUs)
    assert score == pytest.approx(120.0 * 1_800.0 / (1_500.0 * 4.0))
    assert metrics["objective_goodput_per_gpu"] == pytest.approx(30.0)
    assert metrics["duration_ms"] == 1_800_000.0
    # Fails closed instead of scoring 0, which would tie every candidate.
    for bad, match in [
        ({"gpu_hours": 1.0, "duration_ms": 1.0}, "needs an SLA"),
        ({GOODPUT_KEY: 1.0, "gpu_hours": 0.0, "duration_ms": 1.0}, "positive"),
        ({GOODPUT_KEY: math.nan, "gpu_hours": 1.0, "duration_ms": 1.0}, "finite"),
    ]:
        with pytest.raises(ValueError, match=match):
            score_summary(bad, window_s=10.0)


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
    # The baseline keeps the current predictor; other candidates take the
    # winner's fields over the baseline's.
    kalman = {"load_predictor": "kalman", "kalman_r": 5.0}
    assert with_predictor_fields(baseline, baseline, kalman) == baseline
    assert with_predictor_fields(other, baseline, kalman).planner == {
        **current,
        "load_scaling_down_sensitivity": 70,
        **kalman,
    }


def test_request_rejects_unsorted_or_foreign_baseline():
    a, b = sorted([_params(0.0), _params(1.0)], key=RetuneParams.key)
    valid = _request(["t"], [a, b], a)
    with pytest.raises(ValueError, match="sorted"):
        dataclasses.replace(valid, candidates=(b, a))
    with pytest.raises(ValueError, match="baseline"):
        _request(["t"], [a], b)


# ---------------------------------------------------------------------------
# Spawned workers: isolation, deadline and cancellation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(240)
def test_a_dead_worker_fails_only_its_candidate(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        _rows([ORIGIN_MS + 250 * i for i in range(30)] + [ORIGIN_MS + 50_000]),
    )
    ok_a, ok_b = _params(1.0), _params(0.0)
    aborts, raises = _params(1.0, ABORT), _params(1.0, RAISE)
    request = _request(
        [trace],
        [ok_a, ok_b, aborts, raises],
        ok_a,
        engine=_disagg_engine(),
        replicas={"prefill": 1, "decode": 1},
        max_parallel=2,
    )

    result = run_sweep(request, evaluator=_abort_raise_or_evaluate)

    assert result.status == "ok", result.error
    assert result.num_requests == 30
    assert result.row_counts["outside_window"] == 1
    assert result.scores[aborts.key()] is None
    assert result.scores[raises.key()] is None
    assert "exited with code" in result.metrics[aborts.key()]["error"]
    assert "replay exploded" in result.metrics[raises.key()]["error"]
    for ok in (ok_a, ok_b):
        score, metrics = result.scores[ok.key()], result.metrics[ok.key()]
        assert score is not None and score > 0
        assert metrics["completed_requests"] == 30
        # Scored over the fixed 10 s window.
        assert score == pytest.approx(
            metrics["objective_goodput_per_gpu"] * metrics["duration_ms"] / 10_000.0
        )
        # Inner replays run in-process in the worker: no grandchildren.
        assert metrics["children"] == 0
    for key, metrics in result.metrics.items():
        assert RetuneParams(**metrics["params"]).key() == key
    assert result.warmup_observations is None
    assert result.predictor_presweep is None
    assert multiprocessing.active_children() == []


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "timeout_s, cancel_after_s, status, error",
    [
        (2.0, 600.0, "timeout", "sweep exceeded its deadline"),
        (200.0, 1.0, "failed", "sweep cancelled"),
    ],
    ids=["deadline", "cancel"],
)
def test_deadline_kills_workers_and_reports_timeout(
    tmp_path, timeout_s, cancel_after_s, status, error
):
    trace = _write_trace(
        tmp_path / "trace.jsonl", _rows([ORIGIN_MS + 1_000 * i for i in range(5)])
    )
    hangs = _params(1.0, HANG)
    # The deadline or cancel comes long before the worker has imported;
    # nothing waits for the import.
    request = _request([trace], [hangs], hangs, timeout_s=timeout_s)
    cancel = threading.Event()
    timer = threading.Timer(cancel_after_s, cancel.set)

    started = time.monotonic()
    timer.start()
    try:
        result = run_sweep(request, cancel=cancel, evaluator=_fake_evaluator)
    finally:
        timer.cancel()

    assert (result.status, result.error) == (status, error)
    assert result.scores == {}
    assert time.monotonic() - started < 60
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
    # A read error is a failed result too: run_sweep never raises.
    missing = run_sweep(
        _request([str(tmp_path / "missing")], [baseline], baseline),
        evaluator=_fake_evaluator,
    )
    assert missing.status == "failed"
    assert "FileNotFoundError" in missing.error


# ---------------------------------------------------------------------------
# Predictor pre-sweep
# ---------------------------------------------------------------------------


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


# Importing the forecasting libraries and the Prophet fits dominate;
# contended CI runners (arm64) fit Prophet slower.
@pytest.mark.timeout(120)
# Prophet/pandas emit this on newer NumPy; CI turns warnings into errors.
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
)
def test_presweep_ranking_matches_evaluate_preset(monkeypatch):
    predictors = _predictors_module()
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

    # Long history: a forecast does not change the state constant and Kalman
    # predictors carry into the next bucket, so warming the history without
    # forecasting it changes nothing. (ARIMA instead fits once on the whole
    # history.)
    constant = _spy_on_forecasts(monkeypatch, predictors.ConstantPredictor)
    kalman = _spy_on_forecasts(monkeypatch, predictors.KalmanPredictor)
    names = ["constant_last", "kalman_default_raw", "kalman_reactive_log1p"]
    history, window = 300, 20
    long = _wavy(history + window)
    result = predictor_presweep(long, names, 180, eval_steps=window)

    assert (result["warmup_steps"], result["eval_steps"]) == (history, window)
    # Three series x window buckets forecasts per preset, none for history:
    # the first forecast already sees every history bucket.
    assert (len(constant), len(kalman)) == (3 * window, 2 * 3 * window)
    assert constant[:3] == kalman[:3] == [history] * 3
    assert constant[-3:] == kalman[-3:] == [history + window - 1] * 3
    expected = evaluate_preset_losses(long, names, window)
    assert result["losses"] == pytest.approx(expected, rel=1e-12)
    assert ranking(result["losses"]) == ranking(expected)
    assert result["winner"] == ranking(expected)[0]

    # A window shorter than one predictor interval has nothing to score.
    nothing = predictor_presweep(long, ["constant_last"], 180, eval_steps=0)
    assert (nothing["winner"], nothing["fields"]) == (None, {})
    assert nothing["losses"] == {"constant_last": None}

    # Prophet refits on every forecast, so it gets the 5 history points it
    # needs to fit and two window buckets: 3 x 2 fits. (log1p fits on this
    # few points take seconds each.)
    names = ["constant_last", "prophet_w20_raw"]
    short = _wavy(5 + 2)
    result = predictor_presweep(short, names, 180, eval_steps=2)
    expected = evaluate_preset_losses(short, names, 2)
    assert result["losses"] == pytest.approx(expected, rel=1e-9)
    assert ranking(result["losses"]) == ranking(expected)


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
