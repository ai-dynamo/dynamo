# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the sweep module imports it.

"""Unit tests for the retune sweep: scoring, the worker pool and the pre-sweep.

Sweeps that spawn real worker processes live in
``integration/test_retune_sweep_workers.py``.
"""

import dataclasses
import math
import random
import threading
import time
import types

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.core.types import RetuneParams, TrafficObservation
from dynamo.planner.retune import sweep as sweep_module
from dynamo.planner.retune.sweep import (
    GOODPUT_KEY,
    predictor_presweep,
    run_sweep,
    score_summary,
    with_predictor_fields,
)
from dynamo.planner.tests.retune_sweep_helpers import (
    ORIGIN_MS,
    fake_evaluator,
    retune_params,
    sweep_request,
    trace_rows,
    write_trace,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


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
    a, b = sorted([retune_params(0.0), retune_params(1.0)], key=RetuneParams.key)
    valid = sweep_request(["t"], [a, b], a)
    with pytest.raises(ValueError, match="sorted"):
        dataclasses.replace(valid, candidates=(b, a))
    with pytest.raises(ValueError, match="baseline"):
        sweep_request(["t"], [a], b)


# ---------------------------------------------------------------------------
# Worker pool and empty windows (no worker is spawned)
# ---------------------------------------------------------------------------


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
    trace = write_trace(tmp_path / "trace.jsonl", trace_rows([ORIGIN_MS - 5_000]))
    baseline = retune_params(1.0)

    result = run_sweep(
        sweep_request([trace], [baseline], baseline), evaluator=fake_evaluator
    )

    assert result.status == "empty"
    assert result.num_requests == 0
    assert result.row_counts["outside_window"] == 1
    assert result.scores == {}
    # A read error is a failed result too: run_sweep never raises.
    missing = run_sweep(
        sweep_request([str(tmp_path / "missing")], [baseline], baseline),
        evaluator=fake_evaluator,
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
