# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the sweep module imports it.

"""Unit tests for the retune sweep: scoring, the read task, the worker pool
and the pre-sweep.

Sweeps that spawn real worker processes live in
``integration/test_retune_sweep_workers.py``.
"""

import dataclasses
import math
import multiprocessing
import os
import random
import threading
import time
import types
from multiprocessing.reduction import ForkingPickler

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
    score_summary,
    with_predictor_fields,
)
from dynamo.planner.retune.window import (
    ObservationBuckets,
    read_trace_history,
    read_trace_window,
)
from dynamo.planner.tests.retune_sweep_helpers import (
    ORIGIN_MS,
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
# Read task, run in-process here (a sweep runs it in a worker)
# ---------------------------------------------------------------------------


def test_read_task_matches_the_trace_readers(tmp_path):
    # Two warmup intervals of history, then the window; one row per second.
    window_start = ORIGIN_MS
    arrivals = [window_start - 20_000 + 1_000 * i for i in range(30)]
    trace = write_trace(
        tmp_path / "trace.jsonl", trace_rows(arrivals + [window_start + 50_000])
    )
    request = sweep_request(
        [trace],
        [retune_params(1.0)],
        retune_params(1.0),
        history_start_ms=window_start - 20_000,
        base_planner={"enable_throughput_scaling": True},
    )

    def buckets():
        return ObservationBuckets(request.window_end_ms, 10_000.0, 10.0)

    out = tmp_path / "out"
    out.mkdir()
    spec = sweep_module._ReadSpec.from_request(
        request, buckets(), write_observations=True
    )
    inputs = sweep_module._read_sweep_inputs(spec, str(out))

    # The same calls run_sweep made on its own thread before the read moved
    # into a worker task.
    expected_buckets = buckets()
    window = read_trace_window(
        [trace],
        request.window_start_ms,
        request.window_end_ms,
        request.as_of_ms,
        tmp_path / "window.jsonl",
        prune_by_mtime=False,
        buckets=expected_buckets,
    )
    history = read_trace_history(
        [trace],
        request.history_start_ms,
        request.window_start_ms,
        request.as_of_ms,
        tmp_path / "history.jsonl",
        prune_by_mtime=False,
        buckets=expected_buckets,
    )
    expected_counts = {
        **window.row_counts,
        **{f"history_{key}": value for key, value in history.row_counts.items()},
        "history_requests": history.num_requests,
    }
    assert (inputs.num_requests, history.num_requests) == (10, 20)
    assert inputs.row_counts == expected_counts
    assert inputs.observations == tuple(expected_buckets.observations())
    assert len(inputs.observations) == 3
    for name in ("window", "history"):
        produced = getattr(inputs, f"{name}_path")
        assert produced == str(out / f"{name}.jsonl")
        assert (out / f"{name}.jsonl").read_text() == (
            tmp_path / f"{name}.jsonl"
        ).read_text()
    # The pre-sweep workers load the observations from this file.
    assert sweep_module._read_observations(inputs.observations_path) == list(
        inputs.observations
    )

    # An empty window stops before the history; a read error propagates (the
    # worker reports it as the read task's failure).
    empty = sweep_module._read_sweep_inputs(
        dataclasses.replace(
            spec, window_start_ms=ORIGIN_MS + 60_000, window_end_ms=ORIGIN_MS + 70_000
        ),
        str(tmp_path / "empty"),
    )
    assert (empty.num_requests, empty.history_path, empty.observations) == (
        0,
        None,
        None,
    )
    assert empty.row_counts["outside_window"] == 31
    assert "history_requests" not in empty.row_counts
    missing = dataclasses.replace(spec, trace_paths=(str(tmp_path / "missing"),))
    with pytest.raises(FileNotFoundError):
        sweep_module._read_sweep_inputs(missing, str(tmp_path / "missing-out"))


# ---------------------------------------------------------------------------
# Worker pool (no worker is spawned)
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Stands in for a worker process; ``wait`` never sees its sentinel."""

    def __init__(self, name, *, alive=True, dies_on_kill=True):
        self.name, self.pid, self.exitcode = name, None, None
        self.alive, self.dies_on_kill = alive, dies_on_kill
        self.kills, self.joins = 0, []
        self._never_ready, self._writer = os.pipe()
        self.sentinel = self._never_ready

    def is_alive(self):
        return self.alive

    def kill(self):
        self.kills += 1
        self.alive = not self.dies_on_kill

    def join(self, timeout=None):
        self.joins.append(timeout)

    def release(self):
        os.close(self._never_ready)
        os.close(self._writer)


def test_a_task_whose_workers_die_before_accepting_it_fails():
    pool = sweep_module._WorkerPool(1, time.monotonic() + 60, threading.Event())
    spawned = []

    def dead_worker():
        worker = sweep_module._Worker(
            process=_FakeProcess("dead", alive=False),
            conn=types.SimpleNamespace(close=lambda: None),
        )
        pool._workers.append(worker)
        spawned.append(worker)
        return worker

    pool._spawn = dead_worker

    results = pool.run([("task", print, ())])

    ok, error = results["task"]
    assert not ok and "before accepting the task 3 times" in error
    assert len(spawned) == 3 and pool._workers == []
    for worker in spawned:
        worker.process.release()


def test_close_waits_only_for_workers_that_answered(caplog):
    pool = sweep_module._WorkerPool(3, time.monotonic() + 60, threading.Event())
    sent = []
    workers = []
    for name, answered, dies_on_kill in [
        ("answered", True, True),
        ("importing", False, True),
        ("stuck-in-io", False, False),
    ]:
        conn = types.SimpleNamespace(send=sent.append, close=lambda: None)
        process = _FakeProcess(name, dies_on_kill=dies_on_kill)
        workers.append(
            sweep_module._Worker(process=process, conn=conn, answered=answered)
        )
    pool._workers = list(workers)

    pool.close(kill=False)

    answered, importing, stuck = (worker.process for worker in workers)
    # Only the worker that answered a task is asked to exit and waited for.
    assert sent == [None]
    assert answered.joins[0] == sweep_module._JOIN_TIMEOUT_S
    assert (importing.kills, stuck.kills) == (1, 1)
    # Every wait is bounded: a worker that survives SIGKILL is left behind.
    for process in (answered, importing, stuck):
        assert None not in process.joins
        process.release()
    assert "stuck-in-io" in caplog.text and "leaving it behind" in caplog.text
    assert pool._workers == []


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
def test_presweep_ranking_matches_evaluate_preset(monkeypatch, tmp_path):
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
    # A sweep worker reads the same observations from the sweep's file.
    path = sweep_module._write_observations(tmp_path / "observations.json", long)
    via_file = sweep_module._presweep_task(path, names, 180, window)
    assert via_file["losses"] == result["losses"]

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


@pytest.mark.timeout(60)
def test_presweep_dispatch_does_not_wait_for_workers_that_are_not_reading(tmp_path):
    # Four days of 60-s buckets pickle to more than a pipe's socket buffer:
    # sent inline, they would block the sweep thread until a worker still
    # importing reads them, past the deadline and cancel.
    observations = [
        TrafficObservation(60.0, 50.0 + i % 7, 1000.0, 200.0) for i in range(6_000)
    ]
    assert len(ForkingPickler.dumps(tuple(observations))) > 256 * 1024
    path = sweep_module._write_observations(
        tmp_path / "observations.json", observations
    )
    inputs = sweep_module._SweepInputs(
        str(tmp_path / "window.jsonl"),
        1,
        {},
        observations=tuple(observations),
        observations_path=path,
    )
    presets = ("constant_last", "kalman_default_raw")
    request = sweep_request(
        ["t"],
        [retune_params(1.0)],
        retune_params(1.0),
        window_end_ms=ORIGIN_MS + 600_000,
        predictor_presets=presets,
    )
    cancel = threading.Event()
    pool = sweep_module._WorkerPool(2, time.monotonic() + 60, cancel)
    # The workers' pipe ends, never read: workers that are still importing.
    unread = []
    processes = []

    def importing_worker():
        parent, child = multiprocessing.Pipe(duplex=True)
        unread.append(child)
        processes.append(_FakeProcess("importing"))
        worker = sweep_module._Worker(process=processes[-1], conn=parent)
        pool._workers.append(worker)
        return worker

    pool._spawn = importing_worker
    timer = threading.Timer(0.5, cancel.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(sweep_module._SweepInterrupted, match="cancelled"):
            sweep_module._run_presweep(request, pool, inputs, 60_000.0, 60)
    finally:
        timer.cancel()
        pool.close(kill=True)

    assert time.monotonic() - started < 10
    # Every preset was dispatched, as a message that carries only the path.
    messages = [conn.recv() for conn in unread]
    assert sorted(message[0] for message in messages) == [
        f"predictor-presweep:{name}" for name in presets
    ]
    for task_id, function, args in messages:
        assert function is sweep_module._presweep_task
        assert args[0] == path
        assert len(ForkingPickler.dumps((task_id, function, args))) < 1024
    assert sweep_module._read_observations(path) == observations
    for process, conn in zip(processes, unread):
        process.release()
        conn.close()
