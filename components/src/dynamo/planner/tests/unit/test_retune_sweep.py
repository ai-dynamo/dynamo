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
from collections import deque
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
    score_summary,
    with_predictor_fields,
)
from dynamo.planner.retune.window import ObservationBuckets
from dynamo.planner.tests.retune_sweep_helpers import (
    ORIGIN_MS,
    fake_evaluator,
    predictor_candidates,
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


def test_read_task_writes_the_window_and_buckets_the_history(tmp_path):
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

    def read(out, buckets_interval_ms=10_000.0, **changes):
        buckets = ObservationBuckets(
            request.window_end_ms, buckets_interval_ms, buckets_interval_ms / 1000
        )
        spec = sweep_module._ReadSpec.from_request(
            request, buckets, write_observations=True
        )
        return sweep_module._read_sweep_inputs(
            dataclasses.replace(spec, **changes), str(tmp_path / out)
        )

    def warmup(inputs):
        return [
            obs.num_req for obs in sweep_module._read_observations(inputs.warmup_path)
        ]

    inputs = read("out")

    assert inputs.num_requests == 10
    assert inputs.window_path == str(tmp_path / "out" / "window.jsonl")
    assert inputs.row_counts["history_requests"] == 20
    # History and window share the pre-sweep buckets, which end at the
    # window end; the pre-sweep workers load them from a file.
    assert [obs.num_req for obs in inputs.observations] == [10.0, 10.0, 10.0]
    assert sweep_module._read_observations(inputs.observations_path) == list(
        inputs.observations
    )
    # The inner planner's warmup is the history alone, in buckets that end at
    # the window start: off the grid of the window end, a pre-sweep bucket
    # straddles the window start, but no warmup bucket does.
    assert warmup(inputs) == [10.0, 10.0]
    assert warmup(read("off-grid", 7_000.0)) == [6.0, 7.0, 7.0]

    # The history is read only for a planner that scales on throughput (the
    # only one with a load predictor), and not for an empty window.
    no_scaling = read("no-scaling", buckets=None)
    empty = read(
        "empty", window_start_ms=ORIGIN_MS + 60_000, window_end_ms=ORIGIN_MS + 70_000
    )
    for unread, num_requests in [(no_scaling, 10), (empty, 0)]:
        assert unread.num_requests == num_requests
        assert (unread.observations, unread.warmup_path) == (None, None)
        assert "history_requests" not in unread.row_counts
    # A read error propagates: the worker reports it as the task's failure.
    with pytest.raises(ValueError, match="No warmup trace files matched"):
        read("missing", trace_paths=(str(tmp_path / "missing"),))


# ---------------------------------------------------------------------------
# Worker pool (no worker is spawned)
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Stands in for a worker process; ``wait`` never sees its sentinel.

    ``events``, when given, records every kill and join in order.
    """

    def __init__(self, name, *, alive=True, dies_on_kill=True, join_s=0.0, events=None):
        self.name, self.pid, self.exitcode = name, None, None
        self.alive, self.dies_on_kill, self.join_s = alive, dies_on_kill, join_s
        self.joins = []
        self.events = [] if events is None else events
        self._never_ready, self._writer = os.pipe()
        self.sentinel = self._never_ready

    def is_alive(self):
        return self.alive

    def kill(self):
        self.events.append(("kill", self.name))
        self.alive = not self.dies_on_kill

    def join(self, timeout=None):
        self.joins.append(timeout)
        self.events.append(("join", self.name))
        time.sleep(self.join_s)

    def release(self):
        os.close(self._never_ready)
        os.close(self._writer)


def _fake_worker(name, sent, *, alive=True, send=None):
    conn = types.SimpleNamespace(
        send=send or (lambda task: sent.append((name, task[0]))),
        close=lambda: None,
    )
    return sweep_module._Worker(process=_FakeProcess(name, alive=alive), conn=conn)


def test_the_pool_never_runs_more_than_max_parallel_workers(tmp_path):
    pool = sweep_module._WorkerPool(
        2, time.monotonic() + 60, threading.Event(), str(tmp_path)
    )
    sent = []

    def spawn():
        pool._workers.append(_fake_worker(f"w{len(pool._workers)}", sent))
        return pool._workers[-1]

    pool._spawn = spawn

    pool.start(5)
    assert len(pool._workers) == 2
    pending = deque((f"t{i}", print, ()) for i in range(5))
    pool._dispatch(pending)

    assert len(pool._workers) == 2 and len(sent) == 2 and len(pending) == 3
    for worker in pool._workers:
        worker.process.release()


def test_workers_that_died_idle_are_replaced_without_failing_a_task(tmp_path):
    pool = sweep_module._WorkerPool(
        3, time.monotonic() + 60, threading.Event(), str(tmp_path)
    )
    sent: list[tuple[str, str]] = []

    def broken_pipe(task):
        raise BrokenPipeError("worker exited")

    # Two idle workers were killed while another task ran; a third dies
    # between the liveness check and the send.
    workers = [
        _fake_worker("dead-1", sent, alive=False),
        _fake_worker("dead-2", sent, alive=False),
        _fake_worker("dying", sent, send=broken_pipe),
    ]
    pool._workers = list(workers)

    def spawn():
        workers.append(_fake_worker(f"new-{len(workers) - 3}", sent))
        pool._workers.append(workers[-1])
        return workers[-1]

    pool._spawn = spawn
    pending = deque([("a", print, ()), ("b", print, ())])

    pool._dispatch(pending)
    pool._dispatch(pending)

    # No task failed: the one the dying worker did not take went to a new one.
    assert sent == [("new-0", "a"), ("new-1", "b")] and not pending
    assert [worker.process.name for worker in pool._workers] == ["new-0", "new-1"]
    for worker in workers:
        worker.process.release()


def test_close_kills_every_worker_within_one_bounded_wait(tmp_path, caplog):
    pool = sweep_module._WorkerPool(
        3, time.monotonic() + 60, threading.Event(), str(tmp_path)
    )
    events: list[tuple[str, str]] = []
    processes = [
        _FakeProcess("stuck-in-io", dies_on_kill=False, join_s=0.2, events=events),
        _FakeProcess("busy", events=events),
        _FakeProcess("idle", events=events),
    ]
    # The pipes cannot send: an idle worker is not asked to exit.
    pool._workers = [
        sweep_module._Worker(
            process=process, conn=types.SimpleNamespace(close=lambda: None)
        )
        for process in processes
    ]
    pool._workers[1].task_id = "task"

    pool.close()

    stuck, busy, idle = processes
    # Every worker is killed before the first wait, so one that survives
    # SIGKILL does not delay the others' kill.
    assert events[:3] == [("kill", process.name) for process in processes]
    # One wait shared by every worker, so a worker that survives SIGKILL
    # holds the sweep thread for it only once.
    assert stuck.joins[0] <= sweep_module._JOIN_TIMEOUT_S
    assert busy.joins[0] <= sweep_module._JOIN_TIMEOUT_S - 0.2
    assert idle.joins[0] <= busy.joins[0]
    for process in processes:
        process.release()
    assert "stuck-in-io" in caplog.text and "leaving it behind" in caplog.text
    assert pool._workers == []


class _InlinePool:
    """Runs each task in this process, reporting errors as a worker does."""

    def __init__(self, *args):
        pass

    def start(self, count):
        pass

    def close(self):
        pass

    def run(self, tasks):
        results = {}
        for task_id, function, args in tasks:
            try:
                results[task_id] = (True, function(*args))
            except Exception as error:
                results[task_id] = (False, f"{type(error).__name__}: {error}")
        return results


def test_equivalent_candidates_replay_once_and_share_the_score(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_WorkerPool", _InlinePool)
    evaluated = []

    def counting_evaluator(*args):
        evaluated.append(args[3])
        return fake_evaluator(*args)

    trace = write_trace(
        tmp_path / "trace.jsonl", trace_rows([ORIGIN_MS + 100 * i for i in range(5)])
    )
    # Without a pre-sweep winner, ``same`` evaluates as the baseline does.
    baseline = retune_params(
        1.0, load_scaling_down_sensitivity=80, load_predictor="arima"
    )
    same = retune_params(1.0, load_scaling_down_sensitivity=80)
    other = retune_params(1.0, load_scaling_down_sensitivity=70)

    result = sweep_module.run_sweep(
        sweep_request([trace], [baseline, same, other], baseline),
        evaluator=counting_evaluator,
    )

    assert result.status == "ok", result.error
    assert len(evaluated) == 2
    assert result.scores[same.key()] == result.scores[baseline.key()] is not None


def test_a_predictor_winner_is_prepared_in_the_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "_WorkerPool", _InlinePool)
    prepared = []
    tasks: list[str] = []

    def prepare_predictor(config, capabilities, observations):
        tasks.append("prepare")
        prepared.append((config.load_predictor, config.retune, list(observations)))
        if config.kalman_min_points > len(observations):
            raise ValueError(f"needs {config.kalman_min_points} warmup points")

    def evaluator(*args):
        tasks.append("replay")
        return fake_evaluator(*args)

    monkeypatch.setattr(sweep_module, "prepare_predictor", prepare_predictor)
    trace = write_trace(
        tmp_path / "trace.jsonl",
        trace_rows([ORIGIN_MS + 1_000 * i for i in range(-20, 10)]),
    )
    baseline = retune_params(1.0, load_predictor="arima", load_predictor_log1p=False)
    request = sweep_request(
        [trace],
        [baseline, retune_params(0.0)],
        baseline,
        history_start_ms=ORIGIN_MS - 20_000,
        base_planner={
            "mode": "agg",
            "enable_throughput_scaling": True,
            "throughput_adjustment_interval_seconds": 5,
            "retune": {"interval_seconds": 10},
        },
        predictor_candidates=predictor_candidates("constant_last"),
    )

    def run_with_winner(fields):
        presweep = {"winner": "w", "fields": fields, "losses": {"w": 0.1}}
        monkeypatch.setattr(sweep_module, "_run_presweep", lambda *_: presweep)
        tasks.clear()
        return sweep_module.run_sweep(request, evaluator=evaluator)

    # A winner that keeps the current predictor needs no preparation.
    kept = run_with_winner({"load_predictor": "arima", "load_predictor_log1p": False})
    assert kept.status == "ok", kept.error
    assert (kept.predictor_warmup_error, prepared) == (None, [])

    # A new one is prepared once, on the swap's warmup, and its error kept.
    kalman = {"load_predictor": "kalman", "kalman_min_points": 60}
    swapped = run_with_winner(kalman)
    assert swapped.status == "ok", swapped.error
    assert prepared == [("kalman", None, list(swapped.warmup_observations))]
    assert swapped.predictor_warmup_error == "ValueError: needs 60 warmup points"
    # Queued ahead of the replays, so a long fit overlaps them.
    assert tasks == ["prepare", "replay", "replay"]


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


def test_presweep_task_forecasts_only_the_window(monkeypatch, tmp_path):
    # The predictor module imports all three forecasting libraries.
    for name in ("filterpy", "pmdarima", "prophet"):
        pytest.importorskip(name)
    from dynamo.planner.core.load.predictors import ConstantPredictor
    from dynamo.planner.simulation.load_predictor import complete_predictor_preset

    seen: list[int] = []
    predict_next = ConstantPredictor.predict_next

    def spy(self):
        seen.append(len(self.data_buffer))
        return predict_next(self)

    monkeypatch.setattr(ConstantPredictor, "predict_next", spy)
    history, window = 300, 20
    path = sweep_module._write_observations(
        tmp_path / "observations.json", _wavy(history + window)
    )

    mapping = complete_predictor_preset("constant_last")
    loss = sweep_module._presweep_task(path, mapping, 180, window)

    # Three series x window buckets forecasts, none for history: the first
    # forecast already sees every history bucket.
    assert loss is not None
    assert len(seen) == 3 * window and seen[:3] == [history] * 3
    # A window shorter than one predictor interval has nothing to score.
    assert sweep_module._presweep_task(path, mapping, 180, 0) is None


def test_presweep_summary_picks_the_earliest_lowest_loss():
    from dynamo.planner.simulation.load_predictor import (
        complete_predictor_preset,
        predictor_fields,
    )

    # The winner's fields come from its own mapping; ties go to the earliest.
    same = complete_predictor_preset("kalman_default_raw")
    tuned = {**same, "kalman_q_level": 5.0, "kalman_min_points": 3}
    summary = sweep_module._presweep_summary(
        {"constant_last": 0.5, "custom_1": 0.25, "custom_2": None, "custom_3": 0.25},
        {
            "constant_last": complete_predictor_preset("constant_last"),
            "custom_1": tuned,
            "custom_2": same,
            "custom_3": same,
        },
        10,
        30,
        0.0,
    )
    assert summary["winner"] == "custom_1"
    assert summary["fields"] == predictor_fields(tuned)
    assert summary["fields"]["kalman_q_level"] == 5.0

    # Labels key the pre-sweep tasks and records.
    with pytest.raises(ValueError, match="labels must be unique"):
        sweep_request(
            ["t"],
            [retune_params(1.0)],
            retune_params(1.0),
            predictor_candidates=(("a", tuned), ("a", same)),
        )


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
        predictor_candidates=predictor_candidates(*presets),
    )
    cancel = threading.Event()
    pool = sweep_module._WorkerPool(2, time.monotonic() + 60, cancel, str(tmp_path))
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
        pool.close()

    assert time.monotonic() - started < 10
    # Every preset was dispatched, as a message that carries only the path.
    messages = [conn.recv() for conn in unread]
    assert sorted(message[0] for message in messages) == [
        f"predictor-presweep:{name}" for name in presets
    ]
    for task_id, function, args in messages:
        assert function is sweep_module._presweep_task
        # One complete-mapping candidate per task; the 10 window buckets
        # are scored.
        observations_path, entry, interval_s, eval_steps = args
        assert (observations_path, interval_s, eval_steps) == (path, 60, 10)
        assert len(entry) == 7
        assert len(ForkingPickler.dumps((task_id, function, args))) < 1024
    assert sweep_module._read_observations(path) == observations
    for process, conn in zip(processes, unread):
        process.release()
        conn.close()
