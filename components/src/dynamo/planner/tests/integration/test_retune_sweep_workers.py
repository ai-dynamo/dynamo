# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the sweep module imports it.

"""Retune sweeps on real spawned workers: isolation, deadline, cancellation,
the trace read and the pre-sweep.

Each test runs ``run_sweep`` end to end, so it spawns worker processes that
import the evaluators from ``dynamo.planner.tests.retune_sweep_helpers``.
"""

import errno
import multiprocessing
import os
import signal
import threading
import time

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune import sweep as sweep_module
from dynamo.planner.retune.sweep import run_sweep
from dynamo.planner.tests.retune_sweep_helpers import (
    ABORT,
    HANG,
    ORIGIN_MS,
    RAISE,
    abort_raise_or_evaluate,
    disagg_engine,
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
    pytest.mark.integration,
    pytest.mark.planner,
]


@pytest.mark.timeout(240)
def test_a_dead_worker_fails_only_its_candidate(tmp_path):
    trace = write_trace(
        tmp_path / "trace.jsonl",
        trace_rows([ORIGIN_MS + 250 * i for i in range(30)] + [ORIGIN_MS + 50_000]),
    )
    ok_a, ok_b = retune_params(1.0), retune_params(0.0)
    aborts, raises = retune_params(1.0, ABORT), retune_params(1.0, RAISE)
    request = sweep_request(
        [trace],
        [ok_a, ok_b, aborts, raises],
        ok_a,
        engine=disagg_engine(),
        replicas={"prefill": 1, "decode": 1},
        max_parallel=2,
    )

    result = run_sweep(request, evaluator=abort_raise_or_evaluate)

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
    trace = write_trace(
        tmp_path / "trace.jsonl", trace_rows([ORIGIN_MS + 1_000 * i for i in range(5)])
    )
    hangs = retune_params(1.0, HANG)
    # The deadline or cancel comes long before the worker has imported;
    # nothing waits for the import.
    request = sweep_request([trace], [hangs], hangs, timeout_s=timeout_s)
    cancel = threading.Event()
    timer = threading.Timer(cancel_after_s, cancel.set)

    started = time.monotonic()
    timer.start()
    try:
        result = run_sweep(request, cancel=cancel, evaluator=fake_evaluator)
    finally:
        timer.cancel()

    assert (result.status, result.error) == (status, error)
    assert result.scores == {}
    assert time.monotonic() - started < 60
    assert multiprocessing.active_children() == []


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
    warmup_trace = write_trace(
        tmp_path / "warmup.jsonl",
        # The stray in-window row shows windows never read history_paths.
        trace_rows([a for a in arrivals if a < split] + [ORIGIN_MS + 5], prefix="h"),
    )
    trace = write_trace(
        tmp_path / "trace.jsonl", trace_rows([a for a in arrivals if a >= split])
    )
    window_start = ORIGIN_MS
    baseline = retune_params(1.0, load_predictor="arima", load_predictor_log1p=False)
    candidate = retune_params(0.0)
    base_planner = {
        "mode": "agg",
        "enable_throughput_scaling": True,
        "throughput_adjustment_interval_seconds": interval_s,
        "load_predictor_warmup_trace": "/outer/warmup.jsonl",
        "retune": {"interval_seconds": 100},
    }
    request = sweep_request(
        [trace],
        [baseline, candidate],
        baseline,
        window_start_ms=window_start,
        window_end_ms=window_start + 100_000,
        history_start_ms=window_start - 200_000,
        as_of_ms=window_start + 200_000,
        base_planner=base_planner,
        predictor_candidates=predictor_candidates(
            "constant_last", "kalman_default_raw"
        ),
        max_parallel=1,
        history_paths=(warmup_trace,),
    )

    result = run_sweep(request, evaluator=fake_evaluator)

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


@pytest.mark.timeout(240)
def test_empty_window_and_read_errors_end_the_sweep(tmp_path):
    trace = write_trace(tmp_path / "trace.jsonl", trace_rows([ORIGIN_MS - 5_000]))
    baseline = retune_params(1.0)

    result = run_sweep(
        sweep_request([trace], [baseline], baseline), evaluator=fake_evaluator
    )

    assert result.status == "empty", result.error
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
    assert multiprocessing.active_children() == []


def _hold_fifo_open(fifo, reading, stop, fds):
    """Open ``fifo`` for writing once a reader has it open; never write.

    The reader's ``open`` returns and its first ``read`` blocks: a hung read.
    """

    while not stop.is_set():
        try:
            fds.append(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError as error:
            if error.errno != errno.ENXIO:  # ENXIO: no reader yet
                raise
            time.sleep(0.05)
            continue
        reading.set()
        return


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "status, error",
    [("timeout", "sweep exceeded its deadline"), ("failed", "sweep cancelled")],
    ids=["deadline", "cancel"],
)
def test_a_hung_trace_read_ends_at_the_deadline_or_on_cancel(
    tmp_path, monkeypatch, status, error
):
    # The trace is a FIFO no one writes to: the read worker blocks in it,
    # as on a stuck network filesystem.
    fifo = tmp_path / "trace.jsonl"
    os.mkfifo(fifo)
    baseline = retune_params(1.0)
    # The read worker's import fits in the deadline; the test then moves the
    # deadline (or cancels) once the read is blocked.
    request = sweep_request([str(fifo)], [baseline], baseline, timeout_s=200.0)
    pools = []

    class RecordingPool(sweep_module._WorkerPool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

    monkeypatch.setattr(sweep_module, "_WorkerPool", RecordingPool)
    cancel = threading.Event()
    reading, stop, fds = threading.Event(), threading.Event(), []
    blocked_at = []

    def interrupt_once_reading():
        while not reading.wait(timeout=0.1):
            if stop.is_set():
                return
        blocked_at.append(time.monotonic())
        if status == "timeout":
            pools[0]._deadline = time.monotonic() + 1.0
        else:
            cancel.set()

    threads = [
        threading.Thread(
            target=_hold_fifo_open, args=(fifo, reading, stop, fds), daemon=True
        ),
        threading.Thread(target=interrupt_once_reading, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        result = run_sweep(request, cancel=cancel, evaluator=fake_evaluator)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        for fd in fds:
            os.close(fd)

    assert blocked_at, "the read worker never opened the trace"
    assert (result.status, result.error) == (status, error)
    # Stopped while reading: nothing was counted.
    assert (result.num_requests, result.row_counts, result.scores) == (0, {}, {})
    assert time.monotonic() - blocked_at[0] < 30
    assert multiprocessing.active_children() == []


@pytest.mark.timeout(240)
def test_presweep_dispatch_to_stalled_workers_honours_cancel(tmp_path, monkeypatch):
    # 6,000 one-second buckets of history before a 10-s window: inline, the
    # pre-sweep observations would be ~350 KB, above a pipe's socket buffer.
    window_start = ORIGIN_MS
    history_s = 6_000
    trace = write_trace(
        tmp_path / "trace.jsonl",
        trace_rows(
            [window_start - 1_000 * (history_s - i) for i in range(history_s)]
            + [window_start + 1_000 * i for i in range(10)]
        ),
    )
    baseline = retune_params(1.0)
    request = sweep_request(
        [trace],
        [baseline],
        baseline,
        history_start_ms=window_start - 1_000 * history_s,
        base_planner={
            "mode": "agg",
            "enable_throughput_scaling": True,
            "throughput_adjustment_interval_seconds": 1,
        },
        predictor_candidates=predictor_candidates(
            "constant_last", "kalman_default_raw"
        ),
        max_parallel=2,
        timeout_s=200.0,
    )
    cancel = threading.Event()
    seen = {}
    run_presweep = sweep_module._run_presweep

    def stalled_presweep(request, pool, inputs, *args):
        # Freeze every worker, so none reads its pipe: as if each were still
        # importing. The dispatch must not wait for them.
        for worker in pool._workers:
            os.kill(worker.process.pid, signal.SIGSTOP)
        seen["observations"] = len(inputs.observations)
        seen["stalled_at"] = time.monotonic()
        threading.Timer(1.0, cancel.set).start()
        return run_presweep(request, pool, inputs, *args)

    monkeypatch.setattr(sweep_module, "_run_presweep", stalled_presweep)

    result = run_sweep(request, cancel=cancel, evaluator=fake_evaluator)

    assert (result.status, result.error) == ("failed", "sweep cancelled")
    assert seen["observations"] == history_s + 10
    assert time.monotonic() - seen["stalled_at"] < 15
    assert result.num_requests == 10
    assert result.row_counts["history_requests"] == history_s
    assert multiprocessing.active_children() == []
