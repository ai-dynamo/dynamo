# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the sweep module imports it.

"""Retune sweeps on real spawned workers: isolation, deadline, cancellation
and the pre-sweep.

Each test runs ``run_sweep`` end to end, so it spawns worker processes that
import the evaluators from ``dynamo.planner.tests.retune_sweep_helpers``.
"""

import multiprocessing
import threading
import time

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.sweep import run_sweep
from dynamo.planner.tests.retune_sweep_helpers import (
    ABORT,
    HANG,
    ORIGIN_MS,
    RAISE,
    abort_raise_or_evaluate,
    disagg_engine,
    fake_evaluator,
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
        predictor_presets=("constant_last", "kalman_default_raw"),
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
