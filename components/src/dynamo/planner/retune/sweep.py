# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retune sweep: replay one trace window once per candidate and score it.

``run_sweep`` runs on the retune controller's single sweep thread and does
no trace I/O itself. Every phase is a task in one of ``max_parallel`` spawned
worker processes: reading the window into the sweep directory (and, for an
inner planner that scales on throughput, bucketing the history before it),
the optional load-predictor pre-sweep, and one
``dynamo.replay.run_trace_replay`` per candidate. Each worker gets one task
at a time over its own pipe, so a worker that dies (Rust abort, OOM kill)
fails only the task it held and is replaced. At the deadline, or when the
controller closes, every worker is killed, so a hung trace read (a stuck
network filesystem) or a worker stuck in its imports ends the sweep at the
deadline instead of holding the sweep thread. ``multiprocessing.Pool`` is
not used: a Pool worker that dies hard never completes its result and would
hang the sweep until the deadline.

Task messages stay small: bulk data (the window and the observation buckets)
is written to the sweep directory and passed by path. A message larger than
the pipe's socket buffer blocks ``send`` until the worker reads it, which a
worker still importing does not do. Workers also keep their temporary files
in the sweep directory: a killed worker runs no ``atexit`` cleanup, such as
cmdstanpy's for the model files of every Prophet fit, and the directory is
removed with the sweep however its workers ended.

Everything here is module level and picklable: workers are started with the
``spawn`` method and receive their task functions by reference.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import multiprocessing
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal, NamedTuple

from aisimulate.sweeper import OptimizationTarget, objective_value

from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    RetuneParams,
    TrafficObservation,
    WorkerCapabilities,
)
from dynamo.planner.reconfiguration import prepare_predictor
from dynamo.planner.retune.domains import PredictorCandidate
from dynamo.planner.retune.window import (
    ObservationBuckets,
    read_trace_history,
    read_trace_window,
)
from dynamo.planner.simulation.load_predictor import (
    Window,
    evaluate_preset,
    internal_preset,
    predictor_fields,
)
from dynamo.replay import run_trace_replay

logger = logging.getLogger(__name__)

SweepStatus = Literal["ok", "empty", "timeout", "failed"]
Evaluator = Callable[..., tuple[float, dict[str, Any]]]

GOODPUT_KEY = "goodput_output_throughput_tok_s"
WORKER_NICENESS = 10
_POLL_INTERVAL_S = 0.25
_JOIN_TIMEOUT_S = 5.0
_READ_TASK = "read-trace"
_PRESWEEP_TASK = "predictor-presweep"
_PREDICTOR_WARMUP_TASK = "predictor-warmup"
_WINDOW_FILE = "window.jsonl"
_OBSERVATIONS_FILE = "observations.json"
_WARMUP_FILE = "warmup.json"
_WORKER_TEMP_DIR = "tmp"


@dataclass(frozen=True)
class SweepRequest:
    """Everything one sweep needs; picklable and independent of the planner.

    Times are trace milliseconds. ``candidates`` are sorted by
    ``RetuneParams.key`` and contain ``baseline`` (the point in effect at
    launch). ``trace_paths`` hold the windows; ``history_paths`` hold extra
    history that ends before the first window (the replay warmup trace) and
    are read only for the history before a window.
    ``replicas`` maps worker type (``aggregated`` or ``prefill``/``decode``)
    to the ready count at window start. ``base_planner`` is the current
    planner config dump for an inner planner (``None``: fixed replicas, no
    planner). ``predictor_candidates`` are the ``(label, mapping)`` load
    predictors the forecast-loss pre-sweep chooses from, as
    ``retune.domains`` expands ``retune.planner.load_predictor``; its
    winner's predictor fields are fixed into every non-baseline candidate.
    """

    window_index: int
    window_start_ms: float
    window_end_ms: float
    history_start_ms: float
    as_of_ms: float
    trace_paths: tuple[str, ...]
    arrival_speedup_ratio: float
    engine: RetuneEngineSpec
    candidates: tuple[RetuneParams, ...]
    baseline: RetuneParams
    replicas: dict[str, int]
    base_planner: dict[str, Any] | None
    predictor_candidates: tuple[PredictorCandidate, ...] | None
    sla_ttft_ms: float
    sla_itl_ms: float
    max_parallel: int
    timeout_s: float
    history_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        keys = [candidate.key() for candidate in self.candidates]
        if keys != sorted(set(keys)):
            raise ValueError("candidates must be unique and sorted by key")
        if self.baseline not in self.candidates:
            raise ValueError("baseline must be one of the candidates")
        if not self.window_start_ms < self.window_end_ms:
            raise ValueError("window_start_ms must be before window_end_ms")
        if self.history_start_ms > self.window_start_ms:
            raise ValueError("history_start_ms must not be after window_start_ms")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if not self.timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        if self.predictor_candidates:
            labels = [label for label, _ in self.predictor_candidates]
            if len(set(labels)) != len(labels):
                raise ValueError(f"predictor candidate labels must be unique: {labels}")

    @property
    def window_s(self) -> float:
        """Window length on the planner clock (the score's time denominator)."""

        return (self.window_end_ms - self.window_start_ms) / (
            1000.0 * self.arrival_speedup_ratio
        )


@dataclass(frozen=True)
class SweepResult:
    """Outcome of ``run_sweep``.

    ``scores`` and ``metrics`` are keyed by the request candidate's
    ``key()``; a ``None`` score is a failed candidate. ``metrics`` entries
    carry the evaluated ``params`` (the candidate with any pre-sweep
    predictor fields) and either the evaluator's metrics or ``error``.
    ``predictor_presweep`` is ``{winner, fields, losses, eval_steps,
    warmup_steps, wall_s}`` (plus ``errors`` when a candidate's task failed)
    when the pre-sweep ran; ``winner`` and the ``losses`` keys are candidate
    labels. ``warmup_observations`` covers history
    and window, bucketed at the inner planner's throughput interval, for a
    predictor swap. When the winner changes the predictor, the sweep
    prepares it on ``warmup_observations`` as a host does before a swap;
    ``predictor_warmup_error`` is that preparation's error, if any.
    ``row_counts`` holds the window reader's counts and, when the history
    was read, ``history_requests``.
    """

    status: SweepStatus
    num_requests: int = 0
    row_counts: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float | None] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictor_presweep: dict[str, Any] | None = None
    warmup_observations: tuple[TrafficObservation, ...] | None = None
    predictor_warmup_error: str | None = None
    wall_s: float = 0.0
    error: str | None = None


def with_predictor_fields(
    candidate: RetuneParams,
    baseline: RetuneParams,
    fields: Mapping[str, Any] | None,
) -> RetuneParams:
    """The params a candidate is evaluated with after the predictor pre-sweep.

    The baseline keeps the current predictor; every other candidate carries
    the pre-sweep winner's predictor fields. Searched planner keys that a
    candidate does not set (the predictor fields of a grid point) take the
    baseline's values first. That leaves the inner planner unchanged, because
    the base planner already holds the current values, but gives a candidate
    equivalent to the baseline the baseline's key, so it is replayed once.
    """

    if candidate == baseline:
        return candidate
    planner = {**baseline.planner, **candidate.planner, **(fields or {})}
    evaluated = RetuneParams(router=dict(candidate.router), planner=planner)
    return candidate if evaluated == candidate else evaluated


def score_summary(
    summary: Mapping[str, Any], window_s: float
) -> tuple[float, dict[str, Any]]:
    """Score a replay summary as SLA-good output tokens per window-GPU-second.

    ``objective_value(summary, GOODPUT_PER_GPU) * duration_ms / (window_s *
    1000)``: the denominator is the fixed window length, so candidates whose
    drains differ are compared on the same time base. Fails closed: a missing
    goodput key or a non-positive ``gpu_hours``/``duration_ms`` raises instead
    of scoring 0, which would tie every candidate.
    """

    if summary.get(GOODPUT_KEY) is None:
        raise ValueError(f"replay summary has no {GOODPUT_KEY}; scoring needs an SLA")
    goodput, gpu_hours, duration_ms = (
        _finite_number(summary, name)
        for name in (GOODPUT_KEY, "gpu_hours", "duration_ms")
    )
    if gpu_hours <= 0 or duration_ms <= 0:
        raise ValueError(
            "replay summary needs positive gpu_hours and duration_ms, got "
            f"{gpu_hours!r} and {duration_ms!r}"
        )
    if not window_s > 0:
        raise ValueError(f"window_s must be positive, got {window_s!r}")
    objective = objective_value(dict(summary), OptimizationTarget.GOODPUT_PER_GPU)
    score = objective * duration_ms / (window_s * 1000.0)
    return score, {
        "objective_goodput_per_gpu": objective,
        GOODPUT_KEY: goodput,
        "gpu_hours": gpu_hours,
        "duration_ms": duration_ms,
        "completed_requests": summary.get("completed_requests"),
        "goodput_completed_requests": summary.get("goodput_completed_requests"),
    }


def _finite_number(summary: Mapping[str, Any], name: str) -> float:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"replay summary {name} must be numeric, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"replay summary {name} must be finite, got {value!r}")
    return float(value)


def evaluate_candidate(
    window_path: str,
    engine: RetuneEngineSpec,
    arrival_speedup_ratio: float,
    params: RetuneParams,
    replicas: dict[str, int],
    base_planner: dict[str, Any] | None,
    warmup_path: str | None,
    sla_ttft_ms: float,
    sla_itl_ms: float,
    window_s: float,
) -> tuple[float, dict[str, Any]]:
    """Replay the window once with ``params`` and score its summary.

    Without ``base_planner`` the replay keeps ``replicas`` fixed (Scenario 1).
    With it, an inner planner runs from ``base_planner`` with
    ``params.planner`` applied and retune disabled (Scenario 2); its load
    predictor is warmed with the observations in ``warmup_path``, when set.
    The replay uses the canonical router seed and ordinal request ids, so the
    baseline and every candidate see common random numbers, and it does not
    depend on host speed: a candidate that changes no routing or scaling
    decision scores exactly the baseline. With the pinned AISimulate, an inner
    planner in SLA mode is not guaranteed to be bitwise reproducible across
    processes: AISimulate's forward-pass performance model iterates its
    samples in a per-process hash order, so its refits can differ in the last
    bits and flip a scaling decision.
    Raises when the replay fails or its summary lacks the score inputs.
    """

    kwargs = engine.replay_kwargs(params.router)
    if engine.is_disagg:
        kwargs["num_prefill_workers"] = replicas["prefill"]
        kwargs["num_decode_workers"] = replicas["decode"]
    else:
        kwargs["num_workers"] = replicas["aggregated"]
    planner_config = None
    warmup = None
    if base_planner is not None:
        planner_config = {**base_planner, **params.planner, "retune": None}
        if warmup_path is not None:
            warmup = _read_observations(warmup_path)
    elif params.planner:
        raise ValueError("planner parameters need an inner planner (base_planner)")
    report = run_trace_replay(
        [window_path],
        trace_format="dynamo",
        arrival_speedup_ratio=arrival_speedup_ratio,
        planner_config=planner_config,
        _planner_warmup_observations=warmup,
        _deterministic=True,
        sla_ttft_ms=sla_ttft_ms,
        sla_itl_ms=sla_itl_ms,
        capture_per_request=False,
        capture_planner_details=False,
        **kwargs,
    )
    return score_summary(report.summary, window_s)


def _presweep_summary(
    losses: Mapping[str, float | None],
    entries: Mapping[str, dict[str, Any]],
    eval_steps: int,
    warmup_steps: int,
    wall_s: float,
) -> dict[str, Any]:
    winner: str | None = None
    best = math.inf
    for label, loss in losses.items():
        # Strict comparison: the earliest candidate wins ties.
        if loss is not None and loss < best:
            winner, best = label, loss
    return {
        "winner": winner,
        "fields": {} if winner is None else predictor_fields(entries[winner]),
        "losses": dict(losses),
        "eval_steps": eval_steps,
        "warmup_steps": warmup_steps,
        "wall_s": wall_s,
    }


def _write_observations(
    path: str | os.PathLike[str], observations: Sequence[TrafficObservation]
) -> str:
    """Write ``observations`` as a JSON list of ``TrafficObservation`` rows.

    Each row holds the dataclass fields in order. JSON writes a float's
    shortest round-trip repr, so ``_read_observations`` returns the same
    values.
    """

    rows = [dataclasses.astuple(observation) for observation in observations]
    with open(path, "w", encoding="utf-8") as sink:
        json.dump(rows, sink, separators=(",", ":"))
    return os.fspath(path)


def _read_observations(path: str | os.PathLike[str]) -> list[TrafficObservation]:
    """Read observations written by ``_write_observations``."""

    with open(path, encoding="utf-8") as source:
        return [TrafficObservation(*row) for row in json.load(source)]


def _presweep_task(
    observations_path: str,
    entry: dict[str, Any],
    interval_s: int,
    eval_steps: int,
) -> float | None:
    """Mean one-step forecast loss of one pre-sweep candidate on the window.

    The last ``eval_steps`` observations are the window ``W_k``; the ones
    before them are history that only warms the candidate's predictors, with
    no forecast, as a swapped-in predictor is warmed. So a candidate costs
    ``3 * eval_steps`` forecasts however long the history is, and ARIMA
    selects its order once on all of the history. ``None`` when a forecast
    fails or there is nothing to score. The observations travel by path:
    days of history at the planner interval are thousands of buckets, more
    than a pipe's socket buffer holds.
    """

    windows = [
        Window(obs.num_req, obs.isl, obs.osl)
        for obs in _read_observations(observations_path)
    ]
    loss = evaluate_preset(
        windows,
        internal_preset(entry),
        interval_s,
        len(windows) - eval_steps,
        forecast_warmup=False,
    )
    return loss if math.isfinite(loss) else None


def _predictor_warmup_task(planner: dict[str, Any], observations_path: str) -> None:
    """Prepare ``planner``'s load predictor on the sweep's warmup observations.

    The check a host's predictor swap makes (``prepare_predictor`` raises on
    too few effective points or a bad first forecast), run in a worker: an
    ARIMA fit on days of history takes seconds. Raises what it raises.
    """

    prepare_predictor(
        PlannerConfig.model_validate(planner),
        WorkerCapabilities(),
        _read_observations(observations_path),
    )


def _changes_predictor(baseline: RetuneParams, fields: Mapping[str, Any]) -> bool:
    """Whether ``fields`` change the baseline's predictor (bools are not 0/1)."""

    for key, value in fields.items():
        current = baseline.planner.get(key)
        if (isinstance(current, bool), current) != (isinstance(value, bool), value):
            return True
    return False


class _SweepInputs(NamedTuple):
    """What the read task hands back; the files live in the sweep directory.

    ``observations`` are the history and window buckets ending at the window
    end (``None`` without throughput scaling); ``observations_path`` is their
    file, written only for the pre-sweep. ``warmup_path`` holds the history
    buckets ending at the window start, the inner planner's warmup; it is
    ``None`` when no history row was selected.
    """

    window_path: str
    num_requests: int
    row_counts: dict[str, int]
    observations: tuple[TrafficObservation, ...] | None = None
    observations_path: str | None = None
    warmup_path: str | None = None


@dataclass(frozen=True)
class _ReadSpec:
    """The read task's slice of a ``SweepRequest``.

    Only what the readers need: the request also carries every candidate and
    the engine, which would make the task message large. ``buckets`` (empty)
    is set for an inner planner that scales on throughput, the only one with
    a load predictor to warm; the history is read only then.
    """

    trace_paths: tuple[str, ...]
    history_paths: tuple[str, ...]
    window_start_ms: float
    window_end_ms: float
    history_start_ms: float
    as_of_ms: float
    buckets: ObservationBuckets | None
    write_observations: bool

    @classmethod
    def from_request(
        cls,
        request: SweepRequest,
        buckets: ObservationBuckets | None,
        *,
        write_observations: bool,
    ) -> _ReadSpec:
        return cls(
            trace_paths=tuple(request.trace_paths),
            history_paths=tuple(request.history_paths),
            window_start_ms=request.window_start_ms,
            window_end_ms=request.window_end_ms,
            history_start_ms=request.history_start_ms,
            as_of_ms=request.as_of_ms,
            buckets=buckets,
            write_observations=write_observations,
        )


def _read_sweep_inputs(spec: _ReadSpec, work_dir: str) -> _SweepInputs:
    """Read the window into ``work_dir`` and bucket the history before it.

    Runs as a sweep worker task, so the deadline and cancel kill a hung read.
    An empty window stops here: the history is not read. A history row goes
    into ``spec.buckets``, as the window's rows do, and into buckets that end
    at the window start, written as the inner planner's warmup: those never
    take in a window request, wherever the window start falls on the grid of
    ``spec.buckets``. Raises what the readers raise.
    """

    directory = Path(work_dir)
    buckets = spec.buckets
    window = read_trace_window(
        spec.trace_paths,
        spec.window_start_ms,
        spec.window_end_ms,
        spec.as_of_ms,
        directory / _WINDOW_FILE,
        buckets=buckets,
    )
    row_counts = dict(window.row_counts)
    if window.num_requests == 0 or buckets is None:
        return _SweepInputs(window.path, window.num_requests, row_counts)

    warmup_path = None
    if spec.history_start_ms < spec.window_start_ms:
        warmup = ObservationBuckets(
            end_ms=spec.window_start_ms,
            interval_ms=buckets.interval_ms,
            duration_s=buckets.duration_s,
        )
        row_counts["history_requests"] = read_trace_history(
            (*spec.trace_paths, *spec.history_paths),
            spec.history_start_ms,
            spec.window_start_ms,
            spec.as_of_ms,
            buckets=(buckets, warmup),
        )
        if warmup.bucket_count():
            warmup_path = _write_observations(
                directory / _WARMUP_FILE, warmup.observations()
            )
    observations = tuple(buckets.observations())
    observations_path = None
    if spec.write_observations:
        observations_path = _write_observations(
            directory / _OBSERVATIONS_FILE, observations
        )
    return _SweepInputs(
        window_path=window.path,
        num_requests=window.num_requests,
        row_counts=row_counts,
        observations=observations,
        observations_path=observations_path,
        warmup_path=warmup_path,
    )


def run_sweep(
    request: SweepRequest,
    *,
    cancel: threading.Event | None = None,
    evaluator: Evaluator = evaluate_candidate,
) -> SweepResult:
    """Read the window, run the optional pre-sweep, and score every candidate.

    Never raises: a read error or any unexpected failure is status
    ``failed``. An empty window is status ``empty`` and evaluates nothing.
    ``timeout_s`` bounds the whole call in wall-clock time, trace reads
    included; at the deadline every worker is killed and the status is
    ``timeout``. Setting ``cancel`` kills the workers the same way (status
    ``failed``). A sweep stopped while reading has no ``num_requests`` or
    ``row_counts``. ``evaluator`` runs in the workers and must be picklable
    by reference; it is ``evaluate_candidate`` except in tests.
    """

    started = time.monotonic()
    cancel = cancel if cancel is not None else threading.Event()
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"retune-w{request.window_index}-"
        ) as work_dir:
            return _run_sweep(request, Path(work_dir), started, cancel, evaluator)
    except Exception as error:
        # The contract is "never raises": the controller turns this result
        # into a ``failed`` decision, so log the traceback and report it.
        logger.exception("retune sweep for window %d failed", request.window_index)
        return SweepResult(
            status="failed",
            wall_s=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )


class _SweepInterrupted(Exception):
    def __init__(self, status: SweepStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _run_sweep(
    request: SweepRequest,
    work_dir: Path,
    started: float,
    cancel: threading.Event,
    evaluator: Evaluator,
) -> SweepResult:
    deadline = started + request.timeout_s
    interval_s = _inner_throughput_interval(request.base_planner)
    buckets = (
        None
        if interval_s is None
        else ObservationBuckets(
            end_ms=request.window_end_ms,
            interval_ms=interval_s * 1000.0 * request.arrival_speedup_ratio,
            duration_s=float(interval_s),
        )
    )
    predictors = tuple(request.predictor_candidates or ())
    # (bucket length in trace ms, planner interval in s) when the pre-sweep runs
    presweep_interval = (
        (buckets.interval_ms, interval_s)
        if predictors and buckets is not None and interval_s
        else None
    )
    spec = _ReadSpec.from_request(
        request, buckets, write_observations=presweep_interval is not None
    )

    temp_dir = work_dir / _WORKER_TEMP_DIR
    temp_dir.mkdir()
    pool = _WorkerPool(request.max_parallel, deadline, cancel, str(temp_dir))
    inputs: _SweepInputs | None = None
    presweep: dict[str, Any] | None = None
    scores: dict[str, float | None] = {}
    metrics: dict[str, dict[str, Any]] = {}
    try:
        # Start every worker the later rounds can use now, so they import
        # while the first one reads instead of after it.
        pool.start(
            max(len(request.candidates), len(predictors) if presweep_interval else 0)
        )
        read = pool.run([(_READ_TASK, _read_sweep_inputs, (spec, str(work_dir)))])
        ok, value = read[_READ_TASK]
        if not ok:
            logger.error(
                "retune sweep for window %d failed to read its trace: %s",
                request.window_index,
                value,
            )
            return SweepResult(
                status="failed", wall_s=time.monotonic() - started, error=value
            )
        inputs = value
        if inputs.num_requests == 0:
            logger.info(
                "retune sweep for window %d: empty window, row counts %s",
                request.window_index,
                _nonzero(inputs.row_counts),
            )
            return SweepResult(
                status="empty",
                row_counts=dict(inputs.row_counts),
                wall_s=time.monotonic() - started,
            )

        inner_base = None
        if request.base_planner is not None:
            # The inner planner is the current one, warmed from the recorded
            # history (the read task's warmup buckets) instead of the outer
            # run's own warmup trace, never retuned, and without diagnostics
            # reports (one HTML file per candidate). It always scales, even
            # under an advisory outer planner: a candidate is scored on the
            # replicas it would drive. Replay never runs batch scheduling, so
            # it is off, which also drops the Redis settings an actuating
            # planner would require.
            inner_base = {
                **request.base_planner,
                "retune": None,
                "load_predictor_warmup_trace": None,
                "report_interval_hours": None,
                "live_dashboard_port": 0,
                "advisory": False,
                "batch_scheduling": {"enabled": False},
            }
        if presweep_interval is not None:
            presweep = _run_presweep(request, pool, inputs, *presweep_interval)
        fields = None if presweep is None else presweep["fields"]
        by_effective: dict[str, tuple[RetuneParams, list[str]]] = {}
        for candidate in request.candidates:
            effective = with_predictor_fields(candidate, request.baseline, fields)
            entry = by_effective.setdefault(effective.key(), (effective, []))
            entry[1].append(candidate.key())
        tasks: list[tuple[str, Callable[..., Any], tuple[Any, ...]]] = [
            (
                effective_key,
                evaluator,
                (
                    inputs.window_path,
                    request.engine,
                    request.arrival_speedup_ratio,
                    effective,
                    dict(request.replicas),
                    inner_base,
                    inputs.warmup_path,
                    request.sla_ttft_ms,
                    request.sla_itl_ms,
                    request.window_s,
                ),
            )
            for effective_key, (effective, _) in by_effective.items()
        ]
        if (
            fields
            and inner_base is not None
            and _changes_predictor(request.baseline, fields)
        ):
            # Prepare the winner's predictor here, beside the replays, rather
            # than on the planner tick that decides to swap it in. It goes
            # first, so an ARIMA or Prophet fit on days of history overlaps
            # the replays instead of trailing them.
            tasks.insert(
                0,
                (
                    _PREDICTOR_WARMUP_TASK,
                    _predictor_warmup_task,
                    ({**inner_base, **fields}, inputs.observations_path),
                ),
            )
        outcomes = pool.run(tasks)
    except _SweepInterrupted as interrupted:
        logger.warning(
            "retune sweep for window %d stopped: %s",
            request.window_index,
            interrupted,
        )
        return SweepResult(
            status=interrupted.status,
            num_requests=0 if inputs is None else inputs.num_requests,
            row_counts={} if inputs is None else dict(inputs.row_counts),
            predictor_presweep=presweep,
            wall_s=time.monotonic() - started,
            error=str(interrupted),
        )
    finally:
        pool.close()

    ok, value = outcomes.get(_PREDICTOR_WARMUP_TASK, (True, None))
    predictor_warmup_error = None if ok else value
    for effective_key, (effective, original_keys) in by_effective.items():
        ok, value = outcomes[effective_key]
        if ok:
            score, candidate_metrics = value
            entry_score: float | None = float(score)
            entry_metrics = {**candidate_metrics, "params": effective.to_dict()}
        else:
            entry_score = None
            entry_metrics = {"error": value, "params": effective.to_dict()}
        for key in original_keys:
            scores[key] = entry_score
            metrics[key] = entry_metrics
    scored = sum(score is not None for score in scores.values())
    wall_s = time.monotonic() - started
    logger.info(
        "retune sweep for window %d: %d requests, %d/%d candidates scored in "
        "%.1fs, row counts %s",
        request.window_index,
        inputs.num_requests,
        scored,
        len(scores),
        wall_s,
        _nonzero(inputs.row_counts),
    )
    return SweepResult(
        status="ok",
        num_requests=inputs.num_requests,
        row_counts=dict(inputs.row_counts),
        scores=scores,
        metrics=metrics,
        predictor_presweep=presweep,
        warmup_observations=inputs.observations,
        predictor_warmup_error=predictor_warmup_error,
        wall_s=wall_s,
    )


def _nonzero(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: value for key, value in counts.items() if value}


def _inner_throughput_interval(base_planner: Mapping[str, Any] | None) -> int | None:
    """The inner planner's predictor interval, when it scales on throughput."""

    if base_planner is None or not base_planner.get("enable_throughput_scaling"):
        return None
    return int(base_planner["throughput_adjustment_interval_seconds"])


def _run_presweep(
    request: SweepRequest,
    pool: _WorkerPool,
    inputs: _SweepInputs,
    interval_ms: float,
    interval_s: int,
) -> dict[str, Any]:
    """Score every predictor candidate in its own worker task, then pick one.

    ``inputs.observations`` are history and window buckets ending at the
    window end; only the buckets wholly inside ``W_k`` are forecast and
    scored, and the ones before them only warm the predictors. Each task
    carries the path of the observations file, not the observations.
    """

    if inputs.observations is None or inputs.observations_path is None:
        raise RuntimeError("the pre-sweep needs the read task's observations file")
    started = time.monotonic()
    window_steps = int(
        (request.window_end_ms - request.window_start_ms) / interval_ms + 1e-9
    )
    eval_steps = min(len(inputs.observations), window_steps)
    warmup_steps = len(inputs.observations) - eval_steps
    candidates = tuple(request.predictor_candidates or ())
    outcomes = pool.run(
        [
            (
                f"{_PRESWEEP_TASK}:{label}",
                _presweep_task,
                (inputs.observations_path, entry, interval_s, eval_steps),
            )
            for label, entry in candidates
        ]
    )
    losses: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for label, _ in candidates:
        ok, value = outcomes[f"{_PRESWEEP_TASK}:{label}"]
        if ok:
            losses[label] = value
        else:
            # A failed candidate cannot win; the others still compete.
            losses[label] = None
            errors[label] = value
    summary = _presweep_summary(
        losses,
        dict(candidates),
        eval_steps,
        warmup_steps,
        time.monotonic() - started,
    )
    if errors:
        summary["errors"] = errors
    return summary


@dataclass
class _Worker:
    process: BaseProcess
    conn: Connection
    task_id: str | None = None


class _WorkerPool:
    """Spawned worker processes fed one task at a time over their own pipes.

    Workers keep their temporary files in ``temp_dir``.
    """

    def __init__(
        self, size: int, deadline: float, cancel: threading.Event, temp_dir: str
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._size = size
        self._temp_dir = temp_dir
        self._deadline = deadline
        self._cancel = cancel
        self._workers: list[_Worker] = []
        self._spawned = 0

    def run(
        self, tasks: Sequence[tuple[str, Callable[..., Any], tuple[Any, ...]]]
    ) -> dict[str, tuple[bool, Any]]:
        """Run ``(task_id, function, args)`` tasks; ``task_id -> (ok, value)``.

        ``value`` is the function's result, or an error message when it
        raised or its worker died. Raises ``_SweepInterrupted`` at the
        deadline or on cancel.
        """

        pending = deque(tasks)
        results: dict[str, tuple[bool, Any]] = {}
        while len(results) < len(tasks):
            self._check_interrupted()
            self._dispatch(pending)
            busy = [worker for worker in self._workers if worker.task_id is not None]
            remaining = self._deadline - time.monotonic()
            ready = wait(
                [worker.conn for worker in busy]
                + [worker.process.sentinel for worker in busy],
                timeout=max(0.0, min(remaining, _POLL_INTERVAL_S)),
            )
            for worker in busy:
                if worker.conn in ready or worker.process.sentinel in ready:
                    self._collect(worker, results)
        return results

    def start(self, count: int) -> None:
        """Start idle workers up to ``count``, at most the pool size."""

        while len(self._workers) < min(count, self._size):
            self._spawn()

    def close(self) -> None:
        """Kill every worker, then wait for them all for a bounded time.

        An idle worker holds nothing to flush, and one still importing or
        blocked in I/O would not read a request to exit.
        """

        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.kill()
        deadline = time.monotonic() + _JOIN_TIMEOUT_S
        for worker in self._workers:
            _reap(worker, deadline)
        self._workers = []

    def _check_interrupted(self) -> None:
        if self._cancel.is_set():
            raise _SweepInterrupted("failed", "sweep cancelled")
        if time.monotonic() >= self._deadline:
            raise _SweepInterrupted("timeout", "sweep exceeded its deadline")

    def _dispatch(self, pending: deque) -> None:
        # A worker that died idle (killed while another task ran) held no
        # task: replace it without failing one.
        for worker in list(self._workers):
            if worker.task_id is None and not worker.process.is_alive():
                self._discard(worker)
        idle = [worker for worker in self._workers if worker.task_id is None]
        # Start every worker this round needs before sending, so they import
        # in parallel. A send does not wait for the import: task messages
        # are small (bulk data goes by file) and fit in the pipe's buffer.
        while len(idle) < len(pending) and len(self._workers) < self._size:
            idle.append(self._spawn())
        while pending and idle:
            self._check_interrupted()
            worker = idle.pop()
            task = pending.popleft()
            try:
                worker.conn.send(task)
            except OSError:
                # The worker died after the check above; the task waits for
                # the next one.
                self._discard(worker)
                pending.appendleft(task)
                continue
            worker.task_id = task[0]

    def _collect(self, worker: _Worker, results: dict[str, tuple[bool, Any]]) -> None:
        task_id = worker.task_id
        if task_id is None:
            return
        try:
            if worker.conn.poll():
                results[task_id] = worker.conn.recv()
                worker.task_id = None
                return
        except (EOFError, OSError):
            pass
        if worker.process.is_alive():
            return
        worker.process.join()
        results[task_id] = (
            False,
            f"sweep worker exited with code {worker.process.exitcode} "
            "before returning a result",
        )
        logger.warning(
            "retune sweep worker %s died on task %s (exit code %s); replacing it",
            worker.process.name,
            task_id,
            worker.process.exitcode,
        )
        self._discard(worker)

    def _spawn(self) -> _Worker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        self._spawned += 1
        process = self._context.Process(
            target=_worker_main,
            args=(child_conn, self._temp_dir),
            name=f"retune-sweep-{self._spawned}",
            daemon=True,
        )
        process.start()
        child_conn.close()
        worker = _Worker(process=process, conn=parent_conn)
        self._workers.append(worker)
        return worker

    def _discard(self, worker: _Worker) -> None:
        _reap(worker, time.monotonic() + _JOIN_TIMEOUT_S)
        self._workers.remove(worker)


def _reap(worker: _Worker, deadline: float) -> None:
    """Kill ``worker`` if it runs and close its pipe, waiting until ``deadline``.

    A process blocked in uninterruptible I/O (a hard-mounted filesystem)
    does not die on SIGKILL until the I/O returns; it is left behind rather
    than holding the sweep thread.
    """

    if worker.process.is_alive():
        worker.process.kill()
    worker.process.join(timeout=max(0.0, deadline - time.monotonic()))
    if worker.process.is_alive():
        logger.error(
            "retune sweep worker %s (pid %s) did not exit after SIGKILL; "
            "leaving it behind",
            worker.process.name,
            worker.process.pid,
        )
    worker.conn.close()


def _worker_main(conn: Connection, temp_dir: str) -> None:
    """Sweep worker loop: run one task per message until killed.

    Each task is answered with ``(ok, value)``. Temporary files go to
    ``temp_dir``, which is set before any task imports cmdstanpy (Prophet
    imports it lazily): cmdstanpy creates its directory on import.
    """

    os.nice(WORKER_NICENESS)
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir
    while True:
        try:
            _, function, args = conn.recv()
        except EOFError:
            return
        try:
            outcome = (True, function(*args))
        except Exception as error:
            # One candidate's failure must not take down the sweep: the
            # parent records it as that candidate's error.
            outcome = (False, f"{type(error).__name__}: {error}")
        try:
            conn.send(outcome)
        except Exception as error:
            # An unpicklable result is reported as this task's failure.
            conn.send((False, f"unsendable result: {error}"))
