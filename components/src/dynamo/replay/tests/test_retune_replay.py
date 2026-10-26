# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before replay imports.

"""Planner retune through ``run_trace_replay``.

The integration case replays a small generated ``dynamo.request.trace.v1``
workload: a dense 20 s head (25 requests/s) and a sparse tail that keeps the
replay ticking past the first decision. Under the startup router point
(``router_temperature: 100``, near-random worker sampling) the head violates
a tight TTFT SLA far more often than under greedy routing
(``router_temperature: 0``), so the first retune sweep applies the greedy
point and the replay router picks it up mid-run.
"""

from __future__ import annotations

import inspect
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import RetuneParams, TrafficObservation
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.retune import sweep
from dynamo.replay import api, run_synthetic_trace_replay, run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]

_BASE_MS = 1_700_000_000_000

# Ticks every 5 s; window 0 = [0, 120) s launches at the 125 s tick (first
# tick >= 120 s + grace) and is decided one tick later. The interval also caps
# the sweep's wall-clock deadline, which must cover sweep worker start-up.
_INTERVAL_S = 120
_LAUNCH_S = 125.0
_DECIDE_S = 130.0
_HEAD_REQUESTS = 500
_TAIL_REQUESTS = 125
# AISimulate recommend syntax: credit and scale pinned at the router's
# startup values, so the grid is the two temperatures.
_ROUTER_DOMAIN = {
    "overlap_score_credit": 1.0,
    "prefill_load_scale": 1.0,
    "temperature": {"choices": [0.0, 100.0]},
}


def _router_point(temperature: float) -> dict[str, float]:
    return {
        "overlap_score_credit": 1.0,
        "prefill_load_scale": 1.0,
        "router_temperature": temperature,
    }


def _write_trace(
    path: Path,
    *,
    count: int,
    gap_ms: int,
    blocks: int,
    output_tokens: int,
    start_ms: int = _BASE_MS,
    prefix: str = "request",
) -> Path:
    """Write ``count`` requests, ``gap_ms`` apart, each with a unique prompt."""

    with path.open("w", encoding="utf-8") as sink:
        for index in range(count):
            received_ms = start_ms + index * gap_ms
            row = {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": received_ms + 10,
                "request": {
                    "request_id": f"{prefix}-{index}",
                    "request_received_ms": received_ms,
                    "output_tokens": output_tokens,
                    "replay": {
                        "trace_block_size": 64,
                        "input_length": 64 * blocks,
                        "input_sequence_hashes": [
                            index * 1_000 + block for block in range(blocks)
                        ],
                    },
                },
            }
            sink.write(json.dumps(row) + "\n")
    return path


def _write_workload(tmp_path: Path, *, blocks: int, output_tokens: int) -> list[Path]:
    """Dense head (``_HEAD_REQUESTS`` at 40 ms) and a 1 request/s tail, as two
    trace files on one time axis."""

    head = _write_trace(
        tmp_path / "head.jsonl",
        count=_HEAD_REQUESTS,
        gap_ms=40,
        blocks=blocks,
        output_tokens=output_tokens,
    )
    tail = _write_trace(
        tmp_path / "tail.jsonl",
        count=_TAIL_REQUESTS,
        gap_ms=1_000,
        blocks=blocks,
        output_tokens=output_tokens,
        start_ms=_BASE_MS + _HEAD_REQUESTS * 40,
        prefix="tail",
    )
    return [head, tail]


def _engine(**engine) -> dict:
    return normalize_mocker_config({"engine": {"max_num_seqs": 8, **engine}})


def _retune_only(mode: str, **retune) -> dict:
    return {
        "mode": mode,
        "enable_load_scaling": False,
        "enable_throughput_scaling": False,
        "report_interval_hours": None,
        "live_dashboard_port": 0,
        "retune": {
            "interval_seconds": _INTERVAL_S,
            "grace_seconds": 1,
            "max_parallel": 2,
            "router": _ROUTER_DOMAIN,
            **retune,
        },
    }


def _router_revisions(ticks: list[dict]) -> dict[str, list[tuple[float, int]]]:
    revisions: dict[str, list[tuple[float, int]]] = {}
    for tick in ticks:
        for policy in tick["router_policies"]:
            revisions.setdefault(policy["worker_type"], []).append(
                (tick["at_ms"], policy["revision"])
            )
    return revisions


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_retune_lands_router_and_planner_changes_mid_run(tmp_path, monkeypatch):
    # The one end-to-end path: run_trace_replay -> controller -> spawned
    # sweep -> inner load planner -> router_parameters -> Rust revision.
    traces = _write_workload(tmp_path, blocks=16, output_tokens=128)
    seen: list[tuple[float, int]] = []
    original_tick = OrchestratorEngineAdapter.tick

    async def recording_tick(self, scheduled_tick, tick_input):
        seen.append((tick_input.now_s, self._config.load_scaling_down_sensitivity))
        return await original_tick(self, scheduled_tick, tick_input)

    monkeypatch.setattr(OrchestratorEngineAdapter, "tick", recording_tick)
    planner_config = {
        "mode": "agg",
        "optimization_target": "throughput",
        "report_interval_hours": None,
        "live_dashboard_port": 0,
        # Keep the load planner (outer and inner) from scaling below the
        # replay's 4 workers, where routing no longer decides the score.
        "min_endpoint": 4,
        "retune": {
            "interval_seconds": _INTERVAL_S,
            "grace_seconds": 1,
            "max_parallel": 2,
            "sla_ttft_ms": 60.0,
            "sla_itl_ms": 100.0,
            "router": _ROUTER_DOMAIN,
            "planner": {
                "load_sensitivity": {"preset": False},
                "load_scaling_down_sensitivity": {"choices": [70, 90]},
            },
        },
    }

    report = run_trace_replay(
        traces,
        trace_format="dynamo",
        router_mode="kv_router",
        router_config=KvRouterConfig(router_temperature=100.0),
        num_workers=4,
        extra_engine_args=_engine(),
        planner_config=planner_config,
    )

    assert report.summary["completed_requests"] == _HEAD_REQUESTS + _TAIL_REQUESTS
    first = report.planner.retune_events[0]
    assert first["window_index"] == 0
    assert first["outcome"] == "applied", first
    assert first["launched_at_s"] == _LAUNCH_S
    assert first["decided_at_s"] == _DECIDE_S
    assert first["previous"]["router"] == _router_point(100.0)
    assert first["chosen"]["router"] == _router_point(0.0)
    assert first["best_score"] > first["baseline_score"]
    assert report.to_dict()["planner"]["retune_events"] == report.planner.retune_events

    # The planner change is visible from the first tick after the decision.
    changes = first["planner_changes"]
    assert set(changes) == {"load_scaling_down_sensitivity"}
    new_value = changes["load_scaling_down_sensitivity"]
    assert new_value in (70, 90)
    before = [value for now_s, value in seen if now_s <= _DECIDE_S]
    after = [value for now_s, value in seen if now_s > _DECIDE_S]
    assert before and set(before) == {80}
    assert after and set(after) == {new_value}

    # The desired router map flips at the decision tick; each router snapshot
    # reports the router state before that tick's update.
    ticks = report.planner.ticks
    decide_ms = _DECIDE_S * 1000.0
    for tick in ticks:
        desired = tick["runtime_decision"]["router_parameters"]
        expected = 0.0 if tick["at_ms"] >= decide_ms else 100.0
        assert set(desired) == {"aggregated"}
        assert desired["aggregated"]["router_temperature"] == expected
    history = _router_revisions(ticks)["aggregated"]
    assert all(rev == 0 for at_ms, rev in history if at_ms <= decide_ms), history
    revisions_after = [rev for at_ms, rev in history if at_ms > decide_ms]
    assert revisions_after and set(revisions_after) == {1}, history
    later = [tick["router_policies"] for tick in ticks if tick["at_ms"] > decide_ms][0]
    assert {policy["parameters"]["router_temperature"] for policy in later} == {0.0}


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_candidate_replays_share_common_random_numbers(tmp_path):
    # Near-random routing (router_temperature 100) makes every placement a
    # draw from the selector's rng; candidate replays share one canonical
    # seed, so equal decisions give equal scores.
    window = _write_trace(
        tmp_path / "window.jsonl", count=200, gap_ms=40, blocks=16, output_tokens=128
    )
    engine = RetuneEngineSpec(
        extra_engine_args=_engine(),
        router_config=json.loads(KvRouterConfig(router_temperature=100.0).to_json()),
    )

    def evaluate(router: dict) -> tuple[float, dict]:
        return sweep.evaluate_candidate(
            str(window),
            engine,
            1.0,
            RetuneParams(router=router, planner={}),
            {"aggregated": 4},
            None,
            None,
            60.0,
            100.0,
            8.0,
        )

    baseline = evaluate({})
    assert evaluate({}) == baseline
    # Replay reports no shared cache hits, so this candidate changes no
    # routing decision: it ties the baseline, and the controller keeps it.
    assert evaluate({"shared_cache_multiplier": 0.5}) == baseline
    assert evaluate({"router_temperature": 0.0})[0] != baseline[0]


# Report fields measured on the wall clock, not on the simulated one.
_WALL_CLOCK_FIELDS = (
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
)


def _simulated_record(report) -> dict:
    """The report as a dict, without its wall-clock fields."""

    record = report.to_dict()
    for name in _WALL_CLOCK_FIELDS:
        del record["summary"][name]
    return record


def _canonical(record: dict) -> str:
    """Canonical JSON with one field per line, so a failed comparison shows
    the fields that differ."""

    return json.dumps(record, sort_keys=True, indent=1)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_planner_replay_does_not_depend_on_the_tick_deadline(tmp_path):
    # The live tick deadline (scheduling.tick_max_duration_seconds) is wall
    # time, and a replayed tick takes no simulated time, so replay runs every
    # tick to completion. Under the deadline, a tick that outlived it (on a
    # slow or loaded host; here a 1 ns deadline) became skip_tick_timeout and
    # changed the scaling path. The static-threshold load planner
    # (optimization_target throughput) leaves AISimulate's forward-pass
    # performance model out.
    traces = _write_workload(tmp_path, blocks=16, output_tokens=128)

    def replay(**scheduling) -> dict:
        planner_config = {
            "mode": "agg",
            "optimization_target": "throughput",
            "max_gpu_budget": 8,
            "report_interval_hours": None,
            "live_dashboard_port": 0,
        }
        if scheduling:
            planner_config["scheduling"] = scheduling
        report = run_trace_replay(
            traces,
            trace_format="dynamo",
            num_workers=1,
            # 256 blocks of 64 tokens: the dense head fills the KV cache, so
            # the planner scales up, and down again in the tail.
            extra_engine_args=_engine(max_num_seqs=4, num_gpu_blocks=256),
            planner_config=planner_config,
            _deterministic=True,
        )
        record = _simulated_record(report)
        # Digest of the input config, which differs by the deadline itself.
        del record["planner"]["metadata"]["planner_config_digest"]
        return record

    default = replay()
    assert default["planner"]["scaling_events"], "no scaling for the deadline to change"
    assert _canonical(replay(tick_max_duration_seconds=1e-9)) == _canonical(default)


def _planner_candidate_report(traces: list[str]) -> tuple[int, str]:
    """Replay ``traces`` as a sweep candidate with an inner load planner.

    Returns this process's id and the canonical report without its
    wall-clock fields. Module level: the test runs it in spawned processes,
    as the sweep runs its candidates.
    """

    report = run_trace_replay(
        traces,
        trace_format="dynamo",
        router_mode="kv_router",
        num_workers=4,
        extra_engine_args=_engine(max_num_seqs=16),
        planner_config={
            "mode": "agg",
            "optimization_target": "sla",
            "enable_load_scaling": True,
            "enable_throughput_scaling": False,
            "ttft_ms": 500.0,
            "itl_ms": 30.0,
            "min_endpoint": 1,
            "max_gpu_budget": 8,
            # A small FPM store fills within a few ticks, so the planner's
            # AIS regression evicts samples and refits for most of the run.
            "max_num_fpm_samples": 8,
            "report_interval_hours": None,
            "live_dashboard_port": 0,
        },
        _deterministic=True,
        sla_ttft_ms=500.0,
        sla_itl_ms=30.0,
        capture_planner_details=True,
    )
    return os.getpid(), _canonical(_simulated_record(report))


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.xfail(
    strict=True,
    # Only the report comparison may fail as expected; a crash, a timeout or
    # a failed precondition below is a real failure.
    raises=AssertionError,
    reason=(
        "the pinned aisimulate keeps the planner's FPM samples in a std "
        "HashMap (BucketedSamples), so a per-process hasher seed orders every "
        "regression refit and breaks eviction ties; remove this marker once "
        "the pin orders the buckets"
    ),
)
def test_planner_candidate_replay_is_identical_across_processes(tmp_path):
    # Each sweep candidate replays in its own spawned worker, so common random
    # numbers need the whole report, inner planner included, to be a function
    # of the inputs alone. Six 30 s phases of different rates and shapes move
    # the planner's forward-pass samples across the regression's bucket grid;
    # with hashed buckets, 16 replays in fresh processes gave 16 different
    # reports, and with ordered buckets one.
    traces, start_ms = [], _BASE_MS
    for phase, (count, gap_ms, blocks, output_tokens) in enumerate(
        (
            (750, 40, 16, 128),
            (2_500, 12, 32, 256),
            (500, 60, 8, 64),
            (1_500, 20, 48, 512),
            (1_000, 30, 4, 32),
            (3_000, 10, 24, 128),
        )
    ):
        trace = _write_trace(
            tmp_path / f"phase{phase}.jsonl",
            count=count,
            gap_ms=gap_ms,
            blocks=blocks,
            output_tokens=output_tokens,
            start_ms=start_ms,
            prefix=f"phase{phase}",
        )
        traces.append(str(trace))
        start_ms += count * gap_ms

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        (first_pid, first), (second_pid, second) = pool.map(
            _planner_candidate_report, [traces, traces]
        )

    # Preconditions: pytest.fail is not an AssertionError, so the xfail
    # marker cannot absorb them.
    if first_pid == second_pid:
        pytest.fail("both replays ran in one process")
    planner = json.loads(first)["planner"]
    if not planner["scaling_events"]:
        pytest.fail("the inner planner never scaled")
    if all(
        tick["effects"]["diagnostics"]["estimated_itl_ms"] is None
        for tick in planner["ticks"]
    ):
        pytest.fail("the planner's performance model never estimated ITL")
    assert first == second


def _guard_config() -> dict:
    return _retune_only("agg")


def _throughput_config(warmup_trace: Path) -> dict:
    # Throughput scaling warms the inner planner's predictors from history.
    return _guard_config() | {
        "optimization_target": "sla",
        "enable_throughput_scaling": True,
        "load_predictor_warmup_trace": str(warmup_trace),
    }


def _inputs_kwargs(trace: Path, **overrides) -> dict:
    kwargs = {
        "trace_files": [trace],
        "trace_format": "dynamo",
        "replay_concurrency": None,
        "agentic_lanes": None,
        "agentic_snapshot": None,
        "agentic_warmup": False,
        "agentic_profile": None,
        "max_sim_time_ms": None,
        "extra_engine_args": {"engine": {"max_num_seqs": 8}},
        "prefill_engine_args": None,
        "decode_engine_args": None,
        "router_mode": "kv_router",
        "router_config": None,
        "ais_perf_config": None,
        "performance_model_metadata": None,
        "model_name": "test-model",
        "kv_event_lag_ms": 5.0,
        "benchmark_granularity": 4,
        "arrival_speedup_ratio": 2.0,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "options, messages",
    [
        (
            {"trace_format": "mooncake"},
            ["trace_format='dynamo'", "traffic.source.format: dynamo"],
        ),
        (
            {"replay_concurrency": 2},
            ["replay_concurrency", "traffic.load.type: concurrency"],
        ),
        ({"agentic_warmup": True}, ["agentic_warmup", "traffic.load.agentic_*"]),
        (
            {"max_sim_time_ms": 1_000.0},
            ["max_sim_time_ms", "traffic.stop.max_virtual_time_seconds"],
        ),
    ],
    ids=["mooncake-format", "replay-concurrency", "agentic-warmup", "max-sim-time"],
)
def test_retune_rejects_non_open_loop_inputs(tmp_path, options, messages):
    # Each error names the replay kwarg and the aisimulate predict setting;
    # test_retune_predict_cli.py checks that one of them reaches the CLI.
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=4, gap_ms=100, blocks=1, output_tokens=2
    )
    config = PlannerConfig.model_validate(_guard_config())

    with pytest.raises(ValueError) as error:
        api._retune_inputs(config, **_inputs_kwargs(trace, **options))
    assert all(message in str(error.value) for message in messages), error.value


@pytest.mark.unit
def test_retune_accepts_execution_model_for_a_standard_dynamo_trace(tmp_path):
    # DynamoReplayRunner sets execution_model for every performance-model
    # engine; it only matters for agentic rows, which this trace has none of.
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=3, gap_ms=100, blocks=1, output_tokens=2
    )

    report = run_trace_replay(
        [trace],
        trace_format="dynamo",
        router_mode="kv_router",
        planner_config=_guard_config(),
        execution_model="target-model",
    )

    assert report.summary["completed_requests"] == 3
    assert report.planner is not None and report.planner.retune_events == []

    # An agentic Dynamo trace (rows with agent_context) is rejected.
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        row["agent_context"] = {"session_id": "s1"}
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = PlannerConfig.model_validate(_guard_config())

    with pytest.raises(ValueError, match="3 requests with agent_context"):
        api._retune_inputs(config, **_inputs_kwargs(trace))


@pytest.mark.unit
def test_retune_rejects_an_engine_shape_that_differs_from_the_planner_mode(
    tmp_path,
):
    # Prefill and decode args make the outer replay disaggregated even though
    # the planner mode is agg; retune must not pin a different engine.
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=4, gap_ms=100, blocks=1, output_tokens=2
    )
    config = PlannerConfig.model_validate(_guard_config())
    prefill = {"engine": {"max_num_seqs": 8, "worker_type": "prefill"}}
    decode = {"engine": {"max_num_seqs": 8, "worker_type": "decode"}}
    disagg_args = {
        "extra_engine_args": None,
        "prefill_engine_args": prefill,
        "decode_engine_args": decode,
    }

    inputs = api._retune_inputs(config, **_inputs_kwargs(trace, **disagg_args))
    assert inputs.engine.is_disagg
    assert inputs.engine.extra_engine_args is None
    assert inputs.engine.prefill_engine_args == normalize_mocker_config(prefill)
    assert inputs.engine.decode_engine_args == normalize_mocker_config(decode)
    assert inputs.engine.router_config == json.loads(KvRouterConfig().to_json())
    assert inputs.history_paths == ()

    with pytest.raises(ValueError, match="retune engine must set"):
        run_trace_replay(
            [trace],
            trace_format="dynamo",
            router_mode="kv_router",
            planner_config=_guard_config(),
            **disagg_args,
        )


@pytest.mark.unit
def test_synthetic_replay_rejects_retune():
    with pytest.raises(
        ValueError, match="traffic.source type: trace with format: dynamo"
    ):
        run_synthetic_trace_replay(
            64, 4, 4, request_rate=10.0, planner_config=_guard_config()
        )


@pytest.mark.unit
def test_retune_inputs_pin_the_outer_engine_and_time_axis(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl",
        count=3,
        gap_ms=100,
        blocks=1,
        output_tokens=2,
        start_ms=_BASE_MS + 500,
    )
    history = _write_trace(
        tmp_path / "history.jsonl",
        count=3,
        gap_ms=100,
        blocks=1,
        output_tokens=2,
        prefix="history",
    )
    router_config = KvRouterConfig(router_temperature=0.25)
    config = PlannerConfig.model_validate(_throughput_config(history))

    inputs = api._retune_inputs(
        config, **_inputs_kwargs(trace, router_config=router_config)
    )

    assert inputs.trace_paths == (str(trace),)
    assert inputs.history_paths == (str(history),)
    assert inputs.trace_origin_ms == float(_BASE_MS + 500)
    assert inputs.arrival_speedup_ratio == 2.0
    engine = inputs.engine
    assert engine.extra_engine_args == normalize_mocker_config(
        {"engine": {"max_num_seqs": 8}}
    )
    assert engine.prefill_engine_args is None and engine.decode_engine_args is None
    assert engine.router_mode == "kv_router"
    assert engine.router_config == json.loads(router_config.to_json())
    assert engine.model_name == "test-model"
    assert engine.kv_event_lag_ms == 5.0
    assert engine.benchmark_granularity == 4

    # History and windows share one time axis, so history must end before
    # the first replayed request.
    _write_trace(
        history,
        count=4,
        gap_ms=100,
        blocks=1,
        output_tokens=2,
        start_ms=_BASE_MS + 300,
        prefix="history",
    )
    with pytest.raises(ValueError, match="2 history requests arrive"):
        api._retune_inputs(config, **_inputs_kwargs(trace))


@pytest.mark.unit
def test_only_a_throughput_planner_reads_the_warmup_trace_as_history(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=3, gap_ms=100, blocks=1, output_tokens=2
    )
    mooncake = tmp_path / "mooncake.jsonl"
    mooncake.write_text(
        json.dumps({"timestamp": 0, "input_length": 64, "output_length": 2}) + "\n"
    )
    # Without throughput scaling no predictor is warmed from history, so a
    # Mooncake warmup trace, or one overlapping the replay, is never read.
    for overrides in (
        {},
        {"enable_load_scaling": True, "optimization_target": "throughput"},
    ):
        for warmup in (mooncake, trace):
            payload = _guard_config() | overrides
            payload["load_predictor_warmup_trace"] = str(warmup)
            config = PlannerConfig.model_validate(payload)
            inputs = api._retune_inputs(config, **_inputs_kwargs(trace))
            assert inputs.history_paths == ()

    config = PlannerConfig.model_validate(_throughput_config(mooncake))
    with pytest.raises(ValueError, match="dynamo.request.trace.v1 .* relative"):
        api._retune_inputs(config, **_inputs_kwargs(trace))


# Every run_trace_replay kwarg, classified for retune candidate replays, so a
# new kwarg cannot silently replay candidates on another engine.
_PINNED = set(RetuneEngineSpec.model_fields)  # from the outer replay
_PER_CANDIDATE = {  # set by sweep.evaluate_candidate
    "num_workers",
    "num_prefill_workers",
    "num_decode_workers",
    "trace_format",
    "arrival_speedup_ratio",
    "planner_config",
    "sla_ttft_ms",
    "sla_itl_ms",
    "capture_per_request",
    "capture_planner_details",
    "_planner_warmup_observations",
    "_deterministic",
}
_REJECTED = {  # by _retune_inputs: retune replays complete open-loop windows
    "replay_concurrency",
    "agentic_lanes",
    "agentic_snapshot",
    "agentic_warmup",
    "agentic_profile",
    "max_sim_time_ms",
}
_IGNORED = {  # no effect on a candidate's engine or score
    "replay_mode",  # a planner replay is offline
    "execution_model",  # names the target model of agentic rows only
    "report_jsonl_path",  # outputs of the outer replay
    "telemetry_options",
    "sla_e2e_ms",  # the score uses the TTFT and ITL SLAs
    "trace_block_size",  # a Dynamo trace carries its block size
    "trace_shared_prefix_ratio",  # other trace formats only
    "trace_num_prefix_groups",
    "weka_nested_timestamp_basis",
}


@pytest.mark.unit
def test_every_replay_kwarg_is_classified_for_retune(tmp_path, monkeypatch):
    groups = [_PINNED, _PER_CANDIDATE, _REJECTED, _IGNORED]
    params = set(inspect.signature(run_trace_replay).parameters) - {"trace_files"}
    assert set().union(*groups) == params
    assert sum(len(group) for group in groups) == len(params)

    # A candidate replay passes exactly the pinned and per-candidate kwargs.
    calls: list[dict] = []

    def recording_replay(trace_files, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("recorded")

    monkeypatch.setattr(sweep, "run_trace_replay", recording_replay)
    warmup = [TrafficObservation(60.0, 12.0, 512.0, 64.0)]
    warmup_path = sweep._write_observations(tmp_path / "warmup.json", warmup)
    disagg = RetuneEngineSpec(prefill_engine_args={}, decode_engine_args={})
    for engine, replicas, base_planner in (
        (RetuneEngineSpec(), {"aggregated": 1}, {"mode": "agg"}),
        (disagg, {"prefill": 1, "decode": 1}, None),
    ):
        with pytest.raises(RuntimeError, match="recorded"):
            sweep.evaluate_candidate(
                "window.jsonl",
                engine,
                1.0,
                RetuneParams(router={}, planner={}),
                replicas,
                base_planner,
                warmup_path,
                100.0,
                10.0,
                60.0,
            )
    assert set().union(*calls) == _PINNED | _PER_CANDIDATE
    # Common random numbers; an inner planner is warmed from the read task's
    # buckets.
    assert [call["_deterministic"] for call in calls] == [True, True]
    assert [call["_planner_warmup_observations"] for call in calls] == [warmup, None]


@pytest.mark.unit
def test_run_trace_replay_hands_the_warmup_to_the_inner_planner(monkeypatch):
    received = {}

    def recording_adapter(**kwargs):
        received.update(kwargs)
        raise RuntimeError("recorded")

    monkeypatch.setattr(api, "_planner_replay_adapter", lambda: recording_adapter)
    warmup = [TrafficObservation(60.0, 12.0, 512.0, 64.0)]

    with pytest.raises(RuntimeError, match="recorded"):
        run_trace_replay(
            ["window.jsonl"],
            trace_format="dynamo",
            planner_config={"mode": "agg"},
            _planner_warmup_observations=warmup,
        )

    assert received["warmup_observations"] is warmup
