# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before replay imports.

"""Planner retune through ``run_trace_replay``.

The integration cases replay a small generated ``dynamo.request.trace.v1``
workload: a dense 20 s head (25 requests/s) and a sparse tail that keeps the
replay ticking past the first decision. Under the startup router point
(``router_temperature: 100``, near-random worker sampling) the head violates
a tight TTFT SLA far more often than under greedy routing
(``router_temperature: 0``), so the first retune sweep applies the greedy
point and the replay router picks it up mid-run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.replay import api, run_synthetic_trace_replay, run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]

_BASE_MS = 1_700_000_000_000
_DATA = Path(__file__).resolve().parent / "data"

# Ticks every 5 s; window 0 = [0, 120) s launches at the 125 s tick (first
# tick >= 120 s + grace) and is decided one tick later. The interval also caps
# the sweep's wall-clock deadline, which must cover sweep worker start-up.
_INTERVAL_S = 120
_LAUNCH_S = 125.0
_DECIDE_S = 130.0
_HEAD_REQUESTS = 500
_TAIL_REQUESTS = 125


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
            "router": {"router_temperature": [0.0, 100.0]},
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


def _assert_router_update_lands_at_decision(report, worker_types: set[str]) -> None:
    first = report.planner.retune_events[0]
    assert first["window_index"] == 0
    assert first["outcome"] == "applied", first
    assert first["launched_at_s"] == _LAUNCH_S
    assert first["decided_at_s"] == _DECIDE_S
    assert first["previous"]["router"] == {"router_temperature": 100.0}
    assert first["chosen"]["router"] == {"router_temperature": 0.0}
    assert first["planner_changes"] == {}
    assert first["best_score"] > first["baseline_score"]

    ticks = report.planner.ticks
    decide_ms = _DECIDE_S * 1000.0
    for tick in ticks:
        desired = tick["runtime_decision"]["router_parameters"]
        assert set(desired) == worker_types
        expected = 0.0 if tick["at_ms"] >= decide_ms else 100.0
        assert {params["router_temperature"] for params in desired.values()} == {
            expected
        }
    # Each snapshot reports the router state before that tick's update.
    revisions = _router_revisions(ticks)
    assert set(revisions) == worker_types
    for worker_type, history in revisions.items():
        assert all(rev == 0 for at_ms, rev in history if at_ms <= decide_ms), (
            worker_type,
            history,
        )
        after = [rev for at_ms, rev in history if at_ms > decide_ms]
        assert after and all(rev == 1 for rev in after), (worker_type, history)
    later = [tick["router_policies"] for tick in ticks if tick["at_ms"] > decide_ms][0]
    assert {policy["parameters"]["router_temperature"] for policy in later} == {0.0}

    # Retune-only planners never scale.
    assert report.planner.scaling_events == []
    for tick in ticks:
        decision = tick["runtime_decision"]
        assert decision["target_prefill"] is None
        assert decision["target_decode"] is None


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_scenario1_agg_router_update_lands_mid_run(tmp_path):
    traces = _write_workload(tmp_path, blocks=16, output_tokens=128)

    report = run_trace_replay(
        traces,
        trace_format="dynamo",
        router_mode="kv_router",
        router_config=KvRouterConfig(router_temperature=100.0),
        num_workers=4,
        extra_engine_args=_engine(),
        planner_config=_retune_only("agg", sla_ttft_ms=60.0, sla_itl_ms=100.0),
    )

    assert report.summary["completed_requests"] == _HEAD_REQUESTS + _TAIL_REQUESTS
    _assert_router_update_lands_at_decision(report, {"aggregated"})
    assert report.to_dict()["planner"]["retune_events"] == report.planner.retune_events


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_scenario1_disagg_router_update_lands_mid_run(tmp_path):
    traces = _write_workload(tmp_path, blocks=64, output_tokens=64)

    report = run_trace_replay(
        traces,
        trace_format="dynamo",
        router_mode="kv_router",
        router_config=KvRouterConfig(router_temperature=100.0),
        num_prefill_workers=3,
        num_decode_workers=4,
        prefill_engine_args=_engine(worker_type="prefill"),
        decode_engine_args=_engine(worker_type="decode"),
        planner_config=_retune_only("disagg", sla_ttft_ms=100.0, sla_itl_ms=100.0),
    )

    assert report.summary["completed_requests"] == _HEAD_REQUESTS + _TAIL_REQUESTS
    _assert_router_update_lands_at_decision(report, {"prefill", "decode"})


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_scenario2_planner_change_is_visible_from_the_next_tick(tmp_path, monkeypatch):
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
            "router": {"router_temperature": [0.0, 100.0]},
            "planner": {"load_scaling_down_sensitivity": [70, 90]},
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

    first = report.planner.retune_events[0]
    assert first["outcome"] == "applied", first
    assert first["decided_at_s"] == _DECIDE_S
    assert first["chosen"]["router"] == {"router_temperature": 0.0}
    changes = first["planner_changes"]
    assert set(changes) == {"load_scaling_down_sensitivity"}
    new_value = changes["load_scaling_down_sensitivity"]
    assert new_value in (70, 90)
    decided_at = first["decided_at_s"]
    assert [value for now_s, value in seen if now_s <= decided_at] and all(
        value == 80 for now_s, value in seen if now_s <= decided_at
    )
    after = [value for now_s, value in seen if now_s > decided_at]
    assert after and all(value == new_value for value in after)


def _guard_config() -> dict:
    return _retune_only("agg")


@pytest.mark.unit
@pytest.mark.parametrize(
    "options, message",
    [
        ({"replay_concurrency": 2}, "replay_concurrency"),
        ({"agentic_lanes": 2}, "agentic_lanes"),
        ({"agentic_warmup": True}, "agentic_warmup"),
        ({"max_sim_time_ms": 1_000.0}, "max_sim_time_ms"),
    ],
)
def test_retune_rejects_closed_loop_and_truncated_replays(tmp_path, options, message):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=4, gap_ms=100, blocks=1, output_tokens=2
    )

    with pytest.raises(ValueError, match=message):
        run_trace_replay(
            [trace],
            trace_format="dynamo",
            router_mode="kv_router",
            planner_config=_guard_config(),
            **options,
        )


@pytest.mark.unit
def test_retune_accepts_execution_model_for_a_standard_dynamo_trace(tmp_path):
    # DynamoReplayRunner sets execution_model for every performance-model
    # engine; it only matters for agentic rows, which this trace has none of.
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=4, gap_ms=100, blocks=1, output_tokens=2
    )

    report = run_trace_replay(
        [trace],
        trace_format="dynamo",
        router_mode="kv_router",
        planner_config=_guard_config(),
        execution_model="target-model",
    )

    assert report.summary["completed_requests"] == 4
    assert report.planner is not None and report.planner.retune_events == []


@pytest.mark.unit
def test_retune_rejects_an_agentic_dynamo_trace(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=3, gap_ms=100, blocks=1, output_tokens=2
    )
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
    disagg_args = {
        "extra_engine_args": None,
        "prefill_engine_args": _engine(worker_type="prefill"),
        "decode_engine_args": _engine(worker_type="decode"),
    }

    inputs = api._retune_inputs(config, **_inputs_kwargs(trace, **disagg_args))
    assert inputs.engine.is_disagg

    with pytest.raises(ValueError, match="retune engine must set"):
        run_trace_replay(
            [trace],
            trace_format="dynamo",
            router_mode="kv_router",
            planner_config=_guard_config(),
            **disagg_args,
        )


@pytest.mark.unit
def test_retune_requires_the_dynamo_trace_format():
    with pytest.raises(ValueError, match="trace_format='dynamo'"):
        run_trace_replay(
            _DATA / "mooncake_tiny.jsonl",
            trace_format="mooncake",
            router_mode="kv_router",
            planner_config=_guard_config(),
        )


@pytest.mark.unit
def test_synthetic_replay_rejects_retune():
    with pytest.raises(ValueError, match="only supported by run_trace_replay"):
        run_synthetic_trace_replay(
            64, 4, 4, request_rate=10.0, planner_config=_guard_config()
        )


@pytest.mark.unit
def test_retune_history_must_end_before_the_replayed_trace(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=4, gap_ms=100, blocks=1, output_tokens=2
    )
    history = _write_trace(
        tmp_path / "history.jsonl",
        count=4,
        gap_ms=100,
        blocks=1,
        output_tokens=2,
        start_ms=_BASE_MS - 200,
        prefix="history",
    )
    config = _guard_config()
    config["load_predictor_warmup_trace"] = str(history)

    with pytest.raises(ValueError, match="2 history requests arrive"):
        run_trace_replay(
            [trace],
            trace_format="dynamo",
            router_mode="kv_router",
            planner_config=config,
        )


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
    config = PlannerConfig.model_validate(
        _guard_config() | {"load_predictor_warmup_trace": str(history)}
    )

    inputs = api._retune_inputs(
        config, **_inputs_kwargs(trace, router_config=router_config)
    )

    assert inputs.trace_paths == (str(trace),)
    assert inputs.history_paths == (str(history),)
    assert inputs.trace_origin_ms == float(_BASE_MS + 500)
    assert inputs.arrival_speedup_ratio == 2.0
    assert inputs.prune_by_mtime is False
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


@pytest.mark.unit
def test_retune_inputs_default_router_config_and_disagg_engine(tmp_path):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=2, gap_ms=100, blocks=1, output_tokens=2
    )
    prefill = {"engine": {"worker_type": "prefill"}}
    decode = {"engine": {"worker_type": "decode"}}
    config = PlannerConfig.model_validate(_retune_only("disagg"))

    inputs = api._retune_inputs(
        config,
        **_inputs_kwargs(
            trace,
            extra_engine_args=None,
            prefill_engine_args=prefill,
            decode_engine_args=decode,
        ),
    )

    assert inputs.engine.extra_engine_args is None
    assert inputs.engine.prefill_engine_args == normalize_mocker_config(prefill)
    assert inputs.engine.decode_engine_args == normalize_mocker_config(decode)
    assert inputs.engine.router_config == json.loads(KvRouterConfig().to_json())
    assert inputs.history_paths == ()


@pytest.mark.unit
def test_retune_inputs_warn_when_the_live_engine_snapshot_differs(tmp_path, caplog):
    trace = _write_trace(
        tmp_path / "trace.jsonl", count=2, gap_ms=100, blocks=1, output_tokens=2
    )
    config_payload = _guard_config()
    config_payload["retune"]["live"] = {
        "engine": {"router_mode": "kv_router", "model_name": "deployed-model"}
    }
    config = PlannerConfig.model_validate(config_payload)

    with caplog.at_level(logging.WARNING, logger="dynamo.replay.api"):
        api._retune_inputs(config, **_inputs_kwargs(trace))

    assert "planner.retune.live.engine differs" in caplog.text
    assert "extra_engine_args" in caplog.text and "model_name" in caplog.text
