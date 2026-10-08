# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for replay-owned Arena telemetry ingestion."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


@pytest.fixture
def sims_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import the simulation wrapper without a built Dynamo runtime."""

    dynamo = _package("dynamo")
    dynamo_core = types.ModuleType("dynamo._core")
    dynamo_core.run_mocker_trace_replay = lambda *args, **kwargs: {}
    replay = _package("dynamo.replay")
    replay_config = types.ModuleType("dynamo.replay.config")
    replay_config.load_engine_args = lambda value: value
    replay_planner = types.ModuleType("dynamo.replay.planner")
    replay_planner._engine_caps = lambda value: value
    replay_planner._generate_ais_decode_fpms = lambda *args, **kwargs: []
    replay_planner._generate_ais_prefill_fpms = lambda *args, **kwargs: []
    replay_planner._ais_performance_model_configs = lambda metadata, mode: {}
    replay_planner._ais_session_kwargs = lambda config, args: None
    replay_planner._ais_fpm_digest = lambda prefill, decode: ""
    replay_planner.create_session = lambda **kwargs: object()
    replay_report = types.ModuleType("dynamo.replay.report")
    planner = _package("dynamo.planner")
    planner_config_package = _package("dynamo.planner.config")
    planner_core = _package("dynamo.planner.core")
    planner_offline = _package("dynamo.planner.offline")

    planner_config = types.ModuleType("dynamo.planner.config.planner_config")
    planner_config.PlannerConfig = object
    planner_engine_protocol = types.ModuleType("dynamo.planner.core.engine_protocol")
    planner_engine_protocol.EngineProtocol = object
    planner_types = types.ModuleType("dynamo.planner.core.types")
    planner_types.TrafficObservation = object
    planner_types.WorkerCapabilities = object
    replay_adapter = types.ModuleType("dynamo.planner.offline.replay_adapter")
    replay_adapter.ReplayPlannerAdapter = object
    trace_data = types.ModuleType("dynamo.planner.offline.trace_data")
    trace_data.extract_traffic_observations_from_trace = lambda path, interval: []

    class PlannerReplayDetails:
        def __init__(
            self,
            *,
            ticks=None,
            scaling_events=None,
            total_ticks: int = 0,
            html_report_path: str | None = None,
        ) -> None:
            self.ticks = ticks if ticks is not None else []
            self.scaling_events = scaling_events if scaling_events is not None else []
            self.total_ticks = total_ticks
            self.html_report_path = html_report_path

    class ReplayReport:
        def __init__(
            self, *, summary, per_request, coverage, planner, telemetry=None
        ) -> None:
            self.summary = summary
            self.per_request = per_request
            self.coverage = coverage
            self.planner = planner
            self.telemetry = telemetry

    replay_report.PlannerReplayDetails = PlannerReplayDetails
    replay_report.ReplayReport = ReplayReport
    replay_reporting = types.ModuleType("dynamo.replay.reporting")
    replay_reporting.write_report_json = lambda *args, **kwargs: None

    stubs = {
        "dynamo": dynamo,
        "dynamo._core": dynamo_core,
        "dynamo.replay": replay,
        "dynamo.replay.config": replay_config,
        "dynamo.replay.planner": replay_planner,
        "dynamo.replay.report": replay_report,
        "dynamo.replay.reporting": replay_reporting,
        "dynamo.planner": planner,
        "dynamo.planner.config": planner_config_package,
        "dynamo.planner.config.planner_config": planner_config,
        "dynamo.planner.core": planner_core,
        "dynamo.planner.core.engine_protocol": planner_engine_protocol,
        "dynamo.planner.core.types": planner_types,
        "dynamo.planner.offline": planner_offline,
        "dynamo.planner.offline.replay_adapter": replay_adapter,
        "dynamo.planner.offline.trace_data": trace_data,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "autoscaling_arena.runners.sims"
    runners_package = importlib.import_module("autoscaling_arena.runners")
    previous_module = sys.modules.pop(module_name, None)
    had_package_attr = "sims" in vars(runners_package)
    previous_package_attr = vars(runners_package).get("sims")
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if had_package_attr:
            runners_package.sims = previous_package_attr
        else:
            vars(runners_package).pop("sims", None)


def _row(
    worker_id: int,
    *,
    active_blocks: int,
    inactive_blocks: int,
    total_blocks: int,
    running_requests: int,
    waiting_requests: int,
    dp_rank: int = 0,
) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "active_blocks": active_blocks,
        "inactive_blocks": inactive_blocks,
        "total_blocks": total_blocks,
        "active_cache_usage": active_blocks / total_blocks,
        "physical_cache_usage": (active_blocks + inactive_blocks) / total_blocks,
        "running_requests": running_requests,
        "waiting_requests": waiting_requests,
    }


def _telemetry_sample() -> dict[str, Any]:
    return {
        "sample_ordinal": 2,
        "kind": "periodic",
        "interval_start_ms": 0.0,
        "sampled_at_ms": 1_000.0,
        "traffic": {
            "duration_s": 1.0,
            "arriving_requests": 4,
            "completed_requests": 3,
            "avg_isl": 1_024.0,
            "avg_osl": 256.0,
            "avg_ttft_ms": 123.25,
            "avg_itl_ms": 17.75,
            "ttft_count": 2,
            "itl_count": 1,
            "avg_router_kv_hit_rate": 0.625,
            "router_kv_hit_rate_count": 8,
            "avg_accept_length": 2.5,
            "accept_length_forward_count": 6,
        },
        "prefill_scheduler_metrics": [
            _row(
                10,
                active_blocks=10,
                inactive_blocks=5,
                total_blocks=100,
                running_requests=2,
                waiting_requests=2,
                dp_rank=0,
            ),
            _row(
                10,
                active_blocks=30,
                inactive_blocks=5,
                total_blocks=100,
                running_requests=3,
                waiting_requests=3,
                dp_rank=1,
            ),
            _row(
                20,
                active_blocks=99,
                inactive_blocks=0,
                total_blocks=100,
                running_requests=99,
                waiting_requests=99,
            ),
        ],
        "decode_scheduler_metrics": [
            _row(
                30,
                active_blocks=20,
                inactive_blocks=20,
                total_blocks=200,
                running_requests=4,
                waiting_requests=7,
            )
        ],
        "prefill_interval_metrics": {
            "cache_hit_tokens": 30,
            "cache_total_tokens": 200,
            "preemptions": 3,
        },
        "decode_interval_metrics": {
            "cache_hit_tokens": 60,
            "cache_total_tokens": 200,
            "preemptions": 4,
        },
        "router_pending_prefill_requests": 11,
        "router_pending_decode_requests": 13,
        "active_prefill_ids": [10],
        "active_decode_ids": [30],
        "starting_prefill_ids": [],
        "starting_decode_ids": [],
        "draining_prefill_ids": [20],
        "draining_decode_ids": [],
    }


def _contract_sample(
    ordinal: int,
    kind: str,
    *,
    interval_start_ms: float,
    sampled_at_ms: float,
) -> dict[str, Any]:
    sample = copy.deepcopy(_telemetry_sample())
    sample.update(
        sample_ordinal=ordinal,
        kind=kind,
        interval_start_ms=interval_start_ms,
        sampled_at_ms=sampled_at_ms,
    )
    return sample


def _baseline_sample() -> dict[str, Any]:
    return _contract_sample(0, "baseline", interval_start_ms=0.0, sampled_at_ms=0.0)


def _planner_tick() -> dict[str, Any]:
    return {
        "at_ms": 1_000.0,
        "topology": {
            "prefill": {
                "active": [10],
                "starting": [],
                "draining": [20],
            },
            "decode": {
                "active": [30],
                "starting": [],
                "draining": [],
            },
        },
        "runtime_decision": {
            "target_prefill": 2,
            "target_decode": None,
            "next_tick_ms": 2_000.0,
        },
    }


def test_top_level_telemetry_projects_raw_queue_kv_traffic_and_decision(
    sims_module: types.ModuleType,
) -> None:
    planner = sims_module.PlannerReplayDetails(ticks=[_planner_tick()])

    timeline = sims_module._ReplayTelemetryTimeline.build(
        [_telemetry_sample()], planner
    )

    assert len(timeline) == 1
    sample = timeline[0]
    assert sample["timestamp_s"] == 1.0
    assert sample["window_start_s"] == 0.0
    assert sample["arriving_requests"] == 4
    assert sample["completed_requests"] == 3
    assert sample["traffic"] == _telemetry_sample()["traffic"]
    assert sample["interval_duration_s"] == pytest.approx(1.0)
    assert sample["mean_isl"] == pytest.approx(1_024.0)
    assert sample["mean_osl"] == pytest.approx(256.0)
    assert sample["mean_ttft_ms"] == pytest.approx(123.25)
    assert sample["mean_tpot_ms"] == pytest.approx(17.75)
    assert sample["mean_router_kv_hit_rate"] == pytest.approx(0.625)
    assert sample["mean_accept_length"] == pytest.approx(2.5)
    assert sample["accept_length_forward_count"] == 6
    assert sample["active_kv_blocks"] == 60
    assert sample["inactive_kv_blocks"] == 30
    assert sample["total_kv_blocks"] == 400
    assert sample["active_kv_cache_utilization"] == pytest.approx(60 / 400)
    assert sample["physical_kv_cache_utilization"] == pytest.approx(90 / 400)
    assert sample["scheduler_cache_reuse"] == pytest.approx(90 / 400)
    assert sample["preemptions"] == 7
    assert sample["scheduler_waiting_prefill_requests"] == 104
    assert sample["scheduler_waiting_decode_requests"] == 7
    assert sample["router_pending_requests"] == 24
    assert sample["queued_prefill_requests"] == 115
    assert sample["queued_decode_requests"] == 20
    assert sample["total_queued_requests"] == 135
    assert sample["queue_telemetry_semantics"] == (
        "scheduler_waiting_plus_router_pending"
    )
    assert sample["active_prefill_replicas"] == 1
    assert sample["provisioned_prefill_replicas"] == 2
    assert sample["scaling_decision"] is True
    assert sample["requested_prefill_replicas"] == 2
    assert sample["requested_decode_replicas"] == 1


def test_decision_between_samples_is_kept_at_exact_planner_time(
    sims_module: types.ModuleType,
) -> None:
    telemetry = [
        {**_telemetry_sample(), "sampled_at_ms": 5_000.0},
        {**_telemetry_sample(), "sampled_at_ms": 10_000.0},
    ]
    tick = {**_planner_tick(), "at_ms": 7_500.0}

    timeline = sims_module._ReplayTelemetryTimeline.build(
        telemetry, sims_module.PlannerReplayDetails(ticks=[tick])
    )

    assert [row["timestamp_s"] for row in timeline] == [5.0, 7.5, 10.0]
    decision = timeline[1]
    assert decision["decision_only"] is True
    assert decision["scaling_decision"] is True
    assert "total_queued_requests" not in decision


def test_absent_or_partial_telemetry_never_infers_queue(
    sims_module: types.ModuleType,
) -> None:
    assert sims_module._ReplayTelemetryTimeline.build(None, None) == []
    partial = _telemetry_sample()
    partial["active_prefill_ids"] = [10, 999]

    sample = sims_module._ReplayTelemetryTimeline.build([partial], None)[0]

    assert sample["scheduler_metrics_available"] is False
    assert sample["total_queued_requests"] is None
    assert sample["active_kv_cache_utilization"] is None


def test_persisted_telemetry_jsonl_is_the_report_input(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "telemetry.jsonl"
    samples = [
        _baseline_sample(),
        _contract_sample(
            1,
            "periodic",
            interval_start_ms=0.0,
            sampled_at_ms=1_000.0,
        ),
        _contract_sample(
            2,
            "final",
            interval_start_ms=1_000.0,
            sampled_at_ms=1_250.0,
        ),
    ]
    payload = "\n".join(json.dumps(sample) for sample in samples) + "\n"
    path.write_text(payload)

    loaded = sims_module._ingest_telemetry_jsonl(path, None)

    assert loaded.sample_count == 3
    assert loaded.sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert [row["sample_ordinal"] for row in loaded.timeline] == [0, 1, 2]
    assert loaded.timeline[-1]["is_final"] is True


def test_baseline_only_telemetry_is_valid(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(json.dumps(_baseline_sample()) + "\n")

    loaded = sims_module._ingest_telemetry_jsonl(path, None)

    assert loaded.sample_count == 1
    assert loaded.timeline[0]["sample_kind"] == "baseline"


@pytest.mark.parametrize("payload", ["not-json\n", "[]\n", "\n"])
def test_invalid_persisted_telemetry_fails_instead_of_rendering_partial_data(
    sims_module: types.ModuleType, tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(payload)

    with pytest.raises(ValueError, match="dynamo.replay.telemetry.v1"):
        sims_module._ingest_telemetry_jsonl(path, None)


def test_empty_persisted_telemetry_fails_instead_of_hiding_capture_failure(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("")

    with pytest.raises(ValueError, match="file is empty"):
        sims_module._ingest_telemetry_jsonl(path, None)


def test_non_utf8_persisted_telemetry_is_rejected(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_bytes(b"\xff\n")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        sims_module._ingest_telemetry_jsonl(path, None)


_DELETE = object()


@pytest.mark.parametrize(
    ("field_path", "replacement", "error"),
    [
        (("traffic",), _DELETE, "required field is missing"),
        (("traffic", "arriving_requests"), True, "non-negative integer"),
        (("traffic", "avg_ttft_ms"), "fast", "finite non-negative"),
        (("traffic", "avg_accept_length"), {}, "finite non-negative"),
        (
            ("prefill_interval_metrics", "preemptions"),
            1.5,
            "non-negative integer",
        ),
        (("decode_interval_metrics",), [], "must be an object"),
        (("prefill_scheduler_metrics",), {}, "must be an array"),
        (
            ("decode_scheduler_metrics", 0, "worker_id"),
            "30",
            "non-negative integer",
        ),
        (
            ("decode_scheduler_metrics", 0, "active_cache_usage"),
            1.5,
            "at most 1",
        ),
        (("active_prefill_ids",), [True], "non-negative integer"),
        (("router_pending_decode_requests",), -1, "non-negative integer"),
    ],
)
def test_v1_contract_validates_required_nested_shapes_and_types(
    sims_module: types.ModuleType,
    tmp_path: Path,
    field_path: tuple[str | int, ...],
    replacement: Any,
    error: str,
) -> None:
    sample = _baseline_sample()
    target: Any = sample
    for component in field_path[:-1]:
        target = target[component]
    leaf = field_path[-1]
    if replacement is _DELETE:
        del target[leaf]
    else:
        target[leaf] = replacement
    path = tmp_path / "telemetry.jsonl"
    path.write_text(json.dumps(sample) + "\n")

    with pytest.raises(ValueError, match=error):
        sims_module._ingest_telemetry_jsonl(path, None)


@pytest.mark.parametrize(
    ("samples", "error"),
    [
        (
            [
                _contract_sample(
                    0,
                    "periodic",
                    interval_start_ms=0.0,
                    sampled_at_ms=0.0,
                )
            ],
            "first sample must be baseline",
        ),
        (
            [
                _baseline_sample(),
                _contract_sample(
                    2,
                    "periodic",
                    interval_start_ms=0.0,
                    sampled_at_ms=1.0,
                ),
            ],
            "expected contiguous ordinal 1",
        ),
        (
            [
                _baseline_sample(),
                _contract_sample(
                    1,
                    "baseline",
                    interval_start_ms=0.0,
                    sampled_at_ms=1.0,
                ),
            ],
            "baseline is only legal",
        ),
        (
            [
                _baseline_sample(),
                _contract_sample(
                    1,
                    "final",
                    interval_start_ms=0.0,
                    sampled_at_ms=1.0,
                ),
                _contract_sample(
                    2,
                    "periodic",
                    interval_start_ms=1.0,
                    sampled_at_ms=2.0,
                ),
            ],
            "final sample must be the last",
        ),
        (
            [
                _contract_sample(
                    0,
                    "baseline",
                    interval_start_ms=0.0,
                    sampled_at_ms=5.0,
                ),
                _contract_sample(
                    1,
                    "periodic",
                    interval_start_ms=0.0,
                    sampled_at_ms=4.0,
                ),
            ],
            "monotonically non-decreasing",
        ),
        (
            [
                _contract_sample(
                    0,
                    "baseline",
                    interval_start_ms=1.0,
                    sampled_at_ms=0.0,
                )
            ],
            "less than or equal",
        ),
        (
            [
                _contract_sample(
                    0,
                    "snapshot",
                    interval_start_ms=0.0,
                    sampled_at_ms=0.0,
                )
            ],
            "must be one of baseline",
        ),
    ],
)
def test_v1_contract_validates_stream_ordering(
    sims_module: types.ModuleType,
    tmp_path: Path,
    samples: list[dict[str, Any]],
    error: str,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")

    with pytest.raises(ValueError, match=error):
        sims_module._ingest_telemetry_jsonl(path, None)


def test_v1_contract_allows_unknown_fields_and_equal_final_timestamp(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    baseline = _baseline_sample()
    baseline["future_top_level"] = {"new": True}
    baseline["traffic"]["future_traffic_metric"] = 42
    final = _contract_sample(1, "final", interval_start_ms=0.0, sampled_at_ms=0.0)
    final["decode_scheduler_metrics"][0]["future_rank_metric"] = 3
    path = tmp_path / "telemetry.jsonl"
    path.write_text(json.dumps(baseline) + "\n" + json.dumps(final) + "\n")

    loaded = sims_module._ingest_telemetry_jsonl(path, None)

    assert loaded.sample_count == 2
    assert loaded.timeline[-1]["is_final"] is True


def test_arena_replay_result_forwards_dynamo_report(
    sims_module: types.ModuleType,
) -> None:
    planner = sims_module.PlannerReplayDetails(total_ticks=3)
    replay_report = sims_module.ReplayReport(
        summary={"gpu_hours": 1.25},
        per_request=[{"terminal_status": "completed"}],
        coverage={"capture_per_request": True},
        planner=planner,
    )
    result = sims_module.ArenaReplayResult(replay_report=replay_report, timeline=[])

    assert result.planner is planner
    assert result.gpu_hours == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("expected", "argument_name"),
    [
        ("aggregated", "extra_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ],
)
def test_role_neutral_engine_args_infer_role_from_slot(
    sims_module: types.ModuleType, expected: str, argument_name: str
) -> None:
    normalized = sims_module._normalize_engine_args_role(
        '{"speedup_ratio": 2.0}',
        expected=expected,
        argument_name=argument_name,
    )
    assert json.loads(normalized) == {
        "speedup_ratio": 2.0,
        "worker_type": expected,
    }


def test_explicit_engine_role_must_match_slot(
    sims_module: types.ModuleType,
) -> None:
    with pytest.raises(
        ValueError,
        match="prefill_engine_args.worker_type must be 'prefill'",
    ):
        sims_module._normalize_engine_args_role(
            '{"worker_type": "aggregated"}',
            expected="prefill",
            argument_name="prefill_engine_args",
        )


@pytest.mark.parametrize(
    ("telemetry_interval", "expected_telemetry"),
    [(5.0, True), (None, False)],
)
def test_run_arena_replay_ingests_only_the_persisted_telemetry_stream(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    telemetry_interval: float | None,
    expected_telemetry: bool,
) -> None:
    config = types.SimpleNamespace(mode="agg", advisory=False)
    engine_args = types.SimpleNamespace(
        ais_perf_config={"backend": "vllm", "backend_version": "current"}
    )
    capabilities = types.SimpleNamespace(decode="decode-caps")
    engine = object()
    lifecycle: list[str] = []
    runtime_kwargs: dict[str, Any] = {}
    bootstrap_calls: list[dict[str, Any]] = []

    class FakeAdapter:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["planner_config"] is config
            assert kwargs["warmup_observations"] is None
            assert kwargs["benchmark_granularity"] == 8

        def __enter__(self):
            lifecycle.append("enter")
            return self

        def __exit__(self, *exc_info: Any) -> None:
            lifecycle.append("exit")

        def finalize(self, operations: list[dict[str, Any]]) -> Any:
            lifecycle.append("finalize")
            assert operations == [{"operation": "scale_up"}]
            return sims_module.PlannerReplayDetails(ticks=[_planner_tick()])

    def fake_runtime(trace_files: list[str], **kwargs: Any) -> Any:
        runtime_kwargs.update(kwargs)
        lifecycle.append("runtime")
        assert kwargs["scaling_policy"].__class__ is FakeAdapter
        telemetry_path = kwargs["telemetry_jsonl_path"]
        if telemetry_path is not None:
            Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
            Path(telemetry_path).write_text(json.dumps(_baseline_sample()) + "\n")
        return types.SimpleNamespace(
            summary={"gpu_hours": 0.5},
            per_request=[],
            coverage={"capture_per_request": True},
            lifecycle_operations=[{"operation": "scale_up"}],
            # A deliberately conflicting native payload proves Arena does not
            # consume the capture/in-memory contract.
            telemetry={"samples": [{"sampled_at_ms": 999_000.0}]},
        )

    monkeypatch.setattr(sims_module, "_as_config", lambda value: config)
    monkeypatch.setattr(
        sims_module,
        "_load_engine_args",
        lambda value: engine_args if value is not None else None,
    )
    monkeypatch.setattr(sims_module, "_engine_caps", lambda value: "decode-caps")
    monkeypatch.setattr(
        sims_module, "WorkerCapabilities", lambda **kwargs: capabilities
    )
    monkeypatch.setattr(sims_module, "ReplayPlannerAdapter", FakeAdapter)
    monkeypatch.setattr(sims_module, "_run_mocker_trace_replay", fake_runtime)
    monkeypatch.setattr(
        sims_module,
        "_bootstrap_ais_regressions",
        lambda *args, **kwargs: bootstrap_calls.append(kwargs),
    )

    result = sims_module.run_arena_replay(
        trace_file="trace.jsonl",
        autoscaler=lambda config, capabilities: engine,
        substrate_config=config,
        extra_engine_args="{}",
        telemetry_sample_interval_s=telemetry_interval,
        telemetry_jsonl_path=(
            str(tmp_path / "telemetry.jsonl") if expected_telemetry else None
        ),
        capture_per_request=True,
        performance_model_metadata={
            "aggregated": {
                "provider": "ais",
                "config": {
                    "backend": "vllm",
                    "system": "h200_sxm",
                    "model_path": "model",
                    "tp_size": 1,
                },
            }
        },
    )

    assert lifecycle == ["enter", "runtime", "finalize", "exit"]
    assert bootstrap_calls[0]["performance_model_metadata"] == {
        "aggregated": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "backend_version": "current",
                "system": "h200_sxm",
                "model_path": "model",
                "tp_size": 1,
            },
        }
    }
    assert runtime_kwargs["capture_telemetry"] is False
    assert runtime_kwargs["telemetry_sample_interval_ms"] == 5_000.0
    assert "telemetry_callback" not in runtime_kwargs
    assert (runtime_kwargs["telemetry_jsonl_path"] is not None) is expected_telemetry
    assert result.replay_report.telemetry is None
    assert bool(result.timeline) is expected_telemetry
    if expected_telemetry:
        payload = json.dumps(_baseline_sample()) + "\n"
        assert result.timeline[0]["timestamp_s"] == 0.0
        assert result.telemetry_artifact == {
            "contract": "dynamo.replay.telemetry.v1",
            "sample_count": 1,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    else:
        assert result.telemetry_artifact is None


@pytest.mark.parametrize(
    ("trace_kwargs", "error"),
    [
        ({}, "one of trace_file or trace_files is required"),
        (
            {"trace_file": "trace.jsonl", "trace_files": ["trace-000.jsonl"]},
            "trace_file and trace_files are mutually exclusive",
        ),
        (
            {"trace_files": [], "trace_format": "dynamo"},
            "trace_files must contain at least one trace file",
        ),
        (
            {"trace_file": "trace.jsonl", "trace_format": "other"},
            "trace_format must be 'mooncake' or 'dynamo'",
        ),
        (
            {"trace_files": ["one.jsonl", "two.jsonl"]},
            "trace_format='mooncake' requires exactly one trace file",
        ),
    ],
)
def test_run_arena_replay_validates_trace_inputs_before_setup(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    trace_kwargs: dict[str, Any],
    error: str,
) -> None:
    monkeypatch.setattr(
        sims_module,
        "_as_config",
        lambda value: pytest.fail("invalid trace input must fail before setup"),
    )
    with pytest.raises(ValueError, match=error):
        sims_module.run_arena_replay(
            autoscaler=lambda config, capabilities: object(),
            substrate_config={},
            **trace_kwargs,
        )


def test_telemetry_interval_must_be_positive_and_finite(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sims_module,
        "_as_config",
        lambda value: pytest.fail("invalid interval must fail before setup"),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        sims_module.run_arena_replay(
            trace_file="trace.jsonl",
            autoscaler=lambda config, capabilities: object(),
            substrate_config={},
            telemetry_sample_interval_s=0.0,
        )


def test_telemetry_destination_cannot_clobber_trace_or_report(
    sims_module: types.ModuleType, tmp_path: Path
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("trace")
    hard_link = tmp_path / "telemetry-hard-link.jsonl"
    hard_link.hardlink_to(trace)

    with pytest.raises(ValueError, match="replay trace input"):
        sims_module._validate_telemetry_destination(
            hard_link, trace_paths=[str(trace)], report_json=None
        )

    output = tmp_path / "same-output.json"
    with pytest.raises(ValueError, match="must refer to different files"):
        sims_module._validate_telemetry_destination(
            output, trace_paths=[str(trace)], report_json=str(output)
        )


def test_ais_bootstrap_skips_rival_before_generating_fpms(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = types.SimpleNamespace(
        supports_ais_bootstrap=False,
        install_regressions_from_fpms=lambda **kwargs: None,
    )
    adapter = types.SimpleNamespace(
        install_benchmark_fpms=lambda **kwargs: pytest.fail(
            "rival must not receive Planner regression data"
        )
    )
    config = types.SimpleNamespace(optimization_target="sla", mode="agg")
    engine_args = types.SimpleNamespace(
        ais_perf_config={
            "backend": "backend",
            "system": "system",
            "model_path": "model",
        }
    )
    monkeypatch.setattr(
        sims_module,
        "_generate_ais_prefill_fpms",
        lambda *args, **kwargs: pytest.fail("must be skipped"),
    )
    monkeypatch.setattr(
        sims_module,
        "_generate_ais_decode_fpms",
        lambda *args, **kwargs: pytest.fail("must be skipped"),
    )

    sims_module._bootstrap_ais_regressions(
        adapter,
        engine,
        config,
        extra_engine_args=engine_args,
        prefill_engine_args=None,
        decode_engine_args=None,
        performance_model_metadata=None,
        benchmark_granularity=8,
    )


def test_planner_warmup_uses_current_dynamo_observation_contract(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    observations = [object()]
    monkeypatch.setattr(
        sims_module,
        "extract_traffic_observations_from_trace",
        lambda path, interval: (calls.append((path, interval)) or observations),
    )
    config = types.SimpleNamespace(
        load_predictor_warmup_trace="warmup/*.jsonl",
        throughput_adjustment_interval_seconds=60,
    )
    planner_engine = types.SimpleNamespace(
        install_regressions_from_fpms=lambda **kwargs: None
    )

    assert (
        sims_module._planner_warmup_observations(config, planner_engine) is observations
    )
    assert calls == [("warmup/*.jsonl", 60)]

    rival_engine = types.SimpleNamespace(
        supports_ais_bootstrap=False,
        install_regressions_from_fpms=lambda **kwargs: None,
    )
    assert sims_module._planner_warmup_observations(config, rival_engine) is None
    assert calls == [("warmup/*.jsonl", 60)]


def test_resolved_metadata_backfills_each_unpinned_disagg_role_independently(
    sims_module: types.ModuleType,
) -> None:
    metadata = {
        "prefill": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "system": "h200_sxm",
                "model_path": "weights/model",
                "tp_size": 2,
            },
        },
        "decode": {
            "provider": "ais",
            "config": {
                "backend": "sglang",
                "system": "b200_sxm",
                "model_path": "weights/model",
                "tp_size": 4,
            },
        },
    }

    resolved = sims_module._resolved_performance_model_metadata(
        metadata,
        mode="disagg",
        extra_engine_args=None,
        prefill_engine_args=types.SimpleNamespace(
            ais_perf_config={"backend": "vllm", "backend_version": "prefill-current"}
        ),
        decode_engine_args=types.SimpleNamespace(
            ais_perf_config={"backend": "sglang", "backend_version": "decode-current"}
        ),
    )

    assert resolved == {
        "prefill": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "backend_version": "prefill-current",
                "system": "h200_sxm",
                "model_path": "weights/model",
                "tp_size": 2,
            },
        },
        "decode": {
            "provider": "ais",
            "config": {
                "backend": "sglang",
                "backend_version": "decode-current",
                "system": "b200_sxm",
                "model_path": "weights/model",
                "tp_size": 4,
            },
        },
    }
    assert "backend_version" not in metadata["prefill"]["config"]
    assert "backend_version" not in metadata["decode"]["config"]


def test_resolved_metadata_preserves_explicit_or_cross_backend_versions(
    sims_module: types.ModuleType,
) -> None:
    metadata = {
        "prefill": {
            "provider": "ais",
            "config": {"backend": "vllm", "backend_version": "pinned"},
        },
        "decode": {
            "provider": "ais",
            "config": {"backend": "sglang"},
        },
    }

    resolved = sims_module._resolved_performance_model_metadata(
        metadata,
        mode="disagg",
        extra_engine_args=None,
        prefill_engine_args=types.SimpleNamespace(
            ais_perf_config={"backend": "vllm", "backend_version": "prefill-current"}
        ),
        decode_engine_args=types.SimpleNamespace(
            ais_perf_config={"backend": "vllm", "backend_version": "decode-current"}
        ),
    )

    assert resolved == metadata


@pytest.mark.parametrize(
    ("decode_identity", "expected_sessions"),
    [("shared", 1), ("decode", 2)],
)
def test_planner_bootstrap_uses_metadata_and_canonical_session_sharing(
    sims_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decode_identity: str,
    expected_sessions: int,
) -> None:
    metadata = {
        "prefill": {"provider": "ais", "config": {"identity": "shared"}},
        "decode": {
            "provider": "ais",
            "config": {"identity": decode_identity},
        },
    }
    metadata_calls: list[tuple[dict[str, Any], str]] = []
    session_calls: list[dict[str, Any]] = []
    installed: list[dict[str, Any]] = []
    bootstrap_metadata: list[dict[str, Any]] = []

    class FakeAdapter:
        def set_bootstrap_metadata(self, value: dict[str, Any]) -> None:
            bootstrap_metadata.append(value)

        def install_benchmark_fpms(self, **kwargs: Any) -> None:
            installed.append(kwargs)

    monkeypatch.setattr(
        sims_module._replay_planner,
        "_ais_performance_model_configs",
        lambda received, mode: (
            metadata_calls.append((received, mode))
            or {
                "prefill": {"identity": "shared"},
                "decode": {"identity": decode_identity},
            }
        ),
    )
    monkeypatch.setattr(
        sims_module._replay_planner,
        "_ais_session_kwargs",
        lambda config, args: {"identity": config["identity"]},
    )
    monkeypatch.setattr(
        sims_module._replay_planner,
        "create_session",
        lambda **kwargs: session_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        sims_module,
        "_generate_ais_prefill_fpms",
        lambda session, args, granularity: ["prefill-fpm"],
    )
    monkeypatch.setattr(
        sims_module,
        "_generate_ais_decode_fpms",
        lambda session, args, granularity: ["decode-fpm"],
    )
    monkeypatch.setattr(
        sims_module._replay_planner,
        "_ais_fpm_digest",
        lambda prefill, decode: "fpm-digest",
    )

    engine = types.SimpleNamespace(install_regressions_from_fpms=lambda **kwargs: None)
    prefill_args = object()
    decode_args = object()
    sims_module._bootstrap_ais_regressions(
        FakeAdapter(),
        engine,
        types.SimpleNamespace(mode="disagg", optimization_target="sla"),
        extra_engine_args=None,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        performance_model_metadata=metadata,
        benchmark_granularity=4,
    )

    assert metadata_calls == [(metadata, "disagg")]
    assert len(session_calls) == expected_sessions
    assert installed == [
        {
            "prefill_fpms": ["prefill-fpm"],
            "decode_fpms": ["decode-fpm"],
        }
    ]
    assert bootstrap_metadata == [
        {"status": "not_required"},
        {
            "status": "installed",
            "benchmark_granularity": 4,
            "prefill_fpm_count": 1,
            "decode_fpm_count": 1,
            "fpm_sha256": "fpm-digest",
        },
    ]


@pytest.mark.parametrize("role", ["prefill", "decode", "aggregated"])
def test_canonical_performance_config_is_bound_to_deployed_role(sims_module, role):
    raw = {
        "ais_perf_config": {"model": "example", "system": "h200_sxm", "backend": "vllm"}
    }
    normalized = json.loads(
        sims_module._normalize_engine_args_role(
            json.dumps(raw), expected=role, argument_name="engine_args"
        )
    )
    assert normalized["ais_perf_config"]["worker_type"] == role
    assert normalized["worker_type"] == role
    assert "worker_type" not in raw["ais_perf_config"]


def test_conflicting_performance_model_worker_role_fails(sims_module):
    with pytest.raises(ValueError, match="ais_perf_config.worker_type"):
        sims_module._normalize_engine_args_role(
            '{"ais_perf_config": {"worker_type": "aggregated"}}',
            expected="prefill",
            argument_name="engine_args",
        )


def test_invalid_telemetry_destination_does_not_allocate_engine_resources(
    sims_module, monkeypatch, tmp_path
):
    config = types.SimpleNamespace(mode="agg", advisory=False)
    monkeypatch.setattr(sims_module, "_as_config", lambda value: config)
    monkeypatch.setattr(sims_module, "WorkerCapabilities", lambda **kwargs: object())
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n")

    with pytest.raises(ValueError, match="replay trace input"):
        sims_module.run_arena_replay(
            trace_file=str(trace),
            substrate_config={},
            autoscaler=lambda *args: pytest.fail(
                "invalid destinations must fail before creating an engine"
            ),
            telemetry_jsonl_path=str(trace),
        )
