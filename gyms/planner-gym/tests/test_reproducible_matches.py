# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoscaling_arena import match_runner
from autoscaling_arena.match_config import MatchConfigError, parse_match_config
from autoscaling_arena.substrates import Substrate

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def raw_match(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": t,
                    "input_length": 16,
                    "output_length": 2,
                    "hash_ids": [i],
                }
            )
            + "\n"
            for i, t in enumerate([1000, 2000, 3000])
        )
    )
    return {
        "schema_version": 1,
        "name": "controlled",
        "backend": {
            "type": "sim",
            "topology": "agg",
            "gpu_budget": 2,
            "model": {"name": "synthetic-model"},
            "engines": {
                "common": {
                    "system": "synthetic-system",
                    "backend": "vllm",
                    "extra_args": {
                        "timing_model": {
                            "type": "fixed",
                            "prefill_ms": 5,
                            "decode_ms": 1,
                        },
                        "num_gpu_blocks": 64,
                    },
                }
            },
            "replay": {"ais_bootstrap": False},
            "autoscalers": [
                {"name": "fixed", "type": "static", "config": {"num_decode": 1}}
            ],
        },
        "evaluations": {
            "traces": [
                {
                    "name": "input",
                    "path": str(source),
                    "format": "mooncake",
                    "block_size": 16,
                    "presorted": True,
                }
            ]
        },
        "slo_profiles": [{"name": "target", "ttft_ms": 100, "itl_ms": 10}],
        "metrics": {"short_output_itl": "skip"},
        "publish": {
            "artifact_root": str(tmp_path / "out"),
            "destinations": [{"type": "console"}],
        },
    }


def _parse(raw, tmp_path):
    return parse_match_config(raw, source_path=tmp_path / "match.yaml")


@pytest.mark.parametrize("mode", ["memory", "jsonl", "summary"])
def test_preparation_and_capture_modes_reach_scoring(
    raw_match, tmp_path, monkeypatch, mode
):
    raw_match["backend"]["replay"]["request_capture"] = mode
    raw_match["evaluations"]["traces"][0]["prepare"] = {
        "start_ms": 1000,
        "end_ms": 4000,
        "rebase": True,
        "request_scale": 2,
        "seed": 7,
    }
    if mode != "summary":
        raw_match["evaluations"]["defaults"] = {
            "measurement_window": {"start_s": 0, "end_s": 4}
        }
    config = _parse(raw_match, tmp_path)
    calls = []

    def replay(**kwargs):
        calls.append(kwargs)
        inputs = [
            json.loads(line)
            for line in Path(kwargs["trace_file"]).read_text().splitlines()
        ]
        assert len(inputs) == 6
        assert {row["timestamp"] for row in inputs} == {0, 1000, 2000}
        rows = [
            {
                "arrival_time_ms": row["timestamp"],
                "terminal_status": "completed",
                "ttft_ms": 5,
                "itl_ms": 1,
                "e2e_latency_ms": 6,
                "output_length": 2,
            }
            for row in inputs
        ]
        if mode == "jsonl":
            Path(kwargs["report_jsonl_path"]).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        return SimpleNamespace(
            trace_report={
                "num_requests": 6,
                "completed_requests": 6,
                "duration_ms": 3000,
                "gpu_hours": 3 / 3600,
                "goodput_completed_requests": 6,
                "goodput_request_throughput_rps": 2,
            },
            per_request=rows if mode == "memory" else None,
            timeline=[],
            scaling_events=[],
            planner=SimpleNamespace(lifecycle_operations=[]),
        )

    fake = types.ModuleType("autoscaling_arena.runners.sims")
    fake.run_arena_replay = replay
    monkeypatch.setitem(sys.modules, "autoscaling_arena.runners.sims", fake)
    monkeypatch.setattr(
        match_runner, "_build_sim_factory", lambda *args, **kwargs: object()
    )
    report = match_runner.execute_match_config(config)
    (result,) = report["results"]
    assert result["status"] == "ok", result
    assert result["raw_metrics"]["profiles"]["target"]["good_count"] == 6
    expected_rate = 2 if mode == "summary" else 1.5
    assert result["metrics"]["goodput_rps"] == expected_rate
    assert calls[0]["capture_per_request"] is (mode == "memory")
    assert calls[0]["substrate_config"]["ttft_ms"] == 100
    assert calls[0]["substrate_config"]["itl_ms"] == 10
    assert result["evaluation"]["trace"]["transformed"]
    assert result["evaluation"]["preparation"]["output"]["request_count"] == 6
    assert result["execution"]["setup_wall_s"] >= 0
    assert result["execution"]["replay_wall_s"] >= 0


def test_profile_conflicts_and_summary_policy_fail_before_replay(raw_match, tmp_path):
    raw_match["backend"]["autoscalers"] = [
        {"name": "planner", "type": "planner", "start": {"decode": 1}}
    ]
    raw_match["backend"]["planner_config"] = {"ttft_ms": 200}
    with pytest.raises(MatchConfigError, match="conflicts with SLO profile"):
        _parse(raw_match, tmp_path)
    raw_match["backend"].pop("planner_config")
    raw_match["backend"]["replay"]["request_capture"] = "summary"
    raw_match["metrics"]["short_output_itl"] = "fail"
    with pytest.raises(MatchConfigError, match="short_output_itl: skip"):
        _parse(raw_match, tmp_path)
    raw_match["metrics"]["short_output_itl"] = "skip"
    raw_match["evaluations"]["defaults"] = {"measurement_window": {"end_s": 4}}
    with pytest.raises(MatchConfigError, match="requires memory or jsonl"):
        _parse(raw_match, tmp_path)


@pytest.mark.parametrize("topology", ["agg", "disagg"])
@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ({}, 60.0),
        ({"cold_start_delay_s": 0}, 0.0),
        ({"cold_start_delay_s": 12.5}, 12.5),
    ],
)
def test_cold_start_defaults_and_explicit_overrides_reach_replay(
    raw_match, tmp_path, topology, runtime, expected
):
    backend = raw_match["backend"]
    backend["topology"] = topology
    backend["engines"]["common"]["runtime"] = runtime
    if topology == "disagg":
        backend["autoscalers"][0]["config"]["num_prefill"] = 1
    config = _parse(raw_match, tmp_path)
    for role in ("aggregate",) if topology == "agg" else ("prefill", "decode"):
        engine = getattr(config.backend.engines, role)
        assert engine.runtime.cold_start_delay_s == expected
        rendered = json.loads(
            match_runner._render_engine_args(engine, "synthetic-model")
        )
        assert rendered["startup_time"] == expected

    overrides = {"startup_time": runtime["cold_start_delay_s"]} if runtime else {}
    substrate = Substrate(
        "probe",
        "synthetic-model",
        "synthetic-system",
        "vllm",
        extra_engine_args=overrides,
    )
    assert json.loads(substrate.engine_args())["startup_time"] == expected


def test_external_timing_cannot_override_owned_identity(raw_match, tmp_path):
    config = _parse(raw_match, tmp_path)
    external = {
        "type": "external",
        "provider": "ais",
        "config": {"model": "other-model"},
    }
    raw_match["backend"]["engines"]["common"]["extra_args"]["timing_model"] = external
    with pytest.raises(MatchConfigError, match="configure AIS identity"):
        _parse(raw_match, tmp_path)
    engine = replace(
        config.backend.engines.aggregate, extra_args={"timing_model": external}
    )
    with pytest.raises(ValueError, match="AIS identity"):
        match_runner._render_engine_args(engine, "synthetic-model")
    with pytest.raises(ValueError, match="identity belongs"):
        Substrate(
            "probe",
            "synthetic-model",
            "synthetic-system",
            "vllm",
            extra_engine_args={"timing_model": external},
        ).engine_args()
    metadata = match_runner._sim_performance_model_metadata(config.backend)
    assert metadata["aggregated"]["provider"] == "fixed"


@pytest.mark.parametrize(
    "aliases",
    [
        {"aic_nextn": 1},
        {"ais_nextn": 1},
        {"aic_nextn": 1, "ais_nextn": 1},
        {"aic_nextn": None, "ais_nextn": 1},
    ],
)
def test_nextn_identity_matches_native_and_planner(raw_match, tmp_path, aliases):
    extra = raw_match["backend"]["engines"]["common"]["extra_args"]
    extra.pop("timing_model")
    extra.update(aliases)
    config = _parse(raw_match, tmp_path)
    engine = config.backend.engines.aggregate
    rendered = json.loads(match_runner._render_engine_args(engine, "synthetic-model"))
    native = rendered["engine"]
    metadata = match_runner._sim_performance_model_metadata(config.backend)
    assert native["aic_nextn"] == native["timing_model"]["config"]["nextn"] == 1
    assert metadata["aggregated"]["config"]["nextn"] == 1
    assert "ais_nextn" not in native


def test_conflicting_nextn_aliases_fail_before_replay(raw_match, tmp_path):
    raw_match["backend"]["engines"]["common"]["extra_args"].update(
        aic_nextn=1, ais_nextn=2
    )
    with pytest.raises(MatchConfigError, match="aic_nextn conflicts with ais_nextn"):
        _parse(raw_match, tmp_path)


def test_warmup_references_are_case_specific(raw_match, tmp_path, monkeypatch):
    raw_match["backend"]["autoscalers"] = [
        {"name": "planner", "type": "planner", "start": {"decode": 1}},
        {"name": "fixed", "type": "static", "config": {"num_decode": 1}},
    ]
    trace = raw_match["evaluations"]["traces"][0]
    trace["warmup"] = {
        "path": "input.jsonl",
        "prepare": {"start_ms": 0, "end_ms": 2000, "request_scale": 2},
    }
    config = _parse(raw_match, tmp_path)
    calls = []

    def replay(**kwargs):
        calls.append(kwargs)
        warmup = kwargs["substrate_config"].get("load_predictor_warmup_trace")
        if warmup is not None:
            assert len(Path(warmup).read_text().splitlines()) == 2
        return SimpleNamespace(trace_report={}, per_request=None, timeline=[])

    fake = types.ModuleType("autoscaling_arena.runners.sims")
    fake.run_arena_replay = replay
    monkeypatch.setitem(sys.modules, "autoscaling_arena.runners.sims", fake)
    monkeypatch.setattr(
        match_runner, "_build_sim_factory", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "autoscaling_arena.scorecard.scorecard",
        lambda *args, **kwargs: {"profiles": {"target": {"goodput_per_gpu": 1}}},
    )
    report = match_runner.execute_match_config(config)
    assert report["summary"]["succeeded_runs"] == 2
    assert "load_predictor_warmup_trace" in calls[0]["substrate_config"]
    assert "load_predictor_warmup_trace" not in calls[1]["substrate_config"]
    assert report["results"][0]["evaluation"]["warmup"]["applied"] is True
    assert report["results"][1]["evaluation"]["warmup"]["applied"] is False
