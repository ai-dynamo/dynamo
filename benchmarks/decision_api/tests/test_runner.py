# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import time
from unittest.mock import Mock

import pytest
from dynamo_decision_perf.runner import (
    RunSpec,
    build_command,
    campaign_points,
    execute,
    freeze_run,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_screening_is_bounded_and_public_not_streaming(tmp_path):
    spec = RunSpec("oai", "http://127.0.0.1:8000", "model", concurrency=4, requests=16)
    cmd = build_command(spec, tmp_path / "input.json", tmp_path / "out")
    assert cmd[cmd.index("--endpoint-type") + 1] == "decision_oai"
    assert "--streaming" not in cmd
    assert "--no-gpu-telemetry" in cmd
    assert cmd[cmd.index("--custom-dataset-type") + 1] == "inputs_json"
    assert cmd[cmd.index("--export-level") + 1] == "raw"


def test_native_scoring_stream_is_not_token_generation(tmp_path):
    cmd = build_command(
        RunSpec("native_score", "http://localhost:8000", "m"),
        tmp_path / "i",
        tmp_path / "o",
    )
    assert "--streaming" in cmd
    assert "--synthetic-output-tokens-mean" not in cmd


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dialect": "chat"},
        {"url": "https://example.com"},
        {"duration": 1801},
        {"duration": float("nan")},
        {"concurrency": 0},
        {"rate": 0},
        {"warmup": -1},
        {"requests": 0},
        {"gpu": True},
        {"gpu": True, "approved_gpu_hours": 5},
    ],
)
def test_unsafe_specs_rejected(kwargs):
    values = {"dialect": "oai", "url": "http://localhost:8000", "model": "m", **kwargs}
    with pytest.raises(ValueError):
        RunSpec(**values)


def test_sustained_load_is_open_loop(tmp_path):
    spec = RunSpec(
        "systemone",
        "http://localhost:8000",
        "m",
        requests=None,
        duration=300,
        warmup=30,
        rate=1.5,
    )
    cmd = build_command(spec, tmp_path / "i", tmp_path / "o")
    assert "--concurrency" not in cmd
    assert cmd[cmd.index("--request-rate") + 1] == "1.5"
    assert "--request-count" not in cmd


def test_freeze_is_immutable_and_records_payload_hash(tmp_path):
    dataset = tmp_path / "input.json"
    dataset.write_text('{"data": []}')
    profile = {"kind": "cpu_instrument", "series_id": "cpu"}
    spec = RunSpec("oai", "http://localhost:8000", "m")
    frozen = freeze_run(tmp_path / "run", spec, dataset, profile)
    assert len(frozen["workload_sha256"]) == 64
    assert (
        json.loads((tmp_path / "run" / "manifest.json").read_text())["profile"]
        == profile
    )
    with pytest.raises(FileExistsError):
        freeze_run(tmp_path / "run", spec, dataset, profile)


def test_campaign_points_preserve_screening_and_sustained_rules():
    points = campaign_points(2.0)
    assert [
        (p["concurrency"], p["requests"])
        for p in points
        if p["campaign"] == "screening"
    ] == [(1, 4), (2, 8), (4, 16), (8, 32)]
    assert [p["rate"] for p in points if p["campaign"] == "sustained"] == [
        0.5,
        1,
        1.5,
        2,
        2.5,
    ]
    assert all(
        p["warmup"] == 30 and p["duration"] == 300
        for p in points
        if p["campaign"] == "sustained"
    )
    with pytest.raises(ValueError):
        campaign_points(0)


def test_execution_binds_manifest_and_utc(tmp_path):
    dataset = tmp_path / "input.json"
    dataset.write_text('{"data": []}')
    run = tmp_path / "run"
    spec = RunSpec("oai", "http://localhost:8000", "m")
    freeze_run(run, spec, dataset, {"kind": "cpu_instrument"})
    assert (
        execute(run, spec, "/bin/true", client_cpus=(min(os.sched_getaffinity(0)),))
        == 0
    )
    evidence = json.loads((run / "benchmark_execution.json").read_text())
    assert evidence["export_timezone_offset_seconds"] == 0
    assert evidence["manifest_sha256"] == (run / "manifest.sha256").read_text().strip()


def test_budget_cannot_be_extended_by_supplied_deadline(tmp_path):
    spec = RunSpec(
        "oai", "http://localhost:8000", "m", gpu=True, approved_gpu_hours=0.1
    )
    with pytest.raises(ValueError, match="exceeds"):
        execute(
            tmp_path,
            spec,
            "/bin/true",
            budget_deadline=time.time() + 3600,
            budget_started=time.time() - 10,
            client_cpus=(min(os.sched_getaffinity(0)),),
        )
    with pytest.raises(ValueError, match="active"):
        execute(
            tmp_path, spec, "/bin/true", client_cpus=(min(os.sched_getaffinity(0)),)
        )


def test_gpu_freeze_rejects_unqualified_or_relaxed_scheduler(tmp_path):
    spec = RunSpec("oai", "http://localhost:8000", "m", gpu=True, approved_gpu_hours=1)
    with pytest.raises(ValueError, match="missing fields"):
        freeze_run(tmp_path / "r", spec, tmp_path / "i", {})


@pytest.mark.parametrize("defect", ["spec", "workload", "manifest"])
def test_execute_rejects_changed_frozen_inputs(tmp_path, defect):
    dataset = tmp_path / "input.json"
    dataset.write_text('{"data": []}')
    run = tmp_path / "run"
    spec = RunSpec("oai", "http://localhost:8000", "m")
    freeze_run(run, spec, dataset, {"kind": "cpu_instrument"})
    if defect == "spec":
        spec = RunSpec("oai", "http://localhost:8001", "m")
    elif defect == "workload":
        (run / "workload.json").write_text("{}")
    else:
        (run / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        execute(run, spec, "/bin/true")


def test_telemetry_failure_cleans_owned_process_group(tmp_path, monkeypatch):
    from dynamo_decision_perf import runner

    dataset = tmp_path / "input.json"
    dataset.write_text('{"data": []}')
    run = tmp_path / "run"
    spec = RunSpec("oai", "http://localhost:8000", "m")
    freeze_run(run, spec, dataset, {"kind": "cpu_instrument"})
    process = Mock(pid=1234)
    process.poll.return_value = None
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(runner.os, "killpg", kill)
    monkeypatch.setattr(runner, "record_sample", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        execute(run, spec, "/bin/true")
    kill.assert_called_once_with(1234, runner.signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=10)


def test_sigterm_cleans_owned_process_and_records_interrupted_run(
    tmp_path, monkeypatch
):
    from dynamo_decision_perf import runner

    dataset = tmp_path / "input.json"
    dataset.write_text('{"data": []}')
    run = tmp_path / "run"
    spec = RunSpec("oai", "http://localhost:8000", "m")
    freeze_run(run, spec, dataset, {"kind": "cpu_instrument"})
    process = Mock(pid=1234)
    process.poll.return_value = None
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(runner.os, "killpg", kill)
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(*args):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(runner, "record_sample", terminate)
    assert execute(run, spec, "/bin/true") == 128 + signal.SIGTERM
    kill.assert_called_once_with(1234, signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=10)
    assert signal.getsignal(signal.SIGTERM) == previous
    record = json.loads((run / "benchmark_execution.json").read_text())
    assert record["exit_code"] == 128 + signal.SIGTERM
