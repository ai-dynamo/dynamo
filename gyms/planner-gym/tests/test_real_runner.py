# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint configuration and failure reporting without a live deployment."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from autoscaling_arena.runners.real import (
    Endpoint,
    EndpointMatchResult,
    build_aiperf_command,
    format_endpoint_leaderboard,
    load_endpoints,
    run_endpoint_leaderboard,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_endpoints_accepts_json_and_yaml(tmp_path: Path, suffix: str):
    path = tmp_path / f"endpoints{suffix}"
    path.write_text(
        json.dumps(
            {
                "endpoints": [
                    {
                        "name": "static",
                        "url": "http://localhost:8000",
                        "model": "example",
                    }
                ]
            }
        )
        if suffix == ".json"
        else "endpoints:\n  - name: static\n    url: http://localhost:8000\n    model: example\n"
    )
    endpoints = load_endpoints(path)
    assert len(endpoints) == 1
    assert endpoints[0].name == "static"
    assert endpoints[0].model == "example"


def test_committed_endpoint_example_loads():
    path = Path(__file__).resolve().parents[1] / "configs/endpoints.example.yaml"
    assert len(load_endpoints(path)) == 2


def test_failed_endpoint_without_summary_still_renders():
    failure = EndpointMatchResult(
        endpoint="failed",
        workload="flat",
        profile="interactive",
        metrics={},
        returncode=1,
        artifact_dir="artifacts",
    )
    completed = EndpointMatchResult(
        endpoint="completed",
        workload="flat",
        profile="interactive",
        metrics={"goodput_rps": 0.0, "mean_ttft_ms": 100.0},
        returncode=0,
        artifact_dir="artifacts",
    )
    text = format_endpoint_leaderboard([failure, completed])
    assert "[FAILED rc=1]" in text
    assert "n/a" in text
    assert text.index("completed") < text.index("failed")


@pytest.mark.parametrize("block_size", [16, 512])
def test_aiperf_command_uses_workload_block_size(tmp_path, block_size):
    command = build_aiperf_command(
        Endpoint("test", "http://unused", "example"),
        tmp_path / "trace.jsonl",
        tmp_path / "artifacts",
        block_size=block_size,
    )
    assert command[command.index("--isl-block-size") + 1] == str(block_size)


@pytest.mark.parametrize("block_size", [0, -1, True, 1.5])
def test_aiperf_command_rejects_invalid_block_size(tmp_path, block_size):
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        build_aiperf_command(
            Endpoint("test", "http://unused", "example"),
            tmp_path / "trace.jsonl",
            tmp_path / "artifacts",
            block_size=block_size,
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--isl-block-size",
        "--prompt-input-tokens-block-size",
        "--synthetic-input-tokens-block-size",
    ],
)
@pytest.mark.parametrize("equals", [False, True])
def test_aiperf_extra_args_cannot_replace_trace_block_size(tmp_path, flag, equals):
    extra_args = (f"{flag}=512",) if equals else (flag, "512")
    with pytest.raises(ValueError, match="cannot override the trace block_size"):
        build_aiperf_command(
            Endpoint("test", "http://unused", "example"),
            tmp_path / "trace.jsonl",
            tmp_path / "artifacts",
            block_size=16,
            extra_args=extra_args,
        )


def test_endpoint_sweep_keeps_results_after_timeout_and_missing_binary(
    tmp_path, monkeypatch
):
    from autoscaling_arena.runners import real
    from autoscaling_arena.scorecard import DEFAULT_PROFILES

    names = ["completed", "timed-out", "missing", "completed-after-failures"]
    commands = []

    def run(command, **kwargs):
        index = len(commands)
        commands.append(command)
        assert kwargs["timeout"] == 0.1
        if index == 1:
            raise subprocess.TimeoutExpired(command, 0.1, stderr=b"last diagnostic\n")
        if index == 2:
            raise FileNotFoundError("test executable unavailable")
        output = Path(command[command.index("--artifact-dir") + 1])
        (output / "profile_export_aiperf.json").write_text(
            json.dumps({"goodput": {"avg": 4.0}})
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(real.subprocess, "run", run)
    results = run_endpoint_leaderboard(
        endpoints=[Endpoint(name, "http://unused", "example") for name in names],
        workload_traces={"small-blocks": str(tmp_path / "trace.jsonl")},
        workload_block_sizes={"small-blocks": 16},
        profiles=(DEFAULT_PROFILES[0],),
        artifact_root=tmp_path / "artifacts",
        timeout_s=0.1,
    )
    assert [result.endpoint for result in results] == names
    assert [result.returncode for result in results] == [0, 124, 127, 0]
    assert (
        results[0].metrics["goodput_rps"] == results[-1].metrics["goodput_rps"] == 4.0
    )
    assert results[1].metrics == {"_timeout_s": 0.1}
    assert "last diagnostic" in results[1].stderr_tail
    assert "executable not found" in results[2].stderr_tail
    assert all(
        command[command.index("--isl-block-size") + 1] == "16" for command in commands
    )


def test_endpoint_sweep_uses_each_trace_block_size(tmp_path, monkeypatch):
    from autoscaling_arena.runners import real
    from autoscaling_arena.scorecard import DEFAULT_PROFILES

    sizes = []

    def run(command, **kwargs):
        sizes.append(command[command.index("--isl-block-size") + 1])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(real.subprocess, "run", run)
    run_endpoint_leaderboard(
        endpoints=[Endpoint("test", "http://unused", "example")],
        workload_traces={"small": "small.jsonl", "large": "large.jsonl"},
        workload_block_sizes={"small": 16, "large": 512},
        profiles=(DEFAULT_PROFILES[0],),
        artifact_root=tmp_path,
    )
    assert sizes == ["16", "512"]


def test_endpoint_cli_rejects_unknown_profiles_before_loading_endpoints(
    monkeypatch, capsys
):
    script = Path(__file__).resolve().parents[1] / "scripts/run_endpoint_bench.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--endpoints", "missing.json", "--profiles", "interactiv"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "invalid choice: 'interactiv'" in capsys.readouterr().err


def test_endpoint_cli_saves_completed_and_failed_matches(tmp_path, monkeypatch):
    from autoscaling_arena import workloads
    from autoscaling_arena.runners import real
    from autoscaling_arena.workloads.axes import NoSharing, flat

    script = Path(__file__).resolve().parents[1] / "scripts/run_endpoint_bench.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "_REPO", tmp_path)
    endpoints_path = tmp_path / "endpoints.json"
    endpoints_path.write_text(
        json.dumps(
            [
                {"name": "success", "url": "http://unused", "model": "example"},
                {"name": "timeout", "url": "http://unused", "model": "example"},
            ]
        )
    )
    workload = workloads.Workload(
        name="small",
        description="test",
        block_size=16,
        duration_s=1,
        arrival=flat(20),
        shape=lambda rng: (32, 2),
        prefix_factory=NoSharing,
    )
    monkeypatch.setattr(workloads, "get_workload", lambda name: workload)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--isl-block-size") + 1] == "16"
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(command, 0.1)
        output = Path(command[command.index("--artifact-dir") + 1])
        (output / "profile_export_aiperf.json").write_text('{"goodput": {"avg": 2}}')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(real.subprocess, "run", run)
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--endpoints",
            str(endpoints_path),
            "--workloads",
            "small",
            "--profiles",
            "interactive",
            "--timeout-s",
            "0.1",
            "--out",
            str(output),
        ],
    )
    assert main() == 1
    results = json.loads(output.read_text())
    assert len(results) == 2
    assert results[0]["metrics"]["goodput_rps"] == 2
    assert results[1]["returncode"] == 124
