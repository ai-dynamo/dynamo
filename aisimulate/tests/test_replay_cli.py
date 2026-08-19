# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-only single-run CLI and shared parser contracts."""

from __future__ import annotations

import json

import pytest

from aisimulate.replay import cli
from aisimulate.sweeper.replay import ReplayReport

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class _RecordingRunner:
    def __init__(self) -> None:
        self.spec = None
        self.output_requirements = None

    def run(self, spec, *, output_requirements=None):
        self.spec = spec
        self.output_requirements = output_requirements
        return ReplayReport(
            metrics={"completed_requests": 1.0},
            metadata={
                "native_report": {
                    "summary": {
                        "completed_requests": 1,
                        "output_throughput_tok_s": 8.0,
                    },
                    "per_request": [{"request_id": "synthetic-0"}],
                }
            },
        )


class _RecordingFactory:
    def __init__(self, runner: _RecordingRunner) -> None:
        self.runner = runner

    def create(self, worker_id: int) -> _RecordingRunner:
        assert worker_id == 0
        return self.runner


def _engine_args() -> str:
    return json.dumps(
        {
            "engine_type": "vllm",
            "num_gpu_blocks": 32,
            "block_size": 4,
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 2.0,
                "decode_ms": 1.0,
            },
        }
    )


def test_engine_cli_runs_shared_synthetic_schema_and_writes_outputs(
    monkeypatch, tmp_path, capsys
) -> None:
    runner = _RecordingRunner()
    monkeypatch.setattr(
        cli,
        "EngineReplayRunnerFactory",
        lambda: _RecordingFactory(runner),
    )
    report_path = tmp_path / "report.json"
    per_request_path = tmp_path / "requests.jsonl"

    assert (
        cli.main(
            [
                "--extra-engine-args",
                _engine_args(),
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--report-json",
                str(report_path),
                "--per-request-jsonl",
                str(per_request_path),
            ]
        )
        == 0
    )

    assert runner.spec.backend_deployment.deployment_mode == "agg"
    assert runner.spec.workload["request_count"] == 1
    assert runner.output_requirements.include_raw_report is True
    assert runner.output_requirements.capture_per_request is True
    assert json.loads(report_path.read_text()) == {
        "summary": {
            "completed_requests": 1,
            "output_throughput_tok_s": 8.0,
        },
        "per_request": [{"request_id": "synthetic-0"}],
    }
    assert json.loads(per_request_path.read_text()) == {"request_id": "synthetic-0"}
    output = capsys.readouterr().out
    assert "NVIDIA AIPerf | LLM Metrics" in output
    assert f"Saved full report to: {report_path}" in output


def test_engine_cli_rejects_dynamo_only_online_mode() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "--extra-engine-args",
                _engine_args(),
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--replay-mode",
                "online",
            ]
        )


def test_engine_cli_has_no_dynamo_extensions() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.build_parser().parse_args(["--router-mode", "kv_router"])
