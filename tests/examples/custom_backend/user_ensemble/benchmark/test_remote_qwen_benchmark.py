# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.custom_backend.user_ensemble.benchmark.remote_qwen_benchmark import (
    BenchmarkAuditError,
    audit_encoder_log,
    summarize,
    validate_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_encoder_log() -> str:
    capture_lines = "\n".join(
        f"captured CUDA graph: grid=(1, 22, 22) bucket={index}"
        for index in range(1, 15)
    )
    return "\n".join(
        [
            capture_lines,
            "CUDA graph capture complete: grids=[] buckets=[] graphs=14",
            "custom_encoder_dispatch mode=graph batch_size=510 bucket=64 "
            "grid=1x22x22 patch_cost=246840 padded_patch_cost=0",
            "custom_encoder_dispatch mode=graph batch_size=510 bucket=64 "
            "grid=1x36x36 patch_cost=660960 padded_patch_cost=0",
        ]
    )


def test_encoder_log_audits_patches_grids_and_graphs(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(_valid_encoder_log(), encoding="utf-8")

    audit = audit_encoder_log(server_log)

    assert audit == {
        "dispatch_calls": 2,
        "dispatch_modes": {"graph": 2},
        "patch_cost": 907800,
        "grids": ["1x22x22", "1x36x36"],
        "captured_graphs": 14,
    }


def test_encoder_log_rejects_eager_dispatch(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        _valid_encoder_log()
        + "\ncustom_encoder_dispatch mode=eager batch_size=1 bucket=None "
        "patch_cost=1 padded_patch_cost=1 grids=[(1, 22, 22)]\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkAuditError, match="eager dispatch"):
        audit_encoder_log(server_log)


def test_profile_validation_supports_warmup_and_measured_counts(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.json"
    _write_json(
        profile,
        {
            "error_summary": [],
            "was_cancelled": False,
            "request_count": {"avg": 20},
            "input_sequence_length": {"avg": 874.5},
            "output_sequence_length": {"avg": 7},
            "request_throughput": {"avg": 42.0},
            "output_token_throughput": {"avg": 294.0},
            "request_latency": {
                "avg": 10.0,
                "p50": 9.0,
                "p95": 12.0,
                "p99": 14.0,
                "max": 20.0,
            },
        },
    )

    result = validate_profile(profile, expected_requests=20)

    assert result["request_count"] == 20
    with pytest.raises(BenchmarkAuditError, match="exactly 1000 requests"):
        validate_profile(profile, expected_requests=1000)


def test_runner_waits_for_real_generation_without_touching_encoder_counts() -> None:
    runner = (
        Path(__file__).parents[5]
        / "examples/custom_backend/user_ensemble/benchmark/run_qwen_comparison.sh"
    ).read_text(encoding="utf-8")

    assert 'wait_control_plane "$output_dir"\n    wait_generation_ready' in runner
    readiness = runner.split("wait_generation_ready()", maxsplit=1)[1].split(
        "run_cell()", maxsplit=1
    )[0]
    assert '"content":"readiness"' in readiness
    assert "image_url" not in readiness
    assert 'RUN_ID="$(printf' in runner
    assert 'rm -f "${zmq_prefix}' not in runner


def _cell(wall_seconds: float, request_throughput: float) -> dict:
    return {
        "full_client_process_wall_s": wall_seconds,
        "full_client_process_throughput_req_s": 1000 / wall_seconds,
        "aiperf": {
            "request_window_throughput_req_s": request_throughput,
            "request_latency_ms": {
                "avg": 10.0,
                "p50": 9.0,
                "p95": 12.0,
                "p99": 14.0,
                "max": 20.0,
            },
        },
        "gpu": {
            "utilization_percent_mean": 75.0,
            "memory_used_mib_max": 40_000.0,
        },
    }


def test_summary_reports_each_rate_and_workflow_gate(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "benchmark_metadata.json",
        {
            "dynamo_commit": "abc123",
            "container_image": "example/runtime:test",
            "gpu": {"name": "NVIDIA H100 80GB HBM3"},
            "benchmark": {
                "request_rates": [40, 50],
                "concurrency_cap": 512,
            },
        },
    )
    _write_json(
        tmp_path / "workload_audit.json",
        {"measured_sha256": "audited-workload"},
    )
    for request_rate in (40, 50):
        for repetition in range(1, 4):
            _write_json(
                tmp_path
                / f"rate-{request_rate}/rep-{repetition}/direct/cell_audit.json",
                _cell(10.0, float(request_rate)),
            )
            _write_json(
                tmp_path
                / f"rate-{request_rate}/rep-{repetition}/workflow/cell_audit.json",
                _cell(10.5, request_rate * 0.95),
            )
            _write_json(
                tmp_path
                / f"rate-{request_rate}/rep-{repetition}/workflow/joined_smoke.json",
                {
                    "classifier_scores": {
                        "relevant": 0.75,
                        "not_relevant": 0.25,
                    },
                    "classifier_score_sum": 1.0,
                },
            )

    result = summarize(tmp_path)

    assert result["rates"]["40"]["comparison"] == {
        "workflow_to_direct_request_window_ratio": 0.95,
        "workflow_request_window_delta_percent": pytest.approx(-5.0),
        "minimum_ratio": 0.9,
        "passed": True,
    }
    assert result["rates"]["50"]["comparison"]["passed"] is True
    assert result["gate"] == {
        "minimum_workflow_to_direct_ratio": 0.9,
        "passed": True,
    }
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()
    report = (tmp_path / "report.md").read_text()
    assert "rates: 40/50 req/s; concurrency cap: 512" in report
    assert "Overall >=90% gate: **True**" in report
