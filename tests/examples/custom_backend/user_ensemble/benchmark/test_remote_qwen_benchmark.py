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


def test_summary_reports_both_throughput_boundaries_and_no_gate(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "benchmark_metadata.json",
        {
            "dynamo_commit": "abc123",
            "container_image": "example/runtime:test",
            "gpu": {"name": "NVIDIA H100 80GB HBM3"},
        },
    )
    _write_json(
        tmp_path / "workload_audit.json",
        {"measured_sha256": "audited-workload"},
    )
    for repetition in range(1, 4):
        _write_json(
            tmp_path / f"rep-{repetition}/inline/cell_audit.json",
            _cell(10.0, 100.0),
        )
        _write_json(
            tmp_path / f"rep-{repetition}/remote/cell_audit.json",
            _cell(20.0, 50.0),
        )
        _write_json(
            tmp_path / f"rep-{repetition}/remote/joined_smoke.json",
            {
                "classifier_scores": {"positive-mean": 0.5, "negative-mean": 0.5},
                "classifier_score_sum": 1.0,
            },
        )

    result = summarize(tmp_path)

    assert result["comparison"] == {
        "remote_to_inline_full_client_process_ratio": 0.5,
        "remote_full_client_process_delta_percent": -50.0,
        "remote_to_inline_request_window_ratio": 0.5,
        "remote_request_window_delta_percent": -50.0,
    }
    assert "pass" not in result["comparison"]
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()
