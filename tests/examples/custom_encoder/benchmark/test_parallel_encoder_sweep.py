# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the paired custom-encoder benchmark harness."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from examples.custom_encoder.benchmark.run_parallel_encoder_sweep import (
    COMBINED_ROLE,
    CONTROL_ARM,
    ENCODER_ONLY_ROLE,
    PARALLEL_ARM,
    ProcessResult,
    _aiperf_command,
    _arm_summary,
    _patch_sweep_dispatch_summary,
    _patch_sweep_encoder_service,
    _write_timing,
    audit_pool_disjointness,
    run_barrier_pair,
    validate_aiperf,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_aiperf_command_uses_explicit_external_warmup(tmp_path: Path) -> None:
    command = _aiperf_command(
        role=COMBINED_ROLE,
        port=8000,
        concurrency=8,
        requests=1000,
        input_file=tmp_path / "inputs.json",
        artifact_dir=tmp_path / "artifacts",
    )

    assert "--warmup-request-count" not in command
    assert "--use-server-token-count" in command
    zmq_dir = command[command.index("--zmq-ipc-path") + 1]
    assert zmq_dir.startswith("/tmp/aiperf-combined-")
    assert len(f"{zmq_dir}/event_bus_proxy_frontend.ipc") <= 107


def test_patch_sweep_service_separates_patch_and_item_limits(tmp_path: Path) -> None:
    service = _patch_sweep_encoder_service(tmp_path, 41_472, 32)

    assert service.env["DYN_QWEN2_VL_MAX_BATCH_PATCHES"] == "41472"
    assert service.env["DYN_QWEN2_VL_MAX_BATCH_ITEMS"] == "32"
    assert service.env["DYN_QWEN2_VL_GRAPH_IMAGE_SIZES"] == "300x300,500x500"


def test_patch_sweep_dispatch_summary_reconciles_padded_work(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "CUDA graph capture complete: device_memory_delta_gib=20.250\n"
        "custom_encoder_dispatch mode=graph batch_size=31 bucket=32 "
        "grid=1x36x36 patch_cost=40176 padded_patch_cost=41472\n"
        "custom_encoder_dispatch mode=graph batch_size=2 bucket=2 "
        "grid=1x22x22 patch_cost=968 padded_patch_cost=968\n",
        encoding="utf-8",
    )

    result = _patch_sweep_dispatch_summary(log_path)

    assert result["calls"] == 2
    assert result["actual_patches_including_warmup"] == 41_144
    assert result["padded_patches_including_warmup"] == 42_440
    assert result["capture_memory_delta_gib"] == 20.25


def _write_manifest(root: Path, encoded: str, decoded: str) -> None:
    root.mkdir(parents=True)
    (root / "workload_manifest.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "encoded_sha256": encoded,
                        "decoded_rgb_sha256": decoded,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_pool_audit_accepts_disjoint_hashes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_manifest(first, "encoded-a", "decoded-a")
    _write_manifest(second, "encoded-b", "decoded-b")

    result = audit_pool_disjointness([first, second])

    assert result == {
        "pools": 2,
        "images": 2,
        "unique_encoded_sha256": 2,
        "unique_decoded_rgb_sha256": 2,
    }


def test_pool_audit_rejects_cross_pool_reuse(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_manifest(first, "encoded-a", "decoded-a")
    _write_manifest(second, "encoded-a", "decoded-b")

    with pytest.raises(AssertionError, match="overlap"):
        audit_pool_disjointness([first, second])


def test_barrier_pair_releases_clients_together(tmp_path: Path) -> None:
    commands = {
        COMBINED_ROLE: [sys.executable, "-c", "import time; time.sleep(0.15)"],
        ENCODER_ONLY_ROLE: [
            sys.executable,
            "-c",
            "import time; time.sleep(0.15)",
        ],
    }
    artifacts = {
        COMBINED_ROLE: tmp_path / COMBINED_ROLE,
        ENCODER_ONLY_ROLE: tmp_path / ENCODER_ONLY_ROLE,
    }

    started = time.monotonic()
    results = run_barrier_pair(commands, artifacts)
    elapsed = time.monotonic() - started

    releases = [result.released_ns for result in results.values()]
    assert elapsed < 0.5
    assert (max(releases) - min(releases)) / 1_000_000 < 100
    assert all(result.returncode == 0 for result in results.values())
    assert all((path / "command.txt").is_file() for path in artifacts.values())


def test_barrier_pair_stops_peer_after_failure(tmp_path: Path) -> None:
    commands = {
        COMBINED_ROLE: [sys.executable, "-c", "raise SystemExit(7)"],
        ENCODER_ONLY_ROLE: [sys.executable, "-c", "import time; time.sleep(5)"],
    }
    artifacts = {
        COMBINED_ROLE: tmp_path / COMBINED_ROLE,
        ENCODER_ONLY_ROLE: tmp_path / ENCODER_ONLY_ROLE,
    }

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="combined=exit7"):
        run_barrier_pair(commands, artifacts)

    assert time.monotonic() - started < 2


def test_joint_timing_waits_for_both_parallel_clients(tmp_path: Path) -> None:
    results = {
        COMBINED_ROLE: ProcessResult(
            COMBINED_ROLE,
            0,
            1_000_000_000,
            11_000_000_000,
            ["combined"],
            tmp_path / COMBINED_ROLE,
        ),
        ENCODER_ONLY_ROLE: ProcessResult(
            ENCODER_ONLY_ROLE,
            0,
            1_010_000_000,
            21_000_000_000,
            ["encoder"],
            tmp_path / ENCODER_ONLY_ROLE,
        ),
    }

    timing = _write_timing(PARALLEL_ARM, 8, 1, tmp_path, results)

    assert timing["joint_duration_s"] == 20.0
    assert timing["client_completion_rate_min_request_s"] == pytest.approx(1000 / 19.99)
    assert timing["client_completion_rate_max_request_s"] == 100.0
    assert timing["start_skew_ms"] == 10.0
    assert timing["completion_skew_ms"] == 10_000.0


def _metric(value: float) -> dict[str, float]:
    return {
        "min": value,
        "avg": value,
        "max": value,
        "p50": value,
        "p95": value,
        "p99": value,
    }


def test_aiperf_validation_distinguishes_dummy_output(tmp_path: Path) -> None:
    artifact = tmp_path / "encoder"
    artifact.mkdir()
    command = (
        "aiperf profile --concurrency 8 --conversation-num 1000 --streaming "
        "--zmq-ipc-path /tmp/encoder"
    )
    (artifact / "command.txt").write_text(command + "\n", encoding="utf-8")
    document = {
        "request_count": {"avg": 1000},
        "error_summary": [],
        "was_cancelled": False,
        "input_config": {
            "endpoint": {"streaming": True},
            "loadgen": {"concurrency": 8},
        },
        # AIPerf uses the text tokenizer for the encoder-only service. The
        # workload manifest separately audits the multimodal processor ISL 644.
        "input_sequence_length": _metric(312),
        "output_sequence_length": _metric(1),
        "request_throughput": {"avg": 50},
        "request_latency": _metric(20),
        "time_to_first_token": _metric(20),
    }
    result_path = artifact / "profile_export_aiperf.json"
    result_path.write_text(json.dumps(document), encoding="utf-8")

    result = validate_aiperf(result_path, ENCODER_ONLY_ROLE, 8)

    assert result["accepted"]
    assert result["input_sequence_length_avg"] == 312
    assert result["request_throughput"] == 50


def test_mixed_shape_aiperf_allows_variable_text_token_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "encoder"
    artifact.mkdir()
    (artifact / "command.txt").write_text(
        "aiperf profile --concurrency 64 --streaming\n", encoding="utf-8"
    )
    document = {
        "request_count": {"avg": 1000},
        "error_summary": [],
        "was_cancelled": False,
        "input_config": {
            "endpoint": {"streaming": True},
            "loadgen": {"concurrency": 64},
        },
        "input_sequence_length": {"min": 320, "avg": 421.5, "max": 523},
        "output_sequence_length": _metric(1),
        "request_throughput": {"avg": 50},
        "request_latency": _metric(20),
        "time_to_first_token": _metric(20),
    }
    result_path = artifact / "profile_export_aiperf.json"
    result_path.write_text(json.dumps(document), encoding="utf-8")

    result = validate_aiperf(
        result_path,
        ENCODER_ONLY_ROLE,
        64,
        require_fixed_client_isl=False,
    )

    assert result["accepted"]


def test_arm_summary_uses_median_across_confirmations() -> None:
    timings = [
        {
            "arm": CONTROL_ARM,
            "concurrency_per_client": 16,
            "joint_duration_s": 1000 / throughput,
            "clients": {COMBINED_ROLE: {"wall_throughput_request_s": throughput}},
        }
        for throughput in (90.0, 100.0, 130.0)
    ]

    result = _arm_summary(timings, CONTROL_ARM)

    assert result[0]["runs"] == 3
    assert result[0]["median_min_completion_rate"] == 100.0
    assert result[0]["representative_min_completion_rate"] == 100.0
    assert result[0]["representative_max_completion_rate"] == 100.0


def test_arm_summary_reports_per_client_completion_range() -> None:
    timings = [
        {
            "arm": PARALLEL_ARM,
            "concurrency_per_client": 64,
            "joint_duration_s": 52.0,
            "clients": {
                COMBINED_ROLE: {"wall_throughput_request_s": 1000 / 52},
                ENCODER_ONLY_ROLE: {"wall_throughput_request_s": 1000 / 48},
            },
        }
    ]

    result = _arm_summary(timings, PARALLEL_ARM)

    assert result[0]["representative_min_completion_rate"] == pytest.approx(1000 / 52)
    assert result[0]["representative_max_completion_rate"] == pytest.approx(1000 / 48)
