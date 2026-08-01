# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.custom_encoder.benchmark.run_parallel_encoder_sweep import (
    COMBINED_ROLE,
    ENCODER_ONLY_ROLE,
    _combined_service,
    _encoder_service,
)
from examples.custom_encoder.benchmark.run_three_arm_benchmark import (
    AGGREGATED_ARM,
    ARM_ORDER,
    PARALLEL_ARM,
    STANDALONE_ARM,
    _schedule_audit,
    summarize_three_arm,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_schedule_audit_records_mixed_random_order(tmp_path: Path) -> None:
    records = [
        {"path": "a", "width": 300, "height": 300},
        {"path": "b", "width": 300, "height": 300},
        {"path": "c", "width": 500, "height": 500},
        {"path": "d", "width": 500, "height": 500},
    ]
    (tmp_path / "workload_manifest.json").write_text(
        json.dumps({"images": records}), encoding="utf-8"
    )
    rows = [
        {"image": "a"},
        {"image": "c"},
        {"image": "b"},
        {"image": "d"},
    ]
    (tmp_path / "image_custom_4_isl644.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    audit = _schedule_audit(tmp_path, 4)

    assert audit["counts"] == {"300x300": 2, "500x500": 2}
    assert audit["transitions"] == 3
    assert audit["longest_same_size_run"] == 1
    assert len(audit["size_order_sha256"]) == 64


def test_services_accept_patch_aware_limits_and_log_destinations(
    tmp_path: Path,
) -> None:
    combined_log = tmp_path / "combined.log"
    encoder_log = tmp_path / "encoder.log"
    combined = _combined_service(
        tmp_path,
        AGGREGATED_ARM,
        max_batch_patches=41_472,
        max_batch_items=64,
        log_path=combined_log,
    )
    encoder = _encoder_service(
        tmp_path,
        STANDALONE_ARM,
        max_batch_patches=41_472,
        max_batch_items=64,
        log_path=encoder_log,
    )

    assert combined.env["DYN_QWEN2_VL_MAX_BATCH_PATCHES"] == "41472"
    assert encoder.env["DYN_QWEN2_VL_MAX_BATCH_PATCHES"] == "41472"
    assert combined.log_path == combined_log
    assert encoder.log_path == encoder_log


def _client(duration: float, throughput: float) -> dict[str, object]:
    return {
        "duration_s": duration,
        "aiperf": {"request_throughput": throughput},
    }


def test_summary_uses_average_parallel_min_and_max_wall_times(
    tmp_path: Path,
) -> None:
    repetitions = 5
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps({"repetitions": repetitions, "concurrency_per_client": 64}),
        encoding="utf-8",
    )
    aggregated = [10.0, 11.0, 9.0, 10.0, 10.0]
    standalone = [5.0, 5.5, 4.5, 5.0, 5.0]
    combined_parallel = [52.0, 54.0, 50.0, 51.0, 53.0]
    encoder_parallel = [48.0, 49.0, 47.0, 48.0, 48.0]
    for repetition in range(1, repetitions + 1):
        rows = {
            AGGREGATED_ARM: {
                "clients": {COMBINED_ROLE: _client(aggregated[repetition - 1], 100.0)},
                "min_wall_time_s": aggregated[repetition - 1],
                "max_wall_time_s": aggregated[repetition - 1],
            },
            STANDALONE_ARM: {
                "clients": {
                    ENCODER_ONLY_ROLE: _client(standalone[repetition - 1], 200.0)
                },
                "min_wall_time_s": standalone[repetition - 1],
                "max_wall_time_s": standalone[repetition - 1],
            },
            PARALLEL_ARM: {
                "clients": {
                    COMBINED_ROLE: _client(combined_parallel[repetition - 1], 19.0),
                    ENCODER_ONLY_ROLE: _client(encoder_parallel[repetition - 1], 21.0),
                },
                "min_wall_time_s": encoder_parallel[repetition - 1],
                "max_wall_time_s": combined_parallel[repetition - 1],
            },
        }
        for order_index, arm in enumerate(ARM_ORDER, start=1):
            row = {
                "arm": arm,
                "repetition": repetition,
                "order_index": order_index,
                "total_requests": 2_000 if arm == PARALLEL_ARM else 1_000,
                "start_skew_ms": 0.0,
                "completion_skew_ms": 0.0,
                **rows[arm],
            }
            cell = tmp_path / arm / f"run{repetition}"
            cell.mkdir(parents=True)
            (cell / "result.json").write_text(json.dumps(row), encoding="utf-8")

    report = summarize_three_arm(tmp_path)
    summaries = {row["arm"]: row for row in report["summaries"]}

    assert summaries[AGGREGATED_ARM]["avg_max_wall_time_s"] == 10.0
    assert summaries[AGGREGATED_ARM]["wall_throughput_lower_request_s"] == 100.0
    assert summaries[PARALLEL_ARM]["avg_min_wall_time_s"] == 48.0
    assert summaries[PARALLEL_ARM]["avg_max_wall_time_s"] == 52.0
    assert summaries[PARALLEL_ARM]["wall_throughput_lower_request_s"] == pytest.approx(
        1_000 / 52
    )
    assert summaries[PARALLEL_ARM]["wall_throughput_upper_request_s"] == pytest.approx(
        1_000 / 48
    )
    assert "does not use 2,000" in (tmp_path / "benchmark.md").read_text(
        encoding="utf-8"
    )
