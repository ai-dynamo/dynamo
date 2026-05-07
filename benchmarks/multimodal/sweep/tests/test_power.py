# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from benchmarks.multimodal.sweep.power import (
    GPU_TELEMETRY_FILENAME,
    POWER_SUMMARY_FILENAME,
    load_power_samples,
    summarize_power_telemetry,
    write_power_summary,
)


def _write_jsonl(path, rows) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_summarize_power_telemetry_integrates_per_gpu_energy(tmp_path):
    telemetry_path = tmp_path / GPU_TELEMETRY_FILENAME
    _write_jsonl(
        telemetry_path,
        [
            {
                "timestamp": 0.0,
                "gpus": [
                    {"gpu_index": 0, "power_watts": 100.0},
                    {"gpu_index": 1, "power_watts": 50.0},
                ],
            },
            {
                "timestamp": 1.0,
                "gpus": [
                    {"gpu_index": 0, "power_watts": 200.0},
                    {"gpu_index": 1, "power_watts": 50.0},
                ],
            },
            {
                "timestamp": 2.0,
                "gpus": [
                    {"gpu_index": 0, "power_watts": 100.0},
                    {"gpu_index": 1, "power_watts": 50.0},
                ],
            },
        ],
    )

    summary = summarize_power_telemetry(telemetry_path)

    assert summary["sample_count"] == 6
    assert summary["gpu_count"] == 2
    assert summary["duration_s"] == 2.0
    assert summary["total_energy_j"] == 400.0
    assert summary["average_power_w"] == 200.0
    assert summary["sum_peak_power_w"] == 250.0

    gpu0, gpu1 = summary["gpus"]
    assert gpu0["gpu_index"] == "0"
    assert gpu0["energy_j"] == 300.0
    assert gpu0["average_power_w"] == 150.0
    assert gpu0["peak_power_w"] == 200.0
    assert gpu1["gpu_index"] == "1"
    assert gpu1["energy_j"] == 100.0
    assert gpu1["average_power_w"] == 50.0


def test_load_power_samples_accepts_nvidia_smi_style_keys(tmp_path):
    telemetry_path = tmp_path / GPU_TELEMETRY_FILENAME
    _write_jsonl(
        telemetry_path,
        [
            {
                "timestamp": "2026-05-07T00:00:00Z",
                "index": "0",
                "power.draw [W]": "123.50 W",
                "utilization.gpu [%]": "87 %",
                "memory.used [MiB]": "1024 MiB",
            }
        ],
    )

    samples = load_power_samples(telemetry_path)

    assert len(samples) == 1
    assert samples[0].gpu_index == "0"
    assert samples[0].power_watts == 123.5
    assert samples[0].timestamp_s == pytest.approx(1778112000.0)
    assert samples[0].utilization_gpu_pct == 87.0
    assert samples[0].memory_used_mib == 1024.0


def test_write_power_summary_writes_json_when_telemetry_exists(tmp_path):
    telemetry_path = tmp_path / GPU_TELEMETRY_FILENAME
    _write_jsonl(
        telemetry_path,
        [
            {"timestamp": 0, "gpu_index": 0, "power_watts": 125},
            {"timestamp": 1, "gpu_index": 0, "power_watts": 125},
        ],
    )

    summary_path = write_power_summary(tmp_path)

    assert summary_path == tmp_path / POWER_SUMMARY_FILENAME
    written = json.loads(summary_path.read_text())
    assert written["total_energy_j"] == 125.0
    assert written["total_energy_wh"] == pytest.approx(0.034722, abs=0.000001)


def test_write_power_summary_skips_missing_telemetry(tmp_path):
    assert write_power_summary(tmp_path) is None
