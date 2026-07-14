# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.custom_encoder.benchmark.summarize_results import _load_rows, _markdown
from examples.custom_encoder.benchmark.validate_results import validate_matrix

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

RUNTIMES = ("vllm-serve", "dynamo-native", "dynamo-custom-encoder")
RATES = (16, 24, 32)


def _latency(value: float) -> dict[str, float | str]:
    return {
        "unit": "ms",
        "avg": value,
        "p50": value,
        "p90": value + 1,
        "p99": value + 2,
    }


def _write_result(root: Path, runtime: str, rate: int) -> None:
    artifact = root / f"input-qps{rate}" / runtime / f"request_rate{rate}"
    artifact.mkdir(parents=True)
    document = {
        "request_count": {"avg": 1000},
        "error_summary": [],
        "was_cancelled": False,
        "input_config": {
            "endpoint": {"streaming": True},
            "loadgen": {"request_rate": rate},
            "cli_command": "aiperf profile --random-seed 42",
        },
        "input_sequence_length": {"min": 515, "avg": 515, "max": 515},
        "output_sequence_length": {"min": 70, "avg": 70, "max": 70},
        "time_to_first_token": _latency(float(rate)),
        "request_latency": _latency(float(rate * 10)),
        "request_throughput": {"avg": float(rate)},
    }
    (artifact / "profile_export_aiperf.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    (artifact / "command.txt").write_text("aiperf profile\n", encoding="utf-8")


def test_validation_and_markdown_cover_nine_cells(tmp_path: Path) -> None:
    for runtime in RUNTIMES:
        for rate in RATES:
            _write_result(tmp_path, runtime, rate)
    metadata = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "dynamo_commit": "abc123",
        "container_image": "test-image",
        "vllm_version": "test",
        "aiperf_version": "0.8.0",
        "gpu": "H100",
        "custom_encoder_class": "examples.custom_encoder.Qwen3VLVisionEncoder",
        "custom_encoder_load": "retains model.visual",
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    assert len(validate_matrix(tmp_path)) == 9
    markdown = _markdown(tmp_path, _load_rows(tmp_path))
    assert "=== TTFT avg (ms) ===" in markdown
    assert "=== E2E latency p99 (ms) ===" in markdown
    assert "=== Throughput (req/s) ===" in markdown
    assert markdown.count("[artifact]") == 9
    assert markdown.count("[command]") == 9
