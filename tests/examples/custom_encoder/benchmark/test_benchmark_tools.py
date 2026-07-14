# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.custom_encoder.benchmark.run_ablation import VARIANTS
from examples.custom_encoder.benchmark.summarize_ablation import (
    _load_rows as _load_ablation_rows,
)
from examples.custom_encoder.benchmark.summarize_ablation import (
    _markdown as _ablation_markdown,
)
from examples.custom_encoder.benchmark.summarize_results import _load_rows, _markdown
from examples.custom_encoder.benchmark.validate_ablation import validate_ablation
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
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dynamo_commit": "abc123",
        "container_image": "test-image",
        "vllm_version": "test",
        "aiperf_version": "0.8.0",
        "gpu": "H100",
        "cuda_visible_devices": "0",
        "custom_encoder_class": (
            "examples.custom_encoder.qwen2_vl_vision_encoder." "Qwen2VLVisionEncoder"
        ),
        "custom_encoder_load": "retains model.visual",
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    assert len(validate_matrix(tmp_path)) == 9
    markdown = _markdown(tmp_path, _load_rows(tmp_path))
    assert "=== TTFT avg (ms) ===" in markdown
    assert "=== E2E latency avg (ms) ===" in markdown
    assert "=== E2E latency p50 (ms) ===" in markdown
    assert "=== E2E latency p90 (ms) ===" in markdown
    assert "=== E2E latency p99 (ms) ===" in markdown
    assert "=== Throughput (req/s) ===" in markdown
    assert markdown.index("=== TTFT p99 (ms) ===") < markdown.index(
        "=== E2E latency avg (ms) ==="
    )
    assert markdown.index("=== E2E latency p99 (ms) ===") < markdown.index(
        "=== Throughput (req/s) ==="
    )
    assert markdown.count("[artifact]") == 9
    assert markdown.count("[command]") == 9


def test_ablation_validation_and_markdown_cover_all_variants(tmp_path: Path) -> None:
    for label, _buckets, _max_batch_cost, _disabled in VARIANTS:
        for rate in RATES:
            _write_result(tmp_path, label, rate)

    assert len(validate_ablation(tmp_path)) == len(VARIANTS) * len(RATES)
    markdown = _ablation_markdown(tmp_path, _load_ablation_rows(tmp_path))
    assert "eager-b1" in markdown
    assert "graph-b8-only" in markdown
    assert "graph-full" in markdown
    assert markdown.count("[artifact]") == len(VARIANTS) * len(RATES)
