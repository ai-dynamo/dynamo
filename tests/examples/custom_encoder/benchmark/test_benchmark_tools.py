# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.custom_encoder.benchmark.generate_workload import _calculate_custom_isl
from examples.custom_encoder.benchmark.run_ablation import VARIANTS
from examples.custom_encoder.benchmark.run_concurrency_sweep import (
    _config as _concurrency_config,
)
from examples.custom_encoder.benchmark.summarize_ablation import (
    _load_rows as _load_ablation_rows,
)
from examples.custom_encoder.benchmark.summarize_ablation import (
    _markdown as _ablation_markdown,
)
from examples.custom_encoder.benchmark.summarize_concurrency_results import (
    _load_rows as _load_concurrency_rows,
)
from examples.custom_encoder.benchmark.summarize_concurrency_results import (
    _markdown as _concurrency_markdown,
)
from examples.custom_encoder.benchmark.summarize_results import _load_rows, _markdown
from examples.custom_encoder.benchmark.validate_ablation import validate_ablation
from examples.custom_encoder.benchmark.validate_concurrency_results import (
    validate_matrix as validate_concurrency_matrix,
)
from examples.custom_encoder.benchmark.validate_results import validate_matrix

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

RUNTIMES = ("vllm-serve", "dynamo-native", "dynamo-custom-encoder")
RATES = (16, 24, 32)


def test_custom_isl_expands_the_single_image_placeholder() -> None:
    class FakeTokenizer:
        def __call__(self, _rendered: str, *, add_special_tokens: bool) -> object:
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[1, 99, 2])

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<|image_pad|>"
            return 99

    class FakeImageProcessor:
        merge_size = 2

        def __call__(self, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {"image_grid_thw": torch.tensor([[1, 4, 4]])}

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        image_processor = FakeImageProcessor()

        @staticmethod
        def apply_chat_template(*_args: object, **_kwargs: object) -> str:
            return "rendered"

    # Three rendered tokens, with one placeholder replaced by four merged
    # image tokens: 3 - 1 + (1 * 4 * 4 / 2**2) == 6.
    assert _calculate_custom_isl(FakeProcessor(), "prompt", object()) == 6


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


def _write_concurrency_result(root: Path, concurrency: int) -> None:
    runtime = "dynamo-custom-encoder"
    artifact = (
        root
        / f"image_custom_concurrency{concurrency}_1000_isl515"
        / runtime
        / f"concurrency{concurrency}"
    )
    artifact.mkdir(parents=True)
    document = {
        "request_count": {"avg": 1000},
        "error_summary": [],
        "was_cancelled": False,
        "input_config": {
            "endpoint": {"streaming": True},
            "loadgen": {"concurrency": concurrency},
            "cli_command": (
                f"aiperf profile --concurrency {concurrency} --random-seed 42"
            ),
        },
        "input_sequence_length": {"min": 515, "avg": 515, "max": 515},
        "output_sequence_length": {"min": 70, "avg": 70, "max": 70},
        "time_to_first_token": _latency(float(concurrency)),
        "request_latency": _latency(float(concurrency * 10)),
        "request_throughput": {"avg": float(concurrency)},
    }
    (artifact / "profile_export_aiperf.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    (artifact / "command.txt").write_text(
        f"aiperf profile --concurrency {concurrency}\n", encoding="utf-8"
    )


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


def test_concurrency_config_uses_closed_loop_axis_and_requested_tuning(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.jsonl"
    input_file.write_text('{"session_id":"one"}\n', encoding="utf-8")
    config = _concurrency_config(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        input_file,
        32,
        tmp_path / "output",
        False,
    )

    assert config.request_rates is None
    assert config.concurrencies == [32]
    assert config.osl == 70
    assert config.env["DYN_QWEN2_VL_PREPROCESS_CONCURRENCY"] == "64"
    assert config.env["DYN_QWEN2_VL_MAX_BATCH_COST"] == "64"
    assert config.env["DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS"] == "1,2,4,8,16,32,64"
    assert config.env["DYN_QWEN2_VL_GRAPH_IMAGE_SIZES"] == "500x500"
    assert config.env["DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE"] == "1536"


def test_concurrency_validation_and_markdown_cover_three_cells(
    tmp_path: Path,
) -> None:
    for concurrency in (8, 16, 32):
        _write_concurrency_result(tmp_path, concurrency)
    metadata = {
        "decoder_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "encoder_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dynamo_commit": "abc123",
        "container_image": "test-image",
        "gpu": "H100",
        "vllm_version": "test",
        "transformers_version": "test",
        "torch_version": "test",
        "aiperf_version": "0.8.0",
        "settings": {
            "preprocess_concurrency": 64,
            "max_batch_cost": 64,
            "queue_wait_ms": 1,
            "graph_buckets": [1, 2, 4, 8, 16, 32, 64],
            "graph_image_sizes": ["500x500"],
        },
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (tmp_path / "sweep.log").write_text(
        "Config: dynamo-custom-encoder concurrency=8\n"
        "custom_encoder_graph selected_bucket=8 actual_cost=7 batch_size=7\n"
        "Config: dynamo-custom-encoder concurrency=16\n"
        "custom_encoder_graph selected_bucket=16 actual_cost=12 batch_size=12\n"
        "Config: dynamo-custom-encoder concurrency=32\n"
        "custom_encoder_graph selected_bucket=32 actual_cost=24 batch_size=24\n",
        encoding="utf-8",
    )

    assert len(validate_concurrency_matrix(tmp_path)) == 3
    markdown = _concurrency_markdown(tmp_path, _load_concurrency_rows(tmp_path))
    assert "Performance-only adapter" in markdown
    assert "Qwen/Qwen2.5-1.5B-Instruct" in markdown
    assert "| 32 | 24 | `[32]` | 24→32: 1 |" in markdown
    assert markdown.count("[artifact]") == 3
    assert markdown.count("[command]") == 3
