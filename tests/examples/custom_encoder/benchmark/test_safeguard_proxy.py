# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.custom_encoder.benchmark.run_safeguard_proxy_sweep import (
    CONCURRENCIES,
    RUNTIME_DELAYS_US,
    build_config,
    summarize,
    validate_matrix,
)
from examples.custom_encoder.benchmark.safeguard_proxy_workload import (
    _calculate_custom_isl_components,
    _calibrate_prompt,
    _generate_jpeg,
    _request_schedule,
    validate_workload,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_request_schedule_reuses_nine_images_across_one_hundred_requests() -> None:
    images = [f"image-{index}" for index in range(9)]
    first = _request_schedule(images, requests=100, seed=42)
    second = _request_schedule(images, requests=100, seed=42)
    counts = sorted(first.count(image) for image in images)
    assert first == second
    assert counts == [11, 11, 11, 11, 11, 11, 11, 11, 12]


def test_request_schedule_reuses_one_image_for_every_request() -> None:
    assert _request_schedule(["image"], requests=1000, seed=42) == ["image"] * 1000


def test_prompt_calibration_requires_exact_target() -> None:
    prompt, observed = _calibrate_prompt(
        644, lambda text: 600 + text.count("benchmark")
    )
    assert observed == 644
    assert prompt.count("benchmark") == 44


def test_jpeg_generator_supports_300_pixel_workload(tmp_path: Path) -> None:
    path = tmp_path / "image_300x300.jpg"
    record = _generate_jpeg(path, seed=42, image_size=(300, 300))
    assert record["width"] == record["height"] == 300
    assert 50 * 1024 <= path.stat().st_size <= 60 * 1024


def test_workload_validation_rejects_requested_size_mismatch(tmp_path: Path) -> None:
    (tmp_path / "workload_manifest.json").write_text(
        json.dumps({"decoded_image": {"width": 500, "height": 500}}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="does not match requested 300x300"):
        validate_workload(tmp_path, expected_image_size=300)


def test_workload_validation_rejects_requested_image_count_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload_manifest.json").write_text(
        json.dumps(
            {
                "decoded_image": {"width": 300, "height": 300},
                "requests_per_concurrency": 100,
                "unique_images": 9,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="has 9 unique images; requested 1"):
        validate_workload(tmp_path, expected_unique_images=1)


def test_custom_isl_replaces_one_placeholder_with_image_tokens() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "rendered"

        def __call__(self, _text: str, *, add_special_tokens: bool) -> object:
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[1, 99, 2])

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|image_pad|>"
            return 99

    class FakeImageProcessor:
        merge_size = 2

        def __call__(self, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {"image_grid_thw": torch.tensor([[1, 4, 4]])}

    observed = _calculate_custom_isl_components(
        FakeTokenizer(), FakeImageProcessor(), "prompt", object()
    )
    assert observed == 6


def test_config_is_closed_loop_and_uses_requested_encoder_limits(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.jsonl"
    input_file.write_text('{"session_id":"one"}\n', encoding="utf-8")
    config = build_config(
        input_file,
        (1, 2, 3),
        tmp_path / "output",
        smoke=False,
        image_size=(300, 300),
    )
    assert config.request_rates is None
    assert config.concurrencies == [1, 2, 3]
    assert config.conversation_num == 1000
    assert config.warmup_count == 20
    assert config.osl == 7
    assert config.env["DYN_QWEN2_VL_PREPROCESS_CONCURRENCY"] == "4"
    assert config.env["DYN_QWEN2_VL_MAX_BATCH_COST"] == "8"
    assert config.env["DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS"] == "1,2,3,4,5,6,7,8"
    assert config.env["DYN_QWEN2_VL_GRAPH_IMAGE_SIZES"] == "300x300"
    assert config.env["DYN_CUSTOM_ENCODER_DISPATCH_LOG"] == "1"
    assert "DYN_CUSTOM_ENCODER_TIMING" not in config.env
    assert [item.label for item in config.configs] == list(RUNTIME_DELAYS_US)
    assert [item.extra_args[-1] for item in config.configs] == ["0", "1000"]
    assert "--use-server-token-count" in config.aiperf_extra_args


def _latency(value: float) -> dict[str, float | str]:
    return {
        "unit": "ms",
        "avg": value,
        "p50": value,
        "p90": value + 1,
        "p95": value + 2,
        "p99": value + 3,
    }


def _write_result(root: Path, runtime: str, concurrency: int) -> None:
    artifact = root / "image_custom_1000_isl644" / runtime / f"concurrency{concurrency}"
    artifact.mkdir(parents=True)
    command = f"aiperf profile --concurrency {concurrency} --random-seed 42"
    document = {
        "request_count": {"avg": 1000},
        "error_summary": [],
        "was_cancelled": False,
        "input_config": {
            "endpoint": {"streaming": True},
            "loadgen": {"concurrency": concurrency},
            "cli_command": command,
        },
        "input_sequence_length": {"min": 644, "avg": 644, "max": 644},
        "output_sequence_length": {"min": 7, "avg": 7, "max": 7},
        "time_to_first_token": _latency(float(concurrency)),
        "request_latency": _latency(float(concurrency * 10)),
        "request_throughput": {"avg": float(concurrency * 2)},
    }
    (artifact / "profile_export_aiperf.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    (artifact / "command.txt").write_text(command + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "concurrencies",
    [CONCURRENCIES, (4, 8, 16, 24, 32)],
)
def test_validation_and_report_cover_selected_cells(
    tmp_path: Path, concurrencies: tuple[int, ...]
) -> None:
    for runtime in RUNTIME_DELAYS_US:
        for concurrency in concurrencies:
            _write_result(tmp_path, runtime, concurrency)
    metadata = {
        "concurrencies": list(concurrencies),
        "dynamo_commit": "abc123",
        "container_image": "test-image",
        "gpu": "H100",
        "decoder_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "encoder_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "vllm_version": "test",
        "transformers_version": "test",
        "torch_version": "test",
        "aiperf_version": "0.8.0",
        "settings": {
            "preprocess_concurrency": 4,
            "max_batch_cost": 8,
            "batching_policy": "bounded queue hold",
            "queue_delays_us": [0, 1000],
            "graph_buckets": list(range(1, 9)),
            "graph_image_sizes": ["300x300"],
            "unique_images": 1,
        },
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    log_lines: list[str] = []
    for runtime in RUNTIME_DELAYS_US:
        for concurrency in concurrencies:
            log_lines.extend(
                [
                    f"[input] Config: {runtime}  concurrency={concurrency}",
                    "custom_encoder_dispatch mode=graph batch_size=1 "
                    "bucket=1 calls=1020",
                ]
            )
    (tmp_path / "sweep.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    assert len(validate_matrix(tmp_path)) == 2 * len(concurrencies)
    markdown = tmp_path / "benchmark.md"
    csv_path = tmp_path / "benchmark.csv"
    summarize(tmp_path, markdown, csv_path)
    report = markdown.read_text(encoding="utf-8")
    assert "Performance proxy only" in report
    assert "all requests share one 300×300 JPEG" in report
    assert "Triton baseline used nine unique images" in report
    assert "queue delays: `[0, 1000]` microseconds" in report
    last = concurrencies[-1]
    assert f"| {last} | {last * 2:.2f} | {last * 2:.2f} | +0.0% |" in report
    assert f"| 1000 us | {last} | 1020 | 1.00 | 100.0% |" in report
    assert report.count("[AIPerf]") == 2 * len(concurrencies)
    assert csv_path.read_text(encoding="utf-8").count("\n") == len(concurrencies) + 1
    assert (tmp_path / "dispatch_distribution.json").is_file()
