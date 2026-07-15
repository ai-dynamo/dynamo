# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the text engine-client benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks.multimodal.sweep.experiments.engine_client.generate_text_workload import (
    generate_workload,
    rendered_token_count,
)
from benchmarks.multimodal.sweep.experiments.engine_client.report_text_results import (
    paired_interval,
)
from benchmarks.multimodal.sweep.experiments.engine_client.text_config import (
    TextSweepConfig,
)
from benchmarks.multimodal.sweep.experiments.engine_client.validate_text_results import (
    validate_result,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/engine_client/text_concurrency1.yaml"
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        assert add_generation_prompt
        token_count = 10 + messages[0]["content"].count(" benchmark")
        return list(range(token_count))


class FakeBatchEncodingTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        return {
            "input_ids": super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        }


def test_rendered_token_count_accepts_batch_encoding_shape() -> None:
    assert rendered_token_count(FakeBatchEncodingTokenizer(), "x benchmark") == 11


def test_generate_workload_is_text_only_and_exact(tmp_path: Path) -> None:
    with patch(
        "benchmarks.multimodal.sweep.experiments.engine_client."
        "generate_text_workload.AutoTokenizer.from_pretrained",
        return_value=FakeTokenizer(),
    ):
        dataset_path, manifest_path = generate_workload(
            tmp_path,
            request_count=3,
            target_isl=15,
        )

    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert all(set(row) == {"session_id", "text"} for row in rows)
    assert len({row["text"] for row in rows}) == 1
    assert manifest["media_fields"] == []


def test_config_locks_concurrency_one_matrix() -> None:
    config = TextSweepConfig.load(CONFIG_PATH)
    assert config.concurrency == 1
    assert config.repeats == 5
    assert set(config.runtimes) == {
        "vllm-serve",
        "dynamo-async",
        "dynamo-sync",
    }


def metric_document() -> dict[str, Any]:
    document: dict[str, Any] = {
        "request_count": {"avg": 1000},
        "input_sequence_length": {"min": 740, "avg": 740, "max": 740},
        "output_sequence_length": {"min": 70, "avg": 70, "max": 70},
        "request_throughput": {"avg": 10},
        "output_token_throughput": {"avg": 700},
        "time_to_first_token": {"avg": 5, "p50": 4, "p90": 6, "p99": 7},
        "inter_token_latency": {"avg": 2, "p50": 1, "p90": 3, "p99": 4},
        "request_latency": {"avg": 150, "p50": 140, "p90": 160, "p99": 170},
        "input_config": {
            "endpoint": {"streaming": True},
            "phases": [{"name": "profiling", "concurrency": 1}],
        },
        "error_summary": [],
        "was_cancelled": False,
    }
    return document


def test_validate_result_accepts_audited_cell(tmp_path: Path) -> None:
    config = TextSweepConfig.load(CONFIG_PATH)
    artifact_dir = tmp_path / "trial-01/dynamo-async/concurrency1"
    artifact_dir.mkdir(parents=True)
    result_path = artifact_dir / "profile_export_aiperf.json"
    result_path.write_text(json.dumps(metric_document()), encoding="utf-8")
    (artifact_dir / "command.txt").write_text(
        "aiperf profile --concurrency 1 --request-count 1000 "
        "--warmup-request-count 20 --use-server-token-count --random-seed 42\n",
        encoding="utf-8",
    )
    (artifact_dir.parent / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_sha256": config.source_sha256,
                "dataset_sha256": "dataset-sha",
            }
        ),
        encoding="utf-8",
    )

    result = validate_result(
        result_path,
        config,
        config.source_sha256,
        "dataset-sha",
    )
    assert result["accepted"]


def test_paired_interval_uses_each_trial() -> None:
    rows: list[dict[str, Any]] = []
    for trial in range(1, 6):
        rows.extend(
            [
                {
                    "trial": trial,
                    "runtime": "dynamo-async",
                    "request_throughput_rps": 10.0,
                },
                {
                    "trial": trial,
                    "runtime": "dynamo-sync",
                    "request_throughput_rps": 11.0,
                },
            ]
        )
    mean, low, high = paired_interval(rows, "dynamo-sync", "request_throughput_rps")
    assert mean == pytest.approx(10.0)
    assert low == pytest.approx(10.0)
    assert high == pytest.approx(10.0)
