# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from dynamo.vllm.benchmark_points import BenchmarkPoints, load_benchmark_points_file

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _points() -> dict:
    return {
        "schema_version": 1,
        "prefill": [
            {
                "total_prefill_tokens": 8,
                "total_kv_read_tokens": 0,
                "batch_size": 1,
            }
        ],
        "decode": [{"total_kv_read_tokens": 32, "batch_size": 2}],
    }


def test_load_benchmark_points_file_preserves_order(tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps(_points()))

    points = load_benchmark_points_file(str(path))

    assert points.model_dump(mode="json") == _points()


@pytest.mark.parametrize(
    "payload",
    [
        {**_points(), "schema_version": True},
        {**_points(), "extra": []},
        {**_points(), "decode": [{"total_kv_read_tokens": 1, "batch_size": 2}]},
        {
            **_points(),
            "prefill": [
                {
                    "total_prefill_tokens": 8,
                    "total_kv_read_tokens": 0,
                    "batch_size": "1",
                }
            ],
        },
    ],
)
def test_manifest_schema_is_strict(payload):
    with pytest.raises(ValidationError):
        BenchmarkPoints.model_validate(payload)


def test_empty_manifest_is_allowed(tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema_version": 1, "prefill": [], "decode": []}))

    points = load_benchmark_points_file(str(path))

    assert points.prefill == []
    assert points.decode == []


def test_load_error_includes_source_path(tmp_path):
    path = tmp_path / "points.json"
    path.write_text("not json")

    with pytest.raises(ValueError, match=str(path)):
        load_benchmark_points_file(str(path))
