# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re

import pytest

from dynamo.vllm.benchmark_points import (
    benchmark_points_digest,
    load_benchmark_points_file,
    normalize_benchmark_points,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _valid_points() -> dict:
    return {
        "schema_version": 1,
        "prefill": [
            {
                "total_prefill_tokens": 8,
                "total_kv_read_tokens": 0,
                "batch_size": 1,
            },
            {
                "total_prefill_tokens": 16,
                "total_kv_read_tokens": 32,
                "batch_size": 2,
            },
        ],
        "decode": [
            {"total_kv_read_tokens": 64, "batch_size": 1},
            {"total_kv_read_tokens": 128, "batch_size": 4},
        ],
    }


def test_load_benchmark_points_file_normalizes_once_and_preserves_order(tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps(_valid_points()), encoding="utf-8")

    points = load_benchmark_points_file(str(path), "agg")

    assert points == _valid_points()
    assert len(benchmark_points_digest(points)) == 64


def test_load_benchmark_points_file_reports_invalid_utf8_with_path(tmp_path):
    path = tmp_path / "points.json"
    path.write_bytes(b"\xff")

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(path))!s}.*not valid UTF-8",
    ):
        load_benchmark_points_file(str(path), "agg")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda _: [], "top level must be an object"),
        (
            lambda data: data.update(schema_version=True),
            "schema_version must be the integer 1",
        ),
        (
            lambda data: data.update(schema_version=2),
            "schema_version must be 1, got 2",
        ),
        (
            lambda data: (data.pop("decode"), None)[1],
            "missing required field.*decode",
        ),
        (lambda data: data.update(extra=[]), "unsupported field.*extra"),
        (lambda data: data.update(prefill={}), "prefill must be an array"),
    ],
)
def test_normalize_benchmark_points_rejects_invalid_document(mutate, message):
    data = _valid_points()
    replacement = mutate(data)
    payload = replacement if replacement is not None else data

    with pytest.raises(ValueError, match=message):
        normalize_benchmark_points(payload, "agg")


@pytest.mark.parametrize(
    ("phase", "replacement", "message"),
    [
        ("prefill", None, r"prefill\[1\] must be an object"),
        (
            "prefill",
            {"total_prefill_tokens": 8, "batch_size": 1},
            r"prefill\[1\].*missing required.*total_kv_read_tokens",
        ),
        (
            "prefill",
            {
                "total_prefill_tokens": 8,
                "total_kv_read_tokens": 0,
                "batch_size": True,
            },
            r"prefill\[1\]\.batch_size must be an integer",
        ),
        (
            "prefill",
            {
                "total_prefill_tokens": 1,
                "total_kv_read_tokens": 0,
                "batch_size": 2,
            },
            r"prefill\[1\]\.total_prefill_tokens must be at least batch_size",
        ),
        (
            "decode",
            {"total_kv_read_tokens": 4, "batch_size": 8},
            r"decode\[1\]\.total_kv_read_tokens must be at least batch_size",
        ),
        (
            "decode",
            {"total_kv_read_tokens": 32, "batch_size": 1, "dp_rank": 0},
            r"decode\[1\].*unsupported field.*dp_rank",
        ),
    ],
)
def test_normalize_benchmark_points_reports_indexed_errors(phase, replacement, message):
    data = _valid_points()
    data[phase][1] = replacement

    with pytest.raises(ValueError, match=message):
        normalize_benchmark_points(data, "agg")


def test_unused_phase_is_still_validated():
    data = _valid_points()
    data["decode"][1]["batch_size"] = "4"

    with pytest.raises(ValueError, match=r"decode\[1\]\.batch_size"):
        normalize_benchmark_points(data, "prefill")


def test_invalid_mode_is_rejected_explicitly():
    with pytest.raises(ValueError, match="benchmark mode must be one of"):
        normalize_benchmark_points(_valid_points(), "mixed")


@pytest.mark.parametrize(
    ("mode", "empty_phase"),
    [("prefill", "prefill"), ("decode", "decode"), ("agg", "prefill")],
)
def test_selected_phases_must_be_nonempty(mode, empty_phase):
    data = _valid_points()
    data[empty_phase] = []

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{empty_phase} must contain at least one point for benchmark mode {mode!r}"
        ),
    ):
        normalize_benchmark_points(data, mode)
