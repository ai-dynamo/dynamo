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

    # exclude_unset so a v1 manifest round-trips as itself: the optional
    # spread fields added by later schema versions would otherwise appear as
    # explicit nulls that were never in the file.
    assert points.model_dump(mode="json", exclude_unset=True) == _points()


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


def _rows_point(rows: list[list[int]], **overrides) -> dict:
    """A schema-v3 prefill point whose totals agree with its rows."""
    point = {
        "batch_size": len(rows),
        "total_prefill_tokens": sum(new for new, _ in rows),
        "total_kv_read_tokens": sum(kv for _, kv in rows),
        "rows": rows,
    }
    point.update(overrides)
    return point


def _manifest(point: dict, schema_version: int = 3) -> dict:
    return {"schema_version": schema_version, "prefill": [point], "decode": []}


def test_rows_are_accepted_at_schema_v3():
    manifest = _manifest(_rows_point([[6, 4], [2, 0]]))
    parsed = BenchmarkPoints.model_validate(manifest)
    assert parsed.prefill[0].rows == [[6, 4], [2, 0]]


@pytest.mark.parametrize("schema_version", [1, 2])
def test_rows_require_schema_v3(schema_version):
    manifest = _manifest(_rows_point([[6, 4], [2, 0]]), schema_version)
    with pytest.raises(ValidationError, match="rows requires schema_version"):
        BenchmarkPoints.model_validate(manifest)


def test_rows_and_partition_are_mutually_exclusive():
    point = _rows_point(
        [[6, 4], [2, 0]],
        partition={"axis": "new", "high_count": 1, "fraction": 0.5},
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        BenchmarkPoints.model_validate(_manifest(point))


def test_rows_must_match_batch_size():
    point = _rows_point([[6, 4], [2, 0]], batch_size=3)
    with pytest.raises(ValidationError, match="batch_size"):
        BenchmarkPoints.model_validate(_manifest(point))


@pytest.mark.parametrize(
    "field, drift",
    [("total_prefill_tokens", 1), ("total_kv_read_tokens", 1)],
)
def test_rows_must_conserve_the_totals(field, drift):
    """The label is a difference against the equal-length batch with the SAME
    totals, so totals that drift from the rows would measure the extra tokens
    as well as the spread."""
    point = _rows_point([[6, 4], [2, 0]])
    point[field] += drift
    with pytest.raises(ValidationError, match="rows sum to"):
        BenchmarkPoints.model_validate(_manifest(point))


def test_a_row_must_carry_a_new_token():
    """A request scheduled with zero new tokens is not a shorter request, it
    is a request vLLM refuses to schedule at all."""
    point = _rows_point([[8, 4], [0, 0]])
    with pytest.raises(ValidationError, match="at least one new token"):
        BenchmarkPoints.model_validate(_manifest(point))


def test_rows_reject_negative_kv():
    point = _rows_point([[6, 4], [2, 0]])
    point["rows"] = [[6, 5], [2, -1]]
    with pytest.raises(ValidationError, match="cannot be negative"):
        BenchmarkPoints.model_validate(_manifest(point))


def test_a_row_needs_exactly_two_entries():
    point = _rows_point([[6, 4], [2, 0]])
    point["rows"] = [[6, 4, 1], [2, 0]]
    with pytest.raises(ValidationError, match="two entries"):
        BenchmarkPoints.model_validate(_manifest(point))


def test_zero_kv_rows_are_allowed():
    """A mixed calibration batch holds its short rows at no prefix at all, so
    a zero here is the point rather than an omission."""
    parsed = BenchmarkPoints.model_validate(_manifest(_rows_point([[6, 8], [2, 0]])))
    assert parsed.prefill[0].rows[1] == [2, 0]
