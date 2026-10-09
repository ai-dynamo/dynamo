# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic checks for preserving trace timing, reuse and provenance."""

from __future__ import annotations

import hashlib
import json

import pytest
from autoscaling_arena.trace_preparation import TracePreparationConfig, prepare_trace

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _write_trace(path, *, timestamps=range(0, 10_001, 100), session=False):
    records = [
        {
            "timestamp": timestamp,
            "input_length": 128,
            "output_length": 16 + ordinal,
            "hash_ids": [7, 8],
            "request_id": f"request-{ordinal}",
            **({"session_id": "session-a"} if session else {}),
        }
        for ordinal, timestamp in enumerate(timestamps)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return records


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_identity_preserves_exact_bytes_and_records_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source, timestamps=[20, 50, 90])
    source.write_bytes(b"\n" + source.read_bytes().rstrip())
    original_bytes = source.read_bytes()

    result = prepare_trace(
        source, tmp_path / "copy.jsonl", config=TracePreparationConfig(), block_size=64
    )

    assert result.path.read_bytes() == source.read_bytes() == original_bytes
    assert (
        result.manifest["source"]["sha256"]
        == hashlib.sha256(original_bytes).hexdigest()
    )
    assert result.manifest["output"]["request_count"] == len(original)
    assert result.manifest["output"]["sha256"] == result.manifest["source"]["sha256"]
    assert str(source) not in result.manifest_path.read_text()


def test_window_is_half_open_and_rebased_without_compressing_gaps(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source, timestamps=[0, 100, 130, 199, 200, 400])
    result = prepare_trace(
        source,
        tmp_path / "slice.jsonl",
        config=TracePreparationConfig(start_ms=100, end_ms=200, rebase=True),
        block_size=64,
    )

    rows = _read(result.path)
    assert [row["timestamp"] for row in rows] == [0, 30, 99]
    assert [row["hash_ids"] for row in rows] == [
        row["hash_ids"] for row in original[1:4]
    ]
    assert [row["output_length"] for row in rows] == [
        row["output_length"] for row in original[1:4]
    ]
    assert result.manifest["window"]["duration_ms"] == 100
    assert result.manifest["window"]["output_end_ms"] == 100
    assert _read(source) == original


def test_scaling_duplicates_requests_and_isolates_copy_prefixes(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source, timestamps=[0, 300, 700])
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(request_scale=3, seed=11),
        block_size=64,
    )

    rows = _read(result.path)
    assert len(rows) == 9
    assert [row["timestamp"] for row in rows] == [0, 0, 0, 300, 300, 300, 700, 700, 700]
    copies = [rows[index::3] for index in range(3)]
    assert copies[0] == original
    for copy in copies:
        assert [row["input_length"] for row in copy] == [128] * 3
        assert [row["output_length"] for row in copy] == [16, 17, 18]
        assert all(row["hash_ids"] == copy[0]["hash_ids"] for row in copy)
    assert len({item for copy in copies for item in copy[0]["hash_ids"]}) == 6
    assert len({row["request_id"] for row in rows}) == 9


def test_fractional_thinning_is_reproducible_and_retains_declared_window(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source)
    config = TracePreparationConfig(
        request_scale=0.35, seed=9, start_ms=0, end_ms=11_000
    )
    first = prepare_trace(
        source, tmp_path / "first.jsonl", config=config, block_size=64
    )
    second = prepare_trace(
        source, tmp_path / "second.jsonl", config=config, block_size=64
    )
    rows = _read(first.path)

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.manifest == second.manifest
    assert 0 < len(rows) < len(original)
    assert all(row in original for row in rows)
    assert first.manifest["window"]["duration_ms"] == 11_000
    assert first.manifest["window"]["output_end_ms"] == 11_000
    assert first.manifest["output"]["last_timestamp_ms"] < 11_000


@pytest.mark.parametrize("session_field", [None, "session_id"])
def test_phase_jitter_is_bounded_sorted_and_preserves_interior_gaps(
    tmp_path, session_field
):
    source = tmp_path / "source.jsonl"
    original = _write_trace(
        source, timestamps=[0, 2000, 3000, 4000, 10_000], session=True
    )
    config = TracePreparationConfig(
        request_scale=2, seed=19, jitter_ms=500, session_id_field=session_field
    )
    first = prepare_trace(
        source, tmp_path / "first.jsonl", config=config, block_size=64
    )
    second = prepare_trace(
        source, tmp_path / "second.jsonl", config=config, block_size=64
    )
    rows = _read(first.path)
    original_rows = [row for row in rows if row["hash_ids"] == [7, 8]]
    extra_rows = [row for row in rows if row["hash_ids"] != [7, 8]]

    assert first.path.read_bytes() == second.path.read_bytes()
    assert original_rows == original
    assert [row["timestamp"] for row in rows] == sorted(
        row["timestamp"] for row in rows
    )
    assert extra_rows[0]["timestamp"] == 0
    assert extra_rows[-1]["timestamp"] == 10_000
    assert [
        extra_rows[index + 1]["timestamp"] - extra_rows[index]["timestamp"]
        for index in (1, 2)
    ] == pytest.approx([1000, 1000])
    assert all(
        abs(copy["timestamp"] - row["timestamp"]) <= 500
        for copy, row in zip(extra_rows, original)
    )
    if session_field:
        assert len({row["session_id"] for row in extra_rows}) == 1
        assert extra_rows[0]["session_id"] != original[0]["session_id"]


def test_session_phases_differ_but_preserve_session_order(tmp_path):
    source = tmp_path / "source.jsonl"
    records = _write_trace(
        source, timestamps=[0, 2000, 2100, 3000, 3100, 10_000], session=True
    )
    for index in (2, 4):
        records[index]["session_id"] = "session-b"
    source.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(
            request_scale=2, seed=19, jitter_ms=500, session_id_field="session_id"
        ),
        block_size=64,
    )
    added = {
        row["output_length"]: row
        for row in _read(result.path)
        if row["hash_ids"] != [7, 8]
    }
    delta_a = added[17]["timestamp"] - records[1]["timestamp"]
    delta_b = added[18]["timestamp"] - records[2]["timestamp"]
    assert delta_a != delta_b
    assert added[19]["timestamp"] - added[17]["timestamp"] == pytest.approx(1000)
    assert added[20]["timestamp"] - added[18]["timestamp"] == pytest.approx(1000)


@pytest.mark.parametrize(
    "options",
    [
        {"request_scale": 0},
        {"request_scale": float("nan")},
        {"request_scale": 10**1000},
        {"jitter_ms": -1},
        {"seed": True},
        {"start_ms": 4, "end_ms": 3},
        {"rebase": "false"},
        {"session_id_field": ""},
        {"session_id_field": "input_length"},
        {"speedup": 4},
    ],
)
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        TracePreparationConfig.from_mapping(options)


def test_existing_output_and_source_are_never_overwritten(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_trace(source)
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="source"):
        prepare_trace(source, source, config=TracePreparationConfig(), block_size=64)
    output = tmp_path / "output.jsonl"
    output.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        prepare_trace(source, output, config=TracePreparationConfig(), block_size=64)
    assert output.read_bytes() == b"keep me"
    assert source.read_bytes() == original_bytes


def test_invalid_trace_and_empty_window_do_not_publish_partial_outputs(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_trace(source, timestamps=[2, 1])
    output = tmp_path / "output.jsonl"
    with pytest.raises(ValueError, match="not sorted"):
        prepare_trace(source, output, config=TracePreparationConfig(), block_size=64)
    _write_trace(source, timestamps=[1, 2])
    with pytest.raises(ValueError, match="no requests"):
        prepare_trace(
            source, output, config=TracePreparationConfig(start_ms=10), block_size=64
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".trace-preparation-*"))


def test_fractional_scale_above_one_keeps_all_full_copies(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source)
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(request_scale=2.5, seed=1),
        block_size=64,
    )
    rows = _read(result.path)
    counts = {row["timestamp"]: 0 for row in original}
    for row in rows:
        counts[row["timestamp"]] += 1
    assert set(counts.values()) == {2, 3}


def test_hash_namespaces_skip_reserved_ids_including_unselected_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = _write_trace(source, timestamps=[0, 100, 200])
    rows[0]["hash_ids"] = [0, 1]
    rows[1]["hash_ids"] = [2, (1 << 64) - 1]
    rows[2]["hash_ids"] = [3, 4]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(start_ms=100, end_ms=200, request_scale=3),
        block_size=64,
    )
    prepared = _read(result.path)
    reserved = {value for row in rows for value in row["hash_ids"]}
    new_hashes = [value for row in prepared[1:] for value in row["hash_ids"]]
    assert not reserved.intersection(new_hashes)
    assert len(new_hashes) == len(set(new_hashes)) == 4
    assert all(0 <= value < 1 << 64 for value in new_hashes)


def test_empty_thinning_leaves_no_partial_trace(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_trace(source, timestamps=[0])
    output = tmp_path / "output.jsonl"
    with pytest.raises(ValueError, match="thinning selected no requests"):
        prepare_trace(
            source,
            output,
            config=TracePreparationConfig(request_scale=1e-100),
            block_size=64,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".manifest.json").exists()
    assert not list(tmp_path.glob(".trace-preparation-*"))
