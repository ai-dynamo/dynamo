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


@pytest.mark.parametrize("session_id_field", [None, "conversation_id"])
def test_scaling_isolates_native_sessions_without_a_session_phase_field(
    tmp_path, session_id_field
):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source, timestamps=[0, 100, 200], session=True)
    original[2]["session_id"] = "session-b"
    for row in original:
        row["conversation_id"] = "conversation-a"
    source.write_text("\n".join(json.dumps(row) for row in original) + "\n")
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(
            request_scale=3, session_id_field=session_id_field
        ),
        block_size=64,
    )

    rows = _read(result.path)
    copies = [rows[index::3] for index in range(3)]
    assert copies[0] == original
    sessions = [{row["session_id"] for row in copy} for copy in copies]
    assert len(set.union(*sessions)) == 6
    for copy in copies:
        assert copy[0]["session_id"] == copy[1]["session_id"]
        assert copy[0]["session_id"] != copy[2]["session_id"]
    if session_id_field is not None:
        assert len({copy[0][session_id_field] for copy in copies}) == 3
        assert all(len({row[session_id_field] for row in copy}) == 1 for copy in copies)


def test_scaling_preserves_absent_and_null_native_sessions(tmp_path):
    source = tmp_path / "source.jsonl"
    original = _write_trace(source, timestamps=[0, 100, 200])
    for row in original[1:]:
        row["session_id"] = None
        row["request_id"] = None
    source.write_text("\n".join(json.dumps(row) for row in original) + "\n")
    result = prepare_trace(
        source,
        tmp_path / "scaled.jsonl",
        config=TracePreparationConfig(request_scale=2, session_id_field="session_id"),
        block_size=64,
    )

    rows = _read(result.path)
    assert rows[::2] == original
    assert "session_id" not in rows[1]
    for row in rows[2:]:
        assert row["session_id"] is None
        assert row["request_id"] is None


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


# --- Mooncake upstream field aliases (Issue #19234) -----------------------


def test_prepare_trace_identity_canonicalizes_aliases_and_records_provenance(tmp_path):
    source = tmp_path / "source_aliased.jsonl"
    aliased_records = [
        {
            "created_time": 0,
            "input_tokens": 128,
            "output_tokens": 16,
            "hash_ids": [7, 8],
            "request_id": "req-0",
            "custom_metadata": "kept-intact",
        },
        {
            "created_time": 100,
            "input_tokens": 64,
            "output_tokens": 32,
            "hash_ids": [9],
            "request_id": "req-1",
            "custom_metadata": "also-kept",
        },
    ]
    source.write_text("\n".join(json.dumps(r) for r in aliased_records) + "\n")
    source_bytes_before = source.read_bytes()

    output = tmp_path / "prepared_canonical.jsonl"
    result = prepare_trace(
        source, output, config=TracePreparationConfig(), block_size=64
    )

    assert source.read_bytes() == source_bytes_before

    rows = _read(result.path)
    assert len(rows) == 2
    for original, row in zip(aliased_records, rows):
        assert "created_time" not in row
        assert "input_tokens" not in row
        assert "output_tokens" not in row
        assert row["timestamp"] == original["created_time"]
        assert row["input_length"] == original["input_tokens"]
        assert row["output_length"] == original["output_tokens"]
        assert row["hash_ids"] == original["hash_ids"]
        assert row["request_id"] == original["request_id"]
        assert row["custom_metadata"] == original["custom_metadata"]

    source_sha = hashlib.sha256(source_bytes_before).hexdigest()
    output_sha = hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.manifest["source"]["sha256"] == source_sha
    assert result.manifest["output"]["sha256"] == output_sha
    # Hashes legitimately differ because fields were canonicalized
    assert source_sha != output_sha
    assert result.manifest["source"]["request_count"] == 2
    assert result.manifest["output"]["request_count"] == 2

    output_repeat = tmp_path / "prepared_repeat.jsonl"
    result_repeat = prepare_trace(
        source, output_repeat, config=TracePreparationConfig(), block_size=64
    )
    assert result_repeat.path.read_bytes() == result.path.read_bytes()
    assert result_repeat.manifest["output"]["sha256"] == output_sha


def test_prepare_trace_rejects_duplicate_semantic_fields(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    record = {
        "timestamp": 0,
        "created_time": 0,
        "input_length": 64,
        "output_length": 16,
        "hash_ids": [1],
    }
    source.write_text(json.dumps(record) + "\n")
    output = tmp_path / "output.jsonl"
    with pytest.raises(
        ValueError, match="line 1: timestamp and its alias created_time are both set"
    ):
        prepare_trace(source, output, config=TracePreparationConfig(), block_size=64)
    assert not output.exists()


def test_prepare_trace_transformed_aliased_matches_canonical(tmp_path):
    canonical_source = tmp_path / "canonical.jsonl"
    aliased_source = tmp_path / "aliased.jsonl"

    canonical_records = [
        {
            "timestamp": t,
            "input_length": 128,
            "output_length": 16 + i,
            "hash_ids": [7, 8],
            "request_id": f"req-{i}",
            "session_id": "session-1",
        }
        for i, t in enumerate(range(0, 1000, 100))
    ]
    aliased_records = [
        {
            "created_time": r["timestamp"],
            "input_tokens": r["input_length"],
            "output_tokens": r["output_length"],
            "hash_ids": r["hash_ids"],
            "request_id": r["request_id"],
            "session_id": r["session_id"],
        }
        for r in canonical_records
    ]

    canonical_source.write_text(
        "\n".join(json.dumps(r) for r in canonical_records) + "\n"
    )
    aliased_source.write_text("\n".join(json.dumps(r) for r in aliased_records) + "\n")

    config = TracePreparationConfig(
        start_ms=100,
        end_ms=600,
        rebase=True,
        request_scale=2.5,
        seed=42,
        jitter_ms=20.0,
        session_id_field="session_id",
    )

    res_canonical = prepare_trace(
        canonical_source,
        tmp_path / "out_canonical.jsonl",
        config=config,
        block_size=64,
    )
    res_aliased = prepare_trace(
        aliased_source,
        tmp_path / "out_aliased.jsonl",
        config=config,
        block_size=64,
    )

    assert _read(res_aliased.path) == _read(res_canonical.path)
    assert res_aliased.path.read_bytes() == res_canonical.path.read_bytes()
    assert res_aliased.manifest["output"] == res_canonical.manifest["output"]
    assert res_aliased.manifest["window"] == res_canonical.manifest["window"]


@pytest.mark.parametrize(
    "alias_field", ["created_time", "input_tokens", "output_tokens"]
)
def test_session_id_field_rejects_alias_names(alias_field):
    with pytest.raises(
        ValueError, match="session_id_field cannot name a replay metric"
    ):
        TracePreparationConfig(session_id_field=alias_field)


def test_prepare_trace_identity_canonical_preserves_blank_lines_and_no_trailing_newline(
    tmp_path,
):
    source = tmp_path / "canonical_messy.jsonl"
    # Leading newline, blank lines, trailing space, no trailing newline
    content = (
        b"\n"
        b'{"timestamp": 0, "input_length": 64, "output_length": 16, "hash_ids": [1]}  \n'
        b"\n\n"
        b'{"timestamp": 100, "input_length": 64, "output_length": 16, "hash_ids": [2]}'
    )
    source.write_bytes(content)
    output = tmp_path / "prepared.jsonl"
    result = prepare_trace(
        source, output, config=TracePreparationConfig(), block_size=64
    )

    assert output.read_bytes() == content
    assert result.manifest["source"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert result.manifest["output"]["sha256"] == result.manifest["source"]["sha256"]
    assert result.manifest["output"]["request_count"] == 2


def test_prepare_trace_identity_aliased_with_blank_lines_and_mixed_spellings(tmp_path):
    source = tmp_path / "aliased_mixed.jsonl"
    lines = [
        "",
        '{"timestamp": 0, "input_tokens": 64, "output_length": 16, "hash_ids": [1]}',
        "",
        '{"created_time": 100, "input_length": 64, "output_tokens": 16, "hash_ids": [2], "meta": 42}',
        "",
    ]
    source.write_text("\n".join(lines))
    output = tmp_path / "prepared.jsonl"
    result = prepare_trace(
        source, output, config=TracePreparationConfig(), block_size=64
    )

    rows = _read(result.path)
    assert len(rows) == 2
    assert [r["timestamp"] for r in rows] == [0, 100]
    assert [r["input_length"] for r in rows] == [64, 64]
    assert [r["output_length"] for r in rows] == [16, 16]
    assert rows[1]["meta"] == 42
    for r in rows:
        assert "created_time" not in r
        assert "input_tokens" not in r
        assert "output_tokens" not in r
    assert result.manifest["source"]["request_count"] == 2
    assert result.manifest["output"]["request_count"] == 2
    assert (
        result.manifest["output"]["sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert (
        result.manifest["source"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert result.manifest["output"]["sha256"] != result.manifest["source"]["sha256"]
