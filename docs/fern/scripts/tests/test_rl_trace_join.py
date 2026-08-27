# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the runnable RL framework-to-Dynamo trace join example."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest
import rl_trace_join

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "rl_trace_join"
FRAMEWORK = FIXTURES / "framework-attempts.jsonl"
TRACE = FIXTURES / "request-trace.jsonl"
EXPECTED = FIXTURES / "expected-summary.json"


def _fixture_records() -> tuple[list[dict], list[dict]]:
    return (
        rl_trace_join.load_jsonl([FRAMEWORK], "framework attempts"),
        rl_trace_join.load_jsonl([TRACE], "request trace"),
    )


def test_synthetic_join_matches_the_checked_in_summary() -> None:
    framework, trace = _fixture_records()
    joined, summary = rl_trace_join.join_records(framework, trace)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert summary == expected
    assert [record["join_status"] for record in joined] == [
        "complete",
        "complete",
        "expected_incomplete",
    ]
    assert all(record["policy_header_matches"] is True for record in joined)
    assert joined[0]["request_end"]["request"]["kv_hit_rate"] == 0.75
    assert "payload" not in joined[0]


def test_cli_writes_joined_rows_and_summary(tmp_path: Path) -> None:
    joined_path = tmp_path / "joined.jsonl"
    summary_path = tmp_path / "summary.json"
    result = rl_trace_join.main(
        [
            "--framework-attempts",
            str(FRAMEWORK),
            "--request-trace",
            str(TRACE),
            "--joined-jsonl",
            str(joined_path),
            "--summary-json",
            str(summary_path),
            "--strict",
        ]
    )
    assert result == 0
    assert len(joined_path.read_text(encoding="utf-8").splitlines()) == 3
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["strict_violations"] == 0
    )


def test_policy_header_mismatch_is_a_strict_violation() -> None:
    framework, trace = _fixture_records()
    framework = copy.deepcopy(framework)
    framework[0]["target_policy_version"] = "policy-3"
    joined, summary = rl_trace_join.join_records(framework, trace)
    assert joined[0]["policy_header_matches"] is False
    assert summary["policy_mismatches"] == 1
    assert summary["strict_violations"] == 1


def test_duplicate_framework_attempt_is_rejected() -> None:
    framework, trace = _fixture_records()
    with pytest.raises(
        rl_trace_join.JoinError, match="duplicate rollout_id/attempt_id"
    ):
        rl_trace_join.join_records([*framework, framework[0]], trace)


def test_compressed_trace_input_is_supported(tmp_path: Path) -> None:
    compressed = tmp_path / "request-trace.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as output:
        output.write(TRACE.read_text(encoding="utf-8"))
    records = rl_trace_join.load_jsonl([compressed], "request trace")
    assert len(records) == 5
