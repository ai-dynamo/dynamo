# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the checked RL request-plane operations report."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest
import rl_ops_report
import rl_trace_join

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "rl_trace_join"
FRAMEWORK = FIXTURES / "framework-attempts.jsonl"
TRACE = FIXTURES / "request-trace.jsonl"
EXPECTED = FIXTURES / "expected-operations-report.json"


def _joined_records() -> list[dict]:
    framework = rl_trace_join.load_jsonl([FRAMEWORK], "framework attempts")
    trace = rl_trace_join.load_jsonl([TRACE], "request trace")
    joined, _ = rl_trace_join.join_records(framework, trace)
    return joined


def _write_joined(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_checked_join_produces_the_expected_operations_report() -> None:
    report = rl_ops_report.build_report(_joined_records())
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert report == expected
    assert rl_ops_report.strict_findings(report) == []


def test_cli_writes_a_body_free_report(tmp_path: Path) -> None:
    joined_path = tmp_path / "joined.jsonl"
    report_path = tmp_path / "report.json"
    _write_joined(joined_path, _joined_records())
    assert (
        rl_ops_report.main(
            [
                "--joined-jsonl",
                str(joined_path),
                "--report-json",
                str(report_path),
                "--strict",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "dynamo.rl.operations-report.v1"
    assert "payload" not in report_path.read_text(encoding="utf-8")


def test_optional_metric_coverage_is_not_imputed_as_zero() -> None:
    records = _joined_records()
    records[0]["request_end"]["request"].pop("kv_hit_rate")
    report = rl_ops_report.build_report(records)
    summary = report["terminal_metrics"]["kv_hit_rate"]
    assert summary["present"] == 1
    assert summary["coverage_percent"] == 50.0
    assert summary["mean"] == 0.0
    absent = report["terminal_metrics"]["kv_transfer_estimated_latency_ms"]
    assert absent["present"] == 0
    assert absent["mean"] is None


def test_policy_mismatch_fails_strict_mode() -> None:
    records = _joined_records()
    records[0]["policy_header_matches"] = False
    report = rl_ops_report.build_report(records)
    assert rl_ops_report.strict_findings(report) == [
        "strict report rejects 1 policy-header mismatch(es)"
    ]


def test_join_violation_fails_strict_mode() -> None:
    records = _joined_records()
    records[0]["join_status"] = "missing_terminal"
    records[0]["request_end"] = None
    report = rl_ops_report.build_report(records)
    assert rl_ops_report.strict_findings(report) == [
        "strict report rejects join statuses: missing_terminal"
    ]


def test_invalid_numeric_telemetry_is_rejected() -> None:
    records = _joined_records()
    records[0]["request_end"]["request"]["queue_depth"] = "busy"
    with pytest.raises(rl_ops_report.ReportError, match="queue_depth must be numeric"):
        rl_ops_report.build_report(records)


def test_fractional_token_count_is_rejected() -> None:
    records = _joined_records()
    records[0]["request_end"]["request"]["output_tokens"] = 24.5
    with pytest.raises(rl_ops_report.ReportError, match="output_tokens must be an integer"):
        rl_ops_report.build_report(records)


def test_worker_and_finish_reason_breakdowns_are_reported() -> None:
    records = copy.deepcopy(_joined_records())
    records[0]["request_end"]["request"]["finish_reason_metadata"] = {
        "finish_reason": "stop"
    }
    report = rl_ops_report.build_report(records)
    assert report["finish_reasons"] == {"stop": 1}
    assert report["worker_activity"]["decode"]["1"]["output_tokens"] == 24


def test_compressed_join_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "joined.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in _joined_records():
            output.write(json.dumps(record) + "\n")
    report = rl_ops_report.build_report(rl_ops_report.load_joined([path]))
    assert report["rows"] == {"joined": 3, "terminal": 2, "without_terminal": 1}


def test_all_incomplete_rows_produce_coverage_and_fail_strict() -> None:
    records = _joined_records()
    for record in records:
        record["join_status"] = "expected_incomplete"
        record["request_end"] = None
    report = rl_ops_report.build_report(records)
    assert report["terminal_metrics"]["ttft_ms"]["coverage_percent"] == 0.0
    assert rl_ops_report.strict_findings(report) == [
        "at least one terminal request is required"
    ]
