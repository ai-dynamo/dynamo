# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from .compare import compare_case_results, compare_files, main


def _result(
    path: Path, *, runtime: str, cases: dict, suite: str = "openrouter"
) -> Path:
    path.write_text(
        json.dumps(
            {
                "suite": suite,
                "model": "example/model",
                "runtime": runtime,
                "summary": {"score": 0.5},
                "case_results": cases,
            }
        )
    )
    return path


def test_compare_splits_failures_into_candidate_only_control_only_and_shared() -> None:
    candidate = {
        "a": {"status": "pass"},
        "b": {"status": "fail", "failure_kind": "schema_violation"},
        "c": {"status": "fail", "failure_kind": "value_mismatch"},
        "d": {"status": "unscored", "runs": 0},
    }
    control = {
        "a": {"status": "fail", "failure_kind": "call_count"},
        "b": {"status": "fail", "failure_kind": "schema_violation"},
        "c": {"status": "pass"},
        "d": {"status": "pass"},
    }

    comparison = compare_case_results(candidate, control)

    assert comparison["candidate_only_failures"] == ["c"]
    assert comparison["control_only_failures"] == ["a"]
    assert comparison["shared_failures"] == ["b"]
    assert comparison["unscored_in_either"] == ["d"]
    assert comparison["candidate_regressions"] == 1
    assert comparison["candidate_only_failure_kinds"] == {"value_mismatch": 1}
    assert comparison["control_only_failure_kinds"] == {"call_count": 1}
    assert comparison["shared_failure_kinds"] == {"schema_violation": 1}


def test_compare_cli_reads_two_suite_results(tmp_path: Path, capsys) -> None:
    candidate = _result(
        tmp_path / "dynamo.json",
        runtime="dynamo-vllm",
        cases={"x": {"status": "fail", "failure_kind": "schema_violation"}},
    )
    control = _result(
        tmp_path / "native.json", runtime="vllm-serve", cases={"x": {"status": "pass"}}
    )
    output = tmp_path / "compare.json"

    exit_code = main(
        [
            "--candidate",
            str(candidate),
            "--control",
            str(control),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    written = json.loads(output.read_text())
    assert written["candidate"]["runtime"] == "dynamo-vllm"
    assert written["control"]["runtime"] == "vllm-serve"
    assert written["candidate_only_failures"] == ["x"]
    assert json.loads(capsys.readouterr().out) == written


def test_compare_rejects_results_without_case_results_or_mismatched_suites(
    tmp_path: Path,
) -> None:
    without = tmp_path / "bfcl.json"
    without.write_text(json.dumps({"suite": "bfcl", "summary": {}}))
    with_cases = _result(
        tmp_path / "or.json", runtime="dynamo-vllm", cases={"x": {"status": "pass"}}
    )
    with pytest.raises(ValueError, match="no case_results"):
        compare_files(with_cases, without)

    other_suite = _result(
        tmp_path / "other.json",
        runtime="vllm-serve",
        cases={"x": {"status": "pass"}},
        suite="custom",
    )
    with pytest.raises(ValueError, match="suite mismatch"):
        compare_files(with_cases, other_suite)
