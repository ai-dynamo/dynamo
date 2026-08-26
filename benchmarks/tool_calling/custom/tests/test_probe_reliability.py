# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import tool_calling_probe as probe  # noqa: E402
import tool_calling_static_report as static_report  # noqa: E402


def qualification_custom_config() -> dict:
    profiles_path = ROOT.parent / "e2e_verifier" / "profiles.json"
    payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    return payload["profiles"]["qualification"]["custom"]


def qualification_generic_case_ids() -> tuple[str, ...]:
    return tuple(sorted(qualification_custom_config()["generic_cases"]))


def qualification_case_ids(profile: str = "generic") -> tuple[str, ...]:
    config = qualification_custom_config()
    return tuple(
        sorted(
            (
                *config["generic_cases"],
                *config["model_specific_cases"].get(profile, ()),
            )
        )
    )


def test_qualification_uses_the_same_24_generic_cases_for_inline_profiles() -> None:
    expected_ids = qualification_generic_case_ids()

    assert len(expected_ids) == 24
    assert len(set(expected_ids)) == 24
    generic_cases = probe.build_cases("generic")
    assert {case.case_id for case in generic_cases} == set(expected_ids)
    assert all(case.profiles == ("generic",) for case in generic_cases)

    for profile in (
        "generic",
        "gemma4",
        "gpt_oss",
        "qwen3_coder",
        "glm5",
        "glm47",
        "minimax_m2",
        "deepseek_v4",
    ):
        selected = probe.select_cases(
            probe.build_cases(profile), ",".join(expected_ids)
        )

        assert tuple(case.case_id for case in selected) == expected_ids


def test_kimi_k2_adds_only_its_model_specific_customer_regressions() -> None:
    generic_ids = set(qualification_generic_case_ids())
    expected_ids = qualification_case_ids("kimi_k2")
    model_specific_ids = set(expected_ids) - generic_ids

    assert probe.model_case_profile("moonshotai/Kimi-K2.6") == "kimi_k2"
    assert model_specific_ids == {
        "customer_kimi_consume_prior_tool_result",
        "customer_kimi_parallel_weather_final_answer",
    }
    selected = probe.select_cases(probe.build_cases("kimi_k2"), ",".join(expected_ids))
    assert tuple(case.case_id for case in selected) == expected_ids
    assert all(
        case.profiles == ("kimi_k2",)
        for case in selected
        if case.case_id in model_specific_ids
    )


def test_customer_regressions_are_in_the_qualification_profile() -> None:
    qualification_ids = set(qualification_case_ids("kimi_k2"))
    customer_cases = {
        case.case_id: case
        for case in probe.build_cases("all")
        if case.case_id.startswith("customer_")
    }

    assert len(customer_cases) == 10
    assert set(customer_cases) == {
        case_id for case_id in qualification_ids if case_id.startswith("customer_")
    }
    assert "customer_calculate_sum_auto" in customer_cases


def test_auto_calculate_sum_prompt_explicitly_requires_the_tool() -> None:
    case = next(
        case
        for case in probe.build_cases("generic")
        if case.case_id == "customer_calculate_sum_auto"
    )

    prompt = case.messages[0]["content"]
    assert case.tool_choice == "auto"
    assert "Use the calculate_sum tool" in prompt
    assert "do not calculate the answer yourself" in prompt


def test_natural_language_fragment_matching_normalizes_unicode_dashes() -> None:
    case = probe.Case(
        case_id="unicode_dash",
        description="Unicode dash variants match ASCII expected fragments",
        messages=(),
        expected_finish_reasons=("stop",),
        expect_no_tool_calls=True,
        min_tool_calls=0,
        expected_any_content_fragments=("multi-step",),
        expected_final_content_fragments=("multi-step",),
    )
    result = probe.ChatResult(
        content="The endpoint supports streaming multi‑step tool calling.",
        finish_reason="stop",
    )

    errors, _warnings = probe.validate_result(case, result)
    assert errors == []
    assert probe.validate_agent_final(case, result) == []


def test_failure_classifier_separates_failure_ownership() -> None:
    required_case = probe.Case(
        case_id="required_call",
        description="required call",
        messages=(),
        expected_finish_reasons=("tool_calls",),
        tool_choice="required",
    )
    parallel_case = probe.Case(
        case_id="parallel_call",
        description="parallel calls",
        messages=(),
        expected_finish_reasons=("tool_calls",),
        tool_choice="required",
        min_tool_calls=2,
    )
    one_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }

    assert (
        probe.classify_failure(
            required_case,
            probe.ChatResult(finish_reason="stop"),
            [probe.error("unexpected_finish_reason", "stopped without a call")],
        )
        == "dynamo_api"
    )
    assert (
        probe.classify_failure(
            parallel_case,
            probe.ChatResult(finish_reason="tool_calls", tool_calls=[one_call]),
            [probe.error("too_few_tool_calls", "expected two calls")],
        )
        == "model_quality"
    )
    assert (
        probe.classify_failure(
            required_case,
            probe.ChatResult(),
            [probe.error("request_error", "connection timed out")],
        )
        == "infrastructure"
    )
    assert (
        probe.classify_failure(
            required_case,
            probe.ChatResult(),
            [probe.error("unexpected_finish_reason", "missing finish reason")],
        )
        == "dynamo_api"
    )
    assert (
        probe.classify_failure(
            required_case,
            probe.ChatResult(),
            [probe.error("new_unknown_failure", "not classified yet")],
        )
        == "unclassified"
    )


def test_failure_classifier_uses_conservative_primary_precedence() -> None:
    case = probe.Case(
        case_id="mixed_failure",
        description="mixed failure",
        messages=(),
        expected_finish_reasons=("stop",),
    )
    result = probe.ChatResult(finish_reason="stop")

    assert (
        probe.classify_failure(
            case,
            result,
            [
                probe.error("wrong_tool_call_count", "model miss"),
                probe.error("invalid_arguments_json", "parser output was invalid"),
            ],
        )
        == "dynamo_api"
    )
    assert (
        probe.classify_failure(
            case,
            result,
            [
                probe.error("invalid_arguments_json", "parser output was invalid"),
                probe.error("request_error", "connection reset"),
            ],
        )
        == "infrastructure"
    )


def test_static_summary_counts_each_failed_record_once() -> None:
    records = [
        {"pass": True},
        {"pass": False, "failure_category": "dynamo_api"},
        {"pass": False, "failure_category": "infrastructure"},
        {"pass": False, "failure_category": "model_quality"},
        {"pass": False, "failure_category": "unknown_future_category"},
    ]

    summary = static_report.summarize(records)

    assert summary["failed"] == 4
    assert summary["failure_categories"] == {
        "dynamo_api": 1,
        "infrastructure": 1,
        "model_quality": 1,
        "unclassified": 1,
    }
    assert sum(summary["failure_categories"].values()) == summary["failed"]
    assert summary["dynamo_errors"] == 1
    assert summary["serving_errors"] == 1


def test_customer_marker_regressions_include_recent_native_formats() -> None:
    assert "]<]minimax[>[" in probe.RAW_TOOL_MARKERS
    assert "<|call|>" in probe.RAW_TOOL_MARKERS


def test_exclusions_cannot_produce_an_empty_run() -> None:
    case = probe.build_cases("generic")[0]

    try:
        probe.select_cases((case,), "all", "*")
    except ValueError as exc:
        assert str(exc) == "case selection is empty after exclusions"
    else:
        raise AssertionError("empty qualification selection should fail")


def test_static_report_applies_the_fixed_qualification_selection() -> None:
    expected_ids = qualification_generic_case_ids()
    args = static_report.build_parser().parse_args(
        [
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--allow-other-base-url",
            "--no-auth",
            "--model",
            "google/gemma-4-31B-it",
            "--case-profile",
            "gemma4",
            "--cases",
            ",".join(expected_ids),
            "--dry-run",
        ]
    )

    report, records = static_report.run_probe(args)

    assert records == []
    assert tuple(report["config"]["case_ids"]) == expected_ids
    assert report["config"]["exclude_cases"] == ""


def test_static_report_writes_the_harness_artifact_contract(tmp_path: Path) -> None:
    records = [
        {
            "case_id": "plain_no_tools",
            "mode": "nonstream",
            "pass": True,
            "errors": [],
        }
    ]
    report = {
        "title": "qualification",
        "summary": {"passed": 1, "failed": 0, "total": 1},
    }

    static_report.write_static_site(
        report,
        records,
        site_dir=tmp_path,
        model_slug="qualification",
        root_alias=False,
    )

    page = tmp_path / "models" / "qualification"
    assert (page / "index.html").is_file()
    assert json.loads((page / "artifacts" / "latest.json").read_text()) == report
    assert (
        json.loads((page / "artifacts" / "results.public.jsonl").read_text())
        == records[0]
    )


def test_shared_python_identifiers_are_model_neutral() -> None:
    shared_modules = (
        ROOT / "model_profiles.py",
        ROOT / "tool_calling_probe.py",
        ROOT / "tool_calling_static_report.py",
        ROOT.parent / "custom_runner.py",
        ROOT.parent / "e2e_verifier" / "cli.py",
    )

    for path in shared_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        identifiers = [
            node.name.casefold()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert not any("kimi" in name or "k3" in name for name in identifiers), path
