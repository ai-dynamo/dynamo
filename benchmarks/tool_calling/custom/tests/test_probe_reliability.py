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

import custom_runner  # noqa: E402
import tool_calling_probe as probe  # noqa: E402
import tool_calling_static_report as static_report  # noqa: E402

EXPECTED_CUSTOMER_REGRESSION_PRS = {
    "https://github.com/ai-dynamo/dynamo/pull/9778",
    "https://github.com/ai-dynamo/dynamo/pull/9804",
    "https://github.com/ai-dynamo/dynamo/pull/9864",
    "https://github.com/ai-dynamo/dynamo/pull/10030",
    "https://github.com/ai-dynamo/dynamo/pull/11045",
    "https://github.com/ai-dynamo/dynamo/pull/11205",
    "https://github.com/ai-dynamo/dynamo/pull/11554",
    "https://github.com/ai-dynamo/dynamo/pull/11653",
    "https://github.com/ai-dynamo/dynamo/pull/12684",
    "https://github.com/ai-dynamo/frontend-crates/pull/133",
    "https://github.com/ai-dynamo/frontend-crates/pull/152",
    "https://github.com/ai-dynamo/frontend-crates/pull/188",
}


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


def test_qualification_uses_the_same_25_generic_cases_for_inline_profiles() -> None:
    expected_ids = qualification_generic_case_ids()

    assert len(expected_ids) == 25
    assert len(set(expected_ids)) == 25
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


def test_customer_regressions_are_pinned_with_pr_provenance() -> None:
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

    sourced_cases = {
        case_id: case for case_id, case in customer_cases.items() if case.regression_prs
    }
    assert len(sourced_cases) == 9
    assert {
        pull_request
        for case in sourced_cases.values()
        for pull_request in case.regression_prs
    } == EXPECTED_CUSTOMER_REGRESSION_PRS
    assert all(
        pull_request.startswith("https://github.com/ai-dynamo/")
        for case in sourced_cases.values()
        for pull_request in case.regression_prs
    )


def test_customer_marker_regressions_include_recent_native_formats() -> None:
    assert "]<]minimax[>[" in probe.RAW_TOOL_MARKERS
    assert "<|call|>" in probe.RAW_TOOL_MARKERS


def test_qualification_adds_shared_cases_to_a_declarative_profile() -> None:
    expected_ids = qualification_generic_case_ids()
    profile_cases = probe.build_cases("kimi_k3")
    assert all(isinstance(case.regression_prs, tuple) for case in profile_cases)

    combined = custom_runner.add_requested_catalog_cases(
        profile_cases,
        expected_ids,
        probe.build_cases("all"),
    )
    selected = probe.select_cases(combined, ",".join(expected_ids))

    assert len(profile_cases) == 76
    assert tuple(case.case_id for case in selected) == expected_ids


def test_declarative_profiles_can_receive_model_specific_catalog_cases() -> None:
    expected_ids = qualification_case_ids("kimi_k2")
    combined = custom_runner.add_requested_catalog_cases(
        probe.build_cases("kimi_k3"),
        expected_ids,
        probe.build_cases("all"),
    )

    selected = probe.select_cases(combined, ",".join(expected_ids))

    assert tuple(case.case_id for case in selected) == expected_ids


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


def test_shared_python_identifiers_are_model_neutral() -> None:
    shared_modules = (
        ROOT / "case_profile_loader.py",
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
