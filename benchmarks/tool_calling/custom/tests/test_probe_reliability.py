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
