#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply Dynamo request contracts to the custom parser qualification matrix."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

RAW_REASONING_MARKERS = ("</think>", "<mm:think>", "</mm:think>")


def apply_request_contract(
    cases: Iterable[Any], contract: dict[str, Any]
) -> tuple[Any, ...]:
    enabled = contract.get("enabled", {})
    disabled = contract.get("disabled", {})
    if not isinstance(enabled, dict) or not isinstance(disabled, dict):
        raise ValueError("request contract enabled/disabled values must be objects")
    if not enabled and not disabled:
        return tuple(cases)
    result: list[Any] = []
    for case in cases:
        overrides = dict(getattr(case, "request_overrides", None) or {})
        current = overrides.get("chat_template_kwargs")
        current = dict(current) if isinstance(current, dict) else {}
        is_disabled = any(value is False for value in current.values())
        expected = disabled if is_disabled else enabled
        if expected:
            current.update(expected)
            overrides["chat_template_kwargs"] = current
        result.append(dataclasses.replace(case, request_overrides=overrides))
    return tuple(result)


def requested_case_ids(arguments: list[str]) -> tuple[str, ...]:
    if "--cases" not in arguments:
        return ()
    index = arguments.index("--cases")
    if index + 1 >= len(arguments):
        raise ValueError("--cases requires a comma-separated value")
    return tuple(
        case_id.strip()
        for case_id in arguments[index + 1].split(",")
        if case_id.strip()
    )


def add_requested_catalog_cases(
    cases: Iterable[Any],
    requested_ids: Iterable[str],
    catalog_cases: Iterable[Any],
) -> tuple[Any, ...]:
    result = list(cases)
    available_ids = {case.case_id for case in result}
    missing_ids = set(requested_ids) - available_ids
    if not missing_ids:
        return tuple(result)
    catalog_by_id = {case.case_id: case for case in catalog_cases}
    result.extend(
        catalog_by_id[case_id]
        for case_id in sorted(missing_ids)
        if case_id in catalog_by_id
    )
    return tuple(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--custom-root", type=Path, required=True)
    parser.add_argument("--request-contract-json", required=True)
    args, report_args = parser.parse_known_args(argv)
    if report_args[:1] == ["--"]:
        report_args = report_args[1:]
    contract = json.loads(args.request_contract_json)
    if not isinstance(contract, dict):
        raise ValueError("request contract must be an object")
    custom_root = args.custom_root.resolve()
    if not (custom_root / "tool_calling_static_report.py").exists():
        raise ValueError(f"custom tool-calling tests are missing: {custom_root}")
    sys.path.insert(0, str(custom_root))
    report = importlib.import_module("tool_calling_static_report")
    original_build_cases = report.probe.build_cases
    original_validate_result = report.probe.validate_result
    declarative_profiles = set(report.probe.available_case_profiles())
    selected_case_ids = requested_case_ids(report_args)

    def contracted_build_cases(profile: str = "generic") -> tuple[Any, ...]:
        cases = list(
            add_requested_catalog_cases(
                original_build_cases(profile),
                selected_case_ids,
                original_build_cases("all"),
            )
        )
        if profile not in declarative_profiles and not any(
            case.case_id == "plain_no_tools_thinking_disabled" for case in cases
        ):
            cases.append(
                report.probe.Case(
                    case_id="plain_no_tools_thinking_disabled",
                    description=(
                        "plain response with thinking disabled must not leak "
                        "reasoning markers"
                    ),
                    messages=(
                        {
                            "role": "user",
                            "content": "What is 2+2? Answer with only the number.",
                        },
                    ),
                    tools=(),
                    tool_choice=None,
                    expected_finish_reasons=("stop",),
                    expect_no_tool_calls=True,
                    min_tool_calls=0,
                    expect_content=True,
                    validate_schema=False,
                    request_overrides={"chat_template_kwargs": {"thinking": False}},
                )
            )
        return apply_request_contract(cases, contract)

    def validate_with_reasoning_markers(case: Any, result: Any) -> tuple[Any, Any]:
        errors, warnings = original_validate_result(case, result)
        raw = json.dumps(result.raw_response, sort_keys=True, ensure_ascii=False)
        for marker in RAW_REASONING_MARKERS:
            marker_found = False
            if marker in (result.content or ""):
                marker_found = True
                errors.append(
                    report.probe.error(
                        "reasoning_marker_leaked_to_content",
                        f"content contains {marker!r}",
                    )
                )
            if marker in (result.reasoning_content or ""):
                marker_found = True
                errors.append(
                    report.probe.error(
                        "reasoning_marker_leaked_to_reasoning",
                        f"reasoning contains {marker!r}",
                    )
                )
            if not marker_found and marker in raw:
                errors.append(
                    report.probe.error(
                        "reasoning_marker_leaked_in_response",
                        f"response contains {marker!r}",
                    )
                )
        return errors, warnings

    report.probe.build_cases = contracted_build_cases
    report.probe.validate_result = validate_with_reasoning_markers
    original_argv = sys.argv
    try:
        sys.argv = [str(custom_root / "tool_calling_static_report.py"), *report_args]
        return int(report.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
