# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import dataclasses
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


def case_by_id(cases: tuple[probe.Case, ...], case_id: str) -> probe.Case:
    return next(case for case in cases if case.case_id == case_id)


def error_kinds(errors: list[dict]) -> set[str]:
    return {str(item.get("kind")) for item in errors}


def test_qualification_uses_the_same_61_generic_cases_for_inline_profiles() -> None:
    expected_ids = qualification_generic_case_ids()

    assert len(expected_ids) == 61
    assert len(set(expected_ids)) == 61
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
        "minimax_m3",
        "kimi_k3",
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


def test_kimi_models_resolve_by_generation() -> None:
    assert probe.model_case_profile("moonshotai/Kimi-K2.5") == "kimi_k2"
    assert probe.model_case_profile("moonshotai/Kimi-K2.6") == "kimi_k2"
    assert probe.model_case_profile("nvidia/Kimi-K2.6-NVFP4") == "kimi_k2"
    assert probe.model_case_profile("moonshotai/Kimi-K3") == "kimi_k3"
    # K3 has no model-specific cases yet, so its profile selects exactly the
    # generic qualification set; the K2 customer regressions stay on kimi_k2.
    assert {case.case_id for case in probe.build_cases("kimi_k3")} == set(
        qualification_generic_case_ids()
    )


def test_minimax_models_resolve_by_generation() -> None:
    assert probe.model_case_profile("MiniMaxAI/MiniMax-M2.7") == "minimax_m2"
    assert probe.model_case_profile("MiniMaxAI/MiniMax-M2.5") == "minimax_m2"
    assert probe.model_case_profile("nvidia/MiniMax-M3-NVFP4") == "minimax_m3"
    assert probe.model_case_profile("MiniMaxAI/MiniMax-M3") == "minimax_m3"
    assert probe.model_case_profile("MiniMaxAI/MiniMax-M1") == "generic"
    # M3 has no model-specific cases yet: its profile selects exactly the
    # generic qualification set, while M2 adds its parser-stress cases.
    assert {case.case_id for case in probe.build_cases("minimax_m3")} == set(
        qualification_generic_case_ids()
    )
    assert {case.case_id for case in probe.build_cases("minimax_m2")} > set(
        qualification_generic_case_ids()
    )


def test_qwen36_27b_resolves_to_the_complete_qwen_xml_profile() -> None:
    assert probe.model_case_profile("Qwen/Qwen3.6-27B") == "qwen3_coder_xml"
    cases = probe.build_cases("qwen3_coder_xml")
    assert len(cases) == 63
    assert all(case.request_overrides == {} for case in cases)


def test_qwen_xml_profile_covers_explicit_strict_policy() -> None:
    cases = probe.build_cases("qwen3_coder_xml")
    strict_true = case_by_id(cases, "qx_policy_explicit_strict_true_auto")
    strict_false = case_by_id(cases, "qx_policy_explicit_strict_false_auto")

    assert strict_true.tool_choice == strict_false.tool_choice == "auto"
    assert strict_true.messages == strict_false.messages
    assert strict_true.tools[0]["function"]["strict"] is True
    assert strict_false.tools[0]["function"]["strict"] is False
    assert strict_true.validate_schema is True
    assert strict_false.validate_schema is False

    for case, expected_strict in ((strict_true, True), (strict_false, False)):
        payload = probe.build_payload(
            case,
            model="Qwen/Qwen3.6-27B",
            stream=False,
            temperature=0.0,
            default_max_tokens=1024,
        )
        assert payload["tools"][0]["function"]["strict"] is expected_strict


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


def test_glm_forced_and_named_cases_require_reasoning_when_enabled() -> None:
    cases = probe.build_cases("glm5")

    for case_id in (
        "customer_required_forces_weather",
        "customer_named_calculator_choice",
    ):
        case = case_by_id(cases, case_id)
        assert case.request_overrides["chat_template_kwargs"] == {
            "enable_thinking": True,
            "thinking": True,
        }
        assert case.expect_reasoning is True

        missing = probe.ChatResult(
            reasoning_content=None,
            finish_reason="tool_calls",
        )
        missing_errors, _warnings = probe.validate_result(case, missing)
        assert "missing_reasoning_content" in error_kinds(missing_errors)

        present = probe.ChatResult(
            reasoning_content="The request requires a forced tool call.",
            finish_reason="tool_calls",
        )
        present_errors, _warnings = probe.validate_result(case, present)
        assert "missing_reasoning_content" not in error_kinds(present_errors)


def test_glm_thinking_disabled_cases_permit_null_reasoning() -> None:
    cases = probe.build_cases("glm5")

    for case_id in (
        "customer_required_forces_weather_thinking_disabled",
        "customer_named_calculator_choice_thinking_disabled",
    ):
        case = case_by_id(cases, case_id)
        assert case.request_overrides["chat_template_kwargs"] == {
            "enable_thinking": False,
            "thinking": False,
        }
        assert case.expect_reasoning is False

        result = probe.ChatResult(
            reasoning_content=None,
            finish_reason="tool_calls",
        )
        errors, _warnings = probe.validate_result(case, result)
        kinds = error_kinds(errors)
        assert "missing_reasoning_content" not in kinds
        assert "unexpected_reasoning_content" not in kinds


def test_reasoning_expectations_cover_every_inline_profile() -> None:
    enabled_case_ids = (
        "customer_required_forces_weather",
        "customer_named_calculator_choice",
    )
    disabled_case_ids = (
        "customer_required_forces_weather_thinking_disabled",
        "customer_named_calculator_choice_thinking_disabled",
        "customer_truncated_tool_markup_hidden",
        "followup_after_tool_result_thinking_disabled",
        "required_truncated_call_reports_length",
        "required_max_completion_tokens_alias_truncates",
    )

    for profile in probe.INLINE_CASE_PROFILES:
        cases = probe.build_cases(profile)
        for case_id in enabled_case_ids:
            assert case_by_id(cases, case_id).expect_reasoning is True

        disabled_expectation = None if profile == "gpt_oss" else False
        for case_id in disabled_case_ids:
            assert case_by_id(cases, case_id).expect_reasoning is disabled_expectation


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


def test_argument_string_normalization_is_opt_in_and_preserves_types() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "report_status",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["body", "count"],
            },
        },
    }
    case = probe.Case(
        case_id="normalized_prose",
        description="normalize only prose-valued string arguments",
        messages=(),
        tools=(tool,),
        expected_finish_reasons=("tool_calls",),
        expected_tool_calls=(
            {"name": "report_status", "arguments": {"body": "All green", "count": 7}},
        ),
        normalize_argument_strings=True,
    )

    def result(arguments: dict) -> probe.ChatResult:
        return probe.ChatResult(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "report_status",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        )

    normalized_errors, _ = probe.validate_result(
        case, result({"body": "all GREEN!", "count": 7})
    )
    assert "unexpected_tool_calls" not in error_kinds(normalized_errors)

    strict_errors, _ = probe.validate_result(
        dataclasses.replace(case, normalize_argument_strings=False),
        result({"body": "all GREEN!", "count": 7}),
    )
    assert "unexpected_tool_calls" in error_kinds(strict_errors)

    wrong_type_errors, _ = probe.validate_result(
        case, result({"body": "All green.", "count": "7"})
    )
    assert "unexpected_tool_calls" in error_kinds(wrong_type_errors)
    # A numeric string standing in for an integer is reported as the typing
    # defect it is, not as generic schema noise.
    assert "argument_stringified_scalar" in error_kinds(wrong_type_errors)
    assert "schema_validation" not in error_kinds(wrong_type_errors)


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


def test_every_known_error_kind_has_exactly_one_owner_and_subcategory() -> None:
    owned = (
        set(probe.DYNAMO_API_ERROR_KINDS)
        | set(probe.MODEL_QUALITY_ERROR_KINDS)
        | set(probe.INFRASTRUCTURE_ERROR_KINDS)
    )
    assert not (probe.DYNAMO_API_ERROR_KINDS & probe.MODEL_QUALITY_ERROR_KINDS)
    for kind in owned:
        assert kind in probe.ERROR_KIND_SUBCATEGORIES, kind
    for kind, subcategory in probe.ERROR_KIND_SUBCATEGORIES.items():
        if kind in probe.DYNAMO_API_ERROR_KINDS:
            category = "dynamo_api"
        elif kind in probe.MODEL_QUALITY_ERROR_KINDS:
            category = "model_quality"
        elif kind in probe.INFRASTRUCTURE_ERROR_KINDS:
            category = "infrastructure"
        elif kind == "probe_exception":
            category = "unclassified"
        elif "marker_leaked" in kind:
            category = "dynamo_api"
        elif kind in ("unexpected_tool_calls", "unexpected_finish_reason"):
            # Ownership depends on tool_choice; the static entry is the
            # model-quality default and the dynamo_api case is derived.
            category = "model_quality"
        else:
            # Kinds that classify by prefix (missing_expected_*, too_few_*).
            assert kind.startswith(("missing_expected_", "too_few_")), kind
            category = "model_quality"
        assert subcategory in probe.FAILURE_SUBCATEGORIES[category], (kind, category)


def test_failure_subcategory_follows_category_precedence() -> None:
    required_case = probe.Case(
        case_id="required_call",
        description="required call",
        messages=(),
        expected_finish_reasons=("tool_calls",),
        tool_choice="required",
    )
    auto_case = probe.Case(
        case_id="auto_call",
        description="auto call",
        messages=(),
        expected_finish_reasons=("tool_calls",),
    )

    assert probe.classify_failure_detail(
        required_case,
        probe.ChatResult(finish_reason="stop"),
        [probe.error("unexpected_finish_reason", "stopped without a call")],
    ) == ("dynamo_api", "tool_choice_enforcement")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(finish_reason="stop", content="I will not."),
        [probe.error("unexpected_finish_reason", "model declined")],
    ) == ("model_quality", "tool_selection")
    assert probe.classify_failure_detail(
        required_case,
        probe.ChatResult(),
        [probe.error("unexpected_finish_reason", "missing finish reason")],
    ) == ("dynamo_api", "response_shape")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(finish_reason="tool_calls"),
        [
            probe.error("schema_validation", "model miss"),
            probe.error("tool_marker_leaked_to_content", "leak"),
            probe.error("invalid_arguments_json", "parser output was invalid"),
        ],
    ) == ("dynamo_api", "argument_encoding")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(),
        [probe.error("http_error", "HTTP 503", status=503)],
    ) == ("infrastructure", "server_error")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(),
        [probe.error("http_error", "HTTP 429", status=429)],
    ) == ("infrastructure", "rate_limited")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(),
        [probe.error("http_error", "HTTP 400", status=400)],
    ) == ("dynamo_api", "request_validation")
    assert probe.classify_failure_detail(
        auto_case,
        probe.ChatResult(),
        [probe.error("new_unknown_failure", "not classified yet")],
    ) == ("unclassified", "unknown_kind")
    assert probe.classify_failure_detail(auto_case, probe.ChatResult(), []) == (
        None,
        None,
    )


def _call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_stringified_scalars_are_a_dynamo_typing_defect_not_schema_noise() -> None:
    case = probe.Case(
        case_id="typed",
        description="typed",
        messages=(),
        tools=(probe.TOOLS["book_flight"],),
        tool_choice="required",
        expected_tool_names=("book_flight",),
    )
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call(
                "book_flight",
                {"destination": "Lisbon", "passengers": "2", "first_class": "false"},
            )
        ],
    )

    errors, _ = probe.validate_result(case, result)
    kinds = [item["kind"] for item in errors]

    assert kinds.count("argument_stringified_scalar") == 2
    assert "schema_validation" not in kinds
    assert probe.classify_failure_detail(case, result, errors) == (
        "dynamo_api",
        "argument_encoding",
    )

    genuinely_wrong = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call("book_flight", {"destination": "Lisbon", "passengers": "two"})
        ],
    )
    wrong_errors, _ = probe.validate_result(case, genuinely_wrong)
    assert "schema_validation" in error_kinds(wrong_errors)
    assert "argument_stringified_scalar" not in error_kinds(wrong_errors)


def test_reasoning_only_and_empty_final_messages_are_dynamo_failures() -> None:
    case = probe.Case(
        case_id="answer",
        description="answer",
        messages=(),
        tools=(probe.TOOLS["get_weather"],),
        expected_finish_reasons=("stop",),
        expect_no_tool_calls=True,
        min_tool_calls=0,
    )

    reasoning_only = probe.ChatResult(
        finish_reason="stop", reasoning_content="The answer is 15 and cloudy."
    )
    errors, _ = probe.validate_result(case, reasoning_only)
    assert "reasoning_only_response" in error_kinds(errors)
    assert probe.classify_failure_detail(case, reasoning_only, errors) == (
        "dynamo_api",
        "reasoning_routing",
    )

    empty = probe.ChatResult(finish_reason=None)
    errors, _ = probe.validate_result(case, empty)
    assert "empty_response" in error_kinds(errors)

    truncated = probe.ChatResult(finish_reason="length", reasoning_content="hmm")
    errors, _ = probe.validate_result(case, truncated)
    assert "reasoning_only_response" not in error_kinds(errors)
    assert "empty_response" not in error_kinds(errors)

    answered = probe.ChatResult(
        finish_reason="stop", content="15 and cloudy.", reasoning_content="hmm"
    )
    errors, _ = probe.validate_result(case, answered)
    assert errors == []


def test_marker_and_json_leaks_into_content_are_reported() -> None:
    case = probe.Case(
        case_id="parallel",
        description="parallel",
        messages=(),
        tools=(probe.TOOLS["get_weather"],),
        expected_tool_names=("get_weather", "get_weather"),
        min_tool_calls=2,
    )
    result = probe.ChatResult(
        finish_reason="tool_calls",
        content='{"name": "get_weather", "arguments": {"location": "Seoul"}}',
        tool_calls=[_call("get_weather", {"location": "Madrid"})],
    )
    errors, _ = probe.validate_result(case, result)
    assert "tool_call_json_leaked_to_content" in error_kinds(errors)
    assert probe.classify_failure_detail(case, result, errors) == (
        "dynamo_api",
        "marker_leak",
    )

    thought_leak = probe.ChatResult(
        finish_reason="stop",
        content="<think>\n</think>It is cloudy.",
    )
    plain_case = probe.Case(
        case_id="plain",
        description="plain",
        messages=(),
        expected_finish_reasons=("stop",),
        expect_no_tool_calls=True,
        min_tool_calls=0,
    )
    errors, _ = probe.validate_result(plain_case, thought_leak)
    assert "reasoning_marker_leaked_to_content" in error_kinds(errors)


def test_tool_call_ids_must_not_repeat_history_ids() -> None:
    cases = probe.build_cases("generic")
    case = case_by_id(cases, "history_second_turn_new_tool_call")
    assert probe.history_tool_call_ids(case.messages) == {"call_history_london_0"}

    reused = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call("get_weather", {"location": "Berlin"}, "call_history_london_0")
        ],
    )
    errors, _ = probe.validate_result(case, reused)
    assert "tool_id_reused_from_history" in error_kinds(errors)
    assert probe.classify_failure_detail(case, reused, errors) == (
        "dynamo_api",
        "tool_call_identity",
    )

    fresh = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("get_weather", {"location": "Berlin"}, "call_berlin_1")],
    )
    errors, _ = probe.validate_result(case, fresh)
    assert errors == []


def test_content_order_expectation_matches_the_minimax_scenario_check() -> None:
    case = case_by_id(probe.build_cases("generic"), "auto_parameter_order_recall")
    assert case.expected_content_order == ("zone", "alias", "firmware", "serial")
    assert list(case.tools[0]["function"]["parameters"]["properties"]) == list(
        case.expected_content_order
    )

    ordered = probe.ChatResult(
        finish_reason="stop", content="Parameters: zone, alias, firmware, serial."
    )
    errors, _ = probe.validate_result(case, ordered)
    assert errors == []

    sorted_by_provider = probe.ChatResult(
        finish_reason="stop", content="alias, firmware, serial, zone"
    )
    errors, _ = probe.validate_result(case, sorted_by_provider)
    assert "content_order_mismatch" in error_kinds(errors)
    assert probe.classify_failure_detail(case, sorted_by_provider, errors) == (
        "model_quality",
        "content_quality",
    )


def test_request_validation_cases_pass_only_on_the_expected_status() -> None:
    case = case_by_id(probe.build_cases("generic"), "reject_empty_tools_array")
    assert case.expected_http_status == 400
    payload = probe.build_payload(
        case,
        model="m",
        stream=False,
        temperature=0.0,
        default_max_tokens=64,
    )
    assert payload["tools"] == []
    assert "tool_choice" not in payload

    null_choice = case_by_id(
        probe.build_cases("generic"), "tool_choice_null_behaves_as_auto"
    )
    null_payload = probe.build_payload(
        null_choice,
        model="m",
        stream=False,
        temperature=0.0,
        default_max_tokens=64,
    )
    assert "tool_choice" in null_payload and null_payload["tool_choice"] is None
    assert null_payload["chat_template_kwargs"] == {"thinking": True}

    no_tools = case_by_id(
        probe.build_cases("generic"), "history_tool_round_trip_without_tools"
    )
    no_tools_payload = probe.build_payload(
        no_tools,
        model="m",
        stream=False,
        temperature=0.0,
        default_max_tokens=64,
    )
    assert "tools" not in no_tools_payload and "tool_choice" not in no_tools_payload


def test_new_generic_cases_cover_every_inline_profile_without_model_hooks() -> None:
    new_ids = {
        "agent_weather_then_final_answer",
        "auto_parameter_order_recall",
        "followup_after_tool_error_result",
        "followup_after_tool_result_plain_answer",
        "followup_after_tool_result_thinking_disabled",
        "history_second_turn_new_tool_call",
        "history_tool_round_trip_without_tools",
        "named_dotted_tool_name",
        "named_literal_control_characters",
        "named_literal_multibyte_text",
        "named_numeric_types_preserved",
        "named_strict_anyof_argument",
        "parallel_tool_calls_false_single_call",
        "reject_empty_tools_array",
        "required_schema_local_definitions_ref",
        "tool_choice_null_behaves_as_auto",
        "auto_long_system_context_tool_call",
        "auto_parallel_three_distinct_tools",
        "auto_system_prompt_directs_tool_use",
        "auto_user_pasted_tool_call_json_is_data",
        "followup_after_two_tool_results_compares",
        "named_array_of_objects_arguments",
        "named_falsy_values_preserved",
        "named_nullable_argument_json_null",
        "auto_stop_string_does_not_cut_tool_call",
        "auto_tool_description_with_markup_is_text",
        "named_optional_argument_omitted_not_null",
        "none_with_tool_history_answers_in_text",
        "plain_usage_counts_consistent",
        "reject_named_tool_choice_not_in_tools",
        "required_truncated_call_reports_length",
        "auto_many_tools_selects_right_one",
        "named_tool_without_parameters_key",
        "followup_after_long_tool_result",
        "history_three_round_trips_then_answer",
        "required_max_completion_tokens_alias_truncates",
        "followup_tool_result_as_content_parts",
    }
    assert new_ids <= set(qualification_generic_case_ids())
    for profile in probe.INLINE_CASE_PROFILES:
        cases = {case.case_id: case for case in probe.build_cases(profile)}
        assert new_ids <= set(cases)
        disabled = cases["followup_after_tool_result_thinking_disabled"]
        assert disabled.expect_reasoning is (None if profile == "gpt_oss" else False)
    dotted = case_by_id(probe.build_cases("generic"), "named_dotted_tool_name")
    assert dotted.tools[0]["function"]["name"] == "weather.get_current"
    ref_case = case_by_id(
        probe.build_cases("generic"), "required_schema_local_definitions_ref"
    )
    assert ref_case.tools[0]["function"]["parameters"]["properties"]["city"] == {
        "$ref": "#/definitions/City"
    }


def test_static_summary_counts_subcategories_and_kinds() -> None:
    records = [
        {"pass": True},
        {
            "pass": False,
            "failure_category": "dynamo_api",
            "failure_subcategory": "marker_leak",
            "failure_kinds": ["tool_marker_leaked_to_content", "too_few_tool_calls"],
        },
        {
            "pass": False,
            "failure_category": "dynamo_api",
            "failure_subcategory": "not_a_real_subcategory",
            "failure_kinds": ["too_few_tool_calls"],
        },
        {"pass": False, "failure_category": "model_quality"},
    ]

    summary = static_report.summarize(records)

    assert summary["failed"] == 3
    assert summary["failure_subcategories"]["dynamo_api.marker_leak"] == 1
    assert summary["failure_subcategories"]["dynamo_api.reasoning_routing"] == 1
    assert summary["failure_subcategories"]["model_quality.agent_loop"] == 1
    assert sum(summary["failure_subcategories"].values()) == summary["failed"]
    assert summary["failure_kinds"] == {
        "too_few_tool_calls": 2,
        "tool_marker_leaked_to_content": 1,
    }


def test_typing_edge_cases_expect_json_null_and_falsy_values_verbatim() -> None:
    cases = probe.build_cases("generic")
    nullable = case_by_id(cases, "named_nullable_argument_json_null")
    assert ("set_reminder", "note", None) in nullable.expected_arg_values
    stringified_null = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call(
                "set_reminder",
                {"title": "standup", "minutes_from_now": 15, "note": "null"},
            )
        ],
    )
    errors, _ = probe.validate_result(nullable, stringified_null)
    assert "missing_expected_argument_value" in error_kinds(errors)

    falsy = case_by_id(cases, "named_falsy_values_preserved")
    ok = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call(
                "update_counter",
                {"count": 0, "enabled": False, "label": "", "priority": 0},
            )
        ],
    )
    errors, _ = probe.validate_result(falsy, ok)
    assert errors == []
    stringified = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[
            _call(
                "update_counter",
                {"count": "0", "enabled": "false", "label": "", "priority": 0},
            )
        ],
    )
    errors, _ = probe.validate_result(falsy, stringified)
    assert error_kinds(errors) >= {
        "argument_stringified_scalar",
        "missing_expected_argument_value",
    }
    assert probe.classify_failure_detail(falsy, stringified, errors) == (
        "dynamo_api",
        "argument_encoding",
    )


def test_usage_expectation_checks_presence_and_arithmetic() -> None:
    case = case_by_id(probe.build_cases("generic"), "plain_usage_counts_consistent")
    assert case.expect_usage is True
    payload = probe.build_payload(
        case, model="m", stream=True, temperature=0.0, default_max_tokens=64
    )
    assert payload["stream_options"] == {"include_usage": True}

    good = probe.ChatResult(
        finish_reason="stop",
        content="A tokenizer splits text into tokens.",
        usage={"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
    )
    errors, _ = probe.validate_result(case, good)
    assert errors == []

    missing = probe.ChatResult(finish_reason="stop", content="Splits text.")
    errors, _ = probe.validate_result(case, missing)
    assert "usage_missing" in error_kinds(errors)
    assert probe.classify_failure_detail(case, missing, errors) == (
        "dynamo_api",
        "response_shape",
    )

    wrong_sum = probe.ChatResult(
        finish_reason="stop",
        content="Splits text.",
        usage={"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 20},
    )
    errors, _ = probe.validate_result(case, wrong_sum)
    assert "usage_inconsistent" in error_kinds(errors)


def test_truncated_required_call_must_report_length() -> None:
    case = case_by_id(
        probe.build_cases("generic"), "required_truncated_call_reports_length"
    )
    assert case.max_tokens == 24
    mislabeled = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("record_literal", {"label": "cut-eof"})],
    )
    errors, _ = probe.validate_result(case, mislabeled)
    assert "unexpected_finish_reason" in error_kinds(errors)
    assert probe.classify_failure_detail(case, mislabeled, errors) == (
        "dynamo_api",
        "finish_reason",
    )
    truncated = probe.ChatResult(finish_reason="length")
    errors, _ = probe.validate_result(case, truncated)
    assert errors == []


def test_fourth_batch_request_shapes() -> None:
    cases = probe.build_cases("generic")
    many = case_by_id(cases, "auto_many_tools_selects_right_one")
    names = [tool["function"]["name"] for tool in many.tools]
    assert len(names) == 20 and len(set(names)) == 20 and "get_weather" in names

    no_params = case_by_id(cases, "named_tool_without_parameters_key")
    assert "parameters" not in no_params.tools[0]["function"]
    ok = probe.ChatResult(
        finish_reason="tool_calls", tool_calls=[_call("ping_service", {})]
    )
    errors, _ = probe.validate_result(no_params, ok)
    assert errors == []

    alias = case_by_id(cases, "required_max_completion_tokens_alias_truncates")
    payload = probe.build_payload(
        alias, model="m", stream=False, temperature=0.0, default_max_tokens=4096
    )
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 24

    long_result = case_by_id(cases, "followup_after_long_tool_result")
    assert len(long_result.messages[-1]["content"]) > 4000

    parts = case_by_id(cases, "followup_tool_result_as_content_parts")
    assert isinstance(parts.messages[-1]["content"], list)

    rounds = case_by_id(cases, "history_three_round_trips_then_answer")
    assert probe.history_tool_call_ids(rounds.messages) == {
        "call_round_0",
        "call_round_1",
        "call_round_2",
    }


def test_every_thinking_disabled_case_asserts_no_reasoning() -> None:
    """A case that turns thinking off must also assert that none happened."""
    for profile in probe.INLINE_CASE_PROFILES:
        expected = None if profile == "gpt_oss" else False
        for case in probe.build_cases(profile):
            overrides = case.request_overrides or {}
            kwargs = overrides.get("chat_template_kwargs") or {}
            disabled = any(
                kwargs.get(key) is False for key in ("thinking", "enable_thinking")
            )
            if disabled:
                assert case.expect_reasoning is expected, (profile, case.case_id)


def test_thinking_disabled_checks_reasoning_tokens_not_only_the_field() -> None:
    cases = probe.build_cases("generic")
    case = case_by_id(cases, "customer_required_forces_weather_thinking_disabled")
    assert case.expect_reasoning is False

    hidden_thinking = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 300,
            "total_tokens": 350,
            "completion_tokens_details": {"reasoning_tokens": 280},
        },
    )
    errors, _ = probe.validate_result(case, hidden_thinking)
    assert "unexpected_reasoning_tokens" in error_kinds(errors)
    assert "unexpected_reasoning_content" not in error_kinds(errors)
    assert probe.classify_failure_detail(case, hidden_thinking, errors) == (
        "dynamo_api",
        "reasoning_routing",
    )

    clean = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    )
    errors, _ = probe.validate_result(case, clean)
    assert errors == []

    # Backends that report no usage details are not penalized.
    no_details = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
        usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    )
    errors, _ = probe.validate_result(case, no_details)
    assert errors == []

    enabled = case_by_id(cases, "customer_required_forces_weather")
    dropped = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 300,
            "total_tokens": 350,
            "completion_tokens_details": {"reasoning_tokens": 280},
        },
    )
    errors, _ = probe.validate_result(enabled, dropped)
    assert {"missing_reasoning_content", "reasoning_dropped"} <= error_kinds(errors)
    assert probe.classify_failure_detail(enabled, dropped, errors) == (
        "dynamo_api",
        "reasoning_routing",
    )


def test_reasoning_folded_into_content_is_caught_without_markers() -> None:
    """dynamo #17521 / #17578: template ignores thinking=false, frontend folds
    the parsed reasoning into content. No marker, maybe no usage detail."""
    case = case_by_id(
        probe.build_cases("generic"),
        "customer_required_forces_weather_thinking_disabled",
    )
    folded = probe.ChatResult(
        finish_reason="tool_calls",
        content=(
            "The user wants me to say hello briefly, but a tool call is required. "
            "I need to call get_weather with some location. Let me pick Paris and "
            "produce the call exactly as the schema asks, with no extra text."
        ),
        tool_calls=[_call("get_weather", {"location": "Paris"})],
        usage={"prompt_tokens": 40, "completion_tokens": 90, "total_tokens": 130},
    )
    errors, _ = probe.validate_result(case, folded)
    assert "reasoning_text_in_content" in error_kinds(errors)
    assert "unexpected_reasoning_content" not in error_kinds(errors)
    assert probe.classify_failure_detail(case, folded, errors) == (
        "dynamo_api",
        "reasoning_routing",
    )

    # MiniMax M3 markers leaking into content are caught by the marker check.
    marked = probe.ChatResult(
        finish_reason="tool_calls",
        content="<mm:think>Let me think.</mm:think>",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
    )
    errors, _ = probe.validate_result(case, marked)
    assert "reasoning_marker_leaked_to_content" in error_kinds(errors)

    # Short narration next to a forced call is not reasoning.
    narrated = probe.ChatResult(
        finish_reason="tool_calls",
        content="Checking the weather now.",
        tool_calls=[_call("get_weather", {"location": "Paris"})],
    )
    errors, _ = probe.validate_result(case, narrated)
    assert "reasoning_text_in_content" not in error_kinds(errors)
    assert not probe.looks_like_reasoning("Sure, I can help with that.")


def test_stream_options_only_travel_on_streaming_requests() -> None:
    case = case_by_id(probe.build_cases("generic"), "plain_usage_counts_consistent")
    streaming = probe.build_payload(
        case, model="m", stream=True, temperature=0.0, default_max_tokens=64
    )
    nonstreaming = probe.build_payload(
        case, model="m", stream=False, temperature=0.0, default_max_tokens=64
    )
    assert streaming["stream_options"] == {"include_usage": True}
    assert "stream_options" not in nonstreaming


def test_truncation_probes_do_not_schema_validate_the_partial_call() -> None:
    cases = probe.build_cases("generic")
    for case_id in (
        "required_truncated_call_reports_length",
        "required_max_completion_tokens_alias_truncates",
    ):
        case = case_by_id(cases, case_id)
        assert case.validate_schema is False
        partial = probe.ChatResult(
            finish_reason="length",
            tool_calls=[_call("record_literal", {"label": "cut-eof"})],
        )
        errors, _ = probe.validate_result(case, partial)
        assert errors == [], (case_id, errors)
