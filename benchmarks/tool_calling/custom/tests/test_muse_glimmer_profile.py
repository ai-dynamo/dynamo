# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_calling_probe as probe  # noqa: E402
import tool_calling_static_report as static_report  # noqa: E402


def test_muse_profile_contains_61_cases_and_122_two_mode_records() -> None:
    cases = probe.build_cases("muse_glimmer")

    assert len(cases) == 61
    assert len(cases) * 2 == 122
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.case_id.startswith("muse_") for case in cases)


def test_muse_profile_never_sends_a_thinking_control() -> None:
    for case in probe.build_cases("muse_glimmer"):
        kwargs = (case.request_overrides or {}).get("chat_template_kwargs", {})
        for key in (
            "thinking",
            "enable_thinking",
            "preserve_thinking",
            "thinking_effort",
            "reasoning_effort",
        ):
            assert key not in kwargs, f"{case.case_id} sends unsupported {key}"


def test_muse_reserved_markers_cover_recipient_framing_and_atem() -> None:
    forbidden = set(probe.build_cases("muse_glimmer")[0].forbidden_output_fragments)

    for marker in ("<|message|>", "<|start|>", "<|eom|>", "<|eot|>"):
        assert marker in forbidden
    for marker in ("<atem:invoke", "</atem:invoke>", "<atem:parameter"):
        assert marker in forbidden


def test_muse_replaces_the_qwen_think_pair_case() -> None:
    case_ids = {case.case_id for case in probe.build_cases("muse_glimmer")}

    assert "muse_reasoning_missing_open_think_tag" not in case_ids
    assert "muse_reasoning_recipient_routed_always_on" in case_ids


def test_muse_names_from_both_engines_select_the_profile() -> None:
    for name in (
        "meta-models/Muse-Glimmer-30B",
        "muse_glimmer",
        "muse",
    ):
        assert probe.model_case_profile(name) == "muse_glimmer"

    args = static_report.build_parser().parse_args(
        ["--model", "meta-models/Muse-Glimmer-30B", "--case-profile", "muse_glimmer"]
    )
    assert args.case_profile == "muse_glimmer"


def test_argument_normalization_is_opt_in_for_one_prose_case() -> None:
    muse_cases = probe.build_cases("muse_glimmer")
    qwen_cases = probe.build_cases("qwen3_coder_xml")

    assert [case.case_id for case in muse_cases if case.normalize_argument_strings] == [
        "muse_schema_all_required_fields"
    ]
    assert [case.case_id for case in qwen_cases if case.normalize_argument_strings] == [
        "qx_schema_all_required_fields"
    ]
