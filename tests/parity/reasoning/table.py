#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the REASONING.* parity table from YAML fixtures."""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import json
import re
import subprocess
import zoneinfo
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tests.parity.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES
from tests.parity.common import (
    build_parity_tooltip_html,
    linkify_text_html,
    parity_cell_class,
)
from tests.parity.markup import colorize_markup

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests/parity/reasoning/fixtures"
PARSER_FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures"
REASONING_CASES_MD = REPO_ROOT / "lib/parsers/REASONING_CASES.md"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = REPO_ROOT / "tests/parity"

DisplayRow = dict[str, str | None]
_IMPL_DISPLAY = {"dynamo": "Dynamo", "vllm": "vLLM", "sglang": "SGLang"}

CASE_GROUPS = [
    (
        "Core",
        ("batch.1.a", "batch.1.b"),
    ),
    (
        "Reasoning extraction",
        (
            "batch.2.a",
            "batch.2.b",
            "batch.2.c",
            "batch.2.d",
            "batch.2.e",
            "batch.2.f",
        ),
    ),
    ("Tool call boundary", ("batch.3.a", "batch.3.b")),
    ("Malformed / recovery", ("batch.4", "batch.5")),
    ("Multi-span", ("batch.6.a", "batch.6.b")),
    (
        "Core",
        ("stream.1.a", "stream.1.b"),
    ),
    ("Reasoning extraction", ("stream.2.a",)),
    ("Multi-span", ("stream.2.b",)),
    ("Chunk boundaries", ("stream.3.a", "stream.3.b", "stream.3.c")),
]
_CASE_GROUP_BY_CASE = {
    case_id: label for label, case_ids in CASE_GROUPS for case_id in case_ids
}
_CASE_GROUP_INDEX_BY_CASE = {
    case_id: group_idx
    for group_idx, (_, case_ids) in enumerate(CASE_GROUPS)
    for case_id in case_ids
}
_CASE_DISPLAY_ORDER = {
    case_id: (group_idx, case_idx)
    for group_idx, (_, case_ids) in enumerate(CASE_GROUPS)
    for case_idx, case_id in enumerate(case_ids)
}

# Public tool-calling parser rows can share a reasoning implementation. Keep this
# map in user-facing parser-family names, not Rust enum names: `deepseek_v3`
# currently routes to `ReasoningParserType::DeepseekR1`, but the parity table
# should still display `deepseek_v3` because that is the public reasoning parser
# family and fixture directory.
PARSER_TO_REASONING_FAMILY = {
    "deepseek_v3": "deepseek_v3",
    "deepseek_v3_1": "deepseek_v3",
    "deepseek_v3_2": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "glm47": "nemotron_deci",
    "harmony": "gpt_oss",
    "kimi_k2": "kimi_k25",
    "minimax_m2": "minimax_append_think",
    "mistral": "mistral",
    "nemotron_deci": "nemotron_deci",
    "nemotron_nano": "deepseek_r1",
    "qwen3_coder": "qwen3",
}

_FAMILY_METADATA = {
    "basic": {
        "models": ["Generic CoT models"],
        "rust_enum": "ReasoningParserType::Basic",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["qwen3", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "qwen3": {
        "models": ["Qwen3.5", "QwQ-32B", "Qwen3-Think", "Qwen3-Coder"],
        "rust_enum": "ReasoningParserType::Qwen",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "deepseek_v4": {
        "models": ["DeepSeek V4 Pro", "DeepSeek V4 Flash"],
        "rust_enum": "ReasoningParserType::DeepSeekV4",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "nemotron_deci", "glm45"],
        "aliases": ["deepseek-v4", "deepseekv4"],
    },
    "nemotron_deci": {
        "models": [
            "Nemotron-Super / -Ultra / -Deci",
            "Llama-Nemotron",
            "GLM-4.5 / GLM-4.7 via glm45 alias",
        ],
        "rust_enum": "ReasoningParserType::NemotronDeci",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "deepseek_v4", "glm45"],
        "aliases": ["glm45"],
    },
    "deepseek_r1": {
        "models": [
            "DeepSeek R1",
            "DeepSeek V3.x aliases",
            "Nemotron force-reasoning aliases",
        ],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_v3",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
    },
    "deepseek_v3": {
        "models": ["DeepSeek V3", "DeepSeek V3.1", "DeepSeek V3.2"],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_r1",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
        "aliases": ["deepseek_v3_1", "deepseek_v3_2"],
    },
    "kimi": {
        "models": ["Kimi K2 Instruct / Thinking using Unicode think delimiters"],
        "rust_enum": "ReasoningParserType::Kimi",
        "implementation": "BasicReasoningParser `◁think▷` / `◁/think▷`",
    },
    "kimi_k25": {
        "models": ["Kimi K2.5 / Kimi K2.6 style `<think>` force-reasoning models"],
        "rust_enum": "ReasoningParserType::KimiK25",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, "
            "force_reasoning=true, Kimi tool-section exit"
        ),
    },
    "mistral": {
        "models": ["Magistral"],
        "rust_enum": "ReasoningParserType::Mistral",
        "implementation": (
            "BasicReasoningParser `[THINK]` / `[/THINK]`, force_reasoning=true"
        ),
    },
    "granite": {
        "models": ["IBM Granite 3.x", "IBM Granite 3.2 language models"],
        "rust_enum": "ReasoningParserType::Granite",
        "implementation": "GraniteReasoningParser",
    },
    "gpt_oss": {
        "models": ["gpt-oss-20b", "gpt-oss-120b"],
        "rust_enum": "ReasoningParserType::GptOss",
        "implementation": "GptOssReasoningParser / Harmony StreamableParser",
    },
    "minimax_append_think": {
        "models": ["MiniMax M2", "MiniMax M2.1"],
        "rust_enum": "ReasoningParserType::MiniMaxAppendThink",
        "implementation": "MiniMaxAppendThinkParser",
    },
    "gemma4": {
        "models": ["Google Gemma 4 thinking models"],
        "rust_enum": "ReasoningParserType::Gemma4",
        "implementation": "Gemma4ReasoningParser",
        "aliases": ["gemma-4"],
    },
}

_REASONING_MODE_METADATA = {
    "basic": {
        "label": "explicit markers",
        "control": "mostly static",
        "summary": (
            "Reasoning starts only after an opening `<think>` marker or a "
            "prompt-injected start state."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "qwen3": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "Qwen-style reasoning uses explicit `<think>` markers; templates "
            "may inject the opening marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v4": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "DeepSeek V4 uses `<think>` markers, but the prompt formatter "
            "controls whether thinking is enabled."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "nemotron_deci": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": "GLM/Nemotron-Deci style parsing uses explicit `<think>` markers.",
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_r1": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Generation may begin already inside reasoning; marker-free text "
            "can be reasoning until an end marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v3": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": "DeepSeek V3.x routes to the R1 force-reasoning implementation in Dynamo.",
        "static": [
            "Rust enum: ReasoningParserType::DeepseekR1",
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "kimi": {
        "label": "explicit unicode",
        "control": "mostly static",
        "summary": "Kimi legacy reasoning uses Unicode delimiters instead of `<think>`.",
        "static": [
            "BasicReasoningParser `◁think▷` / `◁/think▷`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "kimi_k25": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Kimi K2.5 can start in reasoning and uses the tool call section "
            "marker as a reasoning escape boundary."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
            "tool_start_token=`<|tool_calls_section_begin|>`",
        ],
    },
    "mistral": {
        "label": "force reasoning",
        "control": "mostly static",
        "summary": (
            "Mistral/Magistral reasoning uses bracket markers and starts in "
            "force-reasoning mode."
        ),
        "static": [
            "BasicReasoningParser `[THINK]` / `[/THINK]`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "granite": {
        "label": "phrase markers",
        "control": "mostly static",
        "summary": "Granite reasoning is split by natural-language phrase markers.",
        "static": [
            "GraniteReasoningParser",
            "`Here is my thought process:` / `Here is my response:`",
        ],
    },
    "gpt_oss": {
        "label": "Harmony channels",
        "control": "mostly static",
        "summary": (
            "GPT-OSS reasoning is the Harmony `analysis` channel, parsed by "
            "the Harmony stream parser."
        ),
        "static": [
            "GptOssReasoningParser / Harmony StreamableParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
    "minimax_append_think": {
        "label": "append-think",
        "control": "mostly static",
        "summary": (
            "MiniMax append-think has its own contract instead of normal "
            "reasoning extraction."
        ),
        "static": [
            "MiniMaxAppendThinkParser",
            "parser-specific content wrapper behavior",
        ],
    },
    "gemma4": {
        "label": "Gemma channels",
        "control": "frontend-tunable",
        "summary": "Gemma 4 reasoning is wrapped by channel markers with a `thought` role label.",
        "static": [
            "Gemma4ReasoningParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
}

_FAMILY_TO_VLLM_REASONING = {
    "deepseek_r1": "deepseek_r1",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "gpt_oss": "openai_gptoss",
    "granite": "granite",
    "kimi_k25": "kimi_k2",
    "mistral": "mistral",
    "minimax_append_think": "minimax_m2_append_think",
    "nemotron_deci": "glm45",
    "qwen3": "qwen3",
}

_FAMILY_TO_SGLANG_REASONING = {
    "deepseek_r1": "deepseek-r1",
    "gpt_oss": "gpt-oss",
    "kimi": "kimi",
    "qwen3": "qwen3",
}


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=False,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


def _commit_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _case_sort_key(case_id: str) -> tuple[int, int, int, str]:
    doc_id = case_id.replace("REASONING.", "", 1)
    order = _CASE_DISPLAY_ORDER.get(doc_id)
    if order is not None:
        group_idx, case_idx = order
        return (0, group_idx, case_idx, "")
    parts = doc_id.split(".")
    mode = 0 if parts[0] == "batch" else 1
    top = int(parts[1])
    sub = parts[2] if len(parts) > 2 else ""
    return (1, mode, top, sub)


def _normalize_text(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _canonical(d: dict[str, Any]) -> str:
    d = {
        **d,
        "normal_text": _normalize_text(d.get("normal_text")),
        "reasoning_text": _normalize_text(d.get("reasoning_text")),
    }
    d.pop("reason", None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


_DEFAULT_REASONING_MARKUP_RE = re.compile(r"</?think>")
_NO_AUTO_REASONING_MARKUP_RE = re.compile(r"(?!)")
_REASONING_MARKUP_BY_FAMILY: dict[str, re.Pattern[str]] = {
    "deepseek_r1": re.compile(r"</?think>"),
    "deepseek_v3": re.compile(r"</?think>"),
    "deepseek_v4": re.compile(r"</?think>"),
    "gemma4": re.compile(r"<\|channel>|<channel\|>"),
    "gpt_oss": re.compile(r"<\|(?:start|end|return|channel|message)\|>"),
    "granite": re.compile(
        r"Here is my thought process:|Here is my response:|Here's my response:"
    ),
    "kimi": re.compile(r"◁/?think▷"),
    "kimi_k25": re.compile(r"</?think>"),
    "minimax_append_think": _NO_AUTO_REASONING_MARKUP_RE,
    "mistral": re.compile(r"\[/?THINK\]"),
    "nemotron_deci": re.compile(r"</?think>"),
    "qwen3": re.compile(r"</?think>"),
}


def _reasoning_markup_re(family: str | None) -> re.Pattern[str]:
    return _REASONING_MARKUP_BY_FAMILY.get(family or "", _DEFAULT_REASONING_MARKUP_RE)


def _dynamo_leak_reason(expected: dict[str, Any], family: str | None) -> str | None:
    dynamo = expected.get("dynamo", {})
    if not isinstance(dynamo, dict):
        return None
    marker_re = _reasoning_markup_re(family)
    for field in ("reasoning_text", "normal_text"):
        value = dynamo.get(field)
        if isinstance(value, str) and marker_re.search(value):
            return str(
                dynamo.get("reason")
                or f"Dynamo {field} contains reasoning parser markup."
            )
    return None


def _has_dynamo_leak(case: dict[str, Any], family: str | None) -> bool:
    if case.get("dynamo_leak"):
        return True
    expected = case.get("expected")
    return (
        isinstance(expected, dict) and _dynamo_leak_reason(expected, family) is not None
    )


def _load() -> tuple[dict[str, dict[str, Any]], list[str], dict[tuple[str, str], Path]]:
    rows: dict[str, dict[str, Any]] = {}
    columns = set()
    refs = {}
    for fp in sorted(FIXTURES.glob("*/REASONING.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        family = doc["family"]
        row = rows.setdefault(
            family,
            {
                "family": family,
                "model_label": doc.get("model_label", family),
                "cases": {},
            },
        )
        for case_id, case in doc["cases"].items():
            columns.add(case_id)
            row["cases"][case_id] = case
            refs[(family, case_id)] = fp
    return rows, sorted(columns, key=_case_sort_key), refs


def _load_parser_labels() -> dict[str, str]:
    labels = {}
    for fp in sorted(PARSER_FIXTURES.glob("*/TOOLCALLING.batch.yaml")):
        doc = yaml.safe_load(fp.read_text())
        labels[doc["family"]] = doc.get("model_label", doc["family"])
    return labels


def _build_display_groups(
    rows: dict[str, dict[str, Any]],
) -> tuple[list[DisplayRow], list[DisplayRow], list[DisplayRow]]:
    parser_labels = _load_parser_labels()
    parser_families = set(parser_labels)

    def make_row(tool_family: str) -> dict[str, str | None]:
        return {
            "model_label": parser_labels.get(tool_family, tool_family),
            "tool_family": tool_family,
            "reasoning_family": PARSER_TO_REASONING_FAMILY.get(tool_family),
        }

    def has_reasoning_fixture(tool_family: str) -> bool:
        reasoning_family = PARSER_TO_REASONING_FAMILY.get(tool_family)
        return reasoning_family in rows

    top_n = [
        make_row(tool_family)
        for tool_family in TOP_N_FAMILIES
        if tool_family in parser_families and has_reasoning_fixture(tool_family)
    ]
    other_tool_families = sorted(
        (
            tool_family
            for tool_family in parser_families - set(TOP_N_FAMILIES)
            if has_reasoning_fixture(tool_family)
        ),
        key=lambda family: parser_labels.get(family, family).lower(),
    )
    others = [make_row(tool_family) for tool_family in other_tool_families]

    mapped_reasoning = {
        family for family in PARSER_TO_REASONING_FAMILY.values() if family in rows
    }
    reasoning_only = [
        {
            "model_label": rows[family]["model_label"],
            "tool_family": None,
            "reasoning_family": family,
        }
        for family in sorted(set(rows) - mapped_reasoning)
    ]
    return top_n, others, reasoning_only


def _derive_no_peer_sets(rows: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Reasoning families where every expected case marks the peer unavailable."""

    def all_unavailable(cases: dict[str, dict[str, Any]], impl: str) -> bool:
        expected_cases = [
            case for case in cases.values() if isinstance(case.get("expected"), dict)
        ]
        if not expected_cases:
            return False
        for case in expected_cases:
            block = case.get("expected", {}).get(impl)
            if not isinstance(block, dict) or "unavailable" not in block:
                return False
        return True

    no_vllm = {
        family
        for family, row in rows.items()
        if family not in _FAMILY_TO_VLLM_REASONING
        and all_unavailable(row["cases"], "vllm")
    }
    no_sglang = {
        family
        for family, row in rows.items()
        if family not in _FAMILY_TO_SGLANG_REASONING
        and all_unavailable(row["cases"], "sglang")
    }
    return no_vllm, no_sglang


def family_suffix(
    reasoning_family: str | None,
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    if reasoning_family is None:
        return ""
    suffix = ""
    if reasoning_family in no_vllm:
        suffix += "†"
    if reasoning_family in no_sglang:
        suffix += "§"
    return suffix


def _columns_for_mode(columns: list[str], mode: str) -> list[str]:
    return [case_id for case_id in columns if case_id.startswith(f"REASONING.{mode}.")]


def _is_na_stub(case: dict[str, Any]) -> bool:
    return (
        set(case) <= {"description", "reason", "ref", "spec_ref"} and "reason" in case
    )


def _cell(case: dict[str, Any] | None, family: str | None = None) -> tuple[str, str]:
    if case is None:
        return "—", "missing fixture coverage"
    if "expected" not in case:
        if _is_na_stub(case):
            return "n/a", case["reason"]
        return "?", "fixture has no expected block"

    expected = case["expected"]
    dynamo = expected["dynamo"]
    dynamo_leak = _has_dynamo_leak(case, family)
    markers = []
    unavailable = 0
    tooltip_parts = [case.get("description", "")]

    for impl, letter in (("vllm", "V"), ("sglang", "S")):
        spec = expected[impl]
        if "unavailable" in spec:
            unavailable += 1
            tooltip_parts.append(f"{impl}: unavailable — {spec['unavailable']}")
            continue
        if "error" in spec:
            markers.append(f"{letter}!")
            tooltip_parts.append(f"{impl}: expected error — {spec['error']}")
            continue
        if _canonical(spec) == _canonical(dynamo):
            tooltip_parts.append(f"{impl}: matches Dynamo")
            continue
        suffix = "?" if dynamo_leak or not spec.get("reason") else ""
        markers.append(f"{letter}{suffix}")
        reason = (
            "research-needed" if dynamo_leak else spec.get("reason", "research-needed")
        )
        tooltip_parts.append(f"{impl}: diverges — {reason}")

    if unavailable == 2:
        if dynamo_leak:
            return "↯D", "\n".join(p for p in tooltip_parts if p)
        return "D", "\n".join(p for p in tooltip_parts if p)
    if dynamo_leak:
        return "↯" + ("".join(markers) or "?"), "\n".join(p for p in tooltip_parts if p)
    if markers:
        return "".join(markers), "\n".join(p for p in tooltip_parts if p)
    if unavailable:
        return "=", "\n".join(p for p in tooltip_parts if p)
    return "=", "\n".join(p for p in tooltip_parts if p)


def _render_markdown_row(
    row: dict[str, str | None],
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    reasoning_family = row["reasoning_family"]
    tool_family = row["tool_family"]
    parser_label = _parser_label_text(tool_family, reasoning_family, no_vllm, no_sglang)
    cells = [str(row["model_label"]), parser_label]
    for case_id in columns:
        if reasoning_family is None:
            marker = "n/a"
        else:
            marker, _ = _cell(
                rows[reasoning_family]["cases"].get(case_id),
                reasoning_family,
            )
        cells.append(marker)
    return "| " + " | ".join(cells) + " |"


def _markdown(rows: dict[str, dict[str, Any]], columns: list[str]) -> str:
    out = []
    top_n, others, reasoning_only = _build_display_groups(rows)
    no_vllm, no_sglang = _derive_no_peer_sets(rows)
    header = [
        "model",
        "Reasoning family",
        *[_display_case_id(c) for c in columns],
    ]
    out.append("| " + " | ".join(header) + " |")
    out.append("|---|---|" + "|".join([":-:" for _ in columns]) + "|")
    out.append("| **Top-N models** |   |" + "   |" * len(columns))
    for row in top_n:
        out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    out.append("| **Others** |   |" + "   |" * len(columns))
    for row in others:
        out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    if reasoning_only:
        out.append("| **Reasoning-only** |   |" + "   |" * len(columns))
        for row in reasoning_only:
            out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    return "\n".join(out)


def _display_case_id(case_id: str) -> str:
    parts = case_id.split(".")
    return ".".join(parts[2:])


def _case_doc_id(case_id: str) -> str:
    return case_id.replace("REASONING.", "", 1)


def _case_mode(case_id: str) -> str:
    return _case_doc_id(case_id).split(".", 1)[0]


def _case_group_label(case_id: str) -> str:
    return _CASE_GROUP_BY_CASE.get(_case_doc_id(case_id), "Other")


def _case_group_key(case_id: str) -> str:
    label_key = re.sub(r"[^a-z0-9]+", "_", _case_group_label(case_id).lower())
    return f"{_case_mode(case_id)}_{label_key.strip('_')}"


def _case_band_class(case_id: str) -> str:
    group_idx = _CASE_GROUP_INDEX_BY_CASE.get(_case_doc_id(case_id), len(CASE_GROUPS))
    return f"case-band-{group_idx % 2}"


def _case_runs(columns: list[str]) -> list[list[str]]:
    runs = []
    start = 0
    while start < len(columns):
        group_key = _case_group_key(columns[start])
        end = start + 1
        while end < len(columns) and _case_group_key(columns[end]) == group_key:
            end += 1
        runs.append(columns[start:end])
        start = end
    return runs


def _column_placeholder_html(key: str, tag: str = "td") -> str:
    key_attr = html_lib.escape(key)
    return (
        f'<{tag} class="col-placeholder col-hidden" '
        f'data-col-placeholder-group="{key_attr}"></{tag}>'
    )


def _column_control_header_html(
    key: str,
    label: str,
    *,
    default_visible: bool,
    css_class: str = "",
    colspan: int | None = None,
) -> str:
    key_attr = html_lib.escape(key)
    visible = "true" if default_visible else "false"
    classes = " ".join(part for part in ("column-control", css_class) if part)
    span_size = colspan if colspan is not None else 1
    if colspan is not None:
        span_attr = f'colspan="{colspan}" data-expanded-colspan="{colspan}"'
    else:
        span_attr = 'rowspan="2"'
    action = "Collapse" if default_visible else "Expand"
    return (
        f'<th class="{html_lib.escape(classes)}" data-col-control-group="{key_attr}" '
        f"{span_attr}>"
        f'<button type="button" class="col-toggle" data-col-toggle="{key_attr}" '
        f'data-col-label="{html_lib.escape(label)}" data-col-span="{span_size}" '
        f'data-default-visible="{visible}" aria-pressed="{visible}" '
        f'aria-label="{action} {html_lib.escape(label)} column">'
        '<span class="col-toggle-symbol" aria-hidden="true"></span>'
        f'<span class="col-toggle-label">{html_lib.escape(label)}</span>'
        "</button></th>"
    )


def _case_group_headers_html(columns: list[str]) -> str:
    headers = [
        _column_control_header_html("model", "Model", default_visible=True),
        _column_control_header_html("parser", "Reasoning family", default_visible=True),
    ]
    for run in _case_runs(columns):
        label = _case_group_label(run[0])
        group_key = _case_group_key(run[0])
        headers.append(
            _column_control_header_html(
                group_key,
                label,
                default_visible=True,
                css_class=f"case-group {_case_band_class(run[0])}",
                colspan=len(run),
            )
        )
    return "".join(headers)


def _parse_case_descriptions() -> dict[str, str]:
    if not REASONING_CASES_MD.exists():
        return {}
    pat = re.compile(
        r"\*\*`REASONING\.(batch|stream)\.([0-9]+(?:\.[a-z])?)`\*\*\s+(.+)"
    )
    out = {}
    lines = REASONING_CASES_MD.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = pat.search(lines[i])
        if not match:
            i += 1
            continue
        mode, sub, desc = match.groups()
        body_parts = [desc.strip()]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip() or not nxt.startswith(" "):
                break
            if pat.search(nxt):
                break
            body_parts.append(nxt.strip())
            j += 1
        out.setdefault(f"{mode}.{sub}", " ".join(body_parts).rstrip("."))
        i = j
    return out


def _case_header_html(case_id: str, descriptions: dict[str, str]) -> str:
    display = _display_case_id(case_id)
    desc = descriptions.get(_case_doc_id(case_id)) or ""
    href = "../../../lib/parsers/REASONING_CASES.md"
    return (
        f'<th class="case-sub {_case_band_class(case_id)}" '
        f'data-col-hide-group="{html_lib.escape(_case_group_key(case_id))}">'
        f'<a href="{href}" title="{html_lib.escape(desc)}">'
        f"{html_lib.escape(display)}</a></th>"
    )


def _case_headers_html(columns: list[str], descriptions: dict[str, str]) -> str:
    headers = []
    for run in _case_runs(columns):
        headers.extend(_case_header_html(case_id, descriptions) for case_id in run)
        headers.append(_column_placeholder_html(_case_group_key(run[0]), tag="th"))
    return "".join(headers)


def _glossary_groups(
    descriptions: dict[str, str], columns: list[str]
) -> list[dict[str, object]]:
    if not descriptions:
        return []
    return [
        {
            "label": _case_group_label(run[0]),
            "rows": [
                (
                    _case_doc_id(case_id),
                    descriptions.get(_case_doc_id(case_id), ""),
                )
                for case_id in run
            ],
        }
        for run in _case_runs(columns)
    ]


def _markup_family(family: str | None) -> str | None:
    if family == "gpt_oss":
        return "harmony"
    return family


def _colorize_kimi(text: str) -> str:
    pieces = []
    for part in re.split(r"(◁/?think▷)", text):
        if part == "◁think▷" or part == "◁/think▷":
            pieces.append(f'<span class="tt-c0">{html_lib.escape(part)}</span>')
        else:
            pieces.append(html_lib.escape(part))
    return "".join(pieces)


def _colorize_granite(text: str) -> str:
    replacements = {
        "Here is my thought process:": "tt-c0",
        "Here is my response:": "tt-c1",
        "Here's my thought process:": "tt-c0",
        "Here's my response:": "tt-c1",
    }
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    pieces = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(html_lib.escape(text[last : match.start()]))
        marker = match.group(0)
        cls = replacements[marker]
        pieces.append(f'<span class="{cls}">{html_lib.escape(marker)}</span>')
        last = match.end()
    pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


_STREAM_TAG_RE = re.compile(r"<[^<>]+>|\[/?[A-Z][A-Z0-9_]*\]")
_HARMONY_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_HARMONY_TURN_CLOSE = frozenset({"end", "return", "call"})
_HARMONY_SEGMENT_CLASS = {
    "start": "tt-h-start",
    "channel": "tt-h-channel",
    "constrain": "tt-h-constrain",
    "message": "tt-h-message",
    "end": "tt-h-stop",
    "return": "tt-h-stop",
    "call": "tt-h-call",
}
_GEMMA4_CHANNEL_RE = re.compile(r"<\|channel>[A-Za-z_]+(?:\n)?|<channel\|>")
_PIPES = ("|", "｜")
_BEGIN_SUFFIXES = ("_begin", "▁begin")
_END_SUFFIXES = ("_end", "▁end")


def _split_name(value: str) -> str:
    return re.split(r"[\s/>=]", value, 1)[0].rstrip("|")


def _strip_marker_suffix(value: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return None


def _stream_tag_kind(token: str) -> tuple[str | None, str]:
    if token.startswith("[/"):
        return "close", token[2:-1]
    if token.startswith("["):
        return "open", token[1:-1]

    inner = token[1:-1]
    if inner.startswith("/"):
        return "close", _split_name(inner[1:])
    starts_pipe = inner[:1] in _PIPES
    ends_pipe = inner[-1:] in _PIPES
    if starts_pipe and ends_pipe and len(inner) >= 2:
        middle = inner[1:-1]
        stripped = _strip_marker_suffix(middle, _BEGIN_SUFFIXES)
        if stripped is not None:
            return "open", stripped
        stripped = _strip_marker_suffix(middle, _END_SUFFIXES)
        if stripped is not None:
            return "close", stripped
        return "singleton", middle
    if starts_pipe and inner[:1] == "|":
        return "open", _split_name(inner[1:])
    if ends_pipe and inner[-1:] == "|":
        return "close", _split_name(inner[:-1])
    return "open", _split_name(inner)


def _tag_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    color_seq = 0

    def next_class() -> str:
        nonlocal color_seq
        cls = f"tt-c{color_seq % 8}"
        color_seq += 1
        return cls

    for match in _STREAM_TAG_RE.finditer(text):
        idx = len(intervals)
        token = match.group(0)
        kind, name = _stream_tag_kind(token)
        intervals.append({"start": match.start(), "end": match.end(), "class": None})
        if kind == "singleton":
            intervals[idx]["class"] = next_class()
        elif kind == "close":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    intervals[unmatched_idx]["class"] = "tt-orphan"
                cls = next_class()
                intervals[stack[match_at][1]]["class"] = cls
                intervals[idx]["class"] = cls
                del stack[match_at:]
            else:
                intervals[idx]["class"] = "tt-orphan"
        elif kind == "open":
            stack.append((name, idx))
        else:
            intervals[idx]["class"] = "tt-orphan"
    for _, idx in stack:
        intervals[idx]["class"] = "tt-orphan"
    return intervals


def _simple_marker_intervals(
    text: str, markers: dict[str, str]
) -> list[dict[str, Any]]:
    pattern = re.compile("|".join(re.escape(marker) for marker in markers))
    return [
        {"start": m.start(), "end": m.end(), "class": markers[m.group(0)]}
        for m in pattern.finditer(text)
    ]


def _harmony_intervals(text: str) -> list[dict[str, Any]]:
    intervals = []
    for match in _HARMONY_TOKEN_RE.finditer(text):
        token_name = match.group(1)
        if token_name in _HARMONY_TURN_CLOSE:
            end = match.end()
        else:
            next_match = _HARMONY_TOKEN_RE.search(text, match.end())
            end = next_match.start() if next_match else len(text)
        intervals.append(
            {
                "start": match.start(),
                "end": end,
                "class": _HARMONY_SEGMENT_CLASS.get(token_name, "tt-h-other"),
                "token_start": match.start(),
                "token_end": match.end(),
                "harmony": True,
            }
        )
    return intervals


def _gemma4_channel_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    stack: list[int] = []
    color_seq = 0

    def next_class() -> str:
        nonlocal color_seq
        cls = f"tt-c{color_seq % 8}"
        color_seq += 1
        return cls

    for match in _GEMMA4_CHANNEL_RE.finditer(text):
        token = match.group(0)
        idx = len(intervals)
        intervals.append({"start": match.start(), "end": match.end(), "class": None})
        if token.startswith("<|channel>"):
            stack.append(idx)
        elif stack:
            open_idx = stack.pop()
            cls = next_class()
            intervals[open_idx]["class"] = cls
            intervals[idx]["class"] = cls
        else:
            intervals[idx]["class"] = "tt-orphan"

    for idx in stack:
        intervals[idx]["class"] = "tt-orphan"
    return intervals


def _stream_intervals(text: str, family: str | None) -> list[dict[str, Any]]:
    if family == "gpt_oss":
        return _harmony_intervals(text)
    if family == "gemma4":
        return _gemma4_channel_intervals(text)
    if family == "kimi":
        return _simple_marker_intervals(
            text,
            {"◁think▷": "tt-c0", "◁/think▷": "tt-c0"},
        )
    if family == "granite":
        return _simple_marker_intervals(
            text,
            {
                "Here is my thought process:": "tt-c0",
                "Here is my response:": "tt-c1",
                "Here's my thought process:": "tt-c0",
                "Here's my response:": "tt-c1",
            },
        )
    return _tag_intervals(text)


def _render_harmony_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    token_start = interval["token_start"]
    token_end = interval["token_end"]
    parts = []
    cursor = start
    token_slice_start = max(start, token_start)
    token_slice_end = min(end, token_end)
    if cursor < token_slice_start:
        parts.append(html_lib.escape(text[cursor:token_slice_start]))
        cursor = token_slice_start
    if token_slice_start < token_slice_end:
        parts.append(
            '<span class="tt-h-token">'
            f"{html_lib.escape(text[token_slice_start:token_slice_end])}</span>"
        )
        cursor = token_slice_end
    if cursor < end:
        parts.append(html_lib.escape(text[cursor:end]))
    return f'<span class="tt-h {interval["class"]}">{"".join(parts)}</span>'


def _render_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    if interval.get("harmony"):
        return _render_harmony_interval_slice(text, interval, start, end)
    return (
        f'<span class="{interval["class"]}">{html_lib.escape(text[start:end])}</span>'
    )


def _colorize_chunk_slice(
    text: str, intervals: list[dict[str, Any]], start: int, end: int
) -> str:
    pieces = []
    cursor = start
    for interval in intervals:
        istart = interval["start"]
        iend = interval["end"]
        if iend <= start:
            continue
        if istart >= end:
            break
        overlap_start = max(start, istart)
        overlap_end = min(end, iend)
        if cursor < overlap_start:
            pieces.append(html_lib.escape(text[cursor:overlap_start]))
        pieces.append(
            _render_interval_slice(text, interval, overlap_start, overlap_end)
        )
        cursor = overlap_end
    if cursor < end:
        pieces.append(html_lib.escape(text[cursor:end]))
    return "".join(pieces)


def _colorize_stream_chunks(chunks: list[Any], family: str) -> list[str]:
    raw_chunks = [str(chunk) for chunk in chunks]
    text = "".join(raw_chunks)
    intervals = _stream_intervals(text, family)
    rendered = []
    cursor = 0
    for chunk in raw_chunks:
        end = cursor + len(chunk)
        rendered.append(_colorize_chunk_slice(text, intervals, cursor, end))
        cursor = end
    return rendered


def _colorize_gemma4(text: str) -> str:
    return _colorize_chunk_slice(text, _gemma4_channel_intervals(text), 0, len(text))


def _colorize_reasoning_markup(value: str, family: str | None) -> str:
    if family == "gemma4":
        return _colorize_gemma4(value)
    if family == "kimi":
        return _colorize_kimi(value)
    if family == "granite":
        return _colorize_granite(value)
    return colorize_markup(value, _markup_family(family))


def _format_value_html(name: str, value: Any, family: str | None = None) -> str:
    if isinstance(value, str):
        return f'{html_lib.escape(name)}="{_colorize_reasoning_markup(value, family)}"'
    rendered = html_lib.escape(json.dumps(value, ensure_ascii=False))
    return f"{html_lib.escape(name)}={rendered}"


def _display_family_mapping_note(family: str, display_family: str | None) -> str:
    if display_family and display_family != family:
        return f"{display_family!r} (fixture: {family!r})"
    return repr(family)


def _format_unavailable_html(
    message: Any,
    family: str,
    display_family: str | None,
) -> str:
    rendered = str(message)
    if display_family and display_family != family:
        replacement = f"{display_family!r} (fixture: {family!r})"
        replaced = rendered.replace(repr(family), replacement)
        if replaced == rendered:
            replaced = f"{rendered} (row family: {replacement})"
        rendered = replaced
    return html_lib.escape(f"unavailable: {rendered}")


def _format_output_block_html(
    block: dict[str, Any] | None,
    family: str,
    *,
    display_family: str | None = None,
) -> str:
    if block is None:
        return html_lib.escape("(no expectation)")
    if "unavailable" in block:
        return _format_unavailable_html(block["unavailable"], family, display_family)
    if "error" in block:
        return html_lib.escape(f"error matching {block['error']!r}")

    lines = [
        _format_value_html("reasoning_text", block.get("reasoning_text"), family),
        _format_value_html("normal_text", block.get("normal_text"), family),
    ]
    return "\n".join(lines)


def _input_text_html(case: dict[str, Any], family: str) -> str:
    if "chunks" in case:
        chunk_lines = ["chunks=["]
        for i, chunk_html in enumerate(_colorize_stream_chunks(case["chunks"], family)):
            chunk_lines.append(f'  {i}: "{chunk_html}"')
        chunk_lines.append("]")
        return "\n".join(chunk_lines)
    return _format_value_html("input_text", case.get("model_text", ""), family)


def _vllm_parser_name(family: str, case: dict[str, Any]) -> str | None:
    return case.get("vllm_parser") or _FAMILY_TO_VLLM_REASONING.get(family)


def _vllm_chat_template_kwargs(
    parser_name: str | None, case: dict[str, Any]
) -> dict[str, Any]:
    if not parser_name:
        return {}
    kwargs: dict[str, Any] = {}
    if parser_name == "deepseek_v4":
        kwargs["enable_thinking"] = True
    if parser_name == "deepseek_v3":
        kwargs["thinking"] = True
    kwargs.update(case.get("chat_template_kwargs", {}))
    return kwargs


def _sglang_parser_name(family: str, case: dict[str, Any]) -> str | None:
    return case.get("sglang_parser") or _FAMILY_TO_SGLANG_REASONING.get(family)


def _flag_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _presence(value: Any) -> str:
    return "provided" if value else "none"


def _harness_flags_html(
    case_id: str,
    family: str,
    case: dict[str, Any],
    *,
    display_family: str | None = None,
) -> str:
    vllm_parser = _vllm_parser_name(family, case)
    vllm_kwargs = _vllm_chat_template_kwargs(vllm_parser, case)
    sglang_parser = _sglang_parser_name(family, case)

    mode = _case_mode(case_id)
    if mode == "stream":
        dynamo_token_flag = f"token_chunks={_presence(case.get('token_chunks'))}"
        if case.get("token_chunks"):
            vllm_token_flag = "delta_token_ids=fixture token_chunks"
        elif vllm_parser == "openai_gptoss":
            vllm_token_flag = "delta_token_ids=derived from Harmony tokenizer"
        else:
            vllm_token_flag = "delta_token_ids=derived from stub tokenizer"
    else:
        dynamo_token_flag = f"token_ids={_presence(case.get('token_ids'))}"
        vllm_token_flag = "request.skip_special_tokens=false"

    family_flag = (
        f"parser_family={display_family}; fixture_family={family}"
        if display_family and display_family != family
        else f"parser_family={family}"
    )
    lines = [
        "Parity harness flags used for this result:",
        (
            "dynamo: "
            f"{family_flag}; "
            f"in_reasoning={_flag_value(case.get('in_reasoning', False))}; "
            f"{dynamo_token_flag}"
        ),
    ]

    if vllm_parser:
        lines.append(
            "vllm: "
            f"reasoning_parser={vllm_parser}; "
            f"chat_template_kwargs={_flag_value(vllm_kwargs)}; "
            f"{vllm_token_flag}"
        )
    else:
        lines.append(
            "vllm: unavailable for Dynamo parser family "
            f"{_display_family_mapping_note(family, display_family)}"
        )

    if sglang_parser:
        lines.append(
            "sglang: "
            f"model_type={sglang_parser}; "
            "stream_reasoning=true; "
            f"force_reasoning={_flag_value(case.get('force_reasoning', False))}"
        )
    else:
        lines.append(
            "sglang: unavailable for Dynamo parser family "
            f"{_display_family_mapping_note(family, display_family)}"
        )

    lines.extend(
        [
            "",
            "Not set by this parser-level parity harness:",
            "- frontend `tool_choice`",
            "- SGLang `separate_reasoning`",
            "- request `enable_thinking` / `thinking` / `thinking_mode` "
            "unless shown in vLLM chat_template_kwargs above",
        ]
    )
    return html_lib.escape("\n".join(lines))


def _tooltip_for(
    case: dict[str, Any],
    dyn: dict[str, Any],
    family: str | None,
) -> str:
    """Build the peer-divergence explanation used by hover popups."""

    parts: list[str] = []
    dynamo_leak = _has_dynamo_leak(case, family)
    expected = case.get("expected", {})
    for impl in ("vllm", "sglang"):
        block = expected.get(impl)
        if not isinstance(block, dict) or block is dyn:
            continue
        if "unavailable" in block:
            continue
        name = _IMPL_DISPLAY.get(impl, impl)
        if "error" in block:
            parts.append(f"{name}: expected error matching {block['error']!r}")
            continue
        if _canonical(block) == _canonical(dyn):
            continue
        if "reason" in block and not dynamo_leak:
            parts.append(f"{name}: {block['reason']}")
        elif "reasoning_text" in block or "normal_text" in block:
            parts.append(f"{name}: (research-needed — no `reason:` field yet)")
    return "\n".join(parts)


def _tooltip_html(
    case_id: str,
    family: str,
    case: dict[str, Any],
    *,
    display_family: str | None = None,
) -> str:
    head_family = display_family or family
    head = f"{case_id} — {head_family}"
    description = case.get("description")
    extra_sections: list[tuple[str, str]] = [
        (
            "Harness flags",
            _harness_flags_html(
                case_id,
                family,
                case,
                display_family=display_family,
            ),
        )
    ]
    if display_family and display_family != family:
        extra_sections.append(
            (
                "Family mapping",
                html_lib.escape(
                    "This row uses the public GLM/Nemotron reasoning parser "
                    f"family {display_family!r}; fixtures are shared with "
                    f"{family!r} because GLM and Nemotron use the same "
                    "parser implementation here."
                ),
            )
        )
    if "expected" not in case:
        reason = case.get("reason", "fixture has no expected block")
        return build_parity_tooltip_html(
            head=head,
            description=str(description) if description else None,
            extra_sections=[("Why n/a", linkify_text_html(str(reason)))],
            refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
        )

    expected = case["expected"]
    dyn = expected.get("dynamo")
    all_engines_parity = isinstance(dyn, dict) and all(
        isinstance(expected.get(i), dict)
        and not expected[i].get("unavailable")
        and "error" not in expected[i]
        and _canonical(expected[i]) == _canonical(dyn)
        for i in ("dynamo", "vllm", "sglang")
    )

    output_sections: list[tuple[str, str]] = []
    if all_engines_parity:
        output_sections.append(
            (
                "All engines parity",
                _format_output_block_html(
                    dyn,
                    family,
                    display_family=display_family,
                ),
            )
        )
    else:
        for impl in ("dynamo", "vllm", "sglang"):
            output_sections.append(
                (
                    _IMPL_DISPLAY[impl],
                    _format_output_block_html(
                        expected.get(impl),
                        family,
                        display_family=display_family,
                    ),
                )
            )

    dynamo_leak = (
        _dynamo_leak_reason(expected, family)
        if _has_dynamo_leak(case, family)
        else None
    )
    return build_parity_tooltip_html(
        head=head,
        description=str(description) if description else None,
        input_label="Input chunks" if "chunks" in case else "Input",
        input_html=_input_text_html(case, family),
        output_sections=output_sections,
        divergent_reasons=_tooltip_for(case, dyn, family)
        if isinstance(dyn, dict)
        else None,
        leak_label="↯ Dynamo reasoning leaks",
        leak_text=dynamo_leak
        or ("unresolved" if _has_dynamo_leak(case, family) else None),
        extra_sections=extra_sections,
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


def _missing_tooltip_html(
    case_id: str,
    family: str,
    *,
    display_family: str | None = None,
) -> str:
    head_family = display_family or family
    return build_parity_tooltip_html(
        head=f"{case_id} — {head_family}",
        extra_sections=[
            ("Missing fixture", html_lib.escape("missing fixture coverage"))
        ],
    )


def _no_reasoning_tooltip_html(case_id: str, tool_family: str) -> str:
    return build_parity_tooltip_html(
        head=f"{case_id} — {tool_family}",
        extra_sections=[
            (
                "Why n/a",
                html_lib.escape(
                    "This tool calling parser row has no mapped Dynamo reasoning "
                    "parser fixture."
                ),
            )
        ],
    )


def _render_cell_html(
    case: dict[str, Any] | None,
    family: str,
    case_id: str,
    refs: dict[tuple[str, str], Path],
    *,
    display_family: str | None = None,
) -> str:
    if case is None:
        marker = "—"
        href = ""
        tooltip = _missing_tooltip_html(
            case_id,
            family,
            display_family=display_family,
        )
    else:
        marker, _ = _cell(case, family)
        href = str(refs[(family, case_id)].relative_to(SCRIPT_DIR))
        tooltip = _tooltip_html(
            case_id,
            family,
            case,
            display_family=display_family,
        )

    classes = f"cell {parity_cell_class(marker)} {_case_band_class(case_id)}"
    group_key = html_lib.escape(_case_group_key(case_id))
    label = html_lib.escape(marker)
    if href:
        body = f'<a href="{html_lib.escape(href)}">{label}</a>'
    else:
        body = label
    return (
        f'<td class="{classes}" data-col-hide-group="{group_key}">'
        f"{body}{tooltip}</td>"
    )


def _render_no_reasoning_cell_html(tool_family: str, case_id: str) -> str:
    classes = f"cell na {_case_band_class(case_id)}"
    group_key = html_lib.escape(_case_group_key(case_id))
    tooltip = _no_reasoning_tooltip_html(case_id, tool_family)
    return (
        f'<td class="{classes}" data-col-hide-group="{group_key}">' f"n/a{tooltip}</td>"
    )


def _implementation_label(reasoning_family: str | None) -> str:
    if reasoning_family is None:
        return "n/a"
    meta = _FAMILY_METADATA.get(reasoning_family, {})
    implementation = str(meta.get("implementation") or reasoning_family)
    class_name = implementation.split(" ", 1)[0]
    marker = re.search(r"`([^`]+)`", implementation)
    if class_name == "BasicReasoningParser" and marker:
        return f"{class_name} {marker.group(1)}"
    return class_name


def _display_reasoning_family(
    tool_family: str | None,
    reasoning_family: str | None,
) -> str | None:
    if tool_family == "glm47" and reasoning_family == "nemotron_deci":
        return "glm45"
    return reasoning_family


def _parser_label_text(
    tool_family: str | None,
    reasoning_family: str | None,
    no_vllm: set[str] | None = None,
    no_sglang: set[str] | None = None,
) -> str:
    if reasoning_family is None:
        return "n/a"
    suffix = family_suffix(reasoning_family, no_vllm or set(), no_sglang or set())
    return f"`{_implementation_label(reasoning_family)}`{suffix}"


def _reasoning_mode_meta(
    tool_family: str | None,
    reasoning_family: str | None,
) -> dict[str, Any]:
    if reasoning_family is None:
        return {
            "label": "n/a",
            "control": "not configured",
            "summary": (
                "This tool calling parser row has no mapped Dynamo reasoning "
                "parser fixture."
            ),
            "static": [],
        }

    meta = dict(_REASONING_MODE_METADATA.get(reasoning_family, {}))
    if not meta:
        meta = {
            "label": "custom",
            "control": "mostly static",
            "summary": "Custom reasoning parser family.",
            "static": [],
        }

    row_notes = []
    if tool_family and tool_family != reasoning_family:
        row_notes.append(
            f"Tool calling parser row `{tool_family}` maps to reasoning "
            f"family `{reasoning_family}`."
        )
    if tool_family in {"deepseek_v3_1", "deepseek_v3_2"}:
        row_notes.append(
            "DeepSeek V3.1/V3.2 aliases share the DeepSeek V3.x reasoning fixtures."
        )
    if tool_family == "glm47":
        row_notes.append(
            "GLM tool rows use the public `glm45` reasoning alias; that alias "
            "maps to the `nemotron_deci` implementation and fixtures."
        )
    if tool_family == "harmony":
        row_notes.append(
            "Tool calling parser `harmony` pairs with `gpt_oss` reasoning."
        )
    if tool_family == "kimi_k2":
        row_notes.append(
            "Tool calling parser `kimi_k2` pairs with `kimi_k25` reasoning."
        )

    if row_notes:
        meta["row_notes"] = row_notes
    return meta


def _reasoning_parser_tree_lines(
    tool_family: str | None,
    reasoning_family: str | None,
    meta: dict[str, Any],
) -> list[str]:
    if reasoning_family is None:
        return []

    implementation = _implementation_label(reasoning_family)
    shared = [name for name in meta.get("shared_with", []) if name != reasoning_family]
    aliases = [
        name
        for name in meta.get("aliases", [])
        if name != reasoning_family and name not in shared
    ]
    known_children = {*shared, *aliases}
    row_only = (
        [tool_family]
        if tool_family
        and tool_family != reasoning_family
        and tool_family not in known_children
        else []
    )
    children = [
        (reasoning_family, ""),
        *[(name, "") for name in sorted(row_only)],
        *[(name, "") for name in sorted(shared)],
        *[(name, " (alias)") for name in sorted(aliases)],
    ]

    lines = [
        "",
        "Shared implementation tree:",
        html_lib.escape(implementation),
    ]
    for i, (name, suffix) in enumerate(children):
        branch = "└── " if i == len(children) - 1 else "├── "
        name_label = html_lib.escape(name)
        if tool_family == name:
            name_label = f"<strong>{name_label}</strong>"
        elif tool_family is None and name == reasoning_family:
            name_label = f"<strong>{name_label}</strong>"
        lines.append(f"{branch}{name_label}{html_lib.escape(suffix)}")
    return lines


def _parser_cell_html(
    tool_family: str | None,
    reasoning_family: str | None,
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    meta = _FAMILY_METADATA.get(reasoning_family or "", {})
    display_family = _display_reasoning_family(tool_family, reasoning_family)
    implementation_label = _implementation_label(reasoning_family)
    mode_meta = _reasoning_mode_meta(tool_family, reasoning_family)
    activation_name = display_family or "n/a"
    tooltip_lines = [
        html_lib.escape(
            f'This is activated via "--dyn-reasoning-parser {activation_name}".'
        ),
    ]
    if display_family:
        tooltip_lines.append(f"Parser family: {html_lib.escape(display_family)}")
    if tool_family and tool_family != display_family:
        tooltip_lines.append(f"Tool calling row: {html_lib.escape(tool_family)}")
    if display_family:
        if display_family != reasoning_family:
            tooltip_lines.append(
                f"Fixture family: {html_lib.escape(reasoning_family or '')}"
            )
    else:
        tooltip_lines.append(html_lib.escape("Reasoning parser family: n/a"))
    if meta.get("models"):
        tooltip_lines.append(
            "Models: "
            + html_lib.escape(", ".join(str(model) for model in meta["models"]))
        )
    if meta.get("rust_enum"):
        tooltip_lines.append(f"Rust enum: {html_lib.escape(meta['rust_enum'])}")
    if meta.get("implementation"):
        tooltip_lines.append(
            f"Implementation: {html_lib.escape(meta['implementation'])}"
        )
    peer_notes = []
    if reasoning_family in no_vllm:
        peer_notes.append("no vLLM peer reasoning parser")
    if reasoning_family in no_sglang:
        peer_notes.append("no SGLang peer reasoning parser")
    if peer_notes:
        tooltip_lines.append(
            "Peer availability: " + html_lib.escape(", ".join(peer_notes))
        )
    tooltip_lines.extend(
        [
            "",
            "Mode:",
            "- " + html_lib.escape(f"{mode_meta['label']} / {mode_meta['control']}"),
        ]
    )
    static_config = mode_meta.get("static", [])
    config_lines = [
        str(line)
        for line in static_config
        if not str(line).startswith(
            (
                "BasicReasoningParser",
                "Gemma4ReasoningParser",
                "GptOssReasoningParser",
                "GraniteReasoningParser",
                "MiniMaxAppendThinkParser",
            )
        )
    ]
    if config_lines:
        tooltip_lines.extend(
            ["", "Config:"] + [f"- {html_lib.escape(line)}" for line in config_lines]
        )
    tooltip_lines.extend(
        _reasoning_parser_tree_lines(tool_family, reasoning_family, meta)
    )
    parser_label = _parser_label_text(tool_family, reasoning_family, no_vllm, no_sglang)
    tooltip = (
        f'<div class="ttip-head">{html_lib.escape(parser_label)}</div>'
        f'<pre class="ttip-pre">{chr(10).join(tooltip_lines)}</pre>'
    )
    suffix = family_suffix(reasoning_family, no_vllm, no_sglang)
    suffix_html = (
        f'<span class="parser-suffix">{html_lib.escape(suffix)}</span>'
        if suffix
        else ""
    )
    if tool_family is None:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    elif reasoning_family is None:
        label = "<code>n/a</code>"
    elif tool_family == reasoning_family:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    else:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    return (
        '<td class="parser" data-col-hide-group="parser">'
        f'{label}{suffix_html}<div class="ttip">{tooltip}</div></td>'
    )


def _render_row_html(
    row: dict[str, str | None],
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    tool_family = row["tool_family"]
    reasoning_family = row["reasoning_family"]
    cells = [
        f'<tr><td class="model" data-col-hide-group="model">'
        f"{html_lib.escape(str(row['model_label']))}</td>",
        _column_placeholder_html("model"),
        _parser_cell_html(tool_family, reasoning_family, no_vllm, no_sglang),
        _column_placeholder_html("parser"),
    ]
    for run in _case_runs(columns):
        if reasoning_family is None:
            cells.extend(
                _render_no_reasoning_cell_html(str(tool_family), case_id)
                for case_id in run
            )
        else:
            cases = rows[reasoning_family]["cases"]
            display_family = _display_reasoning_family(tool_family, reasoning_family)
            cells.extend(
                _render_cell_html(
                    cases.get(case_id),
                    reasoning_family,
                    case_id,
                    refs,
                    display_family=display_family,
                )
                for case_id in run
            )
        cells.append(_column_placeholder_html(_case_group_key(run[0])))
    cells.append("</tr>")
    return "".join(cells)


def _compute_stats(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    display_rows: list[dict[str, str | None]],
) -> dict[str, int]:
    stats = {
        "families": len(display_rows),
        "sub_cases": len(columns),
        "slots": len(display_rows) * len(columns),
        "real": 0,
        "parity": 0,
        "dynamo_only": 0,
        "documented": 0,
        "research": 0,
        "errors": 0,
        "na": 0,
        "missing": 0,
    }
    for row in display_rows:
        family = row["reasoning_family"]
        for case_id in columns:
            if family is None:
                marker = "n/a"
            else:
                marker, _ = _cell(rows[family]["cases"].get(case_id), family)
            if marker == "—":
                stats["missing"] += 1
            elif marker == "n/a":
                stats["na"] += 1
            else:
                stats["real"] += 1
                if marker == "=":
                    stats["parity"] += 1
                elif marker == "D":
                    stats["dynamo_only"] += 1
                elif "!" in marker:
                    stats["errors"] += 1
                elif "?" in marker:
                    stats["research"] += 1
                else:
                    stats["documented"] += 1
    return stats


def _omitted_tool_calling_rows_html(rows: dict[str, dict[str, Any]]) -> str:
    parser_labels = _load_parser_labels()
    omitted = []
    for tool_family, label in parser_labels.items():
        reasoning_family = PARSER_TO_REASONING_FAMILY.get(tool_family)
        if reasoning_family not in rows:
            omitted.append((label, tool_family))
    if not omitted:
        return ""

    omitted.sort(key=lambda item: item[0].lower())
    rendered = ", ".join(
        f"{html_lib.escape(label)} (<code>{html_lib.escape(family)}</code>)"
        for label, family in omitted
    )
    return (
        "<br><br>"
        "<strong>Omitted tool calling parser rows:</strong> "
        "no corresponding Dynamo reasoning parser fixture for "
        f"{rendered}."
    )


def _has_missing_cells(rows: dict[str, dict[str, Any]], columns: list[str]) -> bool:
    top_n, others, reasoning_only = _build_display_groups(rows)
    for row in [*top_n, *others, *reasoning_only]:
        reasoning_family = row["reasoning_family"]
        if reasoning_family is None:
            continue
        cases = rows[reasoning_family]["cases"]
        if any(case_id not in cases for case_id in columns):
            return True
    return False


def _legend_html(rows: dict[str, dict[str, Any]], columns: list[str]) -> str:
    missing_text = ""
    if _has_missing_cells(rows, columns):
        missing_text = (
            " · " '<span style="color:#8a6d3b">—</span> missing fixture coverage'
        )
    legend = (
        "<strong>Legend:</strong> "
        '<span style="color:#0a7d2c">=</span> all captured peers match Dynamo · '
        '<span style="color:#555">D</span> Dynamo-only fixture '
        "(both peers unavailable) · "
        '<span style="color:#555">V/S</span> divergence '
        "(V = vLLM, S = SGLang; intentional, has <code>reason:</code>) · "
        '<span style="color:#b00">?</span> more research needed '
        "(e.g. V?, S? — diverges with no <code>reason:</code> yet) · "
        '<span style="color:#b00">↯</span> Dynamo leaks reasoning markup '
        "or final-answer text into the wrong field · "
        '<span style="color:#b00">!</span> expected-error suffix '
        "(e.g. V!, S! — engine crashes by design) · "
        '<span style="color:#aaa">n/a</span> not applicable'
        f"{missing_text}."
        "<br>"
        '<span class="parser-suffix">†</span> = no vLLM peer reasoning parser '
        "for this family · "
        '<span class="parser-suffix">§</span> = no SGLang peer reasoning parser '
        "for this family."
        "<br><br>"
        "<strong>Tooltip fields:</strong> "
        "<code>input_text</code>=raw model output or stream chunks fed into "
        "the reasoning parser · "
        "<code>reasoning_text</code>=hidden reasoning content extracted by "
        "the parser · "
        "<code>normal_text</code>=residual text passed onward to response "
        "assembly or downstream parsers · "
        "<code>harness flags</code>=the exact parser-level arguments used for "
        "the parity result."
    )
    return legend + _omitted_tool_calling_rows_html(rows)


def _mode_label(mode: str) -> str:
    if mode == "batch":
        return "REASONING.batch.*"
    if mode == "stream":
        return "REASONING.stream.*"
    return mode


def _html_panel(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    no_vllm: set[str],
    no_sglang: set[str],
    *,
    mode: str,
    active: bool,
) -> dict[str, object]:
    descriptions = _parse_case_descriptions()
    top_n, others, reasoning_only = _build_display_groups(rows)
    display_rows = [*top_n, *others, *reasoning_only]
    body_rows = []
    sections = [
        ("Top-N models", top_n),
        ("Others", others),
        ("Reasoning-only", reasoning_only),
    ]
    for label, section_rows in sections:
        if not section_rows:
            continue
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{2 + len(columns)}">'
            f"{html_lib.escape(label)}</td></tr>"
        )
        for row in section_rows:
            body_rows.append(
                _render_row_html(row, rows, columns, refs, no_vllm, no_sglang)
            )

    return {
        "id": f"tab-{mode}",
        "mode": mode,
        "label": _mode_label(mode),
        "active": active,
        "group_headers": _case_group_headers_html(columns),
        "sub_headers": _case_headers_html(columns, descriptions),
        "body_rows": body_rows,
        "stats": _compute_stats(rows, columns, display_rows),
        "glossary_groups": _glossary_groups(descriptions, columns),
    }


def _html(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    modes: list[str],
) -> str:
    generated = datetime.datetime.now(
        zoneinfo.ZoneInfo("America/Los_Angeles")
    ).strftime("%Y-%m-%d %H:%M %Z")
    sha = _commit_sha()
    no_vllm, no_sglang = _derive_no_peer_sets(rows)
    panels = [
        _html_panel(
            rows,
            _columns_for_mode(columns, mode),
            refs,
            no_vllm,
            no_sglang,
            mode=mode,
            active=i == 0,
        )
        for i, mode in enumerate(modes)
    ]
    command = "python3 tests/parity/generate_parity_table.py reasoning --html"
    output = "tests/parity/reasoning/PARITY.html"
    if len(modes) == 1:
        command += f" --mode {modes[0]}"
        output = f"tests/parity/reasoning/PARITY.{modes[0]}.html"
    tabs = []
    for i, mode in enumerate(modes):
        panel_id = f"tab-{mode}"
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        tabs.append(
            f'<button class="tab-button{active}" id="{panel_id}-button" '
            f'type="button" role="tab" aria-selected="{selected}" '
            f'data-tab-target="{panel_id}">'
            f"{html_lib.escape(_mode_label(mode))}</button>"
        )

    return (
        _make_jinja_env()
        .get_template("parity_table.html.j2")
        .render(
            title="Dynamo Reasoning Parser - Parity Table",
            stamp=generated,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command=command,
            output=output,
            tabs=tabs,
            panels=panels,
            peer_versions=[],
            intro_html="",
            legend_html=_legend_html(rows, columns),
            case_docs_href="../../../lib/parsers/REASONING_CASES.md",
            case_docs_label="lib/parsers/REASONING_CASES.md",
            case_prefix="REASONING.",
        )
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="emit HTML instead of Markdown")
    ap.add_argument(
        "--mode",
        choices=("all", "batch", "stream"),
        help=(
            "Fixture mode to render. For HTML, default is all reasoning modes as "
            "tabs. For Markdown, default is batch."
        ),
    )
    args = ap.parse_args(argv)

    rows, columns, refs = _load()
    if args.html:
        modes = ["batch", "stream"] if args.mode in (None, "all") else [args.mode]
        print(_html(rows, columns, refs, modes))
    else:
        if args.mode == "all":
            ap.error("--mode all is only supported with --html")
        mode = args.mode or "batch"
        print(_markdown(rows, _columns_for_mode(columns, mode)))


if __name__ == "__main__":
    main()
