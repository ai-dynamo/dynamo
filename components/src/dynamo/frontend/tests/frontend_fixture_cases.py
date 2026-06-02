#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared loader + normalize + assert for the FRONTEND.* YAML fixtures.

Fixtures (``fixtures/frontend_*.yaml``) are backend-agnostic. The per-backend
adapters (``_vllm_frontend_adapter.py`` / ``_sglang_frontend_adapter.py``)
build that backend's StreamingPostProcessor, replay a case's ``model_text`` at
each chunk granularity, and hand the assembled OpenAI choice dicts here.
Both backends emit the same choice shape
(``{delta: {content, reasoning_content, tool_calls}, finish_reason}``), so
normalize/assert is shared rather than duplicated.

Consumed by FRONTEND.4 (assembly, tool parser configured) and FRONTEND.6
(detok, fast plain-text path, no parser). This is a *coverage* matrix (write
once, both engines must pass), NOT a behavioral divergence grid -- vLLM/SGLang
expose no callable frontend on the same input, so there is no
expected.{vllm,sglang}.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


@dataclass
class FrontendCase:
    case_id: str
    model_text: str
    expected: dict[str, Any]
    batch_sizes: list[int] = field(default_factory=lambda: [20])
    single_chunk: bool = False
    description: str = ""


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


def load_tools(name: str = "frontend_assembly") -> list[dict[str, Any]]:
    """Canonical tool schemas; each adapter converts to its engine's type."""
    return _load(name).get("tools", [])


def load_cases(name: str = "frontend_assembly") -> list[FrontendCase]:
    data = _load(name)
    cases = []
    for case_id, body in data["cases"].items():
        cases.append(
            FrontendCase(
                case_id=case_id,
                model_text=body["model_text"],
                expected=body["expected"],
                batch_sizes=body.get("batch_sizes", [20]),
                single_chunk=body.get("single_chunk", False),
                description=body.get("description", ""),
            )
        )
    return cases


def params(cases, known_gaps: dict | None = None):
    """Build pytest params (one per case x batch size). ``known_gaps`` maps
    (case_id, batch_size) -> reason and marks that combo a strict xfail, so a
    documented backend gap stays green while flipping to a failure the moment
    it is fixed."""
    known_gaps = known_gaps or {}
    out = []
    for case in cases:
        sizes = [None] if case.single_chunk else case.batch_sizes
        for bs in sizes:
            label = "all" if bs is None else bs
            marks = []
            reason = known_gaps.get((case.case_id, bs))
            if reason:
                marks.append(pytest.mark.xfail(reason=reason, strict=True))
            out.append(
                pytest.param(case, bs, id=f"{case.case_id}-{label}", marks=marks)
            )
    return out


def normalize(choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a per-chunk choice stream into a single comparable result:
    ``{tool_calls: [{name, arguments}], content, reasoning_content,
    finish_reason}``. Streaming tool-call deltas sharing an ``index`` are
    merged (name from whichever delta carries it, arguments concatenated) the
    way an OpenAI client does; ``arguments`` is JSON-parsed when possible."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    by_index: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for choice in choices:
        delta = choice.get("delta", {}) or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        for tc in delta.get("tool_calls", []) or []:
            idx = tc.get("index", 0)
            fn = tc.get("function", {}) or {}
            if idx not in by_index:
                by_index[idx] = {"name": None, "arguments": ""}
                order.append(idx)
            if fn.get("name"):
                by_index[idx]["name"] = fn["name"]
            if fn.get("arguments"):
                by_index[idx]["arguments"] += fn["arguments"]
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    tool_calls: list[dict[str, Any]] = []
    for idx in order:
        raw_args = by_index[idx]["arguments"]
        try:
            args: Any = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            args = raw_args
        tool_calls.append({"name": by_index[idx]["name"], "arguments": args})

    return {
        "tool_calls": tool_calls,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish_reason,
    }


def assert_case(
    result: dict[str, Any], expected: dict[str, Any], *, context: str
) -> None:
    """Assert a normalized result against a case's ``expected`` block."""
    exp_calls = expected.get("tool_calls")
    if exp_calls is not None:
        got = [
            {"name": c["name"], "arguments": c["arguments"]}
            for c in result["tool_calls"]
        ]
        assert (
            got == exp_calls
        ), f"{context}: tool_calls mismatch\n got={got}\n exp={exp_calls}"

    if "content" in expected:
        assert (
            result["content"].strip() == expected["content"].strip()
        ), f"{context}: content mismatch\n got={result['content']!r}\n exp={expected['content']!r}"

    if "content_contains" in expected:
        assert (
            expected["content_contains"] in result["content"]
        ), f"{context}: content missing {expected['content_contains']!r}\n got={result['content']!r}"

    if "reasoning_contains" in expected:
        assert (
            expected["reasoning_contains"] in result["reasoning_content"]
        ), f"{context}: reasoning missing {expected['reasoning_contains']!r}\n got={result['reasoning_content']!r}"

    if "finish_reason" in expected:
        assert (
            result["finish_reason"] == expected["finish_reason"]
        ), f"{context}: finish_reason {result['finish_reason']!r} != {expected['finish_reason']!r}"
