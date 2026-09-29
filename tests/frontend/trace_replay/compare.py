# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade a frontend reply against the trace turn it replays.

Tiers, from strictest to loosest; a reply lands in the first one it satisfies:

    exact        reasoning, content, tool names and argument values identical
    ws_edges     reasoning / content differ only in leading or trailing whitespace
    ws_args      additionally, string argument values differ only at their edges
    ws_internal  reasoning / content equal once every whitespace run is collapsed
    mismatch     anything else; `categories` says what diverged

`ws_args` is reported separately because edge whitespace inside an argument
(`old_str`, `file_text`) can change what an agent's tool does.

Truncated scripts (`finish_reason == "length"`) cannot be matched exactly, so
they grade as `trunc_ok` when the partial reply is a prefix of the full turn,
leaks no raw markup and ends with `length`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tests.frontend.trace_replay.endpoints import JsonDict, Turn

TIERS = ("exact", "ws_edges", "ws_args", "ws_internal", "trunc_ok", "mismatch")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Grade:
    tier: str
    categories: list[str] = field(default_factory=list)
    detail: JsonDict = field(default_factory=dict)


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _strip_strings(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_strip_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_strings(item) for key, item in value.items()}
    return value


def _stringify_scalars(value: Any) -> Any:
    if isinstance(value, list):
        return [_stringify_scalars(item) for item in value]
    if isinstance(value, dict):
        return {key: _stringify_scalars(item) for key, item in value.items()}
    return json.dumps(value) if not isinstance(value, str) else value


def _text_tier(got: str, want: str) -> int:
    """0 exact, 1 edge whitespace, 3 internal whitespace, 4 different."""
    if got == want:
        return 0
    if got.strip() == want.strip():
        return 1
    if _collapse(got) == _collapse(want):
        return 3
    return 4


def _first_divergence(got: str, want: str) -> JsonDict:
    at = next(
        (i for i, (a, b) in enumerate(zip(got, want)) if a != b),
        min(len(got), len(want)),
    )
    return {
        "at": at,
        "got": got[max(0, at - 40) : at + 80],
        "want": want[max(0, at - 40) : at + 80],
        "got_len": len(got),
        "want_len": len(want),
    }


def _parse_arguments(raw: Any) -> tuple[bool, Any]:
    if not isinstance(raw, str):
        return True, raw
    try:
        return True, json.loads(raw)
    except json.JSONDecodeError:
        return False, raw


def _leaks(turn: Turn, markup: list[str]) -> list[str]:
    leaks = []
    for name, text in (("content", turn.content), ("reasoning", turn.reasoning)):
        if any(marker in text for marker in markup):
            leaks.append(f"markup_in_{name}")
    for call in turn.tool_calls:
        if any(marker in str(call["arguments"]) for marker in markup):
            leaks.append("markup_in_arguments")
    return leaks


def _grade_complete(turn: Turn, want: JsonDict, markup: list[str]) -> Grade:
    categories: list[str] = []
    detail: JsonDict = {}
    tier = 0
    for name, got_text, want_text in (
        ("reasoning", turn.reasoning, want["reasoning"]),
        ("content", turn.content, want["content"]),
    ):
        text_tier = _text_tier(got_text, want_text)
        tier = max(tier, text_tier)
        if text_tier == 4:
            categories.append(f"{name}_text")
            detail[name] = _first_divergence(got_text, want_text)
    # Only a leak when the field is wrong: models often restate reasoning as content.
    reasoning, content = want["reasoning"].strip(), want["content"].strip()
    if "content_text" in categories and reasoning and reasoning in turn.content:
        categories.append("reasoning_in_content")
    if "reasoning_text" in categories and content and content in turn.reasoning:
        categories.append("content_in_reasoning")

    want_calls = want["tool_calls"]
    if len(turn.tool_calls) != len(want_calls):
        categories.append("tool_call_count")
        detail["tool_calls"] = {"got": turn.tool_calls, "want": want_calls}
    for got_call, want_call in zip(turn.tool_calls, want_calls):
        if got_call["name"] != want_call["name"]:
            categories.append("tool_call_name")
            continue
        parsed, got_args = _parse_arguments(got_call["arguments"])
        want_args = want_call["arguments"]
        if not parsed:
            categories.append("tool_args_invalid_json")
            detail.setdefault("arguments", []).append(
                {"got": got_args[:300], "want": json.dumps(want_args)[:300]}
            )
        elif got_args == want_args:
            continue
        elif _strip_strings(got_args) == _strip_strings(want_args):
            tier = max(tier, 2)
            detail.setdefault("arguments_whitespace", []).append(
                {
                    key: {"got": got_args.get(key), "want": value}
                    for key, value in want_args.items()
                    if isinstance(got_args, dict) and got_args.get(key) != value
                }
            )
        elif _stringify_scalars(got_args) == _stringify_scalars(want_args):
            categories.append("tool_args_type")
            detail.setdefault("arguments", []).append(
                {"got": got_args, "want": want_args}
            )
        else:
            categories.append("tool_args_value")
            detail.setdefault("arguments", []).append(
                {"got": json.dumps(got_args)[:300], "want": json.dumps(want_args)[:300]}
            )
    if turn.finish_reason != want["finish_reason"]:
        categories.append("finish_reason")
        detail["finish_reason"] = {
            "got": turn.finish_reason,
            "want": want["finish_reason"],
        }
    categories += _leaks(turn, markup)
    if categories:
        return Grade("mismatch", sorted(set(categories)), detail)
    return Grade(("exact", "ws_edges", "ws_args", "ws_internal")[tier], [], detail)


def _grade_truncated(turn: Turn, want: JsonDict, markup: list[str]) -> Grade:
    categories: list[str] = []
    detail: JsonDict = {}
    for name, got_text, want_text in (
        ("reasoning", turn.reasoning, want["reasoning"]),
        ("content", turn.content, want["content"]),
    ):
        if not _collapse(want_text).startswith(_collapse(got_text)):
            categories.append(f"partial_{name}_not_prefix")
            detail[name] = _first_divergence(got_text, want_text)
    want_calls = want["tool_calls"]
    if len(turn.tool_calls) > len(want_calls):
        categories.append("partial_extra_tool_calls")
    for got_call, want_call in zip(turn.tool_calls, want_calls):
        if got_call["name"] != want_call["name"]:
            categories.append("partial_tool_call_name")
            continue
        parsed, got_args = _parse_arguments(got_call["arguments"])
        if not parsed:
            categories.append("partial_tool_args_invalid_json")
            detail.setdefault("arguments", []).append(str(got_args)[-200:])
        elif not isinstance(got_args, dict) or any(
            key not in want_call["arguments"]
            or not _is_prefix(value, want_call["arguments"][key])
            for key, value in got_args.items()
        ):
            categories.append("partial_tool_args_not_prefix")
    if turn.finish_reason != "length":
        categories.append("finish_reason")
        detail["finish_reason"] = {"got": turn.finish_reason, "want": "length"}
    categories += _leaks(turn, markup)
    if categories:
        return Grade("mismatch", sorted(set(categories)), detail)
    return Grade("trunc_ok")


def _is_prefix(got: Any, want: Any) -> bool:
    if isinstance(got, str) and isinstance(want, str):
        return want.strip().startswith(got.strip())
    return got == want


def _grade_raw(turn: Turn, case: JsonDict) -> Grade:
    """Parsers off: content must be the decoded script, nothing else."""
    want = case["expected"]["finish_reason"]
    want_finish = "length" if want == "length" else "stop"
    categories = []
    detail: JsonDict = {}
    text_tier = _text_tier(turn.content, case["raw_text"])
    if text_tier == 4:
        categories.append("raw_text")
        detail["content"] = _first_divergence(turn.content, case["raw_text"])
    if turn.reasoning or turn.tool_calls:
        categories.append("parsed_without_parser")
    if turn.finish_reason != want_finish:
        categories.append("finish_reason")
        detail["finish_reason"] = {"got": turn.finish_reason, "want": want_finish}
    if categories:
        return Grade("mismatch", categories, detail)
    return Grade(("exact", "ws_edges", "ws_args", "ws_internal")[text_tier])


def replay_token_check(turn: Turn, case: JsonDict) -> str | None:
    """Did the mocker emit this case's script? Random fallback tokens change the count."""
    got = turn.completion_tokens
    if got is None:
        return "usage_missing"
    script_len = case["script_len"]
    if case["variant"] == "stop_in_reasoning":
        return None if got < script_len else f"replay_tokens:{got}>={script_len}"
    allowed = (
        {script_len, script_len - 1} if case.get("eos_terminated") else {script_len}
    )
    return None if got in allowed else f"replay_tokens:{got}!={script_len}"


def grade(turn: Turn, case: JsonDict, markup: list[str], raw: bool = False) -> Grade:
    if raw:
        result = _grade_raw(turn, case)
    elif case["expected"]["finish_reason"] == "length":
        result = _grade_truncated(turn, case["expected"], markup)
    else:
        result = _grade_complete(turn, case["expected"], markup)
    token_issue = replay_token_check(turn, case)
    if token_issue is not None and token_issue != "usage_missing":
        result.tier = "mismatch"
        result.categories = sorted({*result.categories, "replay_tokens"})
        result.detail["replay_tokens"] = token_issue
    return result


def same_turn(a: Turn, b: Turn) -> list[str]:
    """Fields on which two replies of the same case disagree (stream vs unary, etc)."""
    differences = []
    if _collapse(a.reasoning) != _collapse(b.reasoning):
        differences.append("reasoning")
    if _collapse(a.content) != _collapse(b.content):
        differences.append("content")
    if [c["name"] for c in a.tool_calls] != [c["name"] for c in b.tool_calls]:
        differences.append("tool_names")
    elif [_parse_arguments(c["arguments"]) for c in a.tool_calls] != [
        _parse_arguments(c["arguments"]) for c in b.tool_calls
    ]:
        differences.append("tool_arguments")
    if a.finish_reason != b.finish_reason:
        differences.append("finish_reason")
    return differences
