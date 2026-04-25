#!/usr/bin/env python3
"""DIS-1850 — Generate synthetic gap fixtures and verify Jinja parity.

The 4 V4 + 3 V3.2 fixtures don't cover:
  - reasoning_effort = "max" prefix
  - wo_eos = true on assistant
  - merge_tool_messages with consecutive tools
  - sort_tool_results_by_call_order with out-of-order tool_results

This script crafts inputs for each, runs encoding_dsv4.py to get expected,
renders my Jinja template, and diffs.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
REF = REPO / "lib/llm/tests/reference"
TEMPLATES = REPO / "lib/llm/src/preprocessor/prompt/templates"


def load_reference(name):
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_v4(messages, thinking_mode, drop_thinking, reasoning_effort):
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    ref = load_reference("encoding_dsv4")
    pre = ref.merge_tool_messages(messages)
    pre = ref.sort_tool_results_by_call_order(pre)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        undefined=ChainableUndefined,
    )
    env.filters["fromjson"] = json.loads
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
    return env.get_template("deepseek_v4.jinja").render(
        messages=pre,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
        add_bos_token=True,
    )


def expected_v4(messages, thinking_mode, drop_thinking, reasoning_effort):
    ref = load_reference("encoding_dsv4")
    return ref.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
        add_default_bos_token=True,
    )


def diff(e, a):
    n = min(len(e), len(a))
    for i in range(n):
        if e[i] != a[i]:
            return i, e[max(0, i - 60) : i + 60], a[max(0, i - 60) : i + 60]
    if len(e) != len(a):
        return n, e[max(0, n - 60) :], a[max(0, n - 60) :]
    return None, None, None


CASES = [
    {
        "name": "reasoning_effort=max prefix",
        "messages": [
            {"role": "system", "content": "Helpful."},
            {"role": "user", "content": "Solve: 2+2"},
        ],
        "thinking_mode": "thinking",
        "drop_thinking": True,
        "reasoning_effort": "max",
    },
    {
        "name": "reasoning_effort=high (no-op per spec)",
        "messages": [
            {"role": "system", "content": "Helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "thinking_mode": "thinking",
        "drop_thinking": True,
        "reasoning_effort": "high",
    },
    {
        "name": "wo_eos=true on final assistant",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there", "wo_eos": True},
        ],
        "thinking_mode": "chat",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "merge_tool_messages — 2 consecutive tools",
        "messages": [
            {"role": "user", "content": "Get weather and time."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"NYC"}'},
                    },
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "time", "arguments": '{"tz":"EST"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp":72}'},
            {"role": "tool", "tool_call_id": "tc2", "content": '"3pm"'},
            {
                "role": "assistant",
                "content": "It's 72°F and 3pm in NYC.",
                "reasoning_content": "Format both.",
            },
        ],
        "thinking_mode": "thinking",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "sort_tool_results — out-of-order tool_results",
        "messages": [
            {"role": "user", "content": "Both."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tcA",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    },
                    {
                        "id": "tcB",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ],
            },
            # Submit B before A on purpose:
            {"role": "tool", "tool_call_id": "tcB", "content": "B-result"},
            {"role": "tool", "tool_call_id": "tcA", "content": "A-result"},
            {"role": "assistant", "content": "ok", "reasoning_content": "merged"},
        ],
        "thinking_mode": "thinking",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "developer role with tools",
        "messages": [
            {
                "role": "developer",
                "content": "Be precise.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ping",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
            },
        ],
        "thinking_mode": "thinking",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "latest_reminder + user",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "latest_reminder", "content": "remember the rules"},
            {"role": "user", "content": "go"},
        ],
        "thinking_mode": "chat",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "response_format on system",
        "messages": [
            {
                "role": "system",
                "content": "Helpful.",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "color",
                        "schema": {
                            "type": "object",
                            "properties": {"hex": {"type": "string"}},
                            "required": ["hex"],
                        },
                    },
                },
            },
            {"role": "user", "content": "blue"},
        ],
        "thinking_mode": "chat",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
    {
        "name": "non-action task token (query)",
        "messages": [
            {"role": "user", "content": "search this", "task": "query"},
            {"role": "assistant", "content": "result", "reasoning_content": "r"},
        ],
        "thinking_mode": "chat",
        "drop_thinking": True,
        "reasoning_effort": None,
    },
]


def main():
    failures = 0
    for case in CASES:
        name = case.pop("name")
        try:
            e = expected_v4(**case)
            a = render_v4(**case)
        except Exception as ex:
            print(f"== {name}: ERROR {type(ex).__name__}: {ex}")
            failures += 1
            continue
        if e == a:
            print(f"== {name}: PASS ({len(a)} bytes)")
            continue
        pos, ectx, actx = diff(e, a)
        print(f"== {name}: FAIL @ byte {pos}, len exp={len(e)} act={len(a)}")
        print(f"   expected: ...{ectx!r}...")
        print(f"   actual:   ...{actx!r}...")
        failures += 1
    print(f"\n{len(CASES)-failures}/{len(CASES)} pass")
    return failures


if __name__ == "__main__":
    sys.exit(main())
