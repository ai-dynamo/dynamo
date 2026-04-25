#!/usr/bin/env python3
"""Verify deepseek_v4_inline.jinja and deepseek_v32_inline.jinja produce
byte-identical output to their modular counterparts (and therefore to the
upstream Python reference) on every fixture — golden + synthetic gap.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DATA_V4 = REPO / "lib/llm/tests/data/deepseek-v4"
DATA_V32 = REPO / "lib/llm/tests/data/deepseek-v3.2"
REF = REPO / "lib/llm/tests/reference"
TEMPLATES = REPO / "lib/llm/src/preprocessor/prompt/templates"


def load_ref(name):
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_env():
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

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
    return env


def load_messages(path):
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        msgs = list(raw.get("messages", []))
        if "tools" in raw and msgs:
            msgs[0] = {**msgs[0], "tools": raw["tools"]}
        return msgs
    return list(raw)


def main():
    refv4 = load_ref("encoding_dsv4")
    env = make_env()
    inline_v4 = env.get_template("deepseek_v4_inline.jinja")
    inline_v32 = env.get_template("deepseek_v32_inline.jinja")

    fails = 0
    total = 0

    # ===== V4 golden =====
    cfg = {
        1: ("thinking", True, None),
        2: ("thinking", True, None),
        3: ("thinking", True, None),
        4: ("chat", True, None),
    }
    for n in [1, 2, 3, 4]:
        msgs = load_messages(DATA_V4 / f"test_input_{n}.json")
        expected = (DATA_V4 / f"test_output_{n}.txt").read_text().rstrip("\n")
        tm, dt, re_ = cfg[n]
        pre = refv4.merge_tool_messages(msgs)
        pre = refv4.sort_tool_results_by_call_order(pre)
        actual = inline_v4.render(
            messages=pre,
            thinking_mode=tm,
            drop_thinking=dt,
            reasoning_effort=re_,
            add_bos_token=True,
        ).rstrip("\n")
        ok = expected == actual
        total += 1
        if not ok:
            fails += 1
        print(
            f"v4-inline golden #{n}: {'PASS' if ok else 'FAIL'} ({len(actual)} bytes)"
        )

    # ===== V3.2 golden =====
    for inp in [
        "test_input.json",
        "test_input_search_w_date.json",
        "test_input_search_wo_date.json",
    ]:
        msgs = load_messages(DATA_V32 / inp)
        out = inp.replace("input", "output").replace(".json", ".txt")
        expected = (DATA_V32 / out).read_text().rstrip("\n")
        actual = inline_v32.render(
            messages=msgs,
            thinking_mode="thinking",
            drop_thinking=True,
            add_bos_token=True,
        ).rstrip("\n")
        ok = expected == actual
        total += 1
        if not ok:
            fails += 1
        print(
            f"v32-inline golden {inp}: {'PASS' if ok else 'FAIL'} ({len(actual)} bytes)"
        )

    # ===== V4 synthetic gap =====
    GAPS = [
        (
            "reasoning_effort=max",
            [
                {"role": "system", "content": "Helpful."},
                {"role": "user", "content": "Solve: 2+2"},
            ],
            "thinking",
            True,
            "max",
        ),
        (
            "reasoning_effort=high",
            [
                {"role": "system", "content": "Helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "thinking",
            True,
            "high",
        ),
        (
            "wo_eos=true",
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there", "wo_eos": True},
            ],
            "chat",
            True,
            None,
        ),
        (
            "merge_tools",
            [
                {"role": "user", "content": "Get weather and time."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"NYC"}',
                            },
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
            "thinking",
            True,
            None,
        ),
        (
            "sort_tool_results",
            [
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
                {"role": "tool", "tool_call_id": "tcB", "content": "B-result"},
                {"role": "tool", "tool_call_id": "tcA", "content": "A-result"},
                {"role": "assistant", "content": "ok", "reasoning_content": "merged"},
            ],
            "thinking",
            True,
            None,
        ),
        (
            "developer-with-tools",
            [
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
            "thinking",
            True,
            None,
        ),
        (
            "latest_reminder+user",
            [
                {"role": "system", "content": "S"},
                {"role": "latest_reminder", "content": "remember the rules"},
                {"role": "user", "content": "go"},
            ],
            "chat",
            True,
            None,
        ),
        (
            "response_format",
            [
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
            "chat",
            True,
            None,
        ),
        (
            "non-action task",
            [
                {"role": "user", "content": "search this", "task": "query"},
                {"role": "assistant", "content": "result", "reasoning_content": "r"},
            ],
            "chat",
            True,
            None,
        ),
    ]
    for name, msgs, tm, dt, re_ in GAPS:
        expected = refv4.encode_messages(
            msgs,
            thinking_mode=tm,
            drop_thinking=dt,
            reasoning_effort=re_,
            add_default_bos_token=True,
        )
        pre = refv4.merge_tool_messages(msgs)
        pre = refv4.sort_tool_results_by_call_order(pre)
        actual = inline_v4.render(
            messages=pre,
            thinking_mode=tm,
            drop_thinking=dt,
            reasoning_effort=re_,
            add_bos_token=True,
        )
        ok = expected == actual
        total += 1
        if not ok:
            fails += 1
        print(f"v4-inline gap {name}: {'PASS' if ok else 'FAIL'} ({len(actual)} bytes)")

    print(f"\n{total-fails}/{total} pass")
    return fails


if __name__ == "__main__":
    sys.exit(main())
