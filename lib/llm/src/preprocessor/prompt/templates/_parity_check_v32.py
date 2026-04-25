#!/usr/bin/env python3
"""DIS-1850 V3.2 Jinja parity harness."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DATA = REPO / "lib/llm/tests/data/deepseek-v3.2"
REF = REPO / "lib/llm/tests/reference"
TEMPLATES = REPO / "lib/llm/src/preprocessor/prompt/templates"


def load_messages(path):
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        msgs = list(raw.get("messages", []))
        if "tools" in raw and msgs:
            msgs[0] = {**msgs[0], "tools": raw["tools"]}
        return msgs
    return list(raw)


def render_via_jinja(messages, thinking_mode, drop_thinking):
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

    tpl = env.get_template("deepseek_v32.jinja")
    return tpl.render(
        messages=messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_bos_token=True,
    )


def diff(expected, actual):
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            ctx = 80
            return {
                "byte_pos": i,
                "expected_len": len(expected),
                "actual_len": len(actual),
                "expected_at": expected[i : i + 1],
                "actual_at": actual[i : i + 1],
                "expected_ctx": expected[max(0, i - ctx) : min(len(expected), i + ctx)],
                "actual_ctx": actual[max(0, i - ctx) : min(len(actual), i + ctx)],
            }
    if len(expected) != len(actual):
        return {
            "byte_pos": n,
            "expected_len": len(expected),
            "actual_len": len(actual),
            "expected_at": expected[n : n + 1] if n < len(expected) else "<EOF>",
            "actual_at": actual[n : n + 1] if n < len(actual) else "<EOF>",
            "expected_ctx": expected[max(0, n - 80) :],
            "actual_ctx": actual[max(0, n - 80) :],
        }
    return None


# Per encoding_dsv32.py — V3.2 doesn't take reasoning_effort.
FIXTURES = [
    ("test_input.json", "test_output.txt", "thinking", True),
    (
        "test_input_search_w_date.json",
        "test_output_search_w_date.txt",
        "thinking",
        True,
    ),
    (
        "test_input_search_wo_date.json",
        "test_output_search_wo_date.txt",
        "thinking",
        True,
    ),
]


def main():
    args = sys.argv[1:]
    targets = FIXTURES if not args else [FIXTURES[int(i)] for i in args]
    for inp, out, tm, dt in targets:
        msgs = load_messages(DATA / inp)
        expected = (DATA / out).read_text()
        try:
            actual = render_via_jinja(msgs, tm, dt)
        except Exception as e:
            print(f"=== {inp}: RENDER_ERROR ===\n  {type(e).__name__}: {e}\n")
            continue
        e2 = expected.rstrip("\n")
        a2 = actual.rstrip("\n")
        if e2 == a2:
            print(f"=== {inp}: PASS ({len(actual)} bytes) ===\n")
            continue
        d = diff(e2, a2)
        print(f"=== {inp}: FAIL ===")
        print(
            f"  byte_pos={d['byte_pos']}, expected_len={d['expected_len']}, actual_len={d['actual_len']}"
        )
        print(f"  expected_at={d['expected_at']!r}, actual_at={d['actual_at']!r}")
        print(f"  expected_ctx: ...{d['expected_ctx']!r}...")
        print(f"  actual_ctx:   ...{d['actual_ctx']!r}...\n")


if __name__ == "__main__":
    main()
