#!/usr/bin/env python3
"""
DIS-1850 Jinja parity harness.

Render `deepseek_v4.jinja` for each fixture and diff vs the saved test_output.

Usage:
    python3 _parity_check.py [fixture_index]   # default: all
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DATA = REPO / "lib/llm/tests/data/deepseek-v4"
REF = REPO / "lib/llm/tests/reference"
TEMPLATES = REPO / "lib/llm/src/preprocessor/prompt/templates"


def load_reference():
    spec = importlib.util.spec_from_file_location("dsv4_ref", REF / "encoding_dsv4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_messages(path):
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        msgs = list(raw.get("messages", []))
        if "tools" in raw and msgs:
            msgs[0] = {**msgs[0], "tools": raw["tools"]}
        return msgs
    elif isinstance(raw, list):
        return raw
    raise ValueError(f"Unexpected fixture shape: {path}")


def render_via_jinja(messages, thinking_mode, drop_thinking, reasoning_effort, ref):
    """Run preprocessing in Python (per ticket — pre-pass is acceptable),
    then render through Jinja."""
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

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
    # encoding_dsv4.py uses json.dumps(value, ensure_ascii=False); preserve input order.
    # Override tojson to plain json.dumps — jinja2's default htmlsafe_json_dumps
    # escapes ' < > & to \uXXXX for HTML embedding, which we don't want.
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}

    tpl = env.get_template("deepseek_v4.jinja")
    return tpl.render(
        messages=pre,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
        add_bos_token=True,
    )


def first_diff_position(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def diff_report(expected, actual):
    pos = first_diff_position(expected, actual)
    if pos is None:
        return None
    ctx = 80
    lo = max(0, pos - ctx)
    hi_e = min(len(expected), pos + ctx)
    hi_a = min(len(actual), pos + ctx)
    return {
        "byte_pos": pos,
        "expected_len": len(expected),
        "actual_len": len(actual),
        "expected_ctx": expected[lo:hi_e],
        "actual_ctx": actual[lo:hi_a],
        "expected_at_pos": expected[pos : pos + 1] if pos < len(expected) else "<EOF>",
        "actual_at_pos": actual[pos : pos + 1] if pos < len(actual) else "<EOF>",
    }


# Fixture-specific config based on lib/llm/tests/deepseek_v4_encoding.rs.
FIXTURE_CONFIG = {
    1: {"thinking_mode": "thinking", "drop_thinking": True, "reasoning_effort": None},
    2: {"thinking_mode": "thinking", "drop_thinking": True, "reasoning_effort": None},
    3: {"thinking_mode": "thinking", "drop_thinking": True, "reasoning_effort": None},
    4: {"thinking_mode": "chat", "drop_thinking": True, "reasoning_effort": None},
}


def run_fixture(n, ref):
    inp = DATA / f"test_input_{n}.json"
    out = DATA / f"test_output_{n}.txt"
    cfg = FIXTURE_CONFIG[n]

    msgs = load_messages(inp)
    expected = out.read_text()

    try:
        actual = render_via_jinja(
            msgs,
            cfg["thinking_mode"],
            cfg["drop_thinking"],
            cfg["reasoning_effort"],
            ref,
        )
    except Exception as e:
        return {
            "fixture": n,
            "status": "RENDER_ERROR",
            "error": f"{type(e).__name__}: {e}",
        }

    # Strip trailing newline like the existing Rust test does (trim_end).
    e2 = expected.rstrip("\n")
    a2 = actual.rstrip("\n")

    if e2 == a2:
        return {"fixture": n, "status": "PASS", "bytes": len(actual)}

    return {"fixture": n, "status": "FAIL", "diff": diff_report(e2, a2)}


def main():
    ref = load_reference()
    args = sys.argv[1:]
    targets = [int(a) for a in args] if args else [1, 2, 3, 4]

    for n in targets:
        result = run_fixture(n, ref)
        print(f"=== fixture {result['fixture']}: {result['status']} ===")
        if result["status"] == "PASS":
            print(f"  bytes={result['bytes']}")
        elif result["status"] == "RENDER_ERROR":
            print(f"  {result['error']}")
        else:
            d = result["diff"]
            print(
                f"  byte_pos={d['byte_pos']}, expected_len={d['expected_len']}, actual_len={d['actual_len']}"
            )
            print(f"  expected[{d['byte_pos']}]={d['expected_at_pos']!r}")
            print(f"  actual[{d['byte_pos']}]={d['actual_at_pos']!r}")
            print(f"  expected_ctx: ...{d['expected_ctx']!r}...")
            print(f"  actual_ctx:   ...{d['actual_ctx']!r}...")
        print()


if __name__ == "__main__":
    main()
