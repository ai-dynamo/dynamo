#!/usr/bin/env python3
"""DIS-1850 — Render-perf comparison: jinja2 (template path) vs encoding_dsv4.py (Rust-port-equivalent).

Measures wall-clock for N renders on each fixture. The Rust minijinja path is
typically ~2x faster than jinja2, so jinja2 numbers are an upper bound on the
final overhead.
"""

import importlib.util
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
DATA = REPO / "lib/llm/tests/data/deepseek-v4"
DATAV32 = REPO / "lib/llm/tests/data/deepseek-v3.2"
REF = REPO / "lib/llm/tests/reference"
TEMPLATES = REPO / "lib/llm/src/preprocessor/prompt/templates"

ITERS = 1000


def load_reference(name):
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def bench_one(label, fn, iters=ITERS):
    # warmup
    for _ in range(5):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p99 = samples[int(len(samples) * 0.99)]
    mean = statistics.mean(samples)
    print(
        f"  {label:30s}  p50={p50/1000:6.1f}us  p99={p99/1000:7.1f}us  mean={mean/1000:6.1f}us  n={iters}"
    )


def main():
    env = make_env()
    refv4 = load_reference("encoding_dsv4")
    refv32 = load_reference("encoding_dsv32")
    tplv4 = env.get_template("deepseek_v4.jinja")
    tplv32 = env.get_template("deepseek_v32.jinja")

    print(f"[V4 fixtures @ {ITERS} iters]")
    for n in [1, 2, 3, 4]:
        msgs = load_messages(DATA / f"test_input_{n}.json")
        tm = "thinking" if n != 4 else "chat"

        def py():
            return refv4.encode_messages(
                msgs, thinking_mode=tm, drop_thinking=True, add_default_bos_token=True
            )

        def jinja():
            pre = refv4.merge_tool_messages(msgs)
            pre = refv4.sort_tool_results_by_call_order(pre)
            return tplv4.render(
                messages=pre,
                thinking_mode=tm,
                drop_thinking=True,
                reasoning_effort=None,
                add_bos_token=True,
            )

        print(f"\nfixture {n} ({len(py())} bytes output):")
        bench_one("encoding_dsv4.py (Python)", py)
        bench_one("jinja2 + Python pre-pass", jinja)

    print(f"\n[V3.2 fixtures @ {ITERS} iters]")
    for inp, label in [
        ("test_input.json", "base"),
        ("test_input_search_w_date.json", "search w/date"),
        ("test_input_search_wo_date.json", "search wo/date"),
    ]:
        msgs = load_messages(DATAV32 / inp)

        def py():
            return refv32.encode_messages(
                msgs,
                thinking_mode="thinking",
                drop_thinking=True,
                add_default_bos_token=True,
            )

        def jinja():
            return tplv32.render(
                messages=msgs,
                thinking_mode="thinking",
                drop_thinking=True,
                add_bos_token=True,
            )

        print(f"\nfixture {label} ({len(py())} bytes output):")
        bench_one("encoding_dsv32.py (Python)", py)
        bench_one("jinja2 (no preprocess)", jinja)


if __name__ == "__main__":
    main()
