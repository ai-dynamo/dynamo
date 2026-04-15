# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Print TTFT attribution table from JSON extracted by extract_attribution.py.

Reads the JSON, computes derived metrics (template+tokenize, measured totals,
pipeline overhead), and prints a side-by-side comparison table.

Usage:
    python benchmarks/multimodal/sweep/scripts/print_attribution.py \
        attribution.json \
        -o attribution_table.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

STATS = ["avg", "p50", "p90", "p99"]


def g(components: dict[str, Any], name: str, stat: str) -> float | None:
    """Get a stat from components, returning None if missing."""
    comp = components.get(name)
    if comp is None:
        return None
    return comp.get(stat)


def f(v: float | None, w: int = 6) -> str:
    """Format a value right-justified, or dash if None."""
    if v is None:
        return "—".rjust(w)
    return str(int(round(v))).rjust(w)


def print_ttft_comparison(
    bl_name: str, fd_name: str,
    bl_ttft: dict[str, float], fd_ttft: dict[str, float],
    rate: str, out: Any = sys.stdout,
) -> None:
    """Print Table 1: TTFT comparison with delta row."""
    print(file=out)
    print(f"  === TTFT (ms) — rate={rate} ===", file=out)
    print(f"  {'':18s} {'avg':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s}", file=out)

    for name, ttft in [(bl_name, bl_ttft), (fd_name, fd_ttft)]:
        vals = "".join(f"{int(round(ttft.get(s, 0))):>8d}" for s in STATS)
        print(f"  {name:<18s}{vals}", file=out)

    # Delta row (fd relative to bl)
    deltas: list[str] = []
    for s in STATS:
        bl_v = bl_ttft.get(s)
        fd_v = fd_ttft.get(s)
        if bl_v and fd_v and bl_v > 0:
            pct = (fd_v - bl_v) / bl_v * 100
            sign = "+" if pct >= 0 else ""
            deltas.append(f"{sign}{pct:.0f}%".rjust(8))
        else:
            deltas.append("—".rjust(8))
    print(f"  {'delta':<18s}{''.join(deltas)}", file=out)
    print(file=out)


def print_table(data: dict[str, Any], rate: str, out: Any = sys.stdout) -> None:
    configs = data["configs"]
    config_names = list(configs.keys())

    # Determine which is baseline (vllm-serve) and which is dynamo-fd
    bl_name = next((c for c in config_names if "vllm" in c.lower()), config_names[0])
    fd_name = next((c for c in config_names if "dynamo" in c.lower()), config_names[-1])

    bl_rate = configs[bl_name]["rates"].get(rate, {})
    fd_rate = configs[fd_name]["rates"].get(rate, {})
    bl = bl_rate.get("components", {})
    fd = fd_rate.get("components", {})
    bl_ttft = bl_rate.get("ttft", {})
    fd_ttft = fd_rate.get("ttft", {})

    # --- Table 1: TTFT comparison ---
    print_ttft_comparison(bl_name, fd_name, bl_ttft, fd_ttft, rate, out)

    # --- Table 2: Attribution breakdown ---
    def row(label: str, bl_name_key: str | None, fd_name_key: str | None, indent: int = 0) -> None:
        prefix = "  " * indent + label
        bl_vals = " ".join(f(g(bl, bl_name_key, s) if bl_name_key else None) for s in STATS)
        fd_vals = " ".join(f(g(fd, fd_name_key, s) if fd_name_key else None) for s in STATS)
        print(f"  {prefix:<34s} {bl_vals}      {fd_vals}", file=out)

    def row_raw(label: str, bl_dict: dict | None, fd_dict: dict | None, indent: int = 0) -> None:
        prefix = "  " * indent + label
        bl_vals = " ".join(f(bl_dict.get(s) if bl_dict else None) for s in STATS)
        fd_vals = " ".join(f(fd_dict.get(s) if fd_dict else None) for s in STATS)
        print(f"  {prefix:<34s} {bl_vals}      {fd_vals}", file=out)

    def row_single(label: str, bl_val: float | None, fd_val: float | None) -> None:
        prefix = label
        bl_s = f(bl_val)
        fd_s = f(fd_val)
        print(f"  {prefix:<34s} {bl_s} {'':>6s} {'':>6s} {'':>6s}      {fd_s}", file=out)

    W = 110
    print(f"  Attribution Breakdown (rate={rate}, all times in ms)", file=out)
    print(file=out)
    print(f"  {'':34s} {bl_name:>26s}      {fd_name:>26s}", file=out)
    print(f"  {'Component':<34s} {'avg':>6s} {'p50':>6s} {'p90':>6s} {'p99':>6s}      {'avg':>6s} {'p50':>6s} {'p90':>6s} {'p99':>6s}", file=out)
    print("  " + "─" * W, file=out)

    # --- vllm-serve preprocessing ---
    row("openai_preprocess", "openai_preprocess", None, 0)
    row("hf_processor", "hf_processor", None, 1)
    # Derived: template+tokenize = openai_preprocess - hf_processor
    bl_tt: dict[str, float] | None = None
    if g(bl, "openai_preprocess", "avg") is not None and g(bl, "hf_processor", "avg") is not None:
        bl_tt = {s: (g(bl, "openai_preprocess", s) or 0) - (g(bl, "hf_processor", s) or 0) for s in STATS}
    row_raw("template+tokenize (derived)", bl_tt, None, 1)

    # --- dynamo-fd preprocessing ---
    row("rust_preprocess", None, "rust_preprocess", 0)
    row("template", None, "template", 1)
    row("tokenize", None, "tokenize", 1)
    row("gather_mm_data", None, "gather_mm_data", 1)
    row("image_decode (/img)", None, "image_load.decode", 2)
    row("nixl_register (/img)", None, "image_load_register.register", 2)
    row("extract_mm_data", None, "extract_mm_data", 0)
    row("nixl_read (/img)", None, "nixl_read", 1)

    # --- Shared: hf_processor (dynamo-fd side) ---
    row("hf_processor", None, "hf_processor", 0)

    # --- GPU (both sides) ---
    row("vision_encoder (synced)", "vision_encoder", "vision_encoder", 0)
    row("llm_forward_gpu (synced)", "llm_forward_gpu", "llm_forward_gpu", 0)

    print("  " + "─" * W, file=out)

    # --- Measured totals (avg only — percentiles can't be summed) ---
    # vllm-serve: openai_preprocess + vision_encoder + llm_forward_gpu
    bl_measured: float | None = None
    bl_op = g(bl, "openai_preprocess", "avg")
    bl_ve = g(bl, "vision_encoder", "avg")
    bl_fg = g(bl, "llm_forward_gpu", "avg")
    if bl_op is not None and bl_ve is not None and bl_fg is not None:
        bl_measured = bl_op + bl_ve + bl_fg

    # dynamo-fd: rust_preprocess + extract_mm_data + llm_forward (handler)
    # llm_forward (handler) is the non-overlapping wrapper
    fd_rp = g(fd, "rust_preprocess", "avg")
    fd_em = g(fd, "extract_mm_data", "avg")
    fd_lf = g(fd, "llm_forward", "avg")  # handler-level
    fd_measured: float | None = None
    if fd_rp is not None and fd_em is not None and fd_lf is not None:
        fd_measured = fd_rp + fd_em + fd_lf

    row_single("MEASURED TOTAL (avg only)", bl_measured, fd_measured)

    # Unaccounted
    bl_unacct = (bl_ttft.get("avg", 0) - bl_measured) if bl_measured else None
    fd_unacct = (fd_ttft.get("avg", 0) - fd_measured) if fd_measured else None
    row_single("unaccounted", bl_unacct, fd_unacct)

    print("  " + "─" * W, file=out)
    row_raw("TOTAL TTFT (aiperf)", bl_ttft, fd_ttft, 0)
    print(file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print TTFT attribution table from JSON")
    parser.add_argument("input", type=Path, help="attribution.json from extract_attribution.py")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save table to text file")
    parser.add_argument("--rate", type=str, default=None, help="Rate to display (default: first available)")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())

    # Auto-detect rate if not specified
    if args.rate:
        rate = args.rate
    else:
        first_config = next(iter(data["configs"].values()))
        rate = next(iter(first_config["rates"].keys()))

    # Print to stdout
    print_table(data, rate)

    # Optionally save to file
    if args.output:
        buf = StringIO()
        print_table(data, rate, out=buf)
        args.output.write_text(buf.getvalue())
        print(f"Table saved to {args.output}")


if __name__ == "__main__":
    main()
