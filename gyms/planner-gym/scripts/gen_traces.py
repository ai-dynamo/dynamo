#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the workload registry to Mooncake-JSONL traces and print a summary.

Pure-Python — needs neither the Dynamo build nor a venv. Self-bootstrapping.

    python scripts/gen_traces.py [--out DIR] [--seed N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Arena workload traces.")
    p.add_argument("--out", default=str(_REPO / "runs" / "traces"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-recorded",
        action="store_true",
        help="also materialize recorded registry entries (requires their source files)",
    )
    args = p.parse_args()

    from autoscaling_arena.workloads import get_workload, list_workloads

    out = Path(args.out)
    print(f"{'workload':16} {'reqs':>6} {'med ISL':>8} {'med OSL':>8}  kind  -> path")
    for name in list_workloads():
        wl = get_workload(name)
        if not wl.is_synthetic and not args.include_recorded:
            continue
        path = wl.materialize(out, seed=args.seed)
        recs = [
            __import__("json").loads(line) for line in path.read_text().splitlines()
        ]
        med_isl = statistics.median(r["input_length"] for r in recs)
        med_osl = statistics.median(r["output_length"] for r in recs)
        kind = "real" if not wl.is_synthetic else "synth"
        print(
            f"{name:16} {len(recs):6d} {med_isl:8.0f} {med_osl:8.0f}  {kind:5} -> {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
