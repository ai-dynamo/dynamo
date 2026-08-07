# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the planner load-predictor sweep on a mooncake trace and print results.

This utility supports the experimental Sweeper feature and may change with its configuration schema.

python aisimulate/examples/sweeper/tools/run_load_predictor_sweep.py \
    --trace traffic.jsonl --policies throughput_180_5
"""

from __future__ import annotations

import argparse

from dynamo.planner.simulation.load_predictor import (
    LOAD_PREDICTOR_PRESETS,
    sweep_load_predictor,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the load-predictor sweep on a trace")
    p.add_argument("--trace", required=True)
    p.add_argument(
        "--policies",
        nargs="+",
        default=["throughput_180_5"],
        help="Planner scaling_policy candidates",
    )
    a = p.parse_args()

    result = sweep_load_predictor(
        policies=a.policies,
        candidates=list(LOAD_PREDICTOR_PRESETS),
        trace_path=a.trace,
        show_progress=True,
    )

    print(f"\nreason = {result.reason}")
    for iv, best in result.best_by_interval.items():
        print(f"\n== interval {iv}s ==  best = {best}")
        for pid, loss in sorted(result.losses[iv].items(), key=lambda kv: kv[1]):
            mark = "  <-- best" if pid == best else ""
            print(f"  {loss:8.4f}  {pid}{mark}")


if __name__ == "__main__":
    main()
