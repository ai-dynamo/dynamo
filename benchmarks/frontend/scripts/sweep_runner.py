#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Frontend performance sweep runner.

Thin CLI entry point that delegates to sweep_core (pure logic), sweep_executors
(how runs execute), and sweep_k8s (k8s helpers).

Supports two execution modes:
  - local: delegates each run to run_perf.sh (mocker + frontend per run)
  - k8s: DGD-based execution with aiperf against a k8s-deployed frontend

Sweep dimensions (all configurable):
  - tokenizers (hf, fastokens)
  - concurrency levels
  - ISL values
  - worker counts

Usage:
    # Local smoke test (2 runs)
    python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32 --isl 512 \\
        --benchmark-duration 30 --speedup-ratio 1000000

    # Full local sweep with mocker
    python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024,2048

    # K8s sweep with DGD
    python3 sweep_runner.py --mode k8s --dgd-name dynamo-bench-mocker \\
        --tokenizers hf,fastokens --concurrency 50,100 --isl 512

    # K8s with custom deploy template
    python3 sweep_runner.py --mode k8s --deploy-template dgd/templates/vllm.yaml \\
        --tokenizers hf --concurrency 128 --isl 1024

    # Transport saturation sweep
    python3 sweep_runner.py --tokenizers hf --concurrency 4096 \\
        --num-requests 16384,32768 --workers 1,2,4,8 --speedup-ratio 1000000

    # Dry run
    python3 sweep_runner.py --dry-run --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024

    # Emit plan as JSON (for Argo or MCP)
    python3 sweep_runner.py --emit-plan --tokenizers hf --concurrency 50 --isl 512
"""

import sys
from pathlib import Path

# Ensure the scripts directory is on the path for package imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_core.config import build_argument_parser, config_from_args  # noqa: E402
from sweep_core.orchestrator import run as run_sweep  # noqa: E402
from sweep_core.planner import build_plan, print_plan  # noqa: E402


def main():
    parser = build_argument_parser()

    # Add CLI-only flags that don't belong in SweepConfig
    parser.add_argument(
        "--emit-plan",
        action="store_true",
        help="Print the sweep plan as JSON and exit (no execution)",
    )

    args = parser.parse_args()

    # Build typed config from args
    config = config_from_args(args)

    # Build plan
    plan = build_plan(config)
    print_plan(plan)

    # Emit plan JSON mode
    if args.emit_plan:
        print(plan.to_json())
        return

    # Dry run mode
    if config.dry_run:
        for i, run_spec in enumerate(plan.runs, 1):
            print(f"  [{i}/{plan.total_runs}] {run_spec.run_id}")
        return

    # Select executor based on mode
    if config.mode == "local":
        from sweep_executors.local import LocalExecutor

        executor = LocalExecutor()
    elif config.mode == "k8s":
        from sweep_executors.k8s_dgd import K8sDgdExecutor

        executor = K8sDgdExecutor()
    else:
        print(
            f"ERROR: Unknown mode '{config.mode}'. Use 'local' or 'k8s'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run the sweep
    run_sweep(plan, executor)


if __name__ == "__main__":
    main()
