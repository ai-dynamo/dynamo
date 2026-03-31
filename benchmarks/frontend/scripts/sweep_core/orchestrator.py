# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sequential plan runner -- iterates through a SweepPlan using a SweepExecutor.

This module is interface-agnostic: it does not import argparse, subprocess,
or kubectl. It is callable from CLI, MCP server, or test harness.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from sweep_core.artifacts import (
    print_results_table,
    write_csv,
    write_summary,
    write_sweep_config,
)
from sweep_core.failures import FailureTracker
from sweep_core.lifecycle import needs_deploy_or_reset
from sweep_core.models import RunResult, RunSpec, SweepPlan
from sweep_core.reporting import generate_report

if TYPE_CHECKING:
    from sweep_executors.base import SweepExecutor


def run(plan: SweepPlan, executor: "SweepExecutor") -> List[RunResult]:
    """Execute a SweepPlan sequentially using the given executor.

    Args:
        plan: The sweep plan to execute.
        executor: The executor that handles individual runs.

    Returns:
        List of RunResult objects, one per run.
    """
    config = plan.config
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "results.csv"
    summary_path = output_root / "summary.md"

    # Write sweep config
    write_sweep_config(config, output_root, total_runs=plan.total_runs)

    failure_tracker = FailureTracker(config.max_consecutive_fails)
    results: List[RunResult] = []
    previous_run: Optional[RunSpec] = None

    try:
        # Prepare executor inside try so cleanup() runs on prepare failure
        executor.prepare(config)

        for i, run_spec in enumerate(plan.runs, 1):
            deploy = run_spec.deploy
            aiperf = run_spec.aiperf
            run_dir = output_root / run_spec.run_id

            # Check skip policy
            if failure_tracker.should_skip(
                deploy.backend, aiperf.concurrency, deploy.workers
            ):
                result = RunResult(
                    run_spec=run_spec,
                    status="skipped",
                    run_dir=str(run_dir),
                )
                results.append(result)
                print(
                    f"\n  [{i}/{plan.total_runs}] SKIPPED {run_spec.run_id} "
                    f"({config.max_consecutive_fails} consecutive failures)"
                )
                continue

            print(f"\n{'=' * 60}")
            print(f"  [{i}/{plan.total_runs}] {run_spec.run_id}")
            print(f"{'=' * 60}")

            # Deploy or reset if needed
            if needs_deploy_or_reset(run_spec, previous_run, plan.isolation_policy):
                prev_deploy = previous_run.deploy if previous_run else None
                executor.apply_deploy(deploy, prev_deploy)

            # Execute the run
            result = executor.execute_run(run_spec, run_dir)
            results.append(result)
            previous_run = run_spec

            # Update failure tracking
            if result.status == "ok":
                failure_tracker.record_success(
                    deploy.backend, aiperf.concurrency, deploy.workers
                )
                rps = f"{result.req_per_sec:.1f}" if result.req_per_sec else "N/A"
                tp50 = f"{result.ttft_p50_ms:.1f}ms" if result.ttft_p50_ms else "N/A"
                print(f"    OK: {rps} req/s, TTFT p50={tp50}")
            else:
                count = failure_tracker.record_failure(
                    deploy.backend, aiperf.concurrency, deploy.workers
                )
                print(f"    FAIL (consecutive: {count}/{config.max_consecutive_fails})")

            # Generate per-run report
            if not config.no_report and result.status == "ok":
                generate_report(run_dir)

            # Write incremental CSV + summary
            write_csv(results, csv_path, config)
            write_summary(results, summary_path)

            # Cooldown between runs
            if i < plan.total_runs:
                time.sleep(config.cooldown)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")
    finally:
        # Final write
        write_csv(results, csv_path, config)
        write_summary(results, summary_path)
        # Cleanup executor
        executor.cleanup()

    # Print final table
    print_results_table(results)
    print(f"\nResults:  {csv_path}")
    print(f"Summary:  {summary_path}")
    print(f"Per-run:  {output_root}/<run_id>/report.md")

    return results
