#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Run controlled in-process versus HA-Valkey frontend benchmarks."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import os
from datetime import datetime
from typing import Any

from benchmarks.router.valkey_aiperf.arm import run_arm
from benchmarks.router.valkey_aiperf.cli import parse_args, validate_args
from benchmarks.router.valkey_aiperf.provenance import (
    benchmark_provenance,
    make_output_dir,
    provenance_change_errors,
    release_core_provenance_error,
    write_json,
)
from benchmarks.router.valkey_aiperf.schedule import arms_for_args, build_arm_schedule
from benchmarks.router.valkey_aiperf.validation import summarize_results



def main() -> int:
    args = parse_args()
    validate_args(args)
    arms = arms_for_args(args)
    planned_schedule = build_arm_schedule(arms, args.runs)
    provenance = benchmark_provenance(
        args, include_valkey_artifacts="valkey_ha" in arms
    )
    if error := release_core_provenance_error(provenance):
        raise RuntimeError(error)
    output_dir = make_output_dir(args)
    cpu_affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    schedule_path = output_dir / "schedule.json"
    write_json(schedule_path, planned_schedule)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": str(REPO),
        "argv": sys.argv,
        "git_revision": provenance["git"]["revision"],
        "git_dirty": provenance["git"]["dirty"],
        "python_runtime": provenance["python"],
        "provenance": provenance,
        "runner_cpu_affinity": cpu_affinity,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": list(arms),
        "schedule": planned_schedule,
        "schedule_artifact": str(schedule_path),
        "schedule_method": planned_schedule[0]["method"],
        "cpu_caveat": (
            "Per-role affinity is optional. Any role without an affinity option "
            "inherits the runner CPU set, so high-concurrency results can still be "
            "client- or host-CPU-limited rather than a router limit."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)

    results: list[dict[str, Any]] = []
    for planned in planned_schedule:
        run_number = int(planned["run"])
        arm = str(planned["arm"])
        sample_index = int(planned["sample_index"])
        print(
            f"[valkey-router-aiperf] starting sample={sample_index} "
            f"run={run_number} arm={arm}; artifacts={output_dir}",
            flush=True,
        )
        try:
            provenance_before = benchmark_provenance(
                args, include_valkey_artifacts="valkey_ha" in arms
            )
        except Exception as error:
            result = {
                "sample_index": sample_index,
                "run": run_number,
                "arm": arm,
                "schedule_ordinal": planned["ordinal"],
                "status": "provenance_failed",
                "validation_errors": [
                    f"could not capture arm-start provenance: {type(error).__name__}: {error}"
                ],
            }
            results.append(result)
            write_json(
                output_dir / "summary.json",
                {
                    "provenance": provenance,
                    "schedule": planned_schedule,
                    "runs": results,
                    "analysis": summarize_results(results, planned_schedule),
                },
            )
            break
        preflight_errors = provenance_change_errors(
            provenance, provenance_before, provenance_before
        )
        if preflight_errors:
            result = {
                "sample_index": sample_index,
                "run": run_number,
                "arm": arm,
                "schedule_ordinal": planned["ordinal"],
                "status": "provenance_failed",
                "validation_errors": preflight_errors,
                "provenance_before": provenance_before,
            }
            results.append(result)
            write_json(
                output_dir / "summary.json",
                {
                    "provenance": provenance,
                    "schedule": planned_schedule,
                    "runs": results,
                    "analysis": summarize_results(results, planned_schedule),
                },
            )
            break

        result = run_arm(args, arm=arm, run_number=run_number, output_dir=output_dir)
        result.update(
            {
                "sample_index": sample_index,
                "schedule_ordinal": planned["ordinal"],
                "provenance_before": provenance_before,
            }
        )
        provenance_errors: list[str]
        try:
            provenance_after = benchmark_provenance(
                args, include_valkey_artifacts="valkey_ha" in arms
            )
        except Exception as error:
            provenance_after = None
            provenance_errors = [
                f"could not capture arm-end provenance: {type(error).__name__}: {error}"
            ]
        else:
            provenance_errors = provenance_change_errors(
                provenance, provenance_before, provenance_after
            )
        result["provenance_after"] = provenance_after
        if provenance_errors:
            result.setdefault("validation_errors", []).extend(provenance_errors)
            result["status"] = "provenance_changed"
        results.append(result)
        write_json(Path(result["run_dir"]) / "result.json", result)
        analysis = summarize_results(results, planned_schedule)
        write_json(
            output_dir / "summary.json",
            {
                "provenance": provenance,
                "schedule": planned_schedule,
                "runs": results,
                "analysis": analysis,
            },
        )
        print(
            f"[valkey-router-aiperf] completed sample={sample_index} "
            f"run={run_number} arm={arm} status={result['status']}",
            flush=True,
        )
        if provenance_errors:
            break

    analysis = summarize_results(results, planned_schedule)
    write_json(
        output_dir / "summary.json",
        {
            "provenance": provenance,
            "schedule": planned_schedule,
            "runs": results,
            "analysis": analysis,
        },
    )
    print(f"[valkey-router-aiperf] results: {output_dir / 'summary.json'}")
    return 0 if analysis["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
