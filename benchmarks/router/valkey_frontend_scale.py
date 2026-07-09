#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Run an interleaved frontend scale sweep against the Valkey HA harness."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datetime import datetime
from typing import Any

from benchmarks.router.valkey_scale.artifacts import (
    build_summary_document,
    configuration_record,
    write_artifacts,
)
from benchmarks.router.valkey_scale.cli import parse_args
from benchmarks.router.valkey_scale.common import (
    SUMMARY_VERSION,
    build_interleaved_schedule,
)
from benchmarks.router.valkey_scale.plots import make_plot
from benchmarks.router.valkey_scale.sample import (
    require_consistent_child_provenance,
    require_consistent_input_dataset,
    run_sample,
)
from benchmarks.router.valkey_scale.summary import summarize_samples
from benchmarks.router.valkey_scale.validation import (
    git_dirty,
    git_revision,
    prepare_output_dir,
    python_runtime_record,
    validate_args,
    write_json,
)



def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output_dir)
    schedule = build_interleaved_schedule(args.frontend_counts, args.repetitions)
    manifest = {
        "schema_version": SUMMARY_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": str(REPO),
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "python_runtime": python_runtime_record(),
        "argv": sys.argv,
        "configuration": configuration_record(args),
        "schedule": [
            {"sample_index": index, "repetition": repetition, "frontend_count": count}
            for index, (repetition, count) in enumerate(schedule, start=1)
        ],
        "schedule_method": "balanced_cyclic_rotation",
    }
    write_json(args.output_dir / "manifest.json", manifest)

    samples: list[dict[str, Any]] = []
    plot: dict[str, Any] = {
        "status": "pending",
        "png": None,
        "svg": None,
    }
    for sample_index, (repetition, frontend_count) in enumerate(schedule, start=1):
        print(
            "[valkey-frontend-scale] "
            f"starting sample={sample_index}/{len(schedule)} "
            f"repetition={repetition} frontends={frontend_count}; "
            f"artifacts={args.output_dir}",
            flush=True,
        )
        sample = run_sample(
            args,
            sample_index=sample_index,
            repetition=repetition,
            frontend_count=frontend_count,
            output_dir=args.output_dir,
        )
        require_consistent_input_dataset(sample, samples)
        require_consistent_child_provenance(sample, samples)
        samples.append(sample)
        if sample["valid"]:
            rps = sample["metrics"]["request_throughput_rps"]
            print(
                "[valkey-frontend-scale] "
                f"completed sample={sample_index} status=ok rps={rps:.3f}",
                flush=True,
            )
        else:
            print(
                "[valkey-frontend-scale] "
                f"completed sample={sample_index} status=invalid; "
                f"errors={sample['validation_errors']}",
                flush=True,
            )
        document = build_summary_document(args, schedule, samples, plot=plot)
        write_artifacts(args.output_dir, document)
        if not sample["valid"]:
            print(
                "[valkey-frontend-scale] stopping after the first invalid sample; "
                "remaining scale points were not started.",
                file=sys.stderr,
            )
            break

    if any(sample.get("valid") is not True for sample in samples):
        plot["status"] = "not_generated_invalid_samples"
        document = build_summary_document(args, schedule, samples, plot=plot)
        write_artifacts(args.output_dir, document)
        print(
            "[valkey-frontend-scale] invalid sample(s); no aggregate RPS plot was generated. "
            f"Inspect {args.output_dir / 'summary.json'}",
            file=sys.stderr,
        )
        return 1

    summaries = summarize_samples(samples, args.frontend_counts, args.repetitions)
    try:
        png_path, svg_path = make_plot(
            summaries,
            output_dir=args.output_dir,
            baseline_count=args.frontend_counts[0],
            concurrency=args.concurrency,
            isl=args.isl,
            osl=args.osl,
            mocker_processes=args.mocker_processes,
        )
    except Exception as error:
        plot.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        document = build_summary_document(args, schedule, samples, plot=plot)
        write_artifacts(args.output_dir, document)
        print(
            f"[valkey-frontend-scale] plot generation failed: {error}",
            file=sys.stderr,
        )
        return 1

    plot.update(
        {
            "status": "generated",
            "png": str(png_path),
            "svg": str(svg_path),
            "error_bars": "p25-to-p75 request-throughput RPS",
            "ideal_line": f"linear from {args.frontend_counts[0]} frontend(s)",
        }
    )
    document = build_summary_document(args, schedule, samples, plot=plot)
    write_artifacts(args.output_dir, document)
    print(
        "[valkey-frontend-scale] completed valid scale sweep; "
        f"summary={args.output_dir / 'summary.json'} plot={png_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
