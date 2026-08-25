# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo entry point for AI Simulate Sweeper and DGD materialization."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from dynamo.profiler.sweeper.output import write_outputs
from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDMaterializationOptions,
    render_dgd,
)
from dynamo.profiler.sweeper.runner import SweepResult, run_sweep


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-profiler-sweeper",
        description="Run AI Simulate Sweeper with Dynamo Replay and emit deployments.",
    )
    parser.add_argument("--config", required=True, help="SmartSearchConfig YAML")
    parser.add_argument(
        "--backend",
        choices=("sglang", "trtllm", "vllm"),
        required=True,
        help="single backend pinned for this materialization prototype",
    )
    parser.add_argument(
        "--backend-version",
        required=True,
        help="engine version shared by the Candidate performance data and target image",
    )
    parser.add_argument(
        "--backend-image",
        required=True,
        help="backend runtime image written to the generated DGD",
    )
    parser.add_argument(
        "--dynamo-version",
        required=True,
        help="Dynamo version carried by the image and written as runtimeVersionOverride",
    )
    parser.add_argument(
        "-r",
        "--renderer",
        choices=("aic", "direct"),
        default="aic",
        help="DGD lowering implementation (default: aic)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=("dgd", "kustomize"),
        default="dgd",
        help="output form (default: dgd)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory for generated deployment output"
    )
    parser.add_argument("--namespace", help="namespace written to generated DGDs")
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        required=True,
        help="physical GPUs per node",
    )
    parser.add_argument(
        "--name-prefix", default="sweeper-dgd", help="generated DGD name prefix"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable Sweeper progress output"
    )
    return parser


def _render_dgds(
    result: SweepResult,
    options: DGDMaterializationOptions,
    *,
    renderer: str,
) -> list[str]:
    rendered_dgds: list[str] = []
    for candidate_index, candidate in enumerate(result.candidates):
        rendered_dgds.append(
            render_dgd(
                candidate,
                result.config.workload,
                options,
                candidate_index=candidate_index,
                renderer=renderer,
            )
        )
    return rendered_dgds


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        options = DGDMaterializationOptions(
            backend=args.backend,
            backend_version=args.backend_version,
            backend_image=args.backend_image,
            dynamo_version=args.dynamo_version,
            namespace=args.namespace,
            num_gpus_per_node=args.num_gpus_per_node,
            name_prefix=args.name_prefix,
        )
        result = run_sweep(args.config, show_progress=not args.no_progress)
        if not result.candidates:
            parser.error(
                "no feasible candidate found (check backend, workload, SLA, GPU budget, and replay errors)"
            )
        rendered_dgds = _render_dgds(
            result,
            options,
            renderer=args.renderer,
        )
        artifacts = write_outputs(
            rendered_dgds,
            Path(args.output_dir),
            renderer=args.renderer,
            output=args.output,
        )
    except (
        CandidateMaterializationError,
        RuntimeError,
        ValidationError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        print(f"dynamo-profiler-sweeper: error: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {len(artifacts)} {args.output} deployment output(s) "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
