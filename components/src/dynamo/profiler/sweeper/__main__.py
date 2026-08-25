# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo entry point for AI Simulate Sweeper and DGD materialization."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dynamo.profiler.sweeper.dgd import (
    CandidateMaterializationError,
    DGDMaterializationOptions,
    materialize_candidate_dgd,
)
from dynamo.profiler.sweeper.runner import SweepResult, run_sweep


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-profiler-sweeper",
        description="Run AI Simulate Sweeper with Dynamo Replay and emit candidate DGDs.",
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
        "--output-dir", required=True, help="directory for candidate JSON and DGD YAML"
    )
    parser.add_argument("--namespace", help="namespace written to generated DGDs")
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        required=True,
        help="physical GPUs per node",
    )
    parser.add_argument(
        "--name-prefix", default="sweeper-candidate", help="generated DGD name prefix"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable Sweeper progress output"
    )
    return parser


def _candidate_payload(candidate: Any, dgd_file: str) -> dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        payload = candidate.model_dump(mode="json")
    else:
        payload = {
            "config": candidate.config,
            "used_gpus": candidate.used_gpus,
            "score": candidate.score,
            "metrics": candidate.metrics,
            "objectives": candidate.objectives,
        }
    payload["dgd_file"] = dgd_file
    return payload


def _write_outputs(
    result: SweepResult,
    output_dir: Path,
    options: DGDMaterializationOptions,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(result.candidates):
        stem = f"candidate-{candidate_index:03d}"
        dgd_path = output_dir / f"{stem}.dgd.yaml"
        candidate_path = output_dir / f"{stem}.json"
        dgd_yaml = materialize_candidate_dgd(
            candidate,
            result.config.workload,
            options,
            candidate_index=candidate_index,
        )
        dgd_path.write_text(dgd_yaml, encoding="utf-8")
        payload = _candidate_payload(candidate, dgd_path.name)
        candidate_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index.append(payload)

    (output_dir / "index.json").write_text(
        json.dumps({"candidates": index}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


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
        index = _write_outputs(result, Path(args.output_dir), options)
    except (
        CandidateMaterializationError,
        RuntimeError,
        ValidationError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        print(f"dynamo-profiler-sweeper: error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {len(index)} candidate DGD(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
