# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo entry point for AI Simulate Sweeper and DGD generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dynamo.profiler.sweeper.output import write_outputs
from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    render_dgd,
)
from dynamo.profiler.sweeper.runner import SweepResult, load_sweep_config, run_sweep

_DEFAULT_DGD_NAME = "sweeper-dgd"


def _dgd_name(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError(
            "DGD names must be non-empty names without path separators"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-profiler-sweeper",
        description="Run AI Simulate Sweeper with Dynamo Replay and emit deployments.",
    )
    parser.add_argument("--config", required=True, help="SmartSearchConfig YAML")
    parser.add_argument(
        "--dgd-runtime-image",
        required=True,
        help="runtime image written to generated DGD components",
    )
    parser.add_argument(
        "--dgd-runtime-version-override",
        help="explicit runtimeVersionOverride written to generated DGD components",
    )
    parser.add_argument(
        "--dgd-num-gpus-per-node",
        type=int,
        required=True,
        help="physical GPUs per node used to generate DGD resources",
    )
    parser.add_argument(
        "--dgd-namespace",
        help="namespace written to generated DGDs",
    )
    names = parser.add_mutually_exclusive_group()
    names.add_argument(
        "--dgd-name",
        type=_dgd_name,
        help="name of the best DGD generated for a scalar goal",
    )
    names.add_argument(
        "--dgd-name-prefix",
        type=_dgd_name,
        help="name prefix for DGDs generated from a Pareto front",
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
    parser.add_argument(
        "--no-progress", action="store_true", help="disable Sweeper progress output"
    )
    return parser


def _candidate_key(candidate: Any) -> tuple[float, int, str]:
    return (
        float(candidate.score),
        int(candidate.used_gpus),
        json.dumps(candidate.config, sort_keys=True, separators=(",", ":")),
    )


def _best_candidate(candidates: list[Any]) -> Any:
    """Match Sweeper's scalar ranking: highest score, then fewer GPUs."""
    return max(
        candidates, key=lambda candidate: (candidate.score, -candidate.used_gpus)
    )


@dataclass
class _BestDGDPublisher:
    """Atomically publish a newly discovered scalar incumbent."""

    config: Any
    options: DGDGenerationOptions
    dgd_name: str
    renderer: str
    output: str
    output_dir: Path
    best_key: tuple[float, int, str] | None = None
    published_key: tuple[float, int, str] | None = None
    artifacts: list[dict[str, str]] = field(default_factory=list)
    error: Exception | None = None

    def on_round(self, round_number: int, candidates: list[Any]) -> None:
        self.publish(candidates, round_label=str(round_number))

    def publish(self, candidates: list[Any], *, round_label: str) -> None:
        if not candidates:
            return
        best = _best_candidate(candidates)
        best_key = _candidate_key(best)
        if best_key == self.best_key:
            return
        self.best_key = best_key

        try:
            rendered = render_dgd(
                best,
                self.config.workload,
                self.options,
                dgd_name=self.dgd_name,
                renderer=self.renderer,
            )
            artifacts = write_outputs(
                [rendered],
                self.output_dir,
                stems=[self.dgd_name],
                renderer=self.renderer,
                output=self.output,
            )
        except (
            CandidateMaterializationError,
            RuntimeError,
            ValidationError,
            OSError,
            yaml.YAMLError,
            ValueError,
        ) as exc:
            self.error = exc
            retained = (
                f"; retaining {self.output_dir / self.artifacts[0]['path']}"
                if self.artifacts
                else ""
            )
            print(
                f"new best after round {round_label} could not update the DGD"
                f"{retained}: {exc}",
                file=sys.stderr,
            )
            return

        self.artifacts = artifacts
        self.published_key = best_key
        self.error = None
        print(
            f"new best after round {round_label}: score={best.score:.6g}, "
            f"gpus={best.used_gpus} -> {self.output_dir / artifacts[0]['path']}"
        )


def _validate_config(parser: argparse.ArgumentParser, args: Any, config: Any) -> bool:
    backends = list(config.search_space.backend)
    if len(backends) != 1:
        parser.error(
            f"DGD generation requires exactly one search_space.backend; got {backends}"
        )

    is_pareto = bool(config.goal.is_pareto)
    if is_pareto and args.dgd_name is not None:
        parser.error("--dgd-name is only valid for scalar goals; use --dgd-name-prefix")
    if not is_pareto and args.dgd_name_prefix is not None:
        parser.error("--dgd-name-prefix is only valid for Pareto goals; use --dgd-name")
    return is_pareto


def _render_pareto(
    result: SweepResult,
    options: DGDGenerationOptions,
    *,
    name_prefix: str,
    renderer: str,
    output: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    names = [f"{name_prefix}-{index:03d}" for index in range(len(result.candidates))]
    rendered = [
        render_dgd(
            candidate,
            result.config.workload,
            options,
            dgd_name=name,
            renderer=renderer,
        )
        for candidate, name in zip(result.candidates, names, strict=True)
    ]
    return write_outputs(
        rendered,
        output_dir,
        stems=names,
        renderer=renderer,
        output=output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    publisher: _BestDGDPublisher | None = None
    try:
        config = load_sweep_config(args.config)
        is_pareto = _validate_config(parser, args, config)
        options = DGDGenerationOptions(
            runtime_image=args.dgd_runtime_image,
            runtime_version_override=args.dgd_runtime_version_override,
            namespace=args.dgd_namespace,
            num_gpus_per_node=args.dgd_num_gpus_per_node,
        )
        output_dir = Path(args.output_dir)

        if is_pareto:
            result = run_sweep(config, show_progress=not args.no_progress)
            if not result.candidates:
                raise CandidateMaterializationError(
                    "no feasible candidate found (check backend, workload, SLA, GPU budget, and replay errors)"
                )
            artifacts = _render_pareto(
                result,
                options,
                name_prefix=args.dgd_name_prefix or _DEFAULT_DGD_NAME,
                renderer=args.renderer,
                output=args.output,
                output_dir=output_dir,
            )
            print(
                f"wrote {len(artifacts)} {args.output} deployment output(s) "
                f"to {args.output_dir}"
            )
            return 0

        publisher = _BestDGDPublisher(
            config=config,
            options=options,
            dgd_name=args.dgd_name or _DEFAULT_DGD_NAME,
            renderer=args.renderer,
            output=args.output,
            output_dir=output_dir,
        )
        result = run_sweep(
            config,
            show_progress=not args.no_progress,
            on_round=publisher.on_round,
        )
        if not result.candidates:
            raise CandidateMaterializationError(
                "no feasible candidate found (check backend, workload, SLA, GPU budget, and replay errors)"
            )

        # Publish once more in case a test double or future Sweeper skips the final callback.
        publisher.publish(result.candidates, round_label="final")
        if publisher.published_key != publisher.best_key:
            raise CandidateMaterializationError(
                f"best candidate could not be rendered: {publisher.error}"
            )
        print(
            "best known DGD written to "
            f"{publisher.output_dir / publisher.artifacts[0]['path']}"
        )
        return 0
    except KeyboardInterrupt:
        retained = (
            "; best known DGD remains at "
            f"{publisher.output_dir / publisher.artifacts[0]['path']}"
            if publisher is not None and publisher.artifacts
            else ""
        )
        print(f"dynamo-profiler-sweeper: interrupted{retained}", file=sys.stderr)
        return 130
    except (
        CandidateMaterializationError,
        RuntimeError,
        ValidationError,
        OSError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        print(f"dynamo-profiler-sweeper: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
