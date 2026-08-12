# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental ``python -m aisimulate.sweeper`` entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..runner_discovery import RunnerFactoryResolutionError, resolve_runner_factory
from .config import Candidate, SmartSearchConfig
from .reporting import (
    format_sweep_results,
    serialize_sweep_results,
    write_sweep_results,
)


def load_config_or_parser_error(
    parser: argparse.ArgumentParser, path: str
) -> SmartSearchConfig:
    """Load a sweep config with the legacy CLI's concise error messages."""

    try:
        return SmartSearchConfig.from_yaml(path)
    except OSError as exc:  # missing file, a directory, unreadable, etc.
        parser.error(f"could not read config {path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {path}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {path}: {exc}")


def print_candidates_or_exit(
    config: SmartSearchConfig,
    candidates: Sequence[Candidate],
    *,
    top_n: int = 5,
    output_format: str = "table",
) -> None:
    """Preserve concise legacy CLI result and no-candidate behavior for wrappers."""

    if not candidates:
        print(
            "no feasible candidate found "
            "(check backends / SLA / gpu_budget / replay errors)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if output_format == "json":
        print(
            json.dumps(
                serialize_sweep_results(config, candidates),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    print(format_sweep_results(config, candidates, top_n=top_n))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _runner_factory(stack: str):
    return resolve_runner_factory(stack)


def _run_sweep(
    config: SmartSearchConfig,
    *,
    stack: str,
    show_progress: bool,
) -> list[Candidate]:
    # Keep search and its modeling/optimizer dependencies off the import path for
    # ``--help`` and malformed-config errors.
    from .search import Sweeper

    return Sweeper(
        runner_factory=_runner_factory(stack),
        show_progress=show_progress,
    ).run(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate.sweeper",
        description=(
            "[EXPERIMENTAL] Recommend deployment configurations with the "
            "engine-only or optional Dynamo replay stack."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a SmartSearchConfig YAML file"
    )
    parser.add_argument(
        "--stack",
        choices=("engine", "dynamo"),
        default="engine",
        help="Replay composition to use (default: engine)",
    )
    parser.add_argument(
        "--top-n",
        type=_positive_int,
        default=5,
        help="Number of ranked scalar recommendations to print/save (default: 5)",
    )
    parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help="Terminal output format (default: table)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write sweep_results.json and best_config_topn.csv (or pareto.csv) "
            "to this directory"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config_or_parser_error(parser, args.config)
    try:
        candidates = _run_sweep(
            config,
            stack=args.stack,
            show_progress=not args.no_progress,
        )
        print_candidates_or_exit(
            config,
            candidates,
            top_n=args.top_n,
            output_format=args.output_format,
        )
        if args.output_dir is not None:
            paths = write_sweep_results(
                args.output_dir,
                config,
                candidates,
                top_n=args.top_n,
            )
            for path in paths:
                print(f"Saved: {path}", file=sys.stderr)
    except (
        OSError,
        RunnerFactoryResolutionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
