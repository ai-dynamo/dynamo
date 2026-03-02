# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for the multimodal benchmark sweep.

Usage:
    python -m benchmarks.multimodal.sweep --config experiment.yaml
    python -m benchmarks.multimodal.sweep --config experiment.yaml --output-dir /tmp/results
    python -m benchmarks.multimodal.sweep --config experiment.yaml --model MyModel --osl 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, resolve_repo_root
from .orchestrator import run_sweep


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multimodal benchmark sweep from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.multimodal.sweep --config experiments/cache_sweep.yaml\n"
            "  python -m benchmarks.multimodal.sweep --config exp.yaml --osl 200 --skip-plots\n"
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config.",
    )
    parser.add_argument(
        "--concurrencies",
        default=None,
        help="Override concurrency levels (comma-separated, e.g. '1,2,4,8').",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help="Override output sequence length.",
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=None,
        help="Override request count per concurrency level.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=None,
        help="Skip plot generation.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.model is not None:
        overrides["model"] = args.model
    if args.concurrencies is not None:
        overrides["concurrencies"] = [
            int(x.strip()) for x in args.concurrencies.split(",")
        ]
    if args.osl is not None:
        overrides["osl"] = args.osl
    if args.request_count is not None:
        overrides["request_count"] = args.request_count
    if args.skip_plots:
        overrides["skip_plots"] = True

    config = load_config(args.config, cli_overrides=overrides or None)

    repo_root = resolve_repo_root()
    config.validate(repo_root=repo_root)

    run_sweep(config, repo_root=repo_root)


if __name__ == "__main__":
    main()
