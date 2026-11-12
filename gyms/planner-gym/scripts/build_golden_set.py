#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct replay-ready Golden workloads from an external base trace.

The committed YAML recipe defines schedules and composition selectors only.
Source paths are runtime arguments so private datasets never need to live in, or
be named by, this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_GYM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_GYM_DIR / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a trace-agnostic Autoscaling Arena Golden Set."
    )
    parser.add_argument("recipe", type=Path, help="public Golden Set YAML recipe")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="external trace file or homogeneous directory; repeatable",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "dynamo_request_trace_v1", "replay_jsonl"),
        default="auto",
        help="input format (auto rejects mixed collections)",
    )
    parser.add_argument(
        "--source-block-size",
        type=int,
        default=None,
        help="required for replay_jsonl; for raw traces, checks embedded values",
    )
    parser.add_argument(
        "--reference-rps",
        type=float,
        required=True,
        help="requests/second represented by recipe load_factor=1.0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-source-requests",
        type=int,
        default=100_000,
        help="request-count guard; use 0 only after intentionally scoping the source",
    )
    parser.add_argument(
        "--max-source-hashes",
        type=int,
        default=5_000_000,
        help="loaded prefix-hash reference guard; 0 disables it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new output directory (existing directories are never overwritten)",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    from autoscaling_arena.datasets import build_golden_set, load_golden_set_recipe

    try:
        recipe = load_golden_set_recipe(args.recipe)
        result = build_golden_set(
            recipe,
            args.source,
            args.out,
            source_format=args.source_format,
            source_block_size=args.source_block_size,
            reference_rps=args.reference_rps,
            seed=args.seed,
            max_source_requests=(
                None if args.max_source_requests == 0 else args.max_source_requests
            ),
            max_source_hashes=(
                None if args.max_source_hashes == 0 else args.max_source_hashes
            ),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    source = result.manifest["source"]
    print(
        f"source: {source['replayable_requests']} replayable requests, "
        f"block_size={source['block_size']}, format={source['format']}"
    )
    for workload in result.manifest["workloads"]:
        print(
            f"{workload['name']}: {workload['request_count']} requests, "
            f"{workload['duration_ms'] / 1000:.1f}s -> "
            f"{result.output_dir / workload['file']}"
        )
    print(f"manifest: {result.manifest_path}")
    print(f"match config fragment: {result.match_config_fragment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
