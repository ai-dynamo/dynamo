# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental ``python -m dynamo.profiler.spica`` entry point."""

from __future__ import annotations

import argparse
import sys

import yaml
from pydantic import ValidationError

from .config import SmartSearchConfig
from .search import run_smart_search


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.profiler.spica",
        description=(
            "[EXPERIMENTAL] Spica smart sweeper. Its configuration, output, and "
            "optimization behavior may change without notice."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a SmartSearchConfig YAML file"
    )
    args = parser.parse_args()

    try:
        config = SmartSearchConfig.from_yaml(args.config)
    except OSError as exc:  # missing file, a directory, unreadable, etc.
        parser.error(f"could not read config {args.config}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {args.config}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {args.config}: {exc}")

    candidates = run_smart_search(config)
    if not candidates:
        print(
            "no feasible candidate found "
            "(check backends / SLA / gpu_budget / replay errors)",
            file=sys.stderr,
        )
        sys.exit(1)
    if config.goal.is_pareto:
        # The result is a Pareto front: show every objective + the concrete concurrency, since
        # the single `score` (the first objective) hides the tradeoff the front is about.
        print(f"pareto front ({len(candidates)} non-dominated):")
        for i, candidate in enumerate(candidates):
            objectives = ", ".join(
                f"{key}={value:.4g}"
                for key, value in (candidate.objectives or {}).items()
            )
            concurrency = candidate.config.get("concurrency")
            conc = f" concurrency={concurrency}" if concurrency is not None else ""
            print(f"{i}: {objectives}{conc} used_gpus={candidate.used_gpus}")
    else:
        for i, candidate in enumerate(candidates):
            print(f"{i}: score={candidate.score} used_gpus={candidate.used_gpus}")


if __name__ == "__main__":
    main()
