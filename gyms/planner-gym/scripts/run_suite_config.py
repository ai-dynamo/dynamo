#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate or execute independently configured Planner Gym cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ARENA / "src"))

from autoscaling_arena.suite_config import (  # noqa: E402 -- source checkout entrypoint
    SuiteConfigError,
    load_suite_config,
)
from autoscaling_arena.suite_runner import (  # noqa: E402 -- source checkout entrypoint
    execute_suite_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the suite manifest YAML")
    parser.add_argument("--output-dir", type=Path, help="suite artifacts directory")
    parser.add_argument("--case-id", action="append", help="select a case (repeatable)")
    parser.add_argument(
        "--validate-only",
        "--dry-run",
        action="store_true",
        help="validate and list cases",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume checkpoints in --output-dir"
    )
    args = parser.parse_args()
    try:
        config = load_suite_config(args.config)
        selected_ids = set(args.case_id) if args.case_id is not None else None
        cases = config.selected_cases(selected_ids)
        for case in cases:
            print(
                f"{case.case_id}: {case.config.name} "
                f"({case.expected_runs} {case.config.backend.type} runs)",
                flush=True,
            )
        if args.validate_only:
            return 0
        if args.output_dir is None:
            parser.error("--output-dir is required when executing a suite")
        report = execute_suite_config(
            config,
            output_dir=args.output_dir,
            case_ids=selected_ids,
            resume=args.resume,
        )
    except (SuiteConfigError, OSError, ValueError) as exc:
        parser.exit(2, f"Suite error: {exc}\n")
    print(f"Suite report: {args.output_dir.resolve() / 'index.html'}")
    return 0 if report["summary"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
