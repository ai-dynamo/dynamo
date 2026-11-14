# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the same replay implementation in local, GitHub and internal CI jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contract import ReplayError, dumps, write_json
from .runner import RunOptions, _write_junit, prepare_assets, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.frontend.trace_replay")
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser(
        "prepare-assets", help="fetch only the pinned native model assets"
    )
    assets.add_argument("--corpus", type=Path, required=True)
    assets.add_argument("--model-dir", type=Path, required=True)
    execute = commands.add_parser(
        "run", help="exercise the installed frontend and CPU token worker"
    )
    execute.add_argument("--corpus", type=Path, required=True)
    execute.add_argument("--model-dir", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--limit", type=int)
    execute.add_argument("--schedules", default=",".join(RunOptions.schedules))
    execute.add_argument("--modes", default="unary,sse")
    execute.add_argument("--concurrency", type=int, default=4)
    execute.add_argument("--baseline", type=Path)
    execute.add_argument("--request-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-assets":
            print(dumps({"assets": prepare_assets(args.corpus, args.model_dir)}))
            return 0
        code, summary = run(
            RunOptions(
                corpus=args.corpus,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                limit=args.limit,
                schedules=tuple(args.schedules.split(",")),
                modes=tuple(args.modes.split(",")),
                concurrency=args.concurrency,
                baseline=args.baseline,
                request_timeout=args.request_timeout,
            )
        )
    except (ReplayError, OSError) as exc:
        summary = {
            "schema": "dynamo-frontend-replay-result/v1",
            "status": "infra_error",
            "exit_code": 2,
            "infrastructure_errors": [str(exc)],
        }
        if args.command == "run":
            if not (args.output_dir / "summary.json").exists():
                write_json(args.output_dir / "summary.json", summary)
            if not (args.output_dir / "junit.xml").exists():
                _write_junit(args.output_dir / "junit.xml", [], [str(exc)])
        print(dumps(summary))
        return 2
    print(dumps(summary))
    return code


if __name__ == "__main__":
    sys.exit(main())
