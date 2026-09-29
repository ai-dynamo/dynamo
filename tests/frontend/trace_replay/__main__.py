# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI: `python -m tests.frontend.trace_replay {build,run} ...` (see README.md)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from tests.frontend.trace_replay.endpoints import ENDPOINTS
from tests.frontend.trace_replay.runner import (
    Fixtures,
    Replay,
    RunConfig,
    render_markdown,
    write_outputs,
)


def _build(args: argparse.Namespace) -> int:
    # Imported here so `run` works without transformers / tokenizers installed.
    from tests.frontend.trace_replay import fixtures

    stats = fixtures.build(
        args.teacher,
        args.out,
        offset=args.offset,
        min_turns=args.min_turns,
        max_trajectories=args.max_trajectories,
        variant_every=args.variant_every,
        rows_file=args.rows_file,
        source_key=args.source_teacher,
    )
    print(json.dumps(stats, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    config = RunConfig(
        base_url=args.url.rstrip("/"),
        model=args.model,
        endpoints=args.endpoints.split(","),
        modes=args.modes.split(","),
        concurrency=args.concurrency,
        raw=args.raw,
        variants=set(args.variants.split(",")) if args.variants else None,
        max_trajectories=args.max_trajectories,
        disconnect_every=args.disconnect_every,
        raw_dir=args.out / "raw",
    )
    replay = Replay(Fixtures.load(args.fixtures), config)
    report = asyncio.run(replay.run())
    write_outputs(replay, report, args.out)
    print(render_markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.frontend.trace_replay")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="turn dataset trajectories into fixtures")
    build.add_argument("--teacher", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--offset", type=int, default=0, help="first dataset row")
    build.add_argument("--min-turns", type=int, default=1000)
    build.add_argument("--max-trajectories", type=int, default=1000)
    build.add_argument(
        "--variant-every",
        type=int,
        default=4,
        help="every Nth turn also gets truncation and stop-string variants",
    )
    build.add_argument(
        "--rows-file", type=Path, help="JSON list of dataset-shaped rows to use instead"
    )
    build.add_argument(
        "--source-teacher",
        help="take trajectories from this teacher's dataset slice and render them "
        "as --teacher's output (cross-family replay)",
    )
    build.set_defaults(func=_build)

    run = sub.add_parser("run", help="replay fixtures against a live frontend")
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--url", required=True, help="frontend base URL")
    run.add_argument("--model", required=True, help="served model name")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--endpoints", default=",".join(ENDPOINTS))
    run.add_argument("--modes", default="stream,unary")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--variants", help="comma-separated subset of case variants")
    run.add_argument("--max-trajectories", type=int)
    run.add_argument("--disconnect-every", type=int, default=0)
    run.add_argument(
        "--raw", action="store_true", help="the worker registered no parsers"
    )
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
