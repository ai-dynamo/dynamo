# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared single-run parser and the engine-only AISimulate entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..runner import EngineReplayRunnerFactory
from .config import ReplayCliConfig, parse_base_replay_config
from .reporting import format_report_table, write_per_request_jsonl, write_report_json

TRACE_FORMATS = (
    "mooncake",
    "mooncake-delta",
    "agentic_mooncake",
    "applied_compute_agentic",
    "dynamo",
)


def add_base_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every Dynamo-neutral single-run option exactly once."""

    parser.add_argument("trace_files", nargs="*")
    parser.add_argument("--extra-engine-args")
    parser.add_argument("--prefill-engine-args")
    parser.add_argument("--decode-engine-args")
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--request-count", type=int)
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Poisson open-loop request rate in requests per second",
    )
    parser.add_argument(
        "--arrival-interval-ms",
        type=float,
        help="fixed open-loop interval between synthetic requests in milliseconds",
    )
    parser.add_argument("--arrival-seed", type=int, default=42)
    parser.add_argument("--turns-per-session", type=int, default=1)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    parser.add_argument("--num-prefix-groups", type=int, default=0)
    parser.add_argument("--inter-turn-delay-ms", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-prefill-workers", type=int, default=1)
    parser.add_argument("--num-decode-workers", type=int, default=1)
    parser.add_argument("--replay-concurrency", type=int)
    parser.add_argument(
        "--replay-mode",
        choices=("offline", "online"),
        default="offline",
    )
    parser.add_argument("--arrival-speedup-ratio", type=float, default=1.0)
    parser.add_argument("--trace-format", choices=TRACE_FORMATS, default="mooncake")
    parser.add_argument("--trace-block-size", type=int)
    parser.add_argument("--trace-shared-prefix-ratio", type=float, default=0.0)
    parser.add_argument("--trace-num-prefix-groups", type=int, default=0)
    parser.add_argument("--report-json")
    parser.add_argument("--per-request-jsonl")
    parser.add_argument("--max-sim-time-seconds", type=float)
    parser.add_argument("--sla-ttft-ms", type=float)
    parser.add_argument("--sla-itl-ms", type=float)
    parser.add_argument("--sla-e2e-ms", type=float)


def build_parser(
    *, prog: str = "python -m aisimulate.replay"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run one engine-only AISimulate replay.",
    )
    add_base_replay_arguments(parser)
    return parser


def parse_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> ReplayCliConfig:
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return parse_base_replay_config(args)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    config = parse_config(parser, argv)
    if config.replay_mode != "offline":
        parser.error("online replay requires the Dynamo runtime")

    try:
        report = (
            EngineReplayRunnerFactory()
            .create(0)
            .run(
                config.to_replay_spec(),
                output_requirements=config.output_requirements,
            )
        )
        raw_report = report.metadata.get("native_report")
        if not isinstance(raw_report, dict):
            raise RuntimeError("engine replay did not provide its full native report")
        summary = raw_report.get("summary", raw_report)
        if not isinstance(summary, dict):
            raise RuntimeError("engine replay report summary must be a JSON object")
        report_path = write_report_json(raw_report, config.output.report_json)
        if config.output.per_request_jsonl is not None:
            per_request = raw_report.get("per_request")
            if per_request is not None and not isinstance(per_request, list):
                raise RuntimeError("engine replay per_request report must be a list")
            write_per_request_jsonl(config.output.per_request_jsonl, per_request)
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    sys.stdout.write(format_report_table(summary))
    sys.stdout.write("\n")
    sys.stdout.write(f"Saved full report to: {report_path}\n")
    return 0
