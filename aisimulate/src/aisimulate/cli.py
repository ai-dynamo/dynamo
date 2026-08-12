# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ``aisimulate`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .cli_config import CLIConfigError, build_predict_config, build_recommend_config
from .cli_reporting import print_prediction, print_recommendations
from .runner_discovery import RunnerFactoryResolutionError, resolve_runner_factory
from .sweeper.reporting import write_sweep_results
from .sweeper.search import Sweeper

_OBJECTIVES = (
    "throughput",
    "throughput_per_gpu",
    "throughput_per_user",
    "e2e_latency",
    "goodput",
    "goodput_per_gpu",
    "pareto",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stack",
        required=True,
        choices=("engine", "dynamo"),
        help="Simulation composition: standalone engine or full Dynamo system",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "sglang", "trtllm"),
        help="Inference backend; recommend defaults to auto when no config is supplied",
    )
    parser.add_argument("--model", help="Model name or path")
    parser.add_argument("--system", help="Hardware SKU, for example h200_sxm")
    parser.add_argument("--config", help="Simulation/search YAML configuration")
    parser.add_argument(
        "--traffic", help="Workload YAML, either direct or under workload:"
    )
    parser.add_argument("--isl", type=int, help="Synthetic input sequence length")
    parser.add_argument("--osl", type=int, help="Synthetic output sequence length")
    load = parser.add_mutually_exclusive_group()
    load.add_argument("--concurrency", type=int, help="Closed-loop in-flight requests")
    load.add_argument("--request-rate", type=float, help="Open-loop request rate (QPS)")
    load.add_argument(
        "--kv-load-ratio",
        type=float,
        help="Candidate-relative closed-loop KV load",
    )
    parser.add_argument(
        "--num-request-ratio",
        type=float,
        help="Synthetic request-count multiplier (default: 10)",
    )
    parser.add_argument("--objective", choices=_OBJECTIVES)
    parser.add_argument("--sla-ttft-ms", type=float)
    parser.add_argument("--sla-itl-ms", type=float)
    parser.add_argument("--sla-e2e-ms", type=float)
    parser.add_argument(
        "--strict-sla",
        action="store_true",
        help="Reject candidates whose aggregate mean latency misses the SLA",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Result format (default: text)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the public two-command parser."""

    parser = argparse.ArgumentParser(
        prog="aisimulate",
        description="Predict or recommend LLM inference deployment configurations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="Evaluate exactly one resolved deployment configuration",
    )
    _add_shared_arguments(predict)
    predict.add_argument(
        "--deployment-mode",
        choices=("agg", "disagg"),
        help="Deployment topology (default for scalar flags: agg)",
    )
    predict.add_argument("--tp-size", type=int, help="Tensor-parallel size")
    predict.add_argument("--replicas", type=int, help="Replica count (default: 1)")
    predict.add_argument("--max-num-batched-tokens", type=int)
    predict.add_argument("--max-num-seqs", type=int)

    recommend = subparsers.add_parser(
        "recommend",
        help="Search and rank feasible deployment configurations",
    )
    _add_shared_arguments(recommend)
    recommend.add_argument(
        "--deployment-mode",
        action="append",
        choices=("agg", "disagg"),
        help="Topology to search; repeat to select both (default: both)",
    )
    recommend.add_argument("--total-gpus", type=int, help="Maximum GPU budget")
    recommend.add_argument("--max-rounds", type=int)
    recommend.add_argument("--parallel-evals", type=int)
    recommend.add_argument("--candidates-per-round", type=int)
    recommend.add_argument(
        "--top-n",
        type=_positive_int,
        default=5,
        help="Number of scalar recommendations in text and CSV output (default: 5)",
    )
    recommend.add_argument(
        "--output-dir",
        help="Write sweep_results.json and CSV results to this directory",
    )
    return parser


def _run_predict(args: argparse.Namespace) -> int:
    config = build_predict_config(args)
    factory = resolve_runner_factory(args.stack)
    candidates = Sweeper(
        runner_factory=factory,
        show_progress=args.output == "text",
    ).run(config)
    if not candidates:
        print("aisimulate predict: no feasible prediction", file=sys.stderr)
        return 1
    if len(candidates) != 1:
        raise RuntimeError(
            f"predict invariant failed: expected one candidate, got {len(candidates)}"
        )
    print_prediction(stack=args.stack, candidate=candidates[0], output=args.output)
    return 0


def _run_recommend(args: argparse.Namespace) -> int:
    config = build_recommend_config(args)
    factory = resolve_runner_factory(args.stack)
    candidates = Sweeper(
        runner_factory=factory,
        show_progress=args.output == "text",
    ).run(config)
    if not candidates:
        print(
            "aisimulate recommend: no feasible candidate found "
            "(check backends, SLA, GPU budget, and replay errors)",
            file=sys.stderr,
        )
        return 1
    print_recommendations(
        stack=args.stack,
        config=config,
        candidates=candidates,
        output=args.output,
        top_n=args.top_n,
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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            return _run_predict(args)
        return _run_recommend(args)
    except (
        CLIConfigError,
        OSError,
        RunnerFactoryResolutionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2
