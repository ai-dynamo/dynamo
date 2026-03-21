# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

os.environ.setdefault("DYNAMO_SKIP_PYTHON_LOG_INIT", "1")

from dynamo.replay import run_trace_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dynamo.replay")
    parser.add_argument("trace_file")
    parser.add_argument("--extra-engine-args")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--replay-concurrency", type=int)
    parser.add_argument(
        "--replay-mode",
        choices=("offline", "online"),
        default="offline",
    )
    parser.add_argument(
        "--router-mode",
        choices=("round_robin", "kv_router"),
        default="round_robin",
    )
    parser.add_argument("--arrival-speedup-ratio", type=float, default=1.0)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    report = run_trace_replay(
        args.trace_file,
        extra_engine_args=args.extra_engine_args,
        num_workers=args.num_workers,
        replay_concurrency=args.replay_concurrency,
        replay_mode=args.replay_mode,
        router_mode=args.router_mode,
        arrival_speedup_ratio=args.arrival_speedup_ratio,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0
