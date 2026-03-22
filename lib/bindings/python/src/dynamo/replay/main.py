# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("DYNAMO_SKIP_PYTHON_LOG_INIT", "1")

from dynamo.replay import run_trace_replay


def _prepare_extra_engine_args(extra_engine_args, router_queue_policy):
    if router_queue_policy is None:
        return extra_engine_args, None

    payload = {}
    if extra_engine_args is not None:
        payload = json.loads(Path(extra_engine_args).read_text(encoding="utf-8"))
    payload["router_queue_policy"] = router_queue_policy

    temp_dir = tempfile.TemporaryDirectory()
    merged_path = Path(temp_dir.name) / "replay_extra_engine_args.json"
    merged_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(merged_path), temp_dir


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
    parser.add_argument(
        "--router-queue-policy",
        choices=("fcfs", "wspt", "lcfs"),
    )
    parser.add_argument("--arrival-speedup-ratio", type=float, default=1.0)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    merged_extra_engine_args, temp_dir = _prepare_extra_engine_args(
        args.extra_engine_args,
        args.router_queue_policy,
    )

    try:
        report = run_trace_replay(
            args.trace_file,
            extra_engine_args=merged_extra_engine_args,
            num_workers=args.num_workers,
            replay_concurrency=args.replay_concurrency,
            replay_mode=args.replay_mode,
            router_mode=args.router_mode,
            arrival_speedup_ratio=args.arrival_speedup_ratio,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0
