# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .common import (
    DEFAULT_HARNESS,
    DEFAULT_MODEL,
    cpu_list,
    frontend_counts,
    nonnegative_int,
    positive_int,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frontend-counts",
        type=frontend_counts,
        default=(1, 2, 3),
        metavar="COUNTS",
        help="Ordered comma-separated frontend counts to sweep.",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=3,
        help="Fresh, interleaved samples per frontend count.",
    )
    parser.add_argument(
        "--mocker-processes",
        type=positive_int,
        default=1,
        help=(
            "Independent mocker OS processes per child. The fixed four logical "
            "workers and their total routing ranks do not change."
        ),
    )
    parser.add_argument(
        "--frontend-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list forwarded for all child frontends.",
    )
    parser.add_argument(
        "--mocker-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list forwarded for all child mockers.",
    )
    parser.add_argument(
        "--valkey-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list forwarded for both child Valkey servers.",
    )
    parser.add_argument(
        "--aiperf-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list forwarded for aiperf and its workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty directory for all child artifacts and scale results.",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=DEFAULT_HARNESS,
        help="Path to the single-topology Valkey aiperf harness.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch each fresh child harness.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", help="Optional tokenizer passed to aiperf.")
    parser.add_argument("--requests", type=positive_int, default=32768)
    parser.add_argument("--warmup-requests", type=nonnegative_int, default=4096)
    parser.add_argument("--concurrency", type=positive_int, default=4096)
    parser.add_argument("--isl", type=positive_int, default=1024)
    parser.add_argument("--osl", type=positive_int, default=1024)
    parser.add_argument(
        "--valkey-admission-lease-ms",
        type=positive_int,
        default=120000,
        help="Authoritative module reservation lease passed to every child.",
    )
    parser.add_argument(
        "--valkey-gc-interval-ms",
        type=nonnegative_int,
        default=60_000,
        help="Direct-worker lifecycle-GC interval forwarded to every child.",
    )
    parser.add_argument(
        "--valkey-gc-inspection-budget",
        type=positive_int,
        default=256,
        help="Per-tick lifecycle-GC inspection budget forwarded to every child.",
    )
    parser.add_argument(
        "--aiperf-timeout-seconds",
        type=positive_int,
        default=3600,
        help="Per-child aiperf wall-clock timeout, including finalization.",
    )
    parser.add_argument(
        "--aiperf-request-timeout-seconds",
        type=positive_int,
        default=300,
        help="Per-request aiperf timeout for every child.",
    )
    parser.add_argument(
        "--tcp-request-timeout-seconds",
        type=positive_int,
        default=300,
        help="TCP request-plane acknowledgement timeout for every child.",
    )
    parser.add_argument("--ready-timeout", type=positive_int, default=180)
    parser.add_argument("--replica-ready-timeout", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=nonnegative_int, default=2)
    parser.add_argument(
        "--event-plane",
        choices=("nats", "zmq"),
        default="nats",
        help="KV-event plane forwarded to each child harness.",
    )
    parser.add_argument(
        "--etcd-endpoints",
        default=os.environ.get("ETCD_ENDPOINTS", "http://127.0.0.1:2379"),
    )
    parser.add_argument(
        "--nats-server",
        default=os.environ.get("NATS_SERVER", "nats://127.0.0.1:4222"),
    )
    parser.add_argument(
        "--aiperf",
        type=Path,
        help="Optional aiperf executable forwarded to each child harness.",
    )
    parser.add_argument(
        "--aiperf-workers-max",
        type=positive_int,
        help="Optional aiperf client-worker cap forwarded to each child.",
    )
    parser.add_argument(
        "--record-processors",
        type=positive_int,
        help="Optional aiperf record-processor count forwarded to each child.",
    )
    parser.add_argument(
        "--harness-extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "One additional non-topology option forwarded verbatim to the child "
            "harness; repeat as needed, for example "
            "--harness-extra-arg=--speedup-ratio=50000."
        ),
    )
    return parser.parse_args()
