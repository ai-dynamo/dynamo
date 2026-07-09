# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

from .campaign import parse_positive_int_list
from .protocol import (
    HASH_NAMESPACE_SIZE,
    MAX_BLOCKS,
    MAX_HASH_NAMESPACES,
    MAX_LEASE_MS,
    MODES,
    MODE_ALIASES,
    OWNED_MODES,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="apply",
        help="workload; churn_owned counts one STORE+REMOVE pair as one iteration",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=200_000,
        help="logical iterations per sample (churn_owned emits two events each)",
    )
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--connections", type=int, default=64)
    parser.add_argument("--pipeline", type=int, default=128)
    parser.add_argument("--connections-sweep", type=parse_positive_int_list)
    parser.add_argument("--pipeline-sweep", type=parse_positive_int_list)
    parser.add_argument("--blocks-per-event", type=int, default=1)
    parser.add_argument(
        "--churn-prefixes-per-connection",
        type=int,
        default=64,
        help=(
            "bounded prefix-identity ring per connection used by churn_owned; "
            "identities are reused for the full sample"
        ),
    )
    parser.add_argument("--query-blocks", type=int)
    parser.add_argument(
        "--preset",
        choices=("raw", "dynamo", "worker-scale"),
        default="raw",
        help=(
            "topology preset; worker-scale registers 1,024 workers that all own "
            "the queried prefix to expose worker-count scaling"
        ),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--frontends", type=int)
    parser.add_argument("--owners", type=int)
    parser.add_argument("--capacity", type=int, default=16_384)
    parser.add_argument("--lease-ms", type=int, default=120_000)
    parser.add_argument("--worker-lease-ms", type=int, default=600_000)
    parser.add_argument("--latency-sample-limit", type=int, default=200_000)
    parser.add_argument("--appendonly", action="store_true")
    parser.add_argument(
        "--appendfsync", choices=("always", "everysec", "no"), default="everysec"
    )
    parser.add_argument("--auto-aof-rewrite-percentage", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--key", default="dynkv-saturation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--json-indent", type=int)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "events",
        "connections",
        "pipeline",
        "blocks_per_event",
        "repetitions",
        "capacity",
        "lease_ms",
        "worker_lease_ms",
        "latency_sample_limit",
        "churn_prefixes_per_connection",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    for name in ("workers", "frontends", "owners", "query_blocks"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.blocks_per_event > MAX_BLOCKS or (
        args.query_blocks is not None and args.query_blocks > MAX_BLOCKS
    ):
        parser.error(f"block counts cannot exceed {MAX_BLOCKS}")
    if args.churn_prefixes_per_connection * args.blocks_per_event > HASH_NAMESPACE_SIZE:
        parser.error("churn prefix ring exhausts its 48-bit hash namespace")
    if args.capacity > (1 << 32) - 2:
        parser.error("capacity exceeds module range")
    if args.lease_ms > MAX_LEASE_MS or args.worker_lease_ms > MAX_LEASE_MS:
        parser.error(f"lease durations cannot exceed {MAX_LEASE_MS} ms")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("duration-seconds must be positive")
    if args.warmup_seconds < 0:
        parser.error("warmup-seconds cannot be negative")
    if args.auto_aof_rewrite_percentage < 0:
        parser.error("auto-aof-rewrite-percentage cannot be negative")
    connection_values = args.connections_sweep or (args.connections,)
    if max(connection_values) * 2 >= MAX_HASH_NAMESPACES:
        parser.error("connection count exhausts the event hash namespace")
    mode = MODE_ALIASES.get(args.mode, args.mode)
    if (
        mode in OWNED_MODES or args.preset in {"dynamo", "worker-scale"}
    ) and args.duration_seconds is not None:
        total_ms = (args.warmup_seconds + args.duration_seconds) * 1000
        if total_ms >= args.worker_lease_ms * 0.9:
            parser.error("warmup plus duration must remain below 90% of worker lease")
    if not args.server.is_file():
        parser.error(f"Valkey server not found: {args.server}")
    if not args.module.is_file():
        parser.error(f"dynkv module not found: {args.module}")
    if args.data_root is not None:
        args.data_root.mkdir(parents=True, exist_ok=True)
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    args.server = args.server.resolve()
    args.module = args.module.resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.resolve()
    if args.artifact_dir is not None:
        args.artifact_dir = args.artifact_dir.resolve()
