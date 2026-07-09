# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.utils.constants import ROUTER_MODEL_NAME

from .common import (
    DEFAULT_AIPERF,
    DEFAULT_DYNKV_MODULE,
    DEFAULT_LOGICAL_MOCKER_WORKERS,
    DEFAULT_VALKEY_GC_INSPECTION_BUDGET,
    DEFAULT_VALKEY_GC_INTERVAL_MS,
    DEFAULT_VALKEY_SERVER,
    MAX_VALKEY_GC_INSPECTION_BUDGET,
    MAX_VALKEY_GC_INTERVAL_MS,
    MIN_VALKEY_GC_INTERVAL_MS,
    cpu_list,
    nonnegative_int,
    positive_float,
    positive_int,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        choices=(
            "both",
            "matched",
            "inprocess",
            "inprocess_immediate",
            "valkey_ha",
        ),
        default="both",
        help=(
            "Router implementation(s) to measure. 'matched' interleaves only "
            "the immediate in-process control and authoritative Valkey HA arm."
        ),
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=3,
        help=(
            "Interleaved repetitions. Every arm is a complete restart; three "
            "runs are the minimum useful default for a median comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty directory for all run artifacts.",
    )
    parser.add_argument(
        "--aiperf",
        type=Path,
        default=DEFAULT_AIPERF,
        help="aiperf executable; this checkout provides v0.10 by default.",
    )
    parser.add_argument("--model", default=ROUTER_MODEL_NAME)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer path/name for aiperf. Defaults to --model.",
    )
    parser.add_argument(
        "--frontend-count",
        type=positive_int,
        default=3,
        help="Number of independently started frontend processes and aiperf URLs.",
    )
    parser.add_argument(
        "--logical-mocker-workers",
        type=positive_int,
        default=DEFAULT_LOGICAL_MOCKER_WORKERS,
        help="Total independently discoverable logical mock workers.",
    )
    parser.add_argument(
        "--mocker-processes",
        type=positive_int,
        default=1,
        help=(
            "Independent dynamo.mocker OS processes. Logical workers are partitioned "
            "across them without changing worker or DP-rank totals."
        ),
    )
    parser.add_argument(
        "--frontend-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list shared by all frontend processes.",
    )
    parser.add_argument(
        "--mocker-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list shared by all mocker processes.",
    )
    parser.add_argument(
        "--valkey-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list shared by the Valkey primary and replica.",
    )
    parser.add_argument(
        "--aiperf-cpus",
        type=cpu_list,
        metavar="CPU_LIST",
        help="Optional taskset CPU list inherited by aiperf and its client workers.",
    )
    parser.add_argument(
        "--etcd-endpoints",
        default=os.environ.get("ETCD_ENDPOINTS", "http://127.0.0.1:2379"),
        help="Existing etcd endpoint(s) inherited by frontends and workers.",
    )
    parser.add_argument(
        "--event-plane",
        choices=("zmq", "nats"),
        default="nats",
        help=(
            "KV event plane. NATS is the default because instant mocker events can "
            "otherwise race ZMQ frontend subscriptions."
        ),
    )
    parser.add_argument(
        "--nats-server",
        default=os.environ.get("NATS_SERVER", "nats://127.0.0.1:4222"),
        help="NATS endpoint when --event-plane=nats.",
    )
    parser.add_argument("--block-size", type=positive_int, default=16)
    parser.add_argument("--isl", type=positive_int, default=1024)
    parser.add_argument("--osl", type=positive_int, default=1024)
    parser.add_argument("--concurrency", type=positive_int, default=4096)
    parser.add_argument("--requests", type=positive_int, default=16384)
    parser.add_argument(
        "--warmup-requests",
        type=nonnegative_int,
        default=4096,
        help="aiperf warmup request count before the profiling phase.",
    )
    parser.add_argument(
        "--aiperf-workers-max",
        type=positive_int,
        help=(
            "Optional aiperf client-worker cap. Leave unset to retain aiperf's "
            "safe auto-capped default rather than spawning one process per request."
        ),
    )
    parser.add_argument(
        "--record-processors",
        type=positive_int,
        help=(
            "Optional aiperf record-processor count; leave unset for its "
            "automatic value."
        ),
    )
    parser.add_argument(
        "--aiperf-timeout-seconds",
        type=positive_int,
        default=1800,
        help="Wall-clock limit for aiperf, including a potentially stuck finalizer.",
    )
    parser.add_argument(
        "--aiperf-request-timeout-seconds",
        type=positive_int,
        default=300,
        help=(
            "Per-request aiperf HTTP timeout. This bounds failed streams before "
            "the outer aiperf/finalizer timeout is needed."
        ),
    )
    parser.add_argument(
        "--tcp-request-timeout-seconds",
        type=positive_int,
        default=300,
        help=(
            "Internal TCP request-plane acknowledgement timeout propagated to "
            "frontends and mock workers. It must cover queueing at the requested "
            "concurrency; the runtime default of five seconds is too short for "
            "the 4,096-request burst used by this harness."
        ),
    )
    parser.add_argument("--mocker-max-num-seqs", type=positive_int, default=16384)
    parser.add_argument(
        "--mocker-max-num-batched-tokens", type=positive_int, default=16384
    )
    parser.add_argument(
        "--mocker-data-parallel-size",
        type=positive_int,
        default=8,
        help=(
            "Data-parallel ranks per logical mock worker. The default mirrors "
            "the supplied high-capacity mocker command (four workers x eight "
            "ranks = 32 routing ranks)."
        ),
    )
    parser.add_argument("--num-gpu-blocks", type=positive_int, default=131072)
    parser.add_argument("--speedup-ratio", type=positive_float, default=100000.0)
    parser.add_argument("--kv-bytes-per-token", type=positive_int, default=128)
    parser.add_argument(
        "--valkey-server",
        type=Path,
        default=Path(os.environ.get("VALKEY_SERVER", DEFAULT_VALKEY_SERVER)),
    )
    parser.add_argument(
        "--dynkv-module",
        type=Path,
        default=Path(os.environ.get("DYNKV_MODULE", DEFAULT_DYNKV_MODULE)),
    )
    parser.add_argument("--valkey-connection-pool-size", type=positive_int, default=64)
    parser.add_argument(
        "--valkey-event-batching-timeout-ms", type=nonnegative_int, default=1
    )
    parser.add_argument(
        "--valkey-gc-interval-ms",
        type=nonnegative_int,
        default=DEFAULT_VALKEY_GC_INTERVAL_MS,
        help=(
            "Direct-worker lifecycle-GC interval. Zero disables GC; nonzero values "
            "must be in the worker-supported range."
        ),
    )
    parser.add_argument(
        "--valkey-gc-inspection-budget",
        type=positive_int,
        default=DEFAULT_VALKEY_GC_INSPECTION_BUDGET,
        help="Maximum module lifecycle records inspected by each worker GC tick.",
    )
    parser.add_argument(
        "--valkey-authoritative-admission",
        action="store_true",
        help=(
            "Have the replicated Valkey module atomically select and reserve the "
            "worker. This disables frontend-local replica sync for the HA arm."
        ),
    )
    parser.add_argument(
        "--kill-valkey-primary",
        action="store_true",
        help=(
            "Start three Sentinels and SIGKILL the Valkey primary after profiling "
            "begins. This uses the explicit two-node degraded-write policy and "
            "requires --arm valkey_ha --runs 1."
        ),
    )
    parser.add_argument(
        "--sentinel-down-after-ms",
        type=positive_int,
        default=500,
        help="Sentinel primary failure detector used by --kill-valkey-primary.",
    )
    parser.add_argument(
        "--sentinel-failover-timeout-ms",
        type=positive_int,
        default=5_000,
        help="Sentinel failover timeout used by --kill-valkey-primary.",
    )
    parser.add_argument(
        "--fault-after-completed-records",
        type=positive_int,
        default=1,
        help="Completed profiling records observed before SIGKILL injection.",
    )
    parser.add_argument(
        "--fault-timeout-seconds",
        type=positive_float,
        default=120.0,
        help="Deadline for Sentinel promotion after the primary is killed.",
    )
    parser.add_argument(
        "--valkey-admission-lease-ms",
        type=positive_int,
        default=120000,
        help=(
            "Authoritative-admission lease lifetime. The 120s default covers the "
            "1024-token, c=4096 benchmark's observed request lifetime so normal "
            "request cleanup releases capacity before expiry; use a shorter value "
            "only when its renewal traffic and crash-recovery tradeoff are desired."
        ),
    )
    parser.add_argument("--ready-timeout", type=positive_int, default=180)
    parser.add_argument("--replica-ready-timeout", type=positive_float, default=60.0)
    parser.add_argument(
        "--settle-seconds",
        type=nonnegative_int,
        default=2,
        help="Give event subscriptions and direct worker publishers time to converge.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.aiperf = args.aiperf.expanduser().resolve()
    args.valkey_server = args.valkey_server.expanduser().resolve()
    args.dynkv_module = args.dynkv_module.expanduser().resolve()
    args.tokenizer = args.tokenizer or args.model

    if args.mocker_processes > args.logical_mocker_workers:
        raise ValueError(
            "--mocker-processes must be in 1..="
            f"{args.logical_mocker_workers}; "
            f"got {args.mocker_processes}"
        )
    affinities = {
        role: affinity
        for role, affinity in (
            ("frontend", args.frontend_cpus),
            ("mocker", args.mocker_cpus),
            ("valkey", args.valkey_cpus),
            ("aiperf", args.aiperf_cpus),
        )
        if affinity is not None
    }
    if affinities:
        if shutil.which("taskset") is None:
            raise RuntimeError("taskset is required for CPU affinity options")
        for role, affinity in affinities.items():
            validation = subprocess.run(
                [
                    "taskset",
                    "--cpu-list",
                    affinity,
                    sys.executable,
                    "-c",
                    "pass",
                ],
                capture_output=True,
                text=True,
            )
            if validation.returncode != 0:
                detail = validation.stderr.strip() or validation.stdout.strip()
                raise ValueError(
                    f"--{role}-cpus cannot be applied on this host: {detail}"
                )

    if not args.aiperf.is_file() or not os.access(args.aiperf, os.X_OK):
        raise FileNotFoundError(
            f"aiperf executable is not executable: {args.aiperf}. "
            "Pass --aiperf to a v0.10+ installation."
        )
    if args.arm in {"both", "valkey_ha"}:
        if not args.valkey_server.is_file() or not os.access(
            args.valkey_server, os.X_OK
        ):
            raise FileNotFoundError(
                f"Valkey server is not executable: {args.valkey_server}"
            )
        if not args.dynkv_module.is_file():
            raise FileNotFoundError(
                f"DYNKV module was not found: {args.dynkv_module}. "
                "Build it before running this harness."
            )
    if args.valkey_authoritative_admission and args.arm == "inprocess":
        raise ValueError(
            "--valkey-authoritative-admission requires --arm valkey_ha, matched, "
            "inprocess_immediate, or both"
        )
    if args.arm == "matched" and not args.valkey_authoritative_admission:
        raise ValueError("--arm matched requires --valkey-authoritative-admission")
    if args.valkey_authoritative_admission and not (
        10_000 <= args.valkey_admission_lease_ms <= 600_000
    ):
        raise ValueError(
            "--valkey-admission-lease-ms must be in 10000..=600000 for frontend authoritative admission"
        )
    if args.valkey_gc_interval_ms != 0 and not (
        MIN_VALKEY_GC_INTERVAL_MS
        <= args.valkey_gc_interval_ms
        <= MAX_VALKEY_GC_INTERVAL_MS
    ):
        raise ValueError(
            "--valkey-gc-interval-ms must be 0 (disabled) or in "
            f"{MIN_VALKEY_GC_INTERVAL_MS}..={MAX_VALKEY_GC_INTERVAL_MS}"
        )
    if args.valkey_gc_inspection_budget > MAX_VALKEY_GC_INSPECTION_BUDGET:
        raise ValueError(
            "--valkey-gc-inspection-budget must be in 1..="
            f"{MAX_VALKEY_GC_INSPECTION_BUDGET}"
        )
    if args.arm == "inprocess_immediate" and not args.valkey_authoritative_admission:
        raise ValueError(
            "--arm inprocess_immediate is the control for "
            "--valkey-authoritative-admission; enable that option as well"
        )
    if args.kill_valkey_primary:
        if args.arm != "valkey_ha" or args.runs != 1:
            raise ValueError("--kill-valkey-primary requires --arm valkey_ha --runs 1")
        if not args.valkey_authoritative_admission:
            raise ValueError(
                "--kill-valkey-primary requires --valkey-authoritative-admission"
            )
        if args.fault_after_completed_records >= args.requests:
            raise ValueError(
                "--fault-after-completed-records must be smaller than --requests"
            )
