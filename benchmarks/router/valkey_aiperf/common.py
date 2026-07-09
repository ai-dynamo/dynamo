#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""Reproducible aiperf A/B for the Valkey-backed KV router.

Example (uses the intentionally demanding default workload):

    DYNAMO_GPU_PARALLEL_DOWNLOADS_READY=1 \
      .venv/bin/python benchmarks/router/valkey_router_aiperf.py --runs 3

The script creates a fresh topology with configurable frontend and logical
mock-worker counts for every arm and run. The workers share one mocker OS
process by default, or can be split across independent processes without
changing the logical worker/routing-rank count. It uses TCP for the request
plane, a local Valkey primary plus replica for the
``valkey_ha`` arm, and sends one aiperf client across every frontend. When
authoritative admission is enabled,
``--arm both`` also runs an ``inprocess_immediate`` control with the same
immediate-routing policy, so its result is not conflated with a policy change.
An etcd endpoint must already be available (``ETCD_ENDPOINTS`` or the
``--etcd-endpoints`` option), as must a NATS endpoint (``NATS_SERVER``). The
request plane remains TCP; NATS is used by default only for loss-free mocker
KV-event delivery during the A/B comparison. A local NATS server on
``nats://127.0.0.1:4222`` is used unless ``--nats-server`` overrides it.

Results are written beneath ``bench/results/`` by default.  Each arm keeps its
own process logs, Valkey data, aiperf artifact directory, command, and summary
record so an interrupted or failed run remains inspectable.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping


REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.router.helper import (  # noqa: E402
    get_runtime,
    poll_for_worker_instances,
    wait_for_frontend_ready,
)
from tests.router.common import valkey_index_key  # noqa: E402
from tests.router.mocker_process import MockerProcess  # noqa: E402
from tests.router.router_process import (  # noqa: E402
    FrontendRouterProcess,
    ValkeyModuleProcess,
    ValkeySentinelProcess,
)
from tests.utils.constants import ROUTER_MODEL_NAME  # noqa: E402
from tests.utils.port_utils import allocate_ports, deallocate_ports  # noqa: E402


DEFAULT_AIPERF = REPO / "dynamo/bin/aiperf"
DEFAULT_VALKEY_SERVER = Path(
    shutil.which("valkey-server") or "/nonexistent/valkey-server"
)
DEFAULT_DYNKV_MODULE = REPO / "lib/kv-router/valkey-module/dynkv.so"
DEFAULT_LOGICAL_MOCKER_WORKERS = 4
DEFAULT_VALKEY_GC_INTERVAL_MS = 60_000
MIN_VALKEY_GC_INTERVAL_MS = 1_000
MAX_VALKEY_GC_INTERVAL_MS = 86_400_000
DEFAULT_VALKEY_GC_INSPECTION_BUDGET = 256
MAX_VALKEY_GC_INSPECTION_BUDGET = 1_048_576


# These are the non-Valkey policy settings required to make the local router
# behave like phase-one module admission: immediate selection, no frontend
# queue/replica accounting, and no lower-tier cache or temperature credits.
# The normal in-process arm intentionally does *not* use these flags; it is the
# synchronized multi-frontend baseline. It enables replica sync explicitly and
# therefore is not the literal bare-CLI default. The immediate control lets us
# distinguish a module cost/benefit from the cost of changing policy.
IMMEDIATE_LOCAL_ROUTER_FLAGS = (
    "--no-router-replica-sync",
    "--router-queue-threshold",
    "None",
    "--no-router-track-prefill-tokens",
    "--no-router-track-output-blocks",
    "--router-prefill-load-model",
    "none",
    "--router-host-cache-hit-weight",
    "0",
    "--router-disk-cache-hit-weight",
    "0",
    "--shared-cache-multiplier",
    "0",
    "--shared-cache-type",
    "none",
    "--router-kv-overlap-score-credit-decay",
    "0",
    "--router-temperature",
    "0",
)


# Aggregated metrics are saved verbatim in every run result. These selectors
# additionally produce a compact median-of-runs A/B view in summary.json.
SUMMARY_METRICS: dict[str, tuple[str, str, str | None]] = {
    "request_throughput_rps": (
        "request_throughput",
        "avg",
        "request_throughput_rps",
    ),
    "ttft_ms_p50": ("time_to_first_token", "p50", "ttft_ms"),
    "ttft_ms_p95": ("time_to_first_token", "p95", "ttft_ms"),
    "itl_ms_p50": ("inter_token_latency", "p50", "itl_ms"),
    "itl_ms_p95": ("inter_token_latency", "p95", "itl_ms"),
    "request_latency_ms_p50": ("request_latency", "p50", "request_latency_ms"),
    "request_latency_ms_p95": ("request_latency", "p95", "request_latency_ms"),
    "isl_tokens_avg": ("input_sequence_length", "avg", "isl_tokens"),
    "osl_tokens_avg": ("output_sequence_length", "avg", "osl_tokens"),
    "output_token_throughput_tps": ("output_token_throughput", "avg", None),
}


# A benchmark must not silently inherit a developer shell's routing policy.
# The wrappers below add the policy under test explicitly; everything else is
# reset to Dynamo's documented defaults before each arm starts.
ROUTER_ENV_TO_CLEAR = (
    "DYN_ROUTER_LOAD_AWARE",
    "DYN_ROUTER_REPLICA_SYNC",
    "DYN_ROUTER_TRACK_ACTIVE_BLOCKS",
    "DYN_ROUTER_TRACK_OUTPUT_BLOCKS",
    "DYN_ROUTER_ASSUME_KV_REUSE",
    "DYN_ROUTER_TRACK_PREFILL_TOKENS",
    "DYN_ROUTER_PREFILL_LOAD_MODEL",
    "DYN_ROUTER_PREFILL_LOAD_SCALE",
    "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
    "DYN_OVERLAP_SCORE_WEIGHT",
    "DYN_ROUTER_SNAPSHOT_THRESHOLD",
    "DYN_ROUTER_RESET_STATES",
    "DYN_ROUTER_TTL_SECS",
    "DYN_ROUTER_EVENT_THREADS",
    "DYN_ROUTER_QUEUE_POLICY",
    "DYN_USE_REMOTE_INDEXER",
    "DYN_ROUTER_PREDICTED_TTL_SECS",
    "DYN_ROUTER_QUEUE_THRESHOLD",
    "DYN_ROUTER_POLICY_CONFIG",
    "DYN_ROUTER_HOST_CACHE_HIT_WEIGHT",
    "DYN_ROUTER_DISK_CACHE_HIT_WEIGHT",
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",
    "DYN_ROUTER_TEMPERATURE",
    "DYN_ROUTER_SHARED_CACHE_MULTIPLIER",
    "DYN_ROUTER_SHARED_CACHE_TYPE",
    "DYN_SHARED_CACHE_MULTIPLIER",
    "DYN_SHARED_CACHE_TYPE",
    "DYN_ROUTER_DURABLE_KV_EVENTS",
    "DYN_ROUTER_USE_KV_EVENTS",
    "DYN_USE_KV_EVENTS",
    "DYN_ROUTER_VALKEY_CONFIG",
    "DYN_ROUTER_VALKEY_WORKER_LEASE_MS",
    "DYN_ROUTER_VALKEY_GC_INTERVAL_MS",
    "DYN_ROUTER_VALKEY_GC_INSPECTION_BUDGET",
    "DYN_ROUTER_VALKEY_SENTINEL_URLS",
    "DYN_ROUTER_VALKEY_SENTINEL_MASTER_NAME",
    "DYN_ROUTER_VALKEY_SENTINEL_QUORUM",
    "DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT",
    "DYN_ROUTER_VALKEY_ALLOW_DEGRADED_WRITES",
)


# These messages indicate that worker-owned events were lost or that the
# owner lease was surrendered before the publisher drained. They can occur
# after the last Valkey INFO snapshot, so commandstats alone cannot validate
# graceful teardown.
VALKEY_TEARDOWN_FAILURE_MARKERS = (
    "DYNKV_STALE_WORKER_OWNER",
    "Failed to publish event",
    "Direct Valkey KV metadata integrity fault",
    "Direct Valkey APPLY_OWNED failed",
    "entering the worker integrity fence",
    "entered the permanent integrity fence",
    "fencing direct-Valkey metadata",
    "Direct Valkey unregister could not be proved",
    "Valkey lifecycle GC tick failed",
    "Failed to unregister mocker Valkey lease",
    "KV event publisher drain timed out",
    "KV event publisher tasks did not stop after operation cancellation",
    "Skipping mocker Valkey unregister after forced publisher shutdown",
)


class HarnessRequest:
    """Small pytest-request-compatible object used by the process wrappers."""

    def __init__(self, log_dir: Path) -> None:
        self.node = SimpleNamespace(name=str(log_dir))
        self._finalizers: list[Any] = []

    def addfinalizer(self, callback: Any) -> None:
        self._finalizers.append(callback)

    def close(self) -> None:
        while self._finalizers:
            self._finalizers.pop()()


class MockerProcessGroup:
    """Aggregate independently managed mocker processes for readiness checks."""

    def __init__(self, shards: list[MockerProcess]) -> None:
        if not shards:
            raise ValueError("at least one mocker process is required")
        namespaces = {shard.namespace for shard in shards}
        components = {shard.component_name for shard in shards}
        if len(namespaces) != 1 or len(components) != 1:
            raise ValueError("all mocker processes must share one discovery endpoint")
        self.shards = shards
        self.namespace = shards[0].namespace
        self.component_name = shards[0].component_name
        self.num_workers = sum(shard.num_workers for shard in shards)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def cpu_list(value: str) -> str:
    """Validate the common ``taskset --cpu-list`` comma/range syntax."""

    normalized: list[str] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("CPU list contains an empty segment")
        bounds = part.split("-")
        if len(bounds) == 1:
            try:
                cpu = int(bounds[0])
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid CPU number {part!r}"
                ) from error
            if cpu < 0:
                raise argparse.ArgumentTypeError("CPU numbers must be non-negative")
        elif len(bounds) == 2:
            try:
                start, end = (int(bound) for bound in bounds)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid CPU range {part!r}"
                ) from error
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(
                    f"CPU range must satisfy 0 <= start <= end: {part!r}"
                )
        else:
            raise argparse.ArgumentTypeError(f"invalid CPU range {part!r}")
        normalized.append(part)
    return ",".join(normalized)


def partition_logical_workers(
    total_workers: int, process_count: int
) -> list[tuple[int, int]]:
    """Return contiguous ``[start, end)`` worker ordinals for each process."""

    if total_workers < 1:
        raise ValueError("total_workers must be at least 1")
    if not 1 <= process_count <= total_workers:
        raise ValueError(
            f"process_count must be in 1..={total_workers}, got {process_count}"
        )
    base, remainder = divmod(total_workers, process_count)
    partitions: list[tuple[int, int]] = []
    start = 0
    for process_index in range(process_count):
        count = base + (1 if process_index < remainder else 0)
        end = start + count
        partitions.append((start, end))
        start = end
    return partitions


def mocker_process_layout(
    total_workers: int, process_count: int, dp_size: int
) -> list[dict[str, int]]:
    """Describe stable logical-worker and routing-target ownership per process."""

    return [
        {
            "process_index": process_index,
            "logical_worker_start": worker_start,
            "logical_worker_end_exclusive": worker_end,
            "logical_worker_count": worker_end - worker_start,
            # DP ranks are scoped by worker identity in Dynamo. The global
            # ordinal is artifact-only and makes shard coverage unambiguous.
            "worker_local_dp_rank_start": 0,
            "worker_local_dp_rank_end_exclusive": dp_size,
            "routing_target_ordinal_start": worker_start * dp_size,
            "routing_target_ordinal_end_exclusive": worker_end * dp_size,
        }
        for process_index, (worker_start, worker_end) in enumerate(
            partition_logical_workers(total_workers, process_count), start=1
        )
    ]


def command_with_cpu_affinity(command: list[str], cpus: str | None) -> list[str]:
    """Pin a command and all of its descendants, preserving no-affinity defaults."""

    return command if cpus is None else ["taskset", "--cpu-list", cpus, *command]


def apply_managed_process_affinity(process: Any, cpus: str | None) -> None:
    if cpus is not None:
        process.command = command_with_cpu_affinity(process.command, cpus)


def retarget_mocker_process(
    shard: MockerProcess, namespace: str, cpus: str | None
) -> None:
    """Put an independently constructed MockerProcess in a shared namespace."""

    endpoint = f"dyn://{namespace}.{shard.component_name}.generate"
    managed_process = getattr(shard, "_process", None)
    if managed_process is None:
        raise RuntimeError("benchmark mocker shard did not create a managed process")
    try:
        endpoint_index = managed_process.command.index("--endpoint") + 1
    except ValueError as error:
        raise RuntimeError("mocker command has no --endpoint option") from error
    if endpoint_index >= len(managed_process.command):
        raise RuntimeError("mocker command has no --endpoint value")
    managed_process.command[endpoint_index] = endpoint
    shard.namespace = namespace
    shard.endpoint = endpoint
    apply_managed_process_affinity(managed_process, cpus)
