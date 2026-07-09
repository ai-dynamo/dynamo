# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tests.router.helper import get_runtime, poll_for_worker_instances, wait_for_frontend_ready

from .common import MockerProcessGroup, command_with_cpu_affinity

async def wait_for_frontends(
    ports: list[int], workers: MockerProcessGroup, timeout: int
) -> None:
    await asyncio.gather(
        *(
            wait_for_frontend_ready(
                frontend_url=f"http://127.0.0.1:{port}",
                expected_num_workers=workers.num_workers,
                timeout=timeout,
                engine_workers=workers,
                store_backend="etcd",
                request_plane="tcp",
            )
            for port in ports
        )
    )


async def discover_worker_ids(workers: MockerProcessGroup, timeout: int) -> list[int]:
    """Return the unique DRT worker identities for the shared mocker endpoint."""

    runtime = get_runtime(store_backend="etcd", request_plane="tcp")
    try:
        endpoint = runtime.endpoint(
            f"{workers.namespace}.{workers.component_name}.generate"
        )
        worker_ids = await poll_for_worker_instances(
            endpoint, workers.num_workers, max_wait_time=timeout
        )
    finally:
        runtime.shutdown()
    unique_ids = sorted(set(worker_ids))
    if len(unique_ids) != workers.num_workers:
        raise RuntimeError(
            "discovery worker identities are not one-to-one with logical workers: "
            f"expected={workers.num_workers}, discovered={unique_ids}"
        )
    return unique_ids


def build_aiperf_command(
    args: argparse.Namespace, frontend_ports: list[int], artifact_dir: Path
) -> list[str]:
    command = [
        str(args.aiperf),
        "profile",
        "--artifact-dir",
        str(artifact_dir),
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--url-strategy",
        "round_robin",
    ]
    for port in frontend_ports:
        command.extend(("--url", f"http://127.0.0.1:{port}"))
    command.extend(
        (
            "--synthetic-input-tokens-mean",
            str(args.isl),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(args.osl),
            "--output-tokens-stddev",
            "0",
            "--extra-inputs",
            f"max_tokens:{args.osl}",
            "--extra-inputs",
            f"min_tokens:{args.osl}",
            "--extra-inputs",
            "ignore_eos:true",
            "--extra-inputs",
            "repetition_penalty:1.0",
            "--extra-inputs",
            "temperature:0.0",
            "--concurrency",
            str(args.concurrency),
            "--request-count",
            str(args.requests),
            "--request-timeout-seconds",
            str(args.aiperf_request_timeout_seconds),
            "--num-dataset-entries",
            str(max(args.requests, args.concurrency)),
            "--random-seed",
            "100",
            "--export-level",
            "records",
            "--no-gpu-telemetry",
            "--no-server-metrics",
            "--ui",
            "simple",
        )
    )
    if args.aiperf_workers_max is not None:
        command.extend(("--workers-max", str(args.aiperf_workers_max)))
    if args.record_processors is not None:
        command.extend(("--record-processors", str(args.record_processors)))
    if args.warmup_requests:
        command.extend(("--warmup-request-count", str(args.warmup_requests)))
    return command_with_cpu_affinity(command, args.aiperf_cpus)
