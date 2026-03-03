# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from typing import List

# Default concurrency levels - can be overridden with CONCURRENCIES environment variable
DEFAULT_CONCURRENCIES: List[int] = [1, 2, 5, 10, 50, 100, 250]
# Default request count per concurrency level - can be overridden with REQUEST_COUNT env var
# When set to 0 or unset, defaults to max(concurrency * REQUEST_COUNT_SCALE_FACTOR, 10)
# to ensure the concurrency level is fully utilized and each slot runs enough requests
# for stable measurements
DEFAULT_REQUEST_COUNT: int = 0
REQUEST_COUNT_SCALE_FACTOR: int = 3


def get_concurrency_levels() -> List[int]:
    """Get concurrency levels from environment variable or use defaults"""
    concurrencies_env = os.getenv("CONCURRENCIES")
    if concurrencies_env:
        try:
            # Parse comma-separated values
            concurrencies = [int(x.strip()) for x in concurrencies_env.split(",")]
            # Validate all are positive integers
            for c in concurrencies:
                if c <= 0:
                    raise ValueError(f"Concurrency level must be positive, got: {c}")
            return sorted(concurrencies)
        except ValueError as e:
            print(f"WARNING: Invalid CONCURRENCIES environment variable: {e}")
            print(f"Using default concurrency levels: {DEFAULT_CONCURRENCIES}")
            return DEFAULT_CONCURRENCIES

    return DEFAULT_CONCURRENCIES


def get_request_count() -> int:
    """Get request count from environment variable or use default.

    Returns 0 to indicate 'auto' mode (will be computed per concurrency level).
    """
    request_count_env = os.getenv("REQUEST_COUNT")
    if request_count_env:
        try:
            count = int(request_count_env.strip())
            if count < 0:
                raise ValueError(f"Request count must be non-negative, got: {count}")
            return count
        except ValueError as e:
            print(f"WARNING: Invalid REQUEST_COUNT environment variable: {e}")
            return DEFAULT_REQUEST_COUNT
    return DEFAULT_REQUEST_COUNT


CONCURRENCIES: List[int] = get_concurrency_levels()


def run_aiperf(
    service_url: str,
    model_name: str,
    isl: int,
    osl: int,
    stddev: int,
    concurrency: int,
    output_dir: Path,
    request_count: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-compute request count: need enough requests to fully utilize concurrency
    # and run each slot at least REQUEST_COUNT_SCALE_FACTOR times for stable measurements
    if request_count <= 0:
        request_count = max(concurrency * REQUEST_COUNT_SCALE_FACTOR, 10)
    elif request_count < concurrency:
        print(
            f"WARNING: request_count ({request_count}) < concurrency ({concurrency}). "
            f"Actual in-flight concurrency will be capped at {request_count}.",
            flush=True,
        )

    cmd = [
        "aiperf",
        "profile",
        "-m",
        model_name,
        "--endpoint-type",
        "chat",
        "--streaming",
        "-u",
        service_url,
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        str(stddev),
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--output-tokens-mean",
        str(osl),
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--tokenizer",
        model_name,
        "--artifact-dir",
        str(output_dir),
    ]
    print(
        f"Running aiperf with isl {isl}, osl {osl}, concurrency {concurrency}, request_count {request_count}",
        flush=True,
    )

    aip_process = subprocess.Popen(
        cmd,
        cwd=str(output_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = aip_process.communicate()
    if aip_process.returncode == 0:
        print("Aiperf profiling completed successfully", flush=True)
        if stdout:
            print(stdout)
    else:
        print(f"Aiperf failed with error code: {aip_process.returncode}")
        if stderr:
            print(f"stderr: {stderr}")
        raise subprocess.CalledProcessError(
            aip_process.returncode, cmd, output=stdout, stderr=stderr
        )


def run_concurrency_sweep(
    service_url: str, model_name: str, isl: int, osl: int, stddev: int, output_dir: Path
) -> None:
    concurrency_levels = get_concurrency_levels()
    request_count = get_request_count()
    print(
        f"Running concurrency sweep for {model_name} with ISL {isl} and OSL {osl} and standard deviation {stddev}",
        flush=True,
    )
    print(f"Concurrency levels: {concurrency_levels}", flush=True)
    print(
        f"Request count: {request_count if request_count > 0 else f'auto (max(concurrency*{REQUEST_COUNT_SCALE_FACTOR}, 10))'}",
        flush=True,
    )

    for c in concurrency_levels:
        print(f"Starting concurrency level {c}", flush=True)
        run_aiperf(
            service_url,
            model_name,
            isl,
            osl,
            stddev,
            c,
            output_dir / f"c{c}",
            request_count=request_count,
        )
