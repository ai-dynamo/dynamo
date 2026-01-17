# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from typing import List

# Default concurrency levels - can be overridden with CONCURRENCIES environment variable
DEFAULT_CONCURRENCIES: List[int] = [1, 2, 5, 10, 50, 100, 250]


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


CONCURRENCIES: List[int] = get_concurrency_levels()


def run_aiperf(
    service_url: str,
    model_name: str,
    isl: int,
    osl: int,
    stddev: int,
    concurrency: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        f"Running aiperf with isl {isl}, osl {osl}, concurrency {concurrency}",
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
    print(
        f"Running concurrency sweep for {model_name} with ISL {isl} and OSL {osl} and standard deviation {stddev}",
        flush=True,
    )
    print(f"Concurrency levels: {concurrency_levels}", flush=True)

    for c in concurrency_levels:
        print(f"Starting concurrency level {c}", flush=True)
        run_aiperf(service_url, model_name, isl, osl, stddev, c, output_dir / f"c{c}")
