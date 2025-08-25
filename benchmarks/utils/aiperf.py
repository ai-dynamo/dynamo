# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from typing import List

CONCURRENCIES: List[int] = [1, 2, 5, 10, 50, 100, 250]


def run_ai_perf(
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
        output_dir,
        "-vv",
    ]
    print(
        f"Running aiperf with isl {isl}, osl {osl}, concurrency {concurrency}",
        flush=True,
    )

    aip_process = subprocess.Popen(
        cmd,
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = aip_process.communicate()
    if aip_process.returncode == 0:
        print("AIperf profiling completed successfully", flush=True)
        if stdout:
            print(stdout)
    else:
        print(f"AIperf failed with error code: {aip_process.returncode}")
        print(f"stderr: {stderr}")
        raise subprocess.CalledProcessError(aip_process.returncode, cmd)


def run_concurrency_sweep(
    service_url: str, model_name: str, isl: int, osl: int, stddev: int, output_dir: Path
) -> None:
    print(
        f"Running concurrency sweep for {model_name} with ISL {isl} and OSL {osl} and standard deviation {stddev}",
        flush=True,
    )
    for c in CONCURRENCIES:
        print(f"Starting concurrency level {c}", flush=True)
        run_ai_perf(service_url, model_name, isl, osl, stddev, c, output_dir / f"c{c}")
