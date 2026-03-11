# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def _build_aiperf_cmd(
    model: str,
    port: int,
    concurrency: int,
    request_count: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    artifact_dir: Path,
) -> List[str]:
    return [
        "aiperf",
        "profile",
        "-m",
        model,
        "-u",
        f"http://localhost:{port}",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--warmup-request-count",
        str(warmup_count),
        "--input-file",
        input_file,
        "--custom-dataset-type",
        "single_turn",
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "stream:true",
        "--streaming",
        "--artifact-dir",
        str(artifact_dir),
        "--ui",
        "none",
        "--no-server-metrics",
    ]


def run_aiperf_single(
    model: str,
    port: int,
    concurrency: int,
    request_count: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    artifact_dir: Path,
) -> None:
    """Run a single aiperf profile invocation."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_aiperf_cmd(
        model=model,
        port=port,
        concurrency=concurrency,
        request_count=request_count,
        warmup_count=warmup_count,
        input_file=input_file,
        osl=osl,
        artifact_dir=artifact_dir,
    )

    print(f"  aiperf concurrency={concurrency} -> {artifact_dir}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    # Stream aiperf output to stdout in real time (and capture to log file)
    aiperf_log = artifact_dir / "aiperf_stdout.log"
    with open(aiperf_log, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            print(f"    [aiperf] {line}", end="", flush=True)
            log_f.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"  aiperf FAILED (exit {proc.returncode})", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"  aiperf concurrency={concurrency} done.", flush=True)


def run_concurrency_sweep(
    model: str,
    port: int,
    concurrencies: List[int],
    request_count: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    output_dir: Path,
) -> None:
    """Run aiperf across all concurrency levels, writing results under output_dir/c{N}/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in concurrencies:
        run_aiperf_single(
            model=model,
            port=port,
            concurrency=c,
            request_count=request_count,
            warmup_count=warmup_count,
            input_file=input_file,
            osl=osl,
            artifact_dir=output_dir / f"c{c}",
        )

    print(f"Sweep complete. Results in {output_dir}", flush=True)
