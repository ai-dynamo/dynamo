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
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  aiperf FAILED (exit {proc.returncode})", flush=True)
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-10:]:
                print(f"    {line}", flush=True)
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )

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

    for c in sorted(concurrencies):
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
