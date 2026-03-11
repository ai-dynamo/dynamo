# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


def _build_aiperf_cmd(
    model: str,
    port: int,
    request_count: int,
    warmup_count: int,
    osl: int,
    artifact_dir: Path,
    input_file: str = "",
    isl: int = 400,
    concurrency: int = 0,
    qps: float = 0,
    image_width: int = 0,
    image_height: int = 0,
    images_per_request: int = 0,
) -> List[str]:
    cmd = [
        "aiperf",
        "profile",
        "-m",
        model,
        "-u",
        f"http://localhost:{port}",
        "--request-count",
        str(request_count),
        "--warmup-request-count",
        str(warmup_count),
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

    # Dataset: custom JSONL or synthetic
    if input_file:
        cmd.extend(["--input-file", input_file, "--custom-dataset-type", "single_turn"])
    else:
        # Synthetic mode with optional images
        cmd.extend(
            [
                "--synthetic-input-tokens-mean",
                str(isl),
                "--synthetic-input-tokens-stddev",
                "0",
                "--output-tokens-mean",
                str(osl),
            ]
        )
        if images_per_request > 0 and image_width > 0 and image_height > 0:
            cmd.extend(
                [
                    "--image-width-mean",
                    str(image_width),
                    "--image-width-stddev",
                    "0",
                    "--image-height-mean",
                    str(image_height),
                    "--image-height-stddev",
                    "0",
                    "--image-batch-size",
                    str(images_per_request),
                ]
            )

    # Load mode
    if qps > 0:
        cmd.extend(["--request-rate", str(qps)])
    else:
        cmd.extend(["--concurrency", str(concurrency)])

    return cmd


def run_aiperf_single(
    model: str,
    port: int,
    request_count: int,
    warmup_count: int,
    osl: int,
    artifact_dir: Path,
    input_file: str = "",
    isl: int = 400,
    concurrency: int = 0,
    qps: float = 0,
    image_width: int = 0,
    image_height: int = 0,
    images_per_request: int = 0,
) -> None:
    """Run a single aiperf profile invocation.

    Specify either concurrency (fixed in-flight) or qps (fixed arrival rate).
    When input_file is empty, uses aiperf's synthetic mode with optional images.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_aiperf_cmd(
        model=model,
        port=port,
        request_count=request_count,
        warmup_count=warmup_count,
        osl=osl,
        artifact_dir=artifact_dir,
        input_file=input_file,
        isl=isl,
        concurrency=concurrency,
        qps=qps,
        image_width=image_width,
        image_height=image_height,
        images_per_request=images_per_request,
    )

    load_desc = f"qps={qps}" if qps > 0 else f"concurrency={concurrency}"
    print(f"  aiperf {load_desc} -> {artifact_dir}", flush=True)
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

    print(f"  aiperf {load_desc} done.", flush=True)


def run_concurrency_sweep(
    model: str,
    port: int,
    concurrencies: List[int],
    request_count: int,
    warmup_count: int,
    osl: int,
    output_dir: Path,
    input_file: str = "",
    **kwargs,
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
            **kwargs,
        )

    print(f"Sweep complete. Results in {output_dir}", flush=True)


def run_qps_sweep(
    model: str,
    port: int,
    qps_rates: List[float],
    request_count: int,
    warmup_count: int,
    osl: int,
    output_dir: Path,
    input_file: str = "",
    min_duration: int = 60,
    **kwargs,
) -> None:
    """Run aiperf across QPS rates, writing results under output_dir/qps{N}/.

    Request count is auto-scaled per QPS level: max(request_count, qps * min_duration).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for qps in qps_rates:
        scaled_count = max(request_count, int(qps * min_duration))
        run_aiperf_single(
            model=model,
            port=port,
            qps=qps,
            request_count=scaled_count,
            warmup_count=warmup_count,
            input_file=input_file,
            osl=osl,
            artifact_dir=output_dir / f"qps{qps:g}",
            **kwargs,
        )

    print(f"QPS sweep complete. Results in {output_dir}", flush=True)
