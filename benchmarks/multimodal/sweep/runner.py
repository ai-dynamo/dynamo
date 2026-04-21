# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional


def _post_profile(url: str, label: str, *, required: bool) -> None:
    """POST to vllm's /start_profile or /stop_profile.

    When required=True, any failure raises RuntimeError so the sweep fails
    fast instead of producing a misleading empty trace. When required=False,
    failures log a warning and continue (used for stop in a finally block).
    """
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{label} returned HTTP {resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
        msg = f"{label} at {url} failed: {e!r}"
        if required:
            raise RuntimeError(msg) from e
        print(f"WARNING: {msg}", flush=True)


def _build_aiperf_cmd(
    model: str,
    port: int,
    sweep_mode: str,
    sweep_value: int,
    conversation_num: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    artifact_dir: Path,
) -> List[str]:
    if sweep_mode == "concurrency":
        sweep_flag = "--concurrency"
    else:
        sweep_flag = "--request-rate"

    return [
        "aiperf",
        "profile",
        "-m",
        model,
        "-u",
        f"http://localhost:{port}",
        sweep_flag,
        str(sweep_value),
        "--conversation-num",
        str(conversation_num),
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
    sweep_mode: str,
    sweep_value: int,
    conversation_num: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    artifact_dir: Path,
    start_profile_url: Optional[str] = None,
    stop_profile_url: Optional[str] = None,
) -> None:
    """Run a single aiperf profile invocation.

    When ``start_profile_url`` / ``stop_profile_url`` are supplied (typically
    ``http://localhost:{port}/start_profile``), the runner arms / disarms
    vllm's cudaProfilerApi trigger around the aiperf subprocess. Start is
    required-success (fail fast); stop is best-effort in a finally block so
    an aiperf crash can't leave nsys hanging on an un-finalized capture.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_aiperf_cmd(
        model=model,
        port=port,
        sweep_mode=sweep_mode,
        sweep_value=sweep_value,
        conversation_num=conversation_num,
        warmup_count=warmup_count,
        input_file=input_file,
        osl=osl,
        artifact_dir=artifact_dir,
    )

    if start_profile_url:
        _post_profile(start_profile_url, "start_profile", required=True)

    print(f"  aiperf {sweep_mode}={sweep_value} -> {artifact_dir}", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"  aiperf FAILED (exit {proc.returncode})", flush=True)
            for stream_name, stream in [
                ("stderr", proc.stderr),
                ("stdout", proc.stdout),
            ]:
                if stream:
                    for line in stream.strip().splitlines()[-15:]:
                        print(f"    [{stream_name}] {line}", flush=True)
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
            )

        print(f"  aiperf {sweep_mode}={sweep_value} done.", flush=True)
    finally:
        if stop_profile_url:
            _post_profile(stop_profile_url, "stop_profile", required=False)


def run_sweep(
    model: str,
    port: int,
    sweep_mode: str,
    sweep_values: List[int],
    conversation_num: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    output_dir: Path,
) -> None:
    """Run aiperf across all sweep values, writing results under output_dir/{mode}{N}/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for value in sorted(sweep_values):
        run_aiperf_single(
            model=model,
            port=port,
            sweep_mode=sweep_mode,
            sweep_value=value,
            conversation_num=conversation_num,
            warmup_count=warmup_count,
            input_file=input_file,
            osl=osl,
            artifact_dir=output_dir / f"{sweep_mode}{value}",
        )

    print(f"Sweep complete. Results in {output_dir}", flush=True)
