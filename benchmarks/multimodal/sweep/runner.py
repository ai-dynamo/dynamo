# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional


def _build_aiperf_cmd(
    model: str,
    port: int,
    sweep_mode: str,
    sweep_value: int,
    request_count: int,
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


def _post_engine_route(url: str, label: str, *, required: bool) -> None:
    """POST an empty body to a backend /engine/* route.

    ``required=True`` (start): any failure aborts. ``required=False`` (stop):
    failures log a warning — stop is best-effort so an aiperf exception isn't
    masked by a profile finalize error.
    """
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        msg = f"POST {url} ({label}) failed: {e}"
        if required:
            raise RuntimeError(msg) from e
        print(f"  WARN: {msg}", flush=True)
        return

    if status >= 400:
        msg = f"POST {url} ({label}) returned HTTP {status}: {body}"
        if required:
            raise RuntimeError(msg)
        print(f"  WARN: {msg}", flush=True)
        return

    print(f"  {label}: POST {url} -> {status}", flush=True)


def run_aiperf_single(
    model: str,
    port: int,
    sweep_mode: str,
    sweep_value: int,
    request_count: int,
    warmup_count: int,
    input_file: str,
    osl: int,
    artifact_dir: Path,
    start_profile_url: Optional[str] = None,
    stop_profile_url: Optional[str] = None,
) -> None:
    """Run a single aiperf profile invocation.

    If ``start_profile_url`` / ``stop_profile_url`` are set (wired from the
    backend's DYN_SYSTEM_PORT), wrap the aiperf run in
    POST start_profile → aiperf → POST stop_profile, so an nsys-wrapped
    backend captures only the aiperf window.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_aiperf_cmd(
        model=model,
        port=port,
        sweep_mode=sweep_mode,
        sweep_value=sweep_value,
        request_count=request_count,
        warmup_count=warmup_count,
        input_file=input_file,
        osl=osl,
        artifact_dir=artifact_dir,
    )

    print(f"  aiperf {sweep_mode}={sweep_value} -> {artifact_dir}", flush=True)

    if start_profile_url:
        _post_engine_route(start_profile_url, "start_profile", required=True)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if stop_profile_url:
            _post_engine_route(stop_profile_url, "stop_profile", required=False)

    if proc.returncode != 0:
        print(f"  aiperf FAILED (exit {proc.returncode})", flush=True)
        for stream_name, stream in [("stderr", proc.stderr), ("stdout", proc.stdout)]:
            if stream:
                for line in stream.strip().splitlines()[-15:]:
                    print(f"    [{stream_name}] {line}", flush=True)
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )

    print(f"  aiperf {sweep_mode}={sweep_value} done.", flush=True)


def run_sweep(
    model: str,
    port: int,
    sweep_mode: str,
    sweep_values: List[int],
    request_count: int,
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
            request_count=request_count,
            warmup_count=warmup_count,
            input_file=input_file,
            osl=osl,
            artifact_dir=output_dir / f"{sweep_mode}{value}",
        )

    print(f"Sweep complete. Results in {output_dir}", flush=True)
