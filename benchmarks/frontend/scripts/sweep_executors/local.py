# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LocalExecutor -- wraps run_perf.sh for local sweep execution.

This executor delegates each run to run_perf.sh, which handles service
lifecycle (mocker + frontend), observability captures, and aiperf load.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from sweep_core.models import DeployDimension, RunResult, RunSpec, SweepConfig

SCRIPT_DIR = Path(__file__).resolve().parent.parent


class LocalExecutor:
    """Executor that delegates runs to run_perf.sh."""

    def __init__(self) -> None:
        self._config: Optional[SweepConfig] = None
        self._frontend_port: int = 8000

    def prepare(self, config: SweepConfig) -> None:
        """Store config for use during runs."""
        self._config = config
        self._frontend_port = 8000  # local mode always uses 8000

    def apply_deploy(
        self,
        deploy: DeployDimension,
        prev: Optional[DeployDimension],
    ) -> None:
        """In local mode, run_perf.sh handles its own service lifecycle.

        We just wait for the port to be free from the previous run.
        """
        _wait_port_free(self._frontend_port)

    def execute_run(self, run_spec: RunSpec, run_dir: Path) -> RunResult:
        """Execute a single run via run_perf.sh."""
        if self._config is None:
            raise RuntimeError("prepare() must be called before execute_run()")
        config = self._config
        deploy = run_spec.deploy
        aiperf = run_spec.aiperf

        result = RunResult(run_spec=run_spec, run_dir=str(run_dir))

        cmd = [
            str(SCRIPT_DIR / "run_perf.sh"),
            "--model",
            config.model,
            "--isl",
            str(aiperf.isl),
            "--osl",
            str(aiperf.osl),
            "--concurrency",
            str(aiperf.concurrency),
            "--workers",
            str(deploy.workers),
            "--speedup-ratio",
            str(config.speedup_ratio),
            "--num-models",
            str(deploy.num_models),
            "--aiperf-targets",
            config.aiperf_targets,
            "--output-dir",
            str(run_dir),
        ]

        if aiperf.benchmark_duration:
            cmd.extend(["--benchmark-duration", str(aiperf.benchmark_duration)])
        if aiperf.num_requests:
            cmd.extend(["--num-requests", str(aiperf.num_requests)])
        if aiperf.request_rate:
            cmd.extend(["--request-rate", str(aiperf.request_rate)])
        if deploy.tokenizer in ("fast", "fastokens"):
            cmd.append("--fast-tokens")

        # TODO: when run_perf.sh gains --backend vllm support, pass it here
        if deploy.backend == "vllm":
            print(
                "    WARNING: vllm backend not yet supported by run_perf.sh; using mocker"
            )

        # Passthrough args (e.g., --skip-bpf --skip-nsys)
        cmd.extend(config.passthrough_args)

        print(f"    cmd: {' '.join(cmd[:6])}...")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stdout, _ = proc.communicate(timeout=600)

            if proc.returncode == 0:
                result.status = "ok"
            else:
                result.status = "fail"
                print(f"    run_perf.sh failed (rc={proc.returncode})")
                lines = (stdout or "").strip().split("\n")
                for line in lines[-5:]:
                    print(f"      {line}")

        except subprocess.TimeoutExpired:
            result.status = "fail"
            print("    TIMEOUT after 600s")
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(2)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except Exception as e:
            result.status = "fail"
            print(f"    ERROR: {e}")

        # Parse aiperf results
        _parse_aiperf_into_result(result, run_dir)

        return result

    def cleanup(self) -> None:
        """No persistent state to clean up in local mode."""
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_aiperf_json(json_path: Path) -> dict:
    """Parse aiperf profile_export_aiperf.json."""
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text())
        result = {}
        rt = data.get("request_throughput", {})
        result["req_per_sec"] = rt.get("avg", 0)
        ot = data.get("output_token_throughput", {})
        result["output_tok_per_sec"] = ot.get("avg", 0)
        ttft = data.get("time_to_first_token", data.get("ttft", {}))
        if isinstance(ttft, dict):
            result["ttft_p50_ms"] = ttft.get("p50", 0) or 0
            result["ttft_p99_ms"] = ttft.get("p99", 0) or 0
        itl = data.get("inter_token_latency", data.get("itl", {}))
        if isinstance(itl, dict):
            result["itl_p50_ms"] = itl.get("p50", 0) or 0
            result["itl_p99_ms"] = itl.get("p99", 0) or 0
        bd = data.get("benchmark_duration", 0)
        result["duration_sec"] = bd.get("avg", 0) if isinstance(bd, dict) else (bd or 0)
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _parse_aiperf_into_result(result: RunResult, run_dir: Path) -> None:
    """Parse aiperf results from the run directory into the RunResult."""
    aiperf_json = run_dir / "aiperf" / "profile_export_aiperf.json"
    if not aiperf_json.exists():
        # Multi-model: results are in aiperf/<model-name>/
        for candidate in sorted(
            (run_dir / "aiperf").glob("*/profile_export_aiperf.json")
        ):
            aiperf_json = candidate
            break
    metrics = _parse_aiperf_json(aiperf_json)
    if metrics:
        result.req_per_sec = metrics.get("req_per_sec", 0)
        result.output_tok_per_sec = metrics.get("output_tok_per_sec", 0)
        result.ttft_p50_ms = metrics.get("ttft_p50_ms", 0)
        result.ttft_p99_ms = metrics.get("ttft_p99_ms", 0)
        result.itl_p50_ms = metrics.get("itl_p50_ms", 0)
        result.itl_p99_ms = metrics.get("itl_p99_ms", 0)
        result.duration_sec = metrics.get("duration_sec", 0)


def _port_free(port: int) -> bool:
    """Check if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _kill_port(port: int) -> None:
    """Kill any process holding a port."""
    subprocess.run(
        f"fuser -k -TERM {port}/tcp", shell=True, capture_output=True, timeout=5
    )
    time.sleep(2)
    subprocess.run(
        f"fuser -k -KILL {port}/tcp", shell=True, capture_output=True, timeout=5
    )


def _wait_port_free(port: int, timeout: int = 30) -> None:
    """Wait for a port to become free."""
    for i in range(timeout):
        if _port_free(port):
            return
        if i == 0:
            print(f"  Waiting for port {port} to free...")
        time.sleep(1)
    print(f"  Forcing port {port} release...")
    _kill_port(port)
    time.sleep(2)
