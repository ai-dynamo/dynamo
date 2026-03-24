#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Frontend performance sweep runner.

Standalone Python script that orchestrates performance sweeps by delegating
each run to run_perf.sh.  Combines the sweep grid logic of sweep.sh with
the saturation analysis of tasks/sweep.py, and the Prometheus/report
integration of the analysis scripts.

Sweep dimensions (all configurable):
  - tokenizers (hf, fastokens)
  - concurrency levels
  - ISL values
  - worker counts

Backends:
  - mocker (default): fast synthetic backend, no real inference
  - vllm: real vLLM inference server (produces TTFT/ITL metrics)

Each (tokenizer, concurrency, ISL) point is a separate run_perf.sh invocation.
Results are collected into CSV + summary.md + per-run reports.

Usage:
    # Smoke test (2 runs)
    python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32 --isl 512 \\
        --benchmark-duration 30 --speedup-ratio 0

    # Full sweep with mocker
    python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024,2048

    # vLLM backend (real inference)
    python3 sweep_runner.py --backend vllm --tokenizers hf --concurrency 128 --isl 1024

    # Transport saturation sweep (tasks/sweep.py style)
    python3 sweep_runner.py --tokenizers hf --concurrency 4096 \\
        --num-requests 16384,32768 --workers 1,2,4,8 --speedup-ratio 0

    # Dry run
    python3 sweep_runner.py --dry-run --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ANALYSIS_DIR = SCRIPT_DIR / "analysis"

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OSL = 256
DEFAULT_SPEEDUP = 1.0
DEFAULT_BENCHMARK_DURATION = 60
DEFAULT_MAX_CONSECUTIVE_FAILS = 2
DEFAULT_COOLDOWN = 3

TOKENIZER_MAP = {
    "fast": "fastokens",
    "fastokens": "fastokens",
    "hf": "default",
    "default": "default",
}


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a single sweep point."""

    backend: str  # "mocker" or "vllm"
    tokenizer: str  # "hf" or "fastokens"
    concurrency: int
    isl: int
    osl: int
    workers: int
    num_models: int
    aiperf_targets: str  # "first" or "all"
    speedup_ratio: float
    model: str
    benchmark_duration: Optional[int]
    num_requests: Optional[int]
    request_rate: Optional[int]

    @property
    def run_id(self) -> str:
        base = f"{self.tokenizer}_c{self.concurrency}_isl{self.isl}_w{self.workers}"
        if self.num_models > 1:
            base += f"_m{self.num_models}"
        if self.request_rate:
            base += f"_rps{self.request_rate}"
        return base


@dataclass
class RunResult:
    """Result from a single sweep point."""

    config: RunConfig
    status: str = "pending"  # ok, fail, skipped
    req_per_sec: float = 0.0
    output_tok_per_sec: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p99_ms: float = 0.0
    duration_sec: float = 0.0
    run_dir: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _kill_port(port: int):
    """Kill any process holding a port (SIGTERM first, then SIGKILL)."""
    subprocess.run(
        f"fuser -k -TERM {port}/tcp", shell=True, capture_output=True, timeout=5
    )
    time.sleep(2)
    subprocess.run(
        f"fuser -k -KILL {port}/tcp", shell=True, capture_output=True, timeout=5
    )


def _port_free(port: int) -> bool:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex(("127.0.0.1", port)) != 0
    finally:
        s.close()


def _wait_port_free(port: int, timeout: int = 30):
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


def _parse_aiperf_json(json_path: Path) -> dict:
    """Parse aiperf profile_export_aiperf.json."""
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text())
        result = {}
        # Request throughput
        rt = data.get("request_throughput", {})
        result["req_per_sec"] = rt.get("avg", 0)
        # Output token throughput
        ot = data.get("output_token_throughput", {})
        result["output_tok_per_sec"] = ot.get("avg", 0)
        # TTFT (aiperf exports in ms already)
        ttft = data.get("time_to_first_token", data.get("ttft", {}))
        if isinstance(ttft, dict):
            result["ttft_p50_ms"] = ttft.get("p50", 0) or 0
            result["ttft_p99_ms"] = ttft.get("p99", 0) or 0
        # ITL
        itl = data.get("inter_token_latency", data.get("itl", {}))
        if isinstance(itl, dict):
            result["itl_p50_ms"] = itl.get("p50", 0) or 0
            result["itl_p99_ms"] = itl.get("p99", 0) or 0
        # Duration (can be dict with .avg or raw float)
        bd = data.get("benchmark_duration", 0)
        result["duration_sec"] = bd.get("avg", 0) if isinstance(bd, dict) else (bd or 0)
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _run_single(
    cfg: RunConfig,
    run_dir: Path,
    passthrough_args: list[str],
) -> RunResult:
    """Execute a single run_perf.sh invocation."""
    result = RunResult(config=cfg, run_dir=str(run_dir))

    cmd = [
        str(SCRIPT_DIR / "run_perf.sh"),
        "--model",
        cfg.model,
        "--isl",
        str(cfg.isl),
        "--osl",
        str(cfg.osl),
        "--concurrency",
        str(cfg.concurrency),
        "--workers",
        str(cfg.workers),
        "--speedup-ratio",
        str(cfg.speedup_ratio),
        "--num-models",
        str(cfg.num_models),
        "--aiperf-targets",
        cfg.aiperf_targets,
        "--output-dir",
        str(run_dir),
    ]
    if cfg.benchmark_duration:
        cmd.extend(["--benchmark-duration", str(cfg.benchmark_duration)])
    if cfg.num_requests:
        cmd.extend(["--num-requests", str(cfg.num_requests)])
    if cfg.request_rate:
        cmd.extend(["--request-rate", str(cfg.request_rate)])
    if cfg.tokenizer in ("fast", "fastokens"):
        cmd.append("--fast-tokens")
    # TODO: when run_perf.sh gains --backend vllm support, pass it here
    if cfg.backend == "vllm":
        print(
            "    WARNING: vllm backend not yet supported by run_perf.sh; using mocker"
        )

    cmd.extend(passthrough_args)

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
            # Print last few lines of output for debugging
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
            pass  # already exited
    except Exception as e:
        result.status = "fail"
        print(f"    ERROR: {e}")

    # Parse aiperf results -- check both flat and multi-model layouts
    aiperf_json = run_dir / "aiperf" / "profile_export_aiperf.json"
    if not aiperf_json.exists():
        # Multi-model: results are in aiperf/<model-name>/
        for candidate in sorted(
            (run_dir / "aiperf").glob("*/profile_export_aiperf.json")
        ):
            aiperf_json = candidate
            break  # Use the first model's results for the summary row
    metrics = _parse_aiperf_json(aiperf_json)
    if metrics:
        result.req_per_sec = metrics.get("req_per_sec", 0)
        result.output_tok_per_sec = metrics.get("output_tok_per_sec", 0)
        result.ttft_p50_ms = metrics.get("ttft_p50_ms", 0)
        result.ttft_p99_ms = metrics.get("ttft_p99_ms", 0)
        result.itl_p50_ms = metrics.get("itl_p50_ms", 0)
        result.itl_p99_ms = metrics.get("itl_p99_ms", 0)
        result.duration_sec = metrics.get("duration_sec", 0)

    return result


def _generate_report(run_dir: Path):
    """Run create_report.py on a single run directory."""
    try:
        sys.path.insert(0, str(ANALYSIS_DIR))
        from create_report import run_analysis

        report = run_analysis(run_dir)
        (run_dir / "report.md").write_text(report)
    except Exception as e:
        print(f"    Report generation failed: {e}")


# ── Output ───────────────────────────────────────────────────────────────────


def _write_csv(results: list[RunResult], csv_path: Path):
    """Write incremental CSV (called after each run)."""
    fieldnames = [
        "run_id",
        "backend",
        "tokenizer",
        "concurrency",
        "isl",
        "osl",
        "workers",
        "speedup_ratio",
        "status",
        "req_per_sec",
        "output_tok_per_sec",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "duration_sec",
        "run_dir",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                "run_id": r.config.run_id,
                "backend": r.config.backend,
                "tokenizer": r.config.tokenizer,
                "concurrency": r.config.concurrency,
                "isl": r.config.isl,
                "osl": r.config.osl,
                "workers": r.config.workers,
                "speedup_ratio": r.config.speedup_ratio,
                "status": r.status,
                "req_per_sec": f"{r.req_per_sec:.2f}" if r.req_per_sec else "",
                "output_tok_per_sec": f"{r.output_tok_per_sec:.1f}"
                if r.output_tok_per_sec
                else "",
                "ttft_p50_ms": f"{r.ttft_p50_ms:.1f}" if r.ttft_p50_ms else "",
                "ttft_p99_ms": f"{r.ttft_p99_ms:.1f}" if r.ttft_p99_ms else "",
                "itl_p50_ms": f"{r.itl_p50_ms:.1f}" if r.itl_p50_ms else "",
                "itl_p99_ms": f"{r.itl_p99_ms:.1f}" if r.itl_p99_ms else "",
                "duration_sec": f"{r.duration_sec:.1f}" if r.duration_sec else "",
                "run_dir": r.run_dir,
            }
            writer.writerow(row)


def _write_summary(results: list[RunResult], summary_path: Path):
    """Write markdown summary table."""
    lines = ["# Sweep Summary\n"]
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(
        "| Run ID | Req/s | Tok/s | TTFT p50 | TTFT p99 | ITL p50 | Duration | Status |"
    )
    lines.append(
        "|--------|------:|------:|---------:|---------:|--------:|---------:|--------|"
    )

    for r in results:
        rps = f"{r.req_per_sec:.1f}" if r.req_per_sec else "-"
        tps = f"{r.output_tok_per_sec:.0f}" if r.output_tok_per_sec else "-"
        tp50 = f"{r.ttft_p50_ms:.1f}ms" if r.ttft_p50_ms else "-"
        tp99 = f"{r.ttft_p99_ms:.1f}ms" if r.ttft_p99_ms else "-"
        ip50 = f"{r.itl_p50_ms:.1f}ms" if r.itl_p50_ms else "-"
        dur = f"{r.duration_sec:.0f}s" if r.duration_sec else "-"
        lines.append(
            f"| {r.config.run_id} | {rps} | {tps} | {tp50} | {tp99} | {ip50} | {dur} | {r.status} |"
        )

    lines.append("")
    ok = sum(1 for r in results if r.status == "ok")
    fail = sum(1 for r in results if r.status == "fail")
    skip = sum(1 for r in results if r.status == "skipped")
    lines.append(
        f"**Totals:** {ok} passed, {fail} failed, {skip} skipped out of {len(results)}"
    )

    summary_path.write_text("\n".join(lines) + "\n")


def _print_results_table(results: list[RunResult]):
    """Print a compact results table to stdout."""
    print(f"\n{'='*90}")
    print(
        f"  {'Run ID':<30} {'Req/s':>8} {'Tok/s':>8} {'TTFT p50':>10} {'TTFT p99':>10} {'Status':>8}"
    )
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        rps = f"{r.req_per_sec:.1f}" if r.req_per_sec else "N/A"
        tps = f"{r.output_tok_per_sec:.0f}" if r.output_tok_per_sec else "N/A"
        tp50 = f"{r.ttft_p50_ms:.1f}ms" if r.ttft_p50_ms else "N/A"
        tp99 = f"{r.ttft_p99_ms:.1f}ms" if r.ttft_p99_ms else "N/A"
        print(
            f"  {r.config.run_id:<30} {rps:>8} {tps:>8} {tp50:>10} {tp99:>10} {r.status:>8}"
        )
    print(f"{'='*90}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Frontend performance sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Smoke test
  python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32 --isl 512 \\
      --benchmark-duration 30 --speedup-ratio 0

  # Full tokenizer comparison
  python3 sweep_runner.py --tokenizers hf,fastokens --concurrency 32,64 --isl 512,1024,2048

  # vLLM backend (real inference)
  python3 sweep_runner.py --backend vllm --tokenizers hf --concurrency 128 --isl 1024

  # Transport saturation (high concurrency, vary workers)
  python3 sweep_runner.py --tokenizers hf --concurrency 4096 \\
      --num-requests 16384,32768 --workers 1,2,4,8 --speedup-ratio 0

  # With profilers (needs sudo for BPF)
  sudo -E python3 sweep_runner.py --tokenizers hf --concurrency 64 --isl 1024 \\
      -- --with-nsys --with-perf --with-bpf
""",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model path")
    parser.add_argument(
        "--backend",
        choices=["mocker", "vllm"],
        default="mocker",
        help="Engine backend: mocker (synthetic) or vllm (real inference)",
    )
    parser.add_argument(
        "--tokenizers",
        default="hf,fastokens",
        help="Comma-separated tokenizer backends (hf, fastokens)",
    )
    parser.add_argument(
        "--concurrency", default="50,100,200", help="Comma-separated concurrency levels"
    )
    parser.add_argument(
        "--isl", default="512,1024,2048", help="Comma-separated ISL values"
    )
    parser.add_argument(
        "--osl", type=int, default=DEFAULT_OSL, help="Output sequence length"
    )
    parser.add_argument(
        "--workers", default="2", help="Comma-separated worker counts per model"
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        help="Number of model instances (each gets --workers workers, named model-1, model-2, ...)",
    )
    parser.add_argument(
        "--aiperf-targets",
        choices=["first", "all"],
        default="first",
        help="'first': aiperf targets model-1 only (default). 'all': run aiperf for each model.",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=DEFAULT_SPEEDUP,
        help="Mocker speedup (0=infinite)",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=int,
        default=DEFAULT_BENCHMARK_DURATION,
        help="aiperf duration (seconds)",
    )
    parser.add_argument(
        "--num-requests",
        default=None,
        help="Comma-separated request counts (overrides --benchmark-duration)",
    )
    parser.add_argument(
        "--rps",
        default=None,
        help="Comma-separated target request rates (req/s). Sweep dimension when multiple values given.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto timestamped)",
    )
    parser.add_argument(
        "--max-consecutive-fails", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILS
    )
    parser.add_argument(
        "--cooldown", type=int, default=DEFAULT_COOLDOWN, help="Seconds between runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without executing"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip per-run report generation"
    )
    parser.add_argument(
        "passthrough", nargs="*", help="Extra args passed to run_perf.sh (after --)"
    )

    args = parser.parse_args()

    # Parse lists
    tokenizers = [t.strip() for t in args.tokenizers.split(",")]
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    isls = [int(i) for i in args.isl.split(",")]
    worker_counts = [int(w) for w in args.workers.split(",")]
    num_requests_list = (
        [int(n) for n in args.num_requests.split(",")] if args.num_requests else [None]
    )
    rps_list = [int(r) for r in args.rps.split(",")] if args.rps else [None]

    # Build sweep grid
    configs: list[RunConfig] = []
    for tokenizer in tokenizers:
        for workers in worker_counts:
            for concurrency in concurrencies:
                for isl in isls:
                    for nr in num_requests_list:
                        for rps in rps_list:
                            configs.append(
                                RunConfig(
                                    backend=args.backend,
                                    tokenizer=tokenizer,
                                    concurrency=concurrency,
                                    isl=isl,
                                    osl=args.osl,
                                    workers=workers,
                                    num_models=args.num_models,
                                    aiperf_targets=args.aiperf_targets,
                                    speedup_ratio=args.speedup_ratio,
                                    model=args.model,
                                    benchmark_duration=args.benchmark_duration
                                    if nr is None
                                    else None,
                                    num_requests=nr,
                                    request_rate=rps,
                                )
                            )

    # Output directory
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "artifacts" / f"sweep_{ts}"

    total = len(configs)
    print(f"Sweep plan: {total} runs")
    print(f"  Model:          {args.model}")
    print(f"  Backend:        {args.backend}")
    print(f"  Tokenizers:     {tokenizers}")
    print(f"  Concurrencies:  {concurrencies}")
    print(f"  ISLs:           {isls}")
    print(f"  Workers/model:  {worker_counts}")
    print(f"  Models:         {args.num_models}")
    print(f"  Benchmark dur:  {args.benchmark_duration}s")
    if args.num_requests:
        print(f"  Num requests:   {[int(n) for n in args.num_requests.split(',')]}")
    if args.rps:
        print(f"  Request rates:  {[int(r) for r in args.rps.split(',')]} req/s")
    print(f"  Output:         {output_root}")
    print()

    if args.dry_run:
        for i, cfg in enumerate(configs, 1):
            print(f"  [{i}/{total}] {cfg.run_id}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"
    summary_path = output_root / "summary.md"

    # Passthrough args for run_perf.sh (e.g., --skip-bpf --skip-nsys)
    passthrough = args.passthrough or []

    results: list[RunResult] = []
    consecutive_fails: dict[tuple, int] = {}  # (backend, concurrency, workers) -> count

    try:
        for i, cfg in enumerate(configs, 1):
            key = (cfg.backend, cfg.concurrency, cfg.workers)
            run_dir = output_root / cfg.run_id

            # Skip after consecutive failures
            if consecutive_fails.get(key, 0) >= args.max_consecutive_fails:
                result = RunResult(config=cfg, status="skipped", run_dir=str(run_dir))
                results.append(result)
                print(
                    f"\n  [{i}/{total}] SKIPPED {cfg.run_id} ({args.max_consecutive_fails} consecutive failures)"
                )
                continue

            print(f"\n{'='*60}")
            print(f"  [{i}/{total}] {cfg.run_id}")
            print(f"{'='*60}")

            # Wait for port from previous run
            _wait_port_free(8000)

            # Run
            result = _run_single(cfg, run_dir, passthrough)
            results.append(result)

            # Update consecutive failure tracking
            if result.status == "ok":
                consecutive_fails[key] = 0
                rps = f"{result.req_per_sec:.1f}" if result.req_per_sec else "N/A"
                tp50 = f"{result.ttft_p50_ms:.1f}ms" if result.ttft_p50_ms else "N/A"
                print(f"    OK: {rps} req/s, TTFT p50={tp50}")
            else:
                consecutive_fails[key] = consecutive_fails.get(key, 0) + 1
                print(
                    f"    FAIL (consecutive: {consecutive_fails[key]}/{args.max_consecutive_fails})"
                )

            # Generate per-run report
            if not args.no_report and result.status == "ok":
                _generate_report(run_dir)

            # Write incremental CSV + summary
            _write_csv(results, csv_path)
            _write_summary(results, summary_path)

            # Cooldown
            if i < total:
                time.sleep(args.cooldown)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")
    finally:
        _write_csv(results, csv_path)
        _write_summary(results, summary_path)

    # Final output
    _print_results_table(results)
    print(f"\nResults:  {csv_path}")
    print(f"Summary:  {summary_path}")
    print(f"Per-run:  {output_root}/<run_id>/report.md")


if __name__ == "__main__":
    main()
