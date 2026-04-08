#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep benchmark: fast-pool rollout acceleration.

Sweeps through a range of long-tail request fractions, running both a baseline
and an improved configuration at each point, then plots speedup vs fraction.

Baseline:  All workers at normal speed. All requests round-robin across them.
Improved:  Normal workers + fast workers. Normal requests go to normal pool,
           long-tail requests go to fast pool.

The infrastructure (mockers + frontend) is launched once per configuration and
reused across all sweep points.

Usage:
  python benchmarks/fast_pool_sweep.py \
    --model-path Qwen/Qwen3-0.6B \
    --baseline-workers 8 \
    --improved-normal-workers 6 \
    --improved-fast-workers 2 \
    --normal-speedup 10.0 \
    --fast-speedup 16.0 \
    --total-requests 50 \
    --sweep-min 0.0 --sweep-max 0.5 --sweep-steps 6 \
    --output sweep_results.png
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

HEADER_WORKER_ID = "x-worker-instance-id"
HEADER_DP_RANK = "x-dp-rank"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RequestSpec:
    request_id: int
    num_turns: int
    osl_mean: int
    osl_stddev: int
    pool: str  # "normal" or "fast"
    per_turn_max_tokens: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.per_turn_max_tokens:
            self.per_turn_max_tokens = [
                max(1, int(random.gauss(self.osl_mean, self.osl_stddev)))
                for _ in range(self.num_turns)
            ]


class ProgressTracker:
    def __init__(self, total_requests: int, total_turns: int, label: str = ""):
        self.total_requests = total_requests
        self.total_turns = total_turns
        self.label = label
        self._completed_requests = 0
        self._completed_turns = 0
        self._lock = asyncio.Lock()
        self._start = time.monotonic()

    async def turn_done(self, request_id: int, turn: int, num_turns: int):
        async with self._lock:
            self._completed_turns += 1
            elapsed = time.monotonic() - self._start
            sys.stderr.write(
                f"\r  {self.label} [{elapsed:6.1f}s] "
                f"Turns: {self._completed_turns}/{self.total_turns}  "
                f"Requests: {self._completed_requests}/{self.total_requests}  "
                f"(req {request_id} turn {turn + 1}/{num_turns})"
                f"          "
            )
            sys.stderr.flush()

    async def request_done(self, request_id: int, pool: str):
        async with self._lock:
            self._completed_requests += 1
            elapsed = time.monotonic() - self._start
            sys.stderr.write(
                f"\r  {self.label} [{elapsed:6.1f}s] "
                f"Turns: {self._completed_turns}/{self.total_turns}  "
                f"Requests: {self._completed_requests}/{self.total_requests}  "
                f"(req {request_id} [{pool}] done)"
                f"          "
            )
            sys.stderr.flush()

    def finish(self):
        sys.stderr.write("\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class ProcessManager:
    def __init__(self):
        self.procs: list[subprocess.Popen] = []

    def launch(self, cmd: list[str], env: dict | None = None, label: str = ""):
        merged_env = {**os.environ, **(env or {})}
        logger.info(f"Launching [{label}]: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            env=merged_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.procs.append(proc)
        return proc

    def teardown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs.clear()


# ---------------------------------------------------------------------------
# Worker discovery
# ---------------------------------------------------------------------------


def _get_runtime(request_plane: str = "tcp"):
    from dynamo.runtime import DistributedRuntime

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return DistributedRuntime(loop, "etcd", request_plane)


async def wait_for_frontend(url: str, timeout: int = 120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("data"):
                            logger.info("Frontend is ready.")
                            return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise TimeoutError("Frontend did not become ready")


async def wait_for_workers(
    namespace: str, expected: int, request_plane: str = "tcp", timeout: int = 120
):
    runtime = _get_runtime(request_plane)
    endpoint = runtime.endpoint(f"{namespace}.backend.generate")
    client = await endpoint.client()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ids = client.instance_ids()
        if len(ids) >= expected:
            logger.info(f"All {expected} workers registered: {sorted(ids)}")
            return sorted(ids)
        await asyncio.sleep(1)
    raise TimeoutError(
        f"Only {len(client.instance_ids())}/{expected} workers registered"
    )


# ---------------------------------------------------------------------------
# Infrastructure lifecycle
# ---------------------------------------------------------------------------


def launch_infra(
    pm: ProcessManager,
    model_path: str,
    num_normal: int,
    num_fast: int,
    normal_speedup: float,
    fast_speedup: float,
    namespace: str,
    request_plane: str,
    frontend_port: int,
) -> tuple[list[int], list[int]]:
    """Launch mockers + frontend. Returns (normal_ids, fast_ids)."""

    # Normal pool
    pm.launch(
        [
            "python3", "-m", "dynamo.mocker",
            "--model-path", model_path,
            "--speedup-ratio", str(normal_speedup),
            "--num-workers", str(num_normal),
        ],
        env={"DYN_NAMESPACE": namespace},
        label=f"normal-pool ({num_normal} workers, speedup={normal_speedup})",
    )

    logger.info(f"Waiting for {num_normal} normal workers...")
    normal_ids = asyncio.run(
        wait_for_workers(namespace, num_normal, request_plane)
    )

    fast_ids = []
    if num_fast > 0:
        pm.launch(
            [
                "python3", "-m", "dynamo.mocker",
                "--model-path", model_path,
                "--speedup-ratio", str(fast_speedup),
                "--num-workers", str(num_fast),
            ],
            env={"DYN_NAMESPACE": namespace},
            label=f"fast-pool ({num_fast} workers, speedup={fast_speedup})",
        )

        total = num_normal + num_fast
        logger.info(f"Waiting for {num_fast} fast workers...")
        all_ids = asyncio.run(
            wait_for_workers(namespace, total, request_plane)
        )
        fast_ids = sorted(set(all_ids) - set(normal_ids))

    # Frontend
    pm.launch(
        [
            "python3", "-m", "dynamo.frontend",
            "--router-mode", "direct",
            "--http-port", str(frontend_port),
            "--namespace", namespace,
        ],
        env={"DYN_REQUEST_PLANE": request_plane},
        label="frontend",
    )

    logger.info("Waiting for frontend...")
    asyncio.run(wait_for_frontend(f"http://localhost:{frontend_port}"))

    logger.info(f"Normal worker IDs: {normal_ids}")
    logger.info(f"Fast worker IDs:   {fast_ids}")
    return normal_ids, fast_ids


# ---------------------------------------------------------------------------
# Request runner (reused from fast_pool_bench.py)
# ---------------------------------------------------------------------------


async def run_one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    spec: RequestSpec,
    worker_ids: list[int],
    rr_counter: dict[str, int],
    progress: ProgressTracker,
) -> dict:
    url = f"{base_url}/v1/chat/completions"
    messages = []
    turn_times = []
    turn_output_tokens = []
    total_output_tokens = 0

    pool_workers = worker_ids
    idx = rr_counter.get(spec.pool, 0)
    worker_id = pool_workers[idx % len(pool_workers)]
    rr_counter[spec.pool] = idx + 1

    headers = {
        HEADER_WORKER_ID: str(worker_id),
        HEADER_DP_RANK: "0",
    }

    request_start = time.monotonic()

    for turn in range(spec.num_turns):
        max_tokens = spec.per_turn_max_tokens[turn]
        messages.append({
            "role": "user",
            "content": f"Turn {turn + 1} of request {spec.request_id}. "
                       f"Please generate a response.",
        })
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        turn_start = time.monotonic()
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Request {spec.request_id} turn {turn + 1} failed "
                        f"(status {resp.status}): {error_text}"
                    )
                    break
                data = await resp.json()
        except asyncio.TimeoutError:
            logger.error(f"Request {spec.request_id} turn {turn + 1} timed out")
            break
        except Exception as e:
            logger.error(f"Request {spec.request_id} turn {turn + 1} error: {e}")
            break

        turn_elapsed = time.monotonic() - turn_start
        turn_times.append(turn_elapsed)

        choice = data["choices"][0]
        messages.append({"role": "assistant", "content": choice["message"]["content"]})

        usage = data.get("usage", {})
        turn_tokens = usage.get("completion_tokens", 0)
        turn_output_tokens.append(turn_tokens)
        total_output_tokens += turn_tokens

        await progress.turn_done(spec.request_id, turn, spec.num_turns)

    request_elapsed = time.monotonic() - request_start
    await progress.request_done(spec.request_id, spec.pool)

    return {
        "request_id": spec.request_id,
        "pool": spec.pool,
        "worker_id": worker_id,
        "num_turns": spec.num_turns,
        "total_output_tokens": total_output_tokens,
        "total_time_s": request_elapsed,
    }


async def run_batch(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    normal_worker_ids: list[int],
    fast_worker_ids: list[int],
    label: str = "",
) -> tuple[float, list[dict]]:
    """Run a batch. Returns (makespan_seconds, results)."""
    rr_counters: dict[str, int] = {"normal": 0, "fast": 0}
    total_turns = sum(s.num_turns for s in specs)
    progress = ProgressTracker(len(specs), total_turns, label=label)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    client_timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(
        connector=connector, timeout=client_timeout
    ) as session:
        tasks = []
        for spec in specs:
            if spec.pool == "fast" and fast_worker_ids:
                pool_workers = fast_worker_ids
            else:
                pool_workers = normal_worker_ids
            tasks.append(
                run_one_request(
                    session, base_url, model, spec, pool_workers,
                    rr_counters, progress,
                )
            )
        batch_start = time.monotonic()
        results = await asyncio.gather(*tasks)
        batch_time = time.monotonic() - batch_start

    progress.finish()
    return batch_time, list(results)


# ---------------------------------------------------------------------------
# Request spec builder
# ---------------------------------------------------------------------------


def build_specs(
    total_requests: int,
    longtail_fraction: float,
    normal_turns: int,
    longtail_turns: int,
    normal_osl_mean: int,
    normal_osl_stddev: int,
    longtail_osl_mean: int,
    longtail_osl_stddev: int,
) -> list[RequestSpec]:
    num_longtail = int(round(total_requests * longtail_fraction))
    num_normal = total_requests - num_longtail
    specs = []
    rid = 0
    for _ in range(num_normal):
        specs.append(RequestSpec(
            request_id=rid, num_turns=normal_turns,
            osl_mean=normal_osl_mean, osl_stddev=normal_osl_stddev,
            pool="normal",
        ))
        rid += 1
    for _ in range(num_longtail):
        specs.append(RequestSpec(
            request_id=rid, num_turns=longtail_turns,
            osl_mean=longtail_osl_mean, osl_stddev=longtail_osl_stddev,
            pool="fast",
        ))
        rid += 1
    return specs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(sweep_data: list[dict], output_path: str, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = [d["longtail_fraction"] for d in sweep_data]
    baseline_times = [d["baseline_makespan_s"] for d in sweep_data]
    improved_times = [d["improved_makespan_s"] for d in sweep_data]
    speedups = [
        b / i if i > 0 else float("inf")
        for b, i in zip(baseline_times, improved_times)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: makespan comparison
    ax1.plot(fractions, baseline_times, "o-", label="Baseline", color="tab:blue")
    ax1.plot(fractions, improved_times, "s-", label="Improved", color="tab:orange")
    ax1.set_xlabel("Long-tail request fraction")
    ax1.set_ylabel("Batch makespan (s)")
    ax1.set_title("Batch Makespan: Baseline vs Improved")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: speedup
    ax2.plot(fractions, speedups, "D-", color="tab:green")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Long-tail request fraction")
    ax2.set_ylabel("Speedup (baseline / improved)")
    ax2.set_title("Speedup from Fast Pool")
    ax2.grid(True, alpha=0.3)

    subtitle = (
        f"Baseline: {args.baseline_workers}w @ {args.normal_speedup}x | "
        f"Improved: {args.improved_normal_workers}n + {args.improved_fast_workers}f "
        f"@ {args.normal_speedup}x/{args.fast_speedup}x | "
        f"{args.total_requests} total requests"
    )
    fig.suptitle(subtitle, fontsize=9, y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(output_path, dpi=150)
    logger.info(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Fast-pool sweep benchmark")

    # Model
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)

    # Baseline topology
    p.add_argument("--baseline-workers", type=int, required=True,
                   help="Number of workers for the baseline run (all normal speed)")

    # Improved topology
    p.add_argument("--improved-normal-workers", type=int, required=True,
                   help="Number of normal-speed workers in the improved run")
    p.add_argument("--improved-fast-workers", type=int, required=True,
                   help="Number of fast workers in the improved run")

    # Speedup ratios
    p.add_argument("--normal-speedup", type=float, default=10.0)
    p.add_argument("--fast-speedup", type=float, default=16.0)

    # Request parameters
    p.add_argument("--total-requests", type=int, default=50,
                   help="Total number of requests per sweep point")
    p.add_argument("--normal-turns", type=int, default=3)
    p.add_argument("--longtail-turns", type=int, default=10)
    p.add_argument("--normal-osl-mean", type=int, default=128)
    p.add_argument("--normal-osl-stddev", type=int, default=32)
    p.add_argument("--longtail-osl-mean", type=int, default=512)
    p.add_argument("--longtail-osl-stddev", type=int, default=128)

    # Sweep range
    p.add_argument("--sweep-min", type=float, default=0.0,
                   help="Minimum long-tail fraction (inclusive)")
    p.add_argument("--sweep-max", type=float, default=0.5,
                   help="Maximum long-tail fraction (inclusive)")
    p.add_argument("--sweep-steps", type=int, default=6,
                   help="Number of sweep points")

    # Infrastructure
    p.add_argument("--frontend-port", type=int, default=8321)
    p.add_argument("--namespace", default="dynamo")
    p.add_argument("--request-plane", default="tcp", choices=["tcp", "nats"])

    # Output
    p.add_argument("--output", default="fast_pool_sweep.png",
                   help="Output plot path (PNG)")
    p.add_argument("--output-json", default="fast_pool_sweep.json",
                   help="Output raw data path (JSON)")

    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name or args.model_path

    # Build sweep fractions
    if args.sweep_steps < 2:
        fractions = [args.sweep_min]
    else:
        step = (args.sweep_max - args.sweep_min) / (args.sweep_steps - 1)
        fractions = [args.sweep_min + i * step for i in range(args.sweep_steps)]

    logger.info(f"Sweep fractions: {[f'{f:.2f}' for f in fractions]}")
    logger.info(
        f"Baseline: {args.baseline_workers} workers @ {args.normal_speedup}x"
    )
    logger.info(
        f"Improved: {args.improved_normal_workers} normal @ {args.normal_speedup}x "
        f"+ {args.improved_fast_workers} fast @ {args.fast_speedup}x"
    )

    sweep_data = []
    base_url = f"http://localhost:{args.frontend_port}"

    for i, frac in enumerate(fractions):
        logger.info("=" * 60)
        logger.info(f"SWEEP POINT {i + 1}/{len(fractions)}: frac={frac:.2f}")
        logger.info("=" * 60)

        specs = build_specs(
            total_requests=args.total_requests,
            longtail_fraction=frac,
            normal_turns=args.normal_turns,
            longtail_turns=args.longtail_turns,
            normal_osl_mean=args.normal_osl_mean,
            normal_osl_stddev=args.normal_osl_stddev,
            longtail_osl_mean=args.longtail_osl_mean,
            longtail_osl_stddev=args.longtail_osl_stddev,
        )
        num_lt = sum(1 for s in specs if s.pool == "fast")
        logger.info(
            f"  {len(specs)} requests "
            f"({len(specs) - num_lt} normal, {num_lt} long-tail)"
        )

        # --- Baseline run ---
        logger.info("  Launching baseline...")
        bl_pm = ProcessManager()
        try:
            bl_normal_ids, _ = launch_infra(
                pm=bl_pm,
                model_path=args.model_path,
                num_normal=args.baseline_workers,
                num_fast=0,
                normal_speedup=args.normal_speedup,
                fast_speedup=args.normal_speedup,
                namespace=args.namespace,
                request_plane=args.request_plane,
                frontend_port=args.frontend_port,
            )
            bl_makespan, _ = asyncio.run(
                run_batch(
                    base_url, model_name, specs,
                    bl_normal_ids, [],
                    label=f"[BL {frac:.0%}]",
                )
            )
            logger.info(f"  Baseline makespan: {bl_makespan:.3f}s")
        finally:
            bl_pm.teardown()
            time.sleep(2)

        # --- Improved run ---
        logger.info("  Launching improved...")
        imp_pm = ProcessManager()
        try:
            imp_normal_ids, imp_fast_ids = launch_infra(
                pm=imp_pm,
                model_path=args.model_path,
                num_normal=args.improved_normal_workers,
                num_fast=args.improved_fast_workers,
                normal_speedup=args.normal_speedup,
                fast_speedup=args.fast_speedup,
                namespace=args.namespace,
                request_plane=args.request_plane,
                frontend_port=args.frontend_port,
            )
            imp_makespan, _ = asyncio.run(
                run_batch(
                    base_url, model_name, specs,
                    imp_normal_ids, imp_fast_ids,
                    label=f"[IMP {frac:.0%}]",
                )
            )
            logger.info(f"  Improved makespan: {imp_makespan:.3f}s")
        finally:
            imp_pm.teardown()
            time.sleep(2)

        speedup = bl_makespan / imp_makespan if imp_makespan > 0 else float("inf")
        logger.info(f"  Speedup: {speedup:.2f}x")

        sweep_data.append({
            "longtail_fraction": frac,
            "baseline_makespan_s": bl_makespan,
            "improved_makespan_s": imp_makespan,
            "speedup": speedup,
        })

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Fraction':>10} {'Baseline (s)':>14} {'Improved (s)':>14} {'Speedup':>10}")
    print("-" * 70)
    for d in sweep_data:
        print(
            f"{d['longtail_fraction']:>10.2f} "
            f"{d['baseline_makespan_s']:>14.3f} "
            f"{d['improved_makespan_s']:>14.3f} "
            f"{d['speedup']:>10.2f}x"
        )
    print("=" * 70)

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump({
            "config": {
                "baseline_workers": args.baseline_workers,
                "improved_normal_workers": args.improved_normal_workers,
                "improved_fast_workers": args.improved_fast_workers,
                "normal_speedup": args.normal_speedup,
                "fast_speedup": args.fast_speedup,
                "total_requests": args.total_requests,
                "normal_turns": args.normal_turns,
                "longtail_turns": args.longtail_turns,
                "normal_osl_mean": args.normal_osl_mean,
                "normal_osl_stddev": args.normal_osl_stddev,
                "longtail_osl_mean": args.longtail_osl_mean,
                "longtail_osl_stddev": args.longtail_osl_stddev,
                "sweep_min": args.sweep_min,
                "sweep_max": args.sweep_max,
                "sweep_steps": args.sweep_steps,
            },
            "sweep_results": sweep_data,
        }, f, indent=2)
    logger.info(f"Raw data saved to {args.output_json}")

    # Plot
    plot_results(sweep_data, args.output, args)


if __name__ == "__main__":
    main()
