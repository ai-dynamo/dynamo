#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark: fast-pool rollout acceleration.

Launches a frontend (direct routing) + two pools of mockers:
  - "normal" pool  (75% of workers, base speedup ratio)
  - "fast" pool    (25% of workers, higher speedup ratio)

Generates a batch of multi-turn chat requests. Each request is a multi-turn
conversation where the next turn is only sent after the previous turn returns.

"Normal" requests have fewer turns and smaller OSLs per turn.
"Long-tail" requests have more turns and larger OSLs per turn.

Normal requests are round-robined across the normal pool.
Long-tail requests are round-robined across the fast pool.

The script measures total batch completion time (makespan).

Usage:
  python benchmarks/fast_pool_bench.py \
    --model-path /data/models/Qwen3-0.6B \
    --total-workers 8 \
    --normal-speedup 10.0 \
    --fast-speedup 16.0 \
    --normal-requests 40 \
    --longtail-requests 10 \
    --normal-turns 3 \
    --longtail-turns 10 \
    --normal-osl-mean 128 \
    --normal-osl-stddev 32 \
    --longtail-osl-mean 512 \
    --longtail-osl-stddev 128 \
    --frontend-port 8321
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
    """Defines a multi-turn request."""

    request_id: int
    num_turns: int
    osl_mean: int
    osl_stddev: int
    pool: str  # "normal" or "fast"
    # Pre-sampled max_tokens for each turn (populated at construction time)
    per_turn_max_tokens: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.per_turn_max_tokens:
            self.per_turn_max_tokens = [
                max(1, int(random.gauss(self.osl_mean, self.osl_stddev)))
                for _ in range(self.num_turns)
            ]


class ProgressTracker:
    """Thread-safe progress tracker for concurrent requests."""

    def __init__(self, total_requests: int, total_turns: int):
        self.total_requests = total_requests
        self.total_turns = total_turns
        self._completed_requests = 0
        self._completed_turns = 0
        self._lock = asyncio.Lock()
        self._start = time.monotonic()

    async def turn_done(self, request_id: int, turn: int, num_turns: int):
        async with self._lock:
            self._completed_turns += 1
            elapsed = time.monotonic() - self._start
            sys.stderr.write(
                f"\r  [{elapsed:6.1f}s] Turns: {self._completed_turns}/{self.total_turns}  "
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
                f"\r  [{elapsed:6.1f}s] Turns: {self._completed_turns}/{self.total_turns}  "
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
    """Launches and tears down background processes."""

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
    """Create a DistributedRuntime instance."""
    from dynamo.runtime import DistributedRuntime

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return DistributedRuntime(loop, "etcd", request_plane)


async def wait_for_frontend(url: str, timeout: int = 120):
    """Poll the frontend /v1/models endpoint until it's ready."""
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
    """Wait until the expected number of workers are registered."""
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
# Multi-turn request runner
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
    """Execute a single multi-turn request, returning timing info."""
    url = f"{base_url}/v1/chat/completions"
    messages = []
    turn_times = []
    turn_output_tokens = []
    total_output_tokens = 0

    # Round-robin within the assigned pool
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
        messages.append(
            {
                "role": "user",
                "content": f"Turn {turn + 1} of request {spec.request_id}. "
                f"Please generate a response.",
            }
        )
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
            logger.error(
                f"Request {spec.request_id} turn {turn + 1} timed out"
            )
            break
        except Exception as e:
            logger.error(
                f"Request {spec.request_id} turn {turn + 1} error: {e}"
            )
            break

        turn_elapsed = time.monotonic() - turn_start
        turn_times.append(turn_elapsed)

        # Extract assistant response and append to conversation
        choice = data["choices"][0]
        assistant_content = choice["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_content})

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
        "per_turn_max_tokens": spec.per_turn_max_tokens,
        "turn_output_tokens": turn_output_tokens,
        "total_output_tokens": total_output_tokens,
        "total_time_s": request_elapsed,
        "turn_times_s": turn_times,
    }


async def run_batch(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    normal_worker_ids: list[int],
    fast_worker_ids: list[int],
) -> list[dict]:
    """Run the full batch of requests concurrently. Returns results."""
    rr_counters: dict[str, int] = {"normal": 0, "fast": 0}
    total_turns = sum(s.num_turns for s in specs)
    progress = ProgressTracker(len(specs), total_turns)

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

        results = await asyncio.gather(*tasks)

    progress.finish()
    return list(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_request_specs(args) -> list[RequestSpec]:
    """Build the list of RequestSpecs from CLI args."""
    specs = []
    rid = 0
    for _ in range(args.normal_requests):
        specs.append(
            RequestSpec(
                request_id=rid,
                num_turns=args.normal_turns,
                osl_mean=args.normal_osl_mean,
                osl_stddev=args.normal_osl_stddev,
                pool="normal",
            )
        )
        rid += 1
    for _ in range(args.longtail_requests):
        specs.append(
            RequestSpec(
                request_id=rid,
                num_turns=args.longtail_turns,
                osl_mean=args.longtail_osl_mean,
                osl_stddev=args.longtail_osl_stddev,
                pool="fast",
            )
        )
        rid += 1
    return specs


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def print_results(results: list[dict], batch_time: float):
    """Print a summary table and statistics."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    normal_results = [r for r in results if r["pool"] == "normal"]
    fast_results = [r for r in results if r["pool"] == "fast"]

    for label, group in [("Normal pool", normal_results), ("Fast pool", fast_results)]:
        if not group:
            continue
        times = [r["total_time_s"] for r in group]
        tokens = [r["total_output_tokens"] for r in group]
        # Collect all per-turn OSLs across requests in this group
        all_turn_tokens = [
            t for r in group for t in r.get("turn_output_tokens", [])
        ]
        all_turn_max = [
            t for r in group for t in r.get("per_turn_max_tokens", [])
        ]
        print(f"\n{label} ({len(group)} requests):")
        print(f"  Workers used:      {sorted(set(r['worker_id'] for r in group))}")
        print(f"  Avg time/request:  {sum(times) / len(times):.3f}s")
        print(f"  Max time/request:  {max(times):.3f}s")
        print(f"  Min time/request:  {min(times):.3f}s")
        print(f"  Avg total output tokens: {sum(tokens) / len(tokens):.0f}")
        if all_turn_max:
            mean_max = sum(all_turn_max) / len(all_turn_max)
            std_max = _stddev([float(t) for t in all_turn_max])
            print(f"  OSL (max_tokens) per turn: mean={mean_max:.0f} stddev={std_max:.0f}")
        if all_turn_tokens:
            mean_out = sum(all_turn_tokens) / len(all_turn_tokens)
            std_out = _stddev([float(t) for t in all_turn_tokens])
            print(f"  Actual output tokens/turn: mean={mean_out:.0f} stddev={std_out:.0f}")

    print(f"\nTotal batch makespan: {batch_time:.3f}s")

    # The key metric: the tail latency
    all_times = [r["total_time_s"] for r in results]
    all_times.sort()
    p50 = all_times[len(all_times) // 2]
    p90 = all_times[int(len(all_times) * 0.9)]
    p99 = all_times[int(len(all_times) * 0.99)]
    print(f"\nLatency percentiles (per-request):")
    print(f"  p50: {p50:.3f}s  p90: {p90:.3f}s  p99: {p99:.3f}s  max: {max(all_times):.3f}s")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast-pool rollout acceleration benchmark"
    )

    # Model
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model ID or local path (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for API requests (defaults to --model-path)",
    )

    # Worker topology
    parser.add_argument(
        "--normal-workers", type=int, default=6, help="Number of normal pool workers"
    )
    parser.add_argument(
        "--fast-workers", type=int, default=2, help="Number of fast pool workers"
    )

    # Speedup ratios
    parser.add_argument(
        "--normal-speedup",
        type=float,
        default=10.0,
        help="Speedup ratio for normal pool workers",
    )
    parser.add_argument(
        "--fast-speedup",
        type=float,
        default=16.0,
        help="Speedup ratio for fast pool workers",
    )

    # Request generation
    parser.add_argument(
        "--normal-requests",
        type=int,
        default=40,
        help="Number of normal (short) requests",
    )
    parser.add_argument(
        "--longtail-requests",
        type=int,
        default=10,
        help="Number of long-tail requests",
    )
    parser.add_argument(
        "--normal-turns", type=int, default=3, help="Turns per normal request"
    )
    parser.add_argument(
        "--longtail-turns", type=int, default=10, help="Turns per long-tail request"
    )
    parser.add_argument(
        "--normal-osl-mean",
        type=int,
        default=128,
        help="Mean max output tokens per turn for normal requests",
    )
    parser.add_argument(
        "--normal-osl-stddev",
        type=int,
        default=32,
        help="Stddev of max output tokens per turn for normal requests",
    )
    parser.add_argument(
        "--longtail-osl-mean",
        type=int,
        default=512,
        help="Mean max output tokens per turn for long-tail requests",
    )
    parser.add_argument(
        "--longtail-osl-stddev",
        type=int,
        default=128,
        help="Stddev of max output tokens per turn for long-tail requests",
    )

    # Infrastructure
    parser.add_argument(
        "--frontend-port", type=int, default=8321, help="Frontend HTTP port"
    )
    parser.add_argument(
        "--namespace", default="dynamo", help="Dynamo namespace"
    )
    parser.add_argument(
        "--request-plane", default="tcp", choices=["tcp", "nats"], help="Request plane"
    )
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Skip launching frontend/mockers (assume already running). "
        "Provide --normal-worker-ids and --fast-worker-ids.",
    )
    parser.add_argument(
        "--normal-worker-ids",
        type=str,
        default=None,
        help="Comma-separated worker IDs for normal pool (for --skip-infra)",
    )
    parser.add_argument(
        "--fast-worker-ids",
        type=str,
        default=None,
        help="Comma-separated worker IDs for fast pool (for --skip-infra)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name or args.model_path

    num_normal = args.normal_workers
    num_fast = args.fast_workers
    total_workers = num_normal + num_fast

    logger.info(
        f"Worker topology: {num_normal} normal (speedup={args.normal_speedup}) "
        f"+ {num_fast} fast (speedup={args.fast_speedup})"
    )
    logger.info(
        f"Requests: {args.normal_requests} normal ({args.normal_turns} turns, "
        f"OSL ~{args.normal_osl_mean}+/-{args.normal_osl_stddev}) + "
        f"{args.longtail_requests} long-tail ({args.longtail_turns} turns, "
        f"OSL ~{args.longtail_osl_mean}+/-{args.longtail_osl_stddev})"
    )

    base_url = f"http://localhost:{args.frontend_port}"
    pm = ProcessManager()

    try:
        if args.skip_infra:
            # Parse provided worker IDs
            if not args.normal_worker_ids:
                logger.error(
                    "--skip-infra requires at least --normal-worker-ids"
                )
                sys.exit(1)
            normal_ids = [int(x) for x in args.normal_worker_ids.split(",")]
            fast_ids = (
                [int(x) for x in args.fast_worker_ids.split(",")]
                if args.fast_worker_ids
                else []
            )
        else:
            # Launch normal pool first, wait for its workers, then launch
            # fast pool so we can reliably identify which IDs belong to which pool.
            pm.launch(
                [
                    "python3",
                    "-m",
                    "dynamo.mocker",
                    "--model-path",
                    args.model_path,
                    "--speedup-ratio",
                    str(args.normal_speedup),
                    "--num-workers",
                    str(num_normal),
                ],
                env={"DYN_NAMESPACE": args.namespace},
                label=f"normal-pool ({num_normal} workers)",
            )

            logger.info(
                f"Waiting for {num_normal} normal workers to register..."
            )
            normal_ids = asyncio.run(
                wait_for_workers(
                    args.namespace,
                    num_normal,
                    args.request_plane,
                )
            )
            logger.info(f"Normal worker IDs: {normal_ids}")

            fast_ids = []
            if num_fast > 0:
                pm.launch(
                    [
                        "python3",
                        "-m",
                        "dynamo.mocker",
                        "--model-path",
                        args.model_path,
                        "--speedup-ratio",
                        str(args.fast_speedup),
                        "--num-workers",
                        str(num_fast),
                    ],
                    env={"DYN_NAMESPACE": args.namespace},
                    label=f"fast-pool ({num_fast} workers)",
                )

                logger.info(
                    f"Waiting for {num_fast} fast workers to register..."
                )
                all_ids = asyncio.run(
                    wait_for_workers(
                        args.namespace,
                        total_workers,
                        args.request_plane,
                    )
                )
                fast_ids = sorted(set(all_ids) - set(normal_ids))
                logger.info(f"Fast worker IDs: {fast_ids}")

            # Launch frontend
            pm.launch(
                [
                    "python3",
                    "-m",
                    "dynamo.frontend",
                    "--router-mode",
                    "direct",
                    "--http-port",
                    str(args.frontend_port),
                    "--namespace",
                    args.namespace,
                ],
                env={"DYN_REQUEST_PLANE": args.request_plane},
                label="frontend",
            )

            # Wait for frontend
            logger.info("Waiting for frontend to be ready...")
            asyncio.run(wait_for_frontend(base_url))

        logger.info(f"Normal pool worker IDs: {normal_ids}")
        logger.info(f"Fast pool worker IDs:   {fast_ids}")

        # Build request specs
        specs = build_request_specs(args)
        logger.info(f"Running batch of {len(specs)} requests...")

        # Run the benchmark
        batch_start = time.monotonic()
        results = asyncio.run(
            run_batch(base_url, model_name, specs, normal_ids, fast_ids)
        )
        batch_time = time.monotonic() - batch_start

        print_results(results, batch_time)

        # Dump raw results
        output_path = "fast_pool_bench_results.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "total_workers": num_normal + num_fast,
                        "num_normal": num_normal,
                        "num_fast": num_fast,
                        "normal_speedup": args.normal_speedup,
                        "fast_speedup": args.fast_speedup,
                        "normal_requests": args.normal_requests,
                        "longtail_requests": args.longtail_requests,
                        "normal_turns": args.normal_turns,
                        "longtail_turns": args.longtail_turns,
                        "normal_osl_mean": args.normal_osl_mean,
                        "normal_osl_stddev": args.normal_osl_stddev,
                        "longtail_osl_mean": args.longtail_osl_mean,
                        "longtail_osl_stddev": args.longtail_osl_stddev,
                    },
                    "batch_makespan_s": batch_time,
                    "results": results,
                },
                f,
                indent=2,
            )
        logger.info(f"Raw results written to {output_path}")

    finally:
        if not args.skip_infra:
            logger.info("Tearing down processes...")
            pm.teardown()


if __name__ == "__main__":
    main()
