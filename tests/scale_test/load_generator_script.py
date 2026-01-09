# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone load generator script for running inside Kubernetes Jobs.

This script is designed to be mounted into a Kubernetes pod and executed.
Configuration is passed via environment variables:
    - LOAD_GEN_URLS: JSON array of frontend URLs
    - LOAD_GEN_MODEL: Model name for requests
    - LOAD_GEN_DURATION: Test duration in seconds
    - LOAD_GEN_QPS_PER_POD: QPS target for this pod
    - LOAD_GEN_MAX_TOKENS: Maximum tokens per request
    - LOAD_GEN_NUM_PROCESSES: Number of worker processes per pod
    - LOAD_GEN_TOTAL_PODS: Total number of pods in the job
"""

import asyncio
import json
import multiprocessing
import os
import sys
import time

from openai import AsyncOpenAI


def get_config() -> dict:
    """Read configuration from environment variables."""
    urls_json = os.environ.get("LOAD_GEN_URLS", "[]")
    try:
        urls = json.loads(urls_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse LOAD_GEN_URLS: {e}", file=sys.stderr)
        sys.exit(1)

    return {
        "urls": urls,
        "model": os.environ.get("LOAD_GEN_MODEL", ""),
        "duration": int(os.environ.get("LOAD_GEN_DURATION", "60")),
        "qps_per_pod": float(os.environ.get("LOAD_GEN_QPS_PER_POD", "1.0")),
        "max_tokens": int(os.environ.get("LOAD_GEN_MAX_TOKENS", "30")),
        "num_processes": int(os.environ.get("LOAD_GEN_NUM_PROCESSES", "1")),
        "total_pods": int(os.environ.get("LOAD_GEN_TOTAL_PODS", "1")),
    }


async def send_request(client: AsyncOpenAI, model: str, max_tokens: int) -> tuple:
    """Send a single chat completion request."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, how are you today?"}],
            max_tokens=max_tokens,
            stream=False,
        )
        return True, response.usage.completion_tokens if response.usage else 0
    except Exception as e:
        print(f"[Process {os.getpid()}] Request failed: {e}", file=sys.stderr)
        return False, 0


async def run_load_test_worker(
    process_id: int,
    qps: float,
    duration: int,
    urls: list,
    model: str,
    max_tokens: int,
) -> dict:
    """Run load test for a single worker process."""
    print(f"[Process {process_id}] Starting load test...")
    print(f"[Process {process_id}] Duration: {duration}s, QPS: {qps:.2f}")
    print(f"[Process {process_id}] Targets: {len(urls)} frontends")

    # Create clients for each URL
    clients = {}
    for url in urls:
        clients[url] = AsyncOpenAI(
            base_url=f"{url}/v1",
            api_key="not-needed",
            timeout=duration + 30,
        )

    # Track stats per URL
    stats = {url: {"success": 0, "failed": 0, "latencies": [], "tokens": 0} for url in urls}

    interval = 1.0 / qps if qps > 0 else 1.0
    start_time = time.time()
    request_count = 0
    tasks = []

    async def send_and_record(url: str, req_num: int) -> None:
        client = clients[url]
        req_start = time.time()
        success, tokens = await send_request(client, model, max_tokens)
        latency = time.time() - req_start

        if success:
            stats[url]["success"] += 1
            stats[url]["latencies"].append(latency)
            stats[url]["tokens"] += tokens
        else:
            stats[url]["failed"] += 1

    # Generate load for the specified duration
    while time.time() - start_time < duration:
        url = urls[request_count % len(urls)]
        task = asyncio.create_task(send_and_record(url, request_count))
        tasks.append(task)
        request_count += 1

        # Schedule next request using absolute time to prevent drift
        next_request_time = start_time + (request_count * interval)
        sleep_duration = next_request_time - time.time()
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

    # Wait for all pending requests to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    actual_duration = time.time() - start_time

    return {
        "process_id": process_id,
        "stats": stats,
        "actual_duration": actual_duration,
    }


def worker_process(
    process_id: int,
    qps: float,
    duration: int,
    urls: list,
    model: str,
    max_tokens: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Entry point for worker processes in multiprocessing mode."""
    try:
        result = asyncio.run(
            run_load_test_worker(process_id, qps, duration, urls, model, max_tokens)
        )
        result_queue.put(result)
    except Exception as e:
        print(f"[Process {process_id}] Error: {e}", file=sys.stderr)
        result_queue.put({"error": str(e), "process_id": process_id})


def aggregate_results(results: list) -> tuple:
    """Aggregate results from multiple worker processes."""
    if not results:
        return None, 0

    # Collect all URLs from results
    all_urls = set()
    for r in results:
        if "stats" in r:
            all_urls.update(r["stats"].keys())

    agg_stats = {url: {"success": 0, "failed": 0, "latencies": [], "tokens": 0} for url in all_urls}
    total_duration = 0

    for result in results:
        if "error" in result:
            print(f"[Process {result.get('process_id')}] Failed: {result['error']}")
            continue

        total_duration = max(total_duration, result["actual_duration"])

        for url, s in result["stats"].items():
            agg_stats[url]["success"] += s["success"]
            agg_stats[url]["failed"] += s["failed"]
            agg_stats[url]["latencies"].extend(s["latencies"])
            agg_stats[url]["tokens"] += s["tokens"]

    return agg_stats, total_duration


def print_results(agg_stats: dict, actual_duration: float) -> int:
    """Print load test results and return exit code."""
    print("\n" + "=" * 70)
    print("LOAD GENERATION RESULTS (This Pod)")
    print("=" * 70)

    total_requests = 0
    total_errors = 0
    all_latencies = []

    for url, s in sorted(agg_stats.items()):
        total = s["success"] + s["failed"]
        total_requests += total
        total_errors += s["failed"]
        all_latencies.extend(s["latencies"])

        if total > 0:
            avg_lat = sum(s["latencies"]) / len(s["latencies"]) * 1000 if s["latencies"] else 0
            p50_lat = sorted(s["latencies"])[len(s["latencies"]) // 2] * 1000 if s["latencies"] else 0
            p99_idx = int(len(s["latencies"]) * 0.99)
            p99_lat = sorted(s["latencies"])[p99_idx] * 1000 if s["latencies"] else 0

            print(f"\nFrontend {url}:")
            print(f"  Requests: {total}")
            print(f"  Successful: {s['success']}")
            print(f"  Errors: {s['failed']} ({s['failed'] / total * 100:.1f}%)")
            print(f"  Avg latency: {avg_lat:.1f}ms")
            print(f"  P50 latency: {p50_lat:.1f}ms")
            print(f"  P99 latency: {p99_lat:.1f}ms")
            print(f"  Total tokens: {s['tokens']}")

    print("\n" + "-" * 70)
    print(f"Total requests (this pod): {total_requests}")
    print(f"Total errors: {total_errors}")
    print(f"Actual duration: {actual_duration:.1f}s")
    if actual_duration > 0:
        print(f"Actual QPS (this pod): {total_requests / actual_duration:.1f}")
    if total_requests > 0:
        print(f"Overall error rate: {total_errors / total_requests * 100:.1f}%")

    if all_latencies:
        avg_lat = sum(all_latencies) / len(all_latencies) * 1000
        p50_lat = sorted(all_latencies)[len(all_latencies) // 2] * 1000
        p99_idx = int(len(all_latencies) * 0.99)
        p99_lat = sorted(all_latencies)[p99_idx] * 1000
        print(f"\nOverall latency stats (this pod):")
        print(f"  Avg: {avg_lat:.1f}ms")
        print(f"  P50: {p50_lat:.1f}ms")
        print(f"  P99: {p99_lat:.1f}ms")

    print("=" * 70)

    # Return non-zero exit code if there were errors
    return 1 if total_errors > 0 else 0


def main() -> int:
    """Main entry point for the load generator script."""
    config = get_config()

    urls = config["urls"]
    duration = config["duration"]
    qps_per_pod = config["qps_per_pod"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    num_processes = config["num_processes"]
    total_pods = config["total_pods"]

    if not urls:
        print("ERROR: No URLs configured (LOAD_GEN_URLS is empty)", file=sys.stderr)
        return 1

    if not model:
        print("ERROR: No model configured (LOAD_GEN_MODEL is empty)", file=sys.stderr)
        return 1

    print("=" * 70)
    print("LOAD GENERATOR CONFIGURATION")
    print("=" * 70)
    print(f"Pod index: {os.environ.get('JOB_COMPLETION_INDEX', '0')} / {total_pods}")
    print(f"Processes per pod: {num_processes}")
    print(f"QPS per pod: {qps_per_pod:.2f}")
    print(f"QPS per process: {qps_per_pod / num_processes:.2f}")
    print(f"Total target QPS: {qps_per_pod * total_pods:.2f}")
    print(f"Duration: {duration}s")
    print(f"Frontends: {len(urls)}")
    print("=" * 70)
    print()

    if num_processes == 1:
        # Single process mode - no multiprocessing overhead
        print("Running in single-process mode...")
        result = asyncio.run(
            run_load_test_worker(0, qps_per_pod, duration, urls, model, max_tokens)
        )
        results = [result]
    else:
        # Multi-process mode
        print(f"Spawning {num_processes} worker processes...")

        # Calculate QPS per process
        qps_per_process = qps_per_pod / num_processes

        # Create result queue
        result_queue = multiprocessing.Queue()

        # Start worker processes
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=worker_process,
                args=(i, qps_per_process, duration, urls, model, max_tokens, result_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from queue
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

    # Aggregate and print results
    agg_stats, actual_duration = aggregate_results(results)

    if agg_stats is None:
        print("ERROR: No results collected", file=sys.stderr)
        return 1

    return print_results(agg_stats, actual_duration)


if __name__ == "__main__":
    sys.exit(main())

