# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone load generator script for Kubernetes Jobs.

Configuration via environment variables:
    LOAD_GEN_URLS, LOAD_GEN_MODEL, LOAD_GEN_DURATION, LOAD_GEN_QPS_PER_POD,
    LOAD_GEN_MAX_TOKENS, LOAD_GEN_NUM_PROCESSES, LOAD_GEN_TOTAL_PODS
"""

import asyncio
import json
import multiprocessing
import os
import sys
import time

from openai import AsyncOpenAI


def get_config() -> dict:
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
    process_id: int, qps: float, duration: int, urls: list, model: str, max_tokens: int
) -> dict:
    print(
        f"[Process {process_id}] Starting: {duration}s, {qps:.2f} QPS, {len(urls)} targets"
    )

    clients = {
        url: AsyncOpenAI(
            base_url=f"{url}/v1", api_key="not-needed", timeout=duration + 30
        )
        for url in urls
    }
    stats = {
        url: {"success": 0, "failed": 0, "latencies": [], "tokens": 0} for url in urls
    }

    interval = 1.0 / qps if qps > 0 else 1.0
    start_time = time.time()
    request_count = 0
    tasks = []

    async def send_and_record(url: str) -> None:
        req_start = time.time()
        success, tokens = await send_request(clients[url], model, max_tokens)
        latency = time.time() - req_start

        if success:
            stats[url]["success"] += 1
            stats[url]["latencies"].append(latency)
            stats[url]["tokens"] += tokens
        else:
            stats[url]["failed"] += 1

    while time.time() - start_time < duration:
        url = urls[request_count % len(urls)]
        task = asyncio.create_task(send_and_record(url))
        tasks.append(task)
        request_count += 1

        next_request_time = start_time + (request_count * interval)
        sleep_duration = next_request_time - time.time()
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "process_id": process_id,
        "stats": stats,
        "actual_duration": time.time() - start_time,
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
    try:
        result = asyncio.run(
            run_load_test_worker(process_id, qps, duration, urls, model, max_tokens)
        )
        result_queue.put(result)
    except Exception as e:
        print(f"[Process {process_id}] Error: {e}", file=sys.stderr)
        result_queue.put({"error": str(e), "process_id": process_id})


def aggregate_results(results: list) -> tuple:
    if not results:
        return None, 0

    all_urls = set()
    for r in results:
        if "stats" in r:
            all_urls.update(r["stats"].keys())

    agg_stats = {
        url: {"success": 0, "failed": 0, "latencies": [], "tokens": 0}
        for url in all_urls
    }
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
            avg_lat = (
                sum(s["latencies"]) / len(s["latencies"]) * 1000
                if s["latencies"]
                else 0
            )
            p50_lat = (
                sorted(s["latencies"])[len(s["latencies"]) // 2] * 1000
                if s["latencies"]
                else 0
            )
            p99_idx = int(len(s["latencies"]) * 0.99)
            p99_lat = sorted(s["latencies"])[p99_idx] * 1000 if s["latencies"] else 0

            print(f"\nFrontend {url}:")
            print(
                f"  Requests: {total}, Success: {s['success']}, Errors: {s['failed']} ({s['failed'] / total * 100:.1f}%)"
            )
            print(
                f"  Latency: avg={avg_lat:.1f}ms, p50={p50_lat:.1f}ms, p99={p99_lat:.1f}ms"
            )
            print(f"  Tokens: {s['tokens']}")

    print("\n" + "-" * 70)
    print(f"Total requests (this pod): {total_requests}")
    print(f"Total errors: {total_errors}")
    print(f"Actual duration: {actual_duration:.1f}s")
    if actual_duration > 0:
        print(f"Actual QPS (this pod): {total_requests / actual_duration:.1f}")
    if total_requests > 0:
        print(f"Error rate: {total_errors / total_requests * 100:.1f}%")
    print("=" * 70)

    return 1 if total_errors > 0 else 0


def main() -> int:
    config = get_config()

    urls = config["urls"]
    duration = config["duration"]
    qps_per_pod = config["qps_per_pod"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    num_processes = config["num_processes"]
    total_pods = config["total_pods"]

    if not urls:
        print("ERROR: No URLs configured", file=sys.stderr)
        return 1

    if not model:
        print("ERROR: No model configured", file=sys.stderr)
        return 1

    print("=" * 70)
    print(
        f"Pod {os.environ.get('JOB_COMPLETION_INDEX', '0')}/{total_pods}, {num_processes} process(es)"
    )
    print(
        f"QPS: {qps_per_pod:.2f}/pod, {qps_per_pod / num_processes:.2f}/process, {qps_per_pod * total_pods:.2f} total"
    )
    print(f"Duration: {duration}s, Frontends: {len(urls)}")
    print("=" * 70 + "\n")

    if num_processes == 1:
        result = asyncio.run(
            run_load_test_worker(0, qps_per_pod, duration, urls, model, max_tokens)
        )
        results = [result]
    else:
        print(f"Spawning {num_processes} worker processes...")
        qps_per_process = qps_per_pod / num_processes
        result_queue = multiprocessing.Queue()

        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=worker_process,
                args=(
                    i,
                    qps_per_process,
                    duration,
                    urls,
                    model,
                    max_tokens,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

    agg_stats, actual_duration = aggregate_results(results)

    if agg_stats is None:
        print("ERROR: No results collected", file=sys.stderr)
        return 1

    return print_results(agg_stats, actual_duration)


if __name__ == "__main__":
    sys.exit(main())
