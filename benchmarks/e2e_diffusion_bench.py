# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E benchmark for image diffusion postprocessing via Dynamo.

Sends N requests to the /v1/images/generations endpoint and measures:
- Total request latency (network + generation + postprocessing)
- Response size

Requires a running Dynamo frontend + image diffusion worker.

Usage:
    python benchmarks/e2e_diffusion_bench.py [--url URL] [--n N] [--format b64_json|url]
"""

import argparse
import json
import statistics
import time

import requests


def bench(url: str, n: int, response_format: str, model: str, size: str, steps: int):
    payload = {
        "prompt": "A curious raccoon exploring a library",
        "model": model,
        "size": size,
        "response_format": response_format,
        "nvext": {"num_inference_steps": steps},
    }

    print(f"E2E diffusion benchmark: {n} requests, format={response_format}, size={size}, steps={steps}")
    print(f"Target: {url}")
    print("=" * 80)

    # Warmup (1 request)
    print("Warmup...", end=" ", flush=True)
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    warmup_resp = r.json()
    if "error" in warmup_resp:
        print(f"ERROR: {warmup_resp['error']}")
        return
    print(f"OK ({r.elapsed.total_seconds():.2f}s)")

    times = []
    sizes = []

    for i in range(n):
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=300)
        t1 = time.perf_counter()
        r.raise_for_status()

        resp = r.json()
        if "error" in resp:
            print(f"  Request {i+1}: ERROR: {resp['error']}")
            continue

        latency_ms = (t1 - t0) * 1000
        resp_size = len(r.content)
        times.append(latency_ms)
        sizes.append(resp_size)

        data_info = ""
        if resp.get("data"):
            d = resp["data"][0]
            if d.get("b64_json"):
                data_info = f"b64 len={len(d['b64_json'])}"
            elif d.get("url"):
                data_info = f"url={d['url'][:60]}..."

        print(f"  Request {i+1}/{n}: {latency_ms:.0f}ms, resp_size={resp_size:,}B, {data_info}")

    if not times:
        print("No successful requests.")
        return

    print()
    print("Results:")
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 2 else times[0]
    mean = statistics.mean(times)
    avg_size = statistics.mean(sizes)
    print(f"  Latency:  mean={mean:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  min={min(times):.0f}ms  max={max(times):.0f}ms")
    print(f"  Response: avg_size={avg_size/1024:.1f}KB")


def main():
    parser = argparse.ArgumentParser(description="E2E diffusion benchmark")
    parser.add_argument("--url", default="http://localhost:8000/v1/images/generations")
    parser.add_argument("--n", type=int, default=5, help="Number of timed requests (after warmup)")
    parser.add_argument("--format", default="b64_json", choices=["b64_json", "url"])
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    bench(args.url, args.n, args.format, args.model, args.size, args.steps)


if __name__ == "__main__":
    main()
