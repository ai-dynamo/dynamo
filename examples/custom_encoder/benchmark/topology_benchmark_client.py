# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, retry-free OpenAI client for custom-encoder topology benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load_requests(path: Path, model: str, max_tokens: int) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    prepared: list[dict[str, Any]] = []
    dimensions: Counter[str] = Counter()
    encoded_bytes = 0
    image_hashes: set[str] = set()
    for row in rows:
        image_path = Path(row["image"])
        image = image_path.read_bytes()
        encoded_bytes += len(image)
        image_hashes.add(hashlib.sha256(image).hexdigest())
        name_parts = image_path.stem.split("_")
        size = next((part for part in name_parts if "x" in part), "unknown")
        dimensions[size] += 1
        image_uri = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        prepared.append(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": row["text"]},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "min_tokens": max_tokens,
                "ignore_eos": True,
                "temperature": 0,
                "stream": False,
            }
        )
    return prepared, {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unique_images": len(image_hashes),
        "image_size_counts": dict(sorted(dimensions.items())),
        "jpeg_bytes": encoded_bytes,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    requests, audit = _load_requests(args.input, args.model, args.max_tokens)
    if len(requests) != args.expected_requests:
        raise ValueError(
            f"expected {args.expected_requests} requests, found {len(requests)}"
        )
    if audit["unique_images"] != len(requests):
        raise ValueError("every measured request must use a unique image")

    queue: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue()
    for index, request in enumerate(requests):
        queue.put_nowait((index, request))
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    latencies = [0.0] * len(requests)
    completions = [0] * len(requests)
    completion_tokens = [0] * len(requests)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return
                index, request = item
                started = time.perf_counter()
                async with session.post(args.endpoint, json=request) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(
                            f"request {index} failed HTTP {response.status}: {body[:500]}"
                        )
                    payload = json.loads(body)
                latencies[index] = time.perf_counter() - started
                choices = payload.get("choices") or []
                if len(choices) != 1 or choices[0].get("finish_reason") is None:
                    raise RuntimeError(f"request {index} returned no finished choice")
                usage = payload.get("usage") or {}
                observed_tokens = usage.get("completion_tokens")
                if observed_tokens != args.max_tokens:
                    raise RuntimeError(
                        f"request {index} returned {observed_tokens} completion "
                        f"tokens, expected {args.max_tokens}"
                    )
                completion_tokens[index] = int(observed_tokens)
                completions[index] = 1

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        wall_time = time.perf_counter() - started

    successes = sum(completions)
    if successes != len(requests):
        raise RuntimeError(f"only {successes}/{len(requests)} requests completed")
    result = {
        "requests": len(requests),
        "successes": successes,
        "errors": 0,
        "retries": 0,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "completion_tokens": {
            "min": min(completion_tokens),
            "max": max(completion_tokens),
            "total": sum(completion_tokens),
        },
        "wall_time_s": wall_time,
        "request_throughput": len(requests) / wall_time,
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="http://localhost:8000/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=7)
    parser.add_argument("--expected-requests", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.concurrency < 1 or args.expected_requests < 1 or args.max_tokens < 1:
        parser.error("concurrency, expected requests, and max tokens must be positive")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
