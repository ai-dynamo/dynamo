# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep aiohttp vs httpx through dynamo.common.http.

Rate-limited emitter. For each (request_rate, server-processing-time-mean-ms)
pair, brings up a local media server with that processing-time-mean and
issues `--requests` total requests at the target RPS, twice (once per
backend), alternating which backend goes first to remove cold-start bias.
Reports a per-iteration table and a per-(request_rate) cross-iteration
grid.

Usage:
  python -m benchmarks.multimodal.http.sweep \\
      --server-processing-time-means-ms 50,100,200,300,400 \\
      --request-rate 100,500,1000 \\
      --requests 5000 \\
      --timeout 60
"""

from __future__ import annotations

import argparse
import asyncio
import resource
import sys

from .report import print_batch_header, print_grid, print_iteration
from .runner import gen_urls, run_one
from .server_lifecycle import local_media_server
from .stats import Summary, summarize

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep aiohttp vs httpx across server delays and request rates."
    )
    p.add_argument(
        "--server-processing-time-means-ms",
        type=str,
        default="50,100,200,300,400",
        help="comma-separated processing-time-mean-ms values (default: 50,100,200,300,400)",
    )
    p.add_argument(
        "--request-rate",
        type=str,
        default="100,500,1000",
        help="comma-separated target requests-per-second values (default: 100,500,1000)",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=5000,
        help="total number of requests per (rate, mean) cell (default: 5000)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-request timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--port", type=int, default=8233, help="local media server port (default: 8233)"
    )
    p.add_argument(
        "--image-url",
        type=str,
        default=DEFAULT_IMAGE_URL,
        help=f"image URL the server fetches once and re-serves (default: {DEFAULT_IMAGE_URL})",
    )
    return p.parse_args(argv)


def _check_ulimit(peak_inflight_estimate: int) -> None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < max(8192, peak_inflight_estimate + 512):
        print(
            f"[sweep] WARN: ulimit -n is {soft}, may be too low for peak-inflight≈{peak_inflight_estimate}",
            file=sys.stderr,
        )


def _by_backend(summaries: list[Summary], backend: str) -> Summary:
    for s in summaries:
        if s.backend == backend:
            return s
    raise KeyError(f"no summary for backend {backend!r}")


def _parse_float_list(raw: str, flag: str) -> list[float]:
    values = [float(x) for x in raw.split(",") if x.strip()]
    if not values:
        print(f"[sweep] {flag} is empty", file=sys.stderr)
        raise SystemExit(2)
    return values


async def _run_sweep(args: argparse.Namespace) -> int:
    means = _parse_float_list(
        args.server_processing_time_means_ms, "--server-processing-time-means-ms"
    )
    rates = _parse_float_list(args.request_rate, "--request-rate")
    if args.requests <= 0:
        print("[sweep] --requests must be > 0", file=sys.stderr)
        return 2

    peak_inflight = int(max(rates) * (max(means) / 1000.0)) + args.requests // 10
    _check_ulimit(peak_inflight)

    for rate in rates:
        print_batch_header(request_rate=rate, requests=args.requests)
        grid_rows: list[tuple[float, Summary, Summary]] = []
        for i, mean_ms in enumerate(means):
            with local_media_server(
                port=args.port, image_url=args.image_url, mean_ms=mean_ms
            ) as base_url:
                urls = gen_urls([base_url], args.requests)
                order = ["httpx", "aiohttp"] if i % 2 else ["aiohttp", "httpx"]
                print(f"[sweep] mean_ms={mean_ms:g}  order={order}")
                results = [await run_one(b, urls, args.timeout, rate) for b in order]

            summaries = [summarize(r) for r in results]
            print_iteration(mean_ms, summaries)
            grid_rows.append(
                (
                    mean_ms,
                    _by_backend(summaries, "aiohttp"),
                    _by_backend(summaries, "httpx"),
                )
            )

        print_grid(grid_rows)
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_run_sweep(args))


if __name__ == "__main__":
    raise SystemExit(main())
