#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 request load generator.

Drives the frontend with concurrent OpenAI chat requests so the mocker workers
keep doing forward passes — which is what produces event-plane traffic
(KV-cache events + forward-pass-metrics). Prefixes are randomized so the mocker
stores fresh KV blocks (-> kv-events) on most requests.

This is only the load source; the actual latency measurement happens in the
Rust `event_plane_bench_sub` subscriber. We report request throughput here so a
run can be sanity-checked (were the workers actually busy?).
"""
import argparse
import asyncio
import random
import string
import time

import httpx


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()

    url = f"http://localhost:{a.port}/v1/chat/completions"
    sent = 0
    errs = 0
    deadline = time.time() + a.duration

    async with httpx.AsyncClient(timeout=30) as client:
        async def worker() -> None:
            nonlocal sent, errs
            while time.time() < deadline:
                prefix = "".join(
                    random.choice(string.ascii_lowercase)
                    for _ in range(random.randint(5, 30))
                )
                body = {
                    "model": a.model,
                    "messages": [
                        {"role": "user", "content": f"{prefix} please count to forty"}
                    ],
                    "max_tokens": a.max_tokens,
                    "stream": False,
                }
                try:
                    r = await client.post(url, json=body)
                    sent += 1
                    if r.status_code != 200:
                        errs += 1
                except Exception:
                    errs += 1

        await asyncio.gather(*[worker() for _ in range(a.concurrency)])

    print(f"loadgen sent={sent} errs={errs} dur={a.duration}s conc={a.concurrency}")


if __name__ == "__main__":
    asyncio.run(main())
