#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise unary and SSE chat through an existing global endpoint; never changes infrastructure."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(base_url, model, session, timeout):
    results = []
    for stream in (False, True):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json", "x-dynamo-session-id": session}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        started = time.monotonic()
        result = {"stream": stream, "ok": False}
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result["status"] = response.status
                body = response.read(2 * 1024 * 1024).decode()
                if stream:
                    events = [
                        line[5:].strip()
                        for line in body.splitlines()
                        if line.startswith("data:")
                    ]
                    chunks = [
                        json.loads(event) for event in events if event != "[DONE]"
                    ]
                    if "[DONE]" not in events or not any(
                        chunk.get("choices") for chunk in chunks
                    ):
                        raise ValueError("Incomplete SSE response")
                else:
                    if not json.loads(body).get("choices"):
                        raise ValueError("Missing completion choices")
                result["ok"] = True
        except (OSError, ValueError, urllib.error.URLError) as error:
            result["error"] = str(error)
        result["elapsed_seconds"] = time.monotonic() - started
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--session", default="global-router-smoke")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    results = run(args.base_url, args.model, args.session, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"base_url": args.base_url, "model": args.model, "results": results},
            indent=2,
        )
        + "\n"
    )
    print(
        f"{sum(r['ok'] for r in results)}/{len(results)} checks passed; results: {args.output}"
    )
    raise SystemExit(0 if all(r["ok"] for r in results) else 1)
