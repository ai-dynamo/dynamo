#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-playback OpenAI completions stub for the loadgen parity audit (AIPerf environment).

Every request is identified by the SHA-256 of its prompt token IDs (plus ``max_tokens``) and
answered with replay's own timing for that request (``audit_parity.py stub-prep``): the first
token ``ttft_ms`` after receipt and the last ``e2e_ms`` after receipt, then a ``usage`` chunk
with ``nvext.worker_id`` (Dynamo's shapes). Requests that depend on earlier completions (later
turns, closed-loop session starts) therefore become due at the same replay-relative instants as
in replay, plus whatever the load generator adds; ``audit_parity.py stub-check`` measures that.

The ledger has one line per request: receive, first-token and last-token instants (``time_ns``),
the session header, and the payload facts. Standard library plus aiohttp only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import time
from collections import defaultdict


def digest(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def main() -> None:
    from aiohttp import web

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--latency", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # (prompt digest, max_tokens) -> request indices, earliest replay arrival first.
    table: dict = defaultdict(list)
    with open(args.latency) as handle:
        for line in handle:
            row = json.loads(line)
            table[(row["prompt_sha256"], row["max_tokens"])].append(row)
    for rows in table.values():
        rows.sort(key=lambda r: (r["replay_arrival_ms"], r["index"]))
    ledger = open(args.ledger, "a", buffering=1)

    async def completions(request: web.Request) -> web.StreamResponse:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        recv_ns = time.time_ns()
        body = json.loads(await request.read())
        tokens = body.get("prompt") if isinstance(body.get("prompt"), list) else []
        key = (digest(tokens), body.get("max_tokens"))
        candidates = table.get(key) or []
        row = candidates.pop(0) if candidates else None
        entry = {
            "index": None if row is None else row["index"],
            "ambiguous": len(candidates) > 0,
            "recv_ns": recv_ns,
            "prompt_len": len(tokens),
            "max_tokens": body.get("max_tokens"),
            "min_tokens": body.get("min_tokens"),
            "ignore_eos": body.get("ignore_eos"),
            "stream": body.get("stream"),
            "session_header": request.headers.get("X-Dynamo-Session-ID"),
            "other_session_headers": sorted(
                h
                for h in request.headers
                if h.lower()
                in (
                    "x-session-id",
                    "x-claude-code-session-id",
                    "thread-id",
                    "x-session-affinity",
                )
            ),
            "correlation_id": request.headers.get("X-Correlation-ID"),
        }
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        osl = int(body.get("max_tokens") or 1)
        ttft = (row["ttft_ms"] if row else 1.0) / 1000.0
        e2e = (row["e2e_ms"] if row else 2.0) / 1000.0
        model = body.get("model")

        def chunk(text: str, finish=None) -> bytes:
            data = {
                "id": "cmpl-audit",
                "object": "text_completion",
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            }
            return f"data: {json.dumps(data)}\n\n".encode()

        await asyncio.sleep(max(0.0, t0 + ttft - loop.time()))
        entry["first_ns"] = time.time_ns()
        await response.write(chunk(" x"))
        await asyncio.sleep(max(0.0, t0 + e2e - loop.time()))
        if osl > 1:
            await response.write(chunk(" x" * (osl - 1)))
        entry["last_ns"] = time.time_ns()
        final = {
            "id": "cmpl-audit",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": osl,
                "total_tokens": len(tokens) + osl,
            },
            "nvext": {"worker_id": {"decode_worker_id": 0, "prefill_worker_id": 0}},
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        ledger.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return response

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": "Qwen/Qwen3-32B", "object": "model"}]}
        )

    app = web.Application(client_max_size=1 << 30)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/models", models)
    web.run_app(
        app, host=args.host, port=args.port, print=None, access_log=None, backlog=4096
    )


if __name__ == "__main__":
    main()
