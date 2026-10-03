#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loadgen contract test: did AIPerf send what the generator asked for, when it asked for it?

``serve`` is a dummy OpenAI ``/v1/completions`` SSE server that needs no GPU (aiohttp, from the
AIPerf environment). For each request it does three things:

- logs the server-side receive and completion instants, the session header, the request parameters
  and the SHA-256 of the prompt token IDs to a ledger;
- streams exactly ``max_tokens`` chunks, after a deterministic delay of ``ttft_base +
  per_ktok * ISL / 1000``, with a constant inter-chunk gap;
- ends with a ``usage`` chunk and ``nvext.worker_id``, the shapes Dynamo returns.

``check`` compares the ledger with the generator's ``requests.jsonl`` and ``manifest.json``.
These checks must hold exactly:

- **exactly once:** each request was received exactly once, with the exact prompt (matched by
  token digest, so the comparison is independent of AIPerf's own exports);
- **payload:** ``len(prompt) == ISL`` and ``max_tokens == min_tokens == OSL``, with
  ``ignore_eos``, ``stream`` and usage reporting set;
- **sessions:** the ``X-Dynamo-Session-ID`` header is present exactly when replay would pass a
  session, constant within a session, and distinct across sessions;
- **causality:** turn ``k + 1`` of a session arrives after turn ``k`` completed;
- **closed loop:** concurrent sessions never exceed ``C``, the run reaches ``C``, and sessions
  start in the generated order (allowing for starts that land in the same scheduler instant).

Timing is reported, not asserted:

- open-loop **arrival lateness**, against the server-side origin that best fits the generated
  timestamps;
- **think-time residuals**: the arrival of turn ``k + 1``, minus the server's completion of turn
  ``k``, minus the delay.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def prompt_digest(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


# --------------------------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------------------------


def serve(args: argparse.Namespace) -> None:
    from aiohttp import web

    ledger = open(args.ledger, "a", buffering=1)
    stats = Counter()

    async def completions(request: web.Request) -> web.StreamResponse:
        recv_ns = time.time_ns()
        body = await request.json()
        prompt = body.get("prompt")
        tokens = prompt if isinstance(prompt, list) else []
        osl = int(body.get("max_tokens") or 1)
        session = request.headers.get("X-Dynamo-Session-ID")
        worker = (
            int(
                hashlib.sha256(
                    (session or request.headers.get("X-Request-ID", "")).encode()
                ).hexdigest(),
                16,
            )
            % args.workers
        )
        entry = {
            "recv_ns": recv_ns,
            "prompt_sha256": prompt_digest(tokens),
            "prompt_len": len(tokens),
            "max_tokens": body.get("max_tokens"),
            "min_tokens": body.get("min_tokens"),
            "ignore_eos": body.get("ignore_eos"),
            "stream": body.get("stream"),
            "include_usage": (body.get("stream_options") or {}).get("include_usage"),
            "nvext": body.get("nvext"),
            "model": body.get("model"),
            "session_header": session,
            "request_id": request.headers.get("X-Request-ID"),
            "correlation_id": request.headers.get("X-Correlation-ID"),
            "headers": sorted(h.lower() for h in request.headers),
        }
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(
            (args.ttft_base_ms + args.ttft_ms_per_ktok * len(tokens) / 1000.0) / 1000.0
        )
        entry["first_token_ns"] = time.time_ns()
        for i in range(osl):
            if i:
                await asyncio.sleep(args.itl_ms / 1000.0)
            chunk = {
                "id": "c",
                "object": "text_completion",
                "model": body.get("model"),
                "choices": [{"index": 0, "text": " x", "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = {
            "id": "c",
            "object": "text_completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": osl,
                "total_tokens": len(tokens) + osl,
            },
            "nvext": {"worker_id": {"decode_worker_id": 7000 + worker}},
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        entry["done_ns"] = time.time_ns()
        ledger.write(json.dumps(entry) + "\n")
        stats["requests"] += 1
        return response

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": args.model, "object": "model"}]}
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application(client_max_size=1 << 30)
    app.add_routes(
        [
            web.post("/v1/completions", completions),
            web.get("/v1/models", models),
            web.get("/health", health),
        ]
    )
    web.run_app(app, host=args.host, port=args.port, access_log=None, print=None)


# --------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def q(p: float) -> float:
        rank = p / 100 * (len(ordered) - 1)
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    return {
        "n": len(values),
        "min": ordered[0],
        "p50": q(50),
        "p99": q(99),
        "max": ordered[-1],
    }


def check(inputs: Path, ledger_path: Path, same_instant_ms: float = 5.0) -> dict:
    manifest = json.loads((inputs / "manifest.json").read_text())
    requests = [
        json.loads(x)
        for x in (inputs / "requests.jsonl").read_text().splitlines()
        if x.strip()
    ]
    ledger = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
    failures: list[str] = []
    report: dict = {
        "cell_id": manifest.get("cell_id"),
        "k": manifest.get("k"),
        "open_loop": manifest["open_loop"],
        "requests": len(requests),
        "received": len(ledger),
        "failures": failures,
    }
    # Pair by prompt digest; identical prompts (duplicate trace rows) pair in arrival order.
    received: dict[str, list[dict]] = defaultdict(list)
    for entry in sorted(ledger, key=lambda e: e["recv_ns"]):
        received[entry["prompt_sha256"]].append(entry)
    expected: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        expected[request["prompt_sha256"]].append(request)
    for digest, group in expected.items():
        group.sort(
            key=lambda r: (
                r["arrival_ms"] if r["arrival_ms"] is not None else 0.0,
                r["index"],
            )
        )
    matched: dict[tuple[str, int], dict] = {}
    for digest, group in expected.items():
        got = received.get(digest, [])
        if len(got) != len(group):
            failures.append(
                f"prompt {digest[:12]}: expected {len(group)} receipts, got {len(got)}"
            )
        for request, entry in zip(group, got):
            matched[(request["conversation_id"], request["turn_index"])] = entry
    extra = sum(max(len(v) - len(expected.get(d, [])), 0) for d, v in received.items())
    report.update(matched=len(matched), unexpected_receipts=extra)
    if extra:
        failures.append(f"{extra} receipts match no generated prompt")

    payload_errors = Counter()
    for request in requests:
        entry = matched.get((request["conversation_id"], request["turn_index"]))
        if entry is None:
            continue
        if entry["prompt_len"] != request["input_length"]:
            payload_errors["isl"] += 1
        if not (entry["max_tokens"] == entry["min_tokens"] == request["osl_sent"]):
            payload_errors["osl"] += 1
        if (
            entry["ignore_eos"] is not True
            or entry["stream"] is not True
            or entry["include_usage"] is not True
        ):
            payload_errors["flags"] += 1
        if manifest.get("worker_id_field") and (entry.get("nvext") or {}).get(
            "extra_fields"
        ) != ["worker_id"]:
            payload_errors["nvext"] += 1
        if entry["model"] != manifest["model"]:
            payload_errors["model"] += 1
    report["payload_errors"] = dict(payload_errors)
    if payload_errors:
        failures.append(f"payload errors {dict(payload_errors)}")

    sessions: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        sessions[request["conversation_id"]].append(request)
    header_values: dict[str, set] = defaultdict(set)
    present = absent = 0
    for key, turns in sessions.items():
        for request in turns:
            entry = matched.get((key, request["turn_index"]))
            if entry is None:
                continue
            value = entry["session_header"]
            present += value is not None
            absent += value is None
            header_values[key].add(value)
    inconsistent = sum(len(v) > 1 for v in header_values.values())
    owners = Counter(
        next(iter(v)) for v in header_values.values() if len(v) == 1 and None not in v
    )
    shared = sum(n > 1 for n in owners.values())
    report["session_header"] = {
        "expected": manifest["session_header"],
        "present": present,
        "absent": absent,
        "inconsistent_sessions": inconsistent,
        "values_shared_across_sessions": shared,
    }
    if manifest["session_header"] and (absent or inconsistent or shared):
        failures.append(
            "session header missing, inconsistent within a session, or shared"
        )
    if not manifest["session_header"] and present:
        failures.append(
            f"{present} requests carried a session header replay would not pass"
        )

    # Causality and think-time residuals (server clock).
    causality = 0
    residuals = []
    for key, turns in sessions.items():
        for previous, request in zip(turns, turns[1:]):
            a = matched.get((key, previous["turn_index"]))
            b = matched.get((key, request["turn_index"]))
            if a is None or b is None:
                continue
            gap_ms = (b["recv_ns"] - a["done_ns"]) / 1e6
            causality += gap_ms < 0
            residuals.append(gap_ms - request["delay_ms"])
    report["causality_violations"] = causality
    report["think_residual_ms"] = _quantiles(residuals)
    if causality:
        failures.append(
            f"{causality} turns arrived before their previous turn completed"
        )

    if manifest["open_loop"]:
        firsts = [
            (r, matched.get((r["conversation_id"], 0)))
            for r in requests
            if r["arrival_ms"] is not None
        ]
        firsts = [(r, e) for r, e in firsts if e is not None]
        origin = min(e["recv_ns"] - round(r["arrival_ms"] * 1e6) for r, e in firsts)
        lateness = [(e["recv_ns"] - origin) / 1e6 - r["arrival_ms"] for r, e in firsts]
        report["arrival_lateness_ms"] = _quantiles(lateness)
    else:
        cap = int(manifest["concurrency"])
        spans = []
        for key, turns in sessions.items():
            entries = [matched.get((key, r["turn_index"])) for r in turns]
            if any(e is None for e in entries):
                continue
            spans.append((entries[0]["recv_ns"], entries[-1]["done_ns"], key))
        events = sorted(
            [(s, 1) for s, _, _ in spans] + [(e, -1) for _, e, _ in spans],
            key=lambda x: (x[0], x[1]),
        )
        current = peak = 0
        for _, delta in events:
            current += delta
            peak = max(peak, current)
        order = {key: i for i, key in enumerate(sessions)}
        starts = sorted(spans)
        inversions = sum(
            1
            for (sa, _, ka), (sb, _, kb) in zip(starts, starts[1:])
            if order[kb] < order[ka] and (sb - sa) / 1e6 > same_instant_ms
        )
        report["closed"] = {
            "cap": cap,
            "peak_sessions": peak,
            "start_order_inversions": inversions,
        }
        if peak > cap:
            failures.append(f"{peak} concurrent sessions exceed the cap {cap}")
        if peak < min(cap, len(sessions)):
            failures.append(f"peak {peak} never reached the cap {cap}")
        if inversions:
            failures.append(f"{inversions} sessions started out of the generated order")
    report["ok"] = not failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=18991)
    p_serve.add_argument("--ledger", type=Path, required=True)
    p_serve.add_argument("--model", default="Qwen/Qwen3-32B")
    p_serve.add_argument("--ttft-base-ms", type=float, default=5.0)
    p_serve.add_argument("--ttft-ms-per-ktok", type=float, default=2.0)
    p_serve.add_argument("--itl-ms", type=float, default=2.0)
    p_serve.add_argument("--workers", type=int, default=4)
    p_check = sub.add_parser("check")
    p_check.add_argument("--inputs", type=Path, required=True)
    p_check.add_argument("--ledger", type=Path, required=True)
    p_check.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args)
        return 0
    report = check(args.inputs, args.ledger)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
