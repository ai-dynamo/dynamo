# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Measure when the first actionable tool call info arrives in the stream.

With dispatch ON: `event: tool_call_dispatch` fires as soon as tool call is parseable.
With dispatch OFF: tool call info only available at `finish_reason: tool_calls`.

Measures:
- first_token_ms: first reasoning/content delta
- first_tool_info_ms: first tool_call_dispatch event OR first tool_calls delta
- stream_done_ms: finish_reason received
"""

import argparse
import json
import statistics
import sys
import time

import requests


def measure_one(url, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a city description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["city", "description"],
                },
            },
        }
    ]

    body = {
        "model": model,
        "max_tokens": 2000,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are a travel expert. Think about what makes each city special, then call echo.",
            },
            {"role": "user", "content": "Describe Tokyo by calling the echo tool."},
        ],
        "tools": tools,
        "tool_choice": "auto",
    }

    headers = {"Content-Type": "application/json"}
    t0 = time.monotonic()

    r = requests.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers=headers,
        stream=True,
        timeout=60,
    )

    result = {
        "first_token_ms": None,
        "first_tool_info_ms": None,
        "dispatch_event_ms": None,
        "tool_delta_ms": None,
        "finish_reason_ms": None,
        "stream_done_ms": None,
    }

    event_type = None

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            event_type = None
            continue

        now = (time.monotonic() - t0) * 1000

        if line.startswith("event: "):
            event_type = line[7:].strip()
            if (
                event_type == "tool_call_dispatch"
                and result["dispatch_event_ms"] is None
            ):
                result["dispatch_event_ms"] = now
                if result["first_tool_info_ms"] is None:
                    result["first_tool_info_ms"] = now
            continue

        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                result["stream_done_ms"] = now
                break

            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue

            choices = d.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            # First token (reasoning or content)
            if result["first_token_ms"] is None:
                if delta.get("content") or delta.get("reasoning_content"):
                    result["first_token_ms"] = now

            # First tool call delta
            tc = delta.get("tool_calls")
            if tc and result["tool_delta_ms"] is None:
                result["tool_delta_ms"] = now
                if result["first_tool_info_ms"] is None:
                    result["first_tool_info_ms"] = now

            # Finish reason
            if finish and result["finish_reason_ms"] is None:
                result["finish_reason_ms"] = now

        event_type = None

    r.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--label", default="unknown", help="ON or OFF label for output")
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args()

    print(f"Dispatch label: {args.label}", file=sys.stderr)
    print(f"Running {args.runs} measurements...", file=sys.stderr)

    results = []
    for i in range(args.runs):
        r = measure_one(args.url, args.model)
        r["run"] = i + 1
        r["label"] = args.label
        results.append(r)

        dispatch = (
            f"dispatch={r['dispatch_event_ms']:.1f}"
            if r["dispatch_event_ms"]
            else "no dispatch"
        )
        tool = (
            f"tool_delta={r['tool_delta_ms']:.1f}"
            if r["tool_delta_ms"]
            else "no tool delta"
        )
        info = (
            f"first_info={r['first_tool_info_ms']:.1f}"
            if r["first_tool_info_ms"]
            else "no info"
        )
        done = f"done={r['finish_reason_ms']:.1f}" if r["finish_reason_ms"] else "?"

        print(f"  {i+1}: {info}ms  {done}ms  ({dispatch}, {tool})", file=sys.stderr)
        time.sleep(0.3)

    # Summary
    infos = [r["first_tool_info_ms"] for r in results if r["first_tool_info_ms"]]
    dones = [r["finish_reason_ms"] for r in results if r["finish_reason_ms"]]

    if infos:
        print(
            f"\nFirst tool info: mean={statistics.mean(infos):.1f}ms  stdev={statistics.stdev(infos):.1f}ms",
            file=sys.stderr,
        )
    if dones:
        print(
            f"Finish reason:   mean={statistics.mean(dones):.1f}ms  stdev={statistics.stdev(dones):.1f}ms",
            file=sys.stderr,
        )
    if infos and dones:
        print(
            f"Info before done: {statistics.mean(dones) - statistics.mean(infos):.1f}ms avg",
            file=sys.stderr,
        )

    has_dispatch = any(r["dispatch_event_ms"] is not None for r in results)
    print(f"Dispatch events detected: {has_dispatch}", file=sys.stderr)

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {
                            k: round(v, 1) if isinstance(v, float) else v
                            for k, v in r.items()
                        }
                    )
                    + "\n"
                )
        print(f"Saved to {args.jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
