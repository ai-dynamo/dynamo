#!/usr/bin/env python3
"""Measure SSE event timing from a streaming chat completion with tool calls.

Captures timestamps for:
- first token (any delta)
- first reasoning token
- reasoning end (transition to content/tool)
- tool call complete (full name + arguments available)
- finish_reason received
- stream done ([DONE])

Usage:
    python3 measure_stream_timing.py --url http://localhost:8000 --model MODEL --runs N
"""

import argparse
import csv
import json
import sys
import time

import requests


def make_request_body(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 300,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You have access to a calculator tool. When asked math questions, always use it.",
            },
            {"role": "user", "content": "What is 42 * 17?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


def parse_sse_lines(response):
    """Yield (event_type, data_dict) from SSE stream."""
    event_type = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            event_type = None
            continue
        if line.startswith("event: "):
            event_type = line[7:].strip()
            continue
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                yield (event_type or "done", {"done": True})
            else:
                try:
                    yield (event_type, json.loads(raw))
                except json.JSONDecodeError:
                    pass


def measure_one_run(url: str, model: str) -> dict:
    """Run one streaming request and return timing milestones."""
    body = make_request_body(model)
    t0 = time.monotonic()

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=30,
    )
    resp.raise_for_status()

    milestones = {
        "request_sent_ms": 0.0,
        "first_token_ms": None,
        "first_reasoning_ms": None,
        "reasoning_end_ms": None,
        "tool_call_complete_ms": None,
        "finish_reason_ms": None,
        "done_ms": None,
    }

    saw_reasoning = False
    reasoning_ended = False
    tool_call_complete = False

    for event_type, data in parse_sse_lines(resp):
        now = (time.monotonic() - t0) * 1000  # ms since request

        if data.get("done"):
            milestones["done_ms"] = now
            break

        choices = data.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        # First token
        if milestones["first_token_ms"] is None:
            has_content = delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            if has_content:
                milestones["first_token_ms"] = now

        # Reasoning
        if delta.get("reasoning_content") is not None:
            if milestones["first_reasoning_ms"] is None:
                milestones["first_reasoning_ms"] = now
            saw_reasoning = True

        # Reasoning end: first chunk after reasoning that isn't reasoning
        if saw_reasoning and not reasoning_ended:
            if delta.get("reasoning_content") is None and (
                delta.get("content") is not None or delta.get("tool_calls") is not None
            ):
                milestones["reasoning_end_ms"] = now
                reasoning_ended = True

        # Tool call complete
        if delta.get("tool_calls") and not tool_call_complete:
            tc = delta["tool_calls"][0]
            fn = tc.get("function", {})
            if fn.get("name") and fn.get("arguments"):
                milestones["tool_call_complete_ms"] = now
                tool_call_complete = True

        # Dispatch events (if streaming dispatch is enabled)
        if event_type == "tool_call_dispatch":
            milestones["tool_dispatch_event_ms"] = now
        if event_type == "reasoning_dispatch":
            if "reasoning_dispatch_first_ms" not in milestones:
                milestones["reasoning_dispatch_first_ms"] = now

        # Finish reason
        if finish and milestones["finish_reason_ms"] is None:
            milestones["finish_reason_ms"] = now

            # Capture nvext timing if present
            nvext = data.get("nvext", {}).get("timing", {})
            if nvext:
                milestones["server_total_time_ms"] = nvext.get("total_time_ms")

    resp.close()
    return milestones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="-", help="Output CSV path (- for stdout)")
    parser.add_argument("--jsonl", default=None, help="Output JSONL path for raw data")
    args = parser.parse_args()

    results = []
    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}...", file=sys.stderr)
        m = measure_one_run(args.url, args.model)
        m["run"] = i + 1
        results.append(m)
        print(f"  TTFT={m['first_token_ms']:.1f}ms  tool_complete={m.get('tool_call_complete_ms', 'N/A')}  done={m['done_ms']:.1f}ms", file=sys.stderr)
        if i < args.runs - 1:
            time.sleep(0.5)

    # Write JSONL
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Raw data written to {args.jsonl}", file=sys.stderr)

    # Write CSV
    fieldnames = [
        "run", "first_token_ms", "first_reasoning_ms", "reasoning_end_ms",
        "tool_call_complete_ms", "finish_reason_ms", "done_ms", "server_total_time_ms",
    ]
    extra = ["tool_dispatch_event_ms", "reasoning_dispatch_first_ms"]
    for e in extra:
        if any(e in r for r in results):
            fieldnames.append(e)

    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    if args.output != "-":
        out.close()
        print(f"CSV written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
