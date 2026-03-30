# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Harness-in-the-loop dispatch experiment (Section 3, Agentic Harnesses blog).

Variant A (Buffered): wait for finish_reason=tool_calls, then parse & execute.
Variant B (Dispatch-aware): execute tool immediately on event: tool_call_dispatch.
"""

import argparse
import csv
import io
import json
import statistics
import sys
import threading
import time

import requests

# ---------------------------------------------------------------------------
# Tool definition & simulated execution
# ---------------------------------------------------------------------------

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an assistant with access to a calculator tool. "
            "Always use the calculator tool for arithmetic."
        ),
    },
    {"role": "user", "content": "What is 42 * 17?"},
]


def simulate_tool(latency_s: float) -> str:
    """Simulate tool execution with configurable latency."""
    time.sleep(latency_s)
    return json.dumps({"result": 42 * 17})


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _iter_sse_lines(response):
    """Yield (event_type, data_str) tuples from an SSE stream."""
    current_event = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line  # already decoded
        if line == "":
            # blank line = end of event block
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data = line[len("data:") :].strip()
            yield (current_event, data)
            # reset event after yielding
            current_event = None


# ---------------------------------------------------------------------------
# Variant A — Buffered
# ---------------------------------------------------------------------------


def run_buffered(url: str, model: str, tool_latency_s: float) -> dict:
    t = {"request_sent": time.monotonic()}

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": MESSAGES,
            "tools": [CALCULATOR_TOOL],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 512,
        },
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    first_token_seen = False
    accumulated_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}

    for _evt_type, data in _iter_sse_lines(resp):
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if not first_token_seen and delta:
            t["first_token"] = time.monotonic()
            first_token_seen = True

        # Accumulate tool call deltas
        for tc in delta.get("tool_calls") or []:
            idx = tc["index"]
            if idx not in accumulated_tool_calls:
                accumulated_tool_calls[idx] = {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                }
            fn = tc.get("function", {})
            if "name" in fn and fn["name"]:
                accumulated_tool_calls[idx]["name"] = fn["name"]
            if "id" in tc and tc["id"]:
                accumulated_tool_calls[idx]["id"] = tc["id"]
            accumulated_tool_calls[idx]["arguments"] += fn.get("arguments", "")

        if choice.get("finish_reason") == "tool_calls":
            t["tool_detected"] = time.monotonic()

    t["stream_done"] = time.monotonic()

    # Execute tool after stream completes
    t["tool_started"] = time.monotonic()
    simulate_tool(tool_latency_s)
    t["tool_finished"] = time.monotonic()

    resp.close()
    return t


# ---------------------------------------------------------------------------
# Variant B — Dispatch-aware
# ---------------------------------------------------------------------------


def run_dispatch(url: str, model: str, tool_latency_s: float) -> dict:
    t = {"request_sent": time.monotonic()}

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": MESSAGES,
            "tools": [CALCULATOR_TOOL],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 512,
        },
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    first_token_seen = False
    tool_thread = None
    tool_finished_event = threading.Event()
    tool_times: dict[str, float] = {}

    def _exec_tool():
        tool_times["started"] = time.monotonic()
        simulate_tool(tool_latency_s)
        tool_times["finished"] = time.monotonic()
        tool_finished_event.set()

    for evt_type, data in _iter_sse_lines(resp):
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # tool_call_dispatch events may not have choices
        if evt_type == "tool_call_dispatch":
            t["tool_detected"] = time.monotonic()
            if tool_thread is None:
                tool_thread = threading.Thread(target=_exec_tool, daemon=True)
                tool_thread.start()
            continue

        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if not first_token_seen and delta:
            t["first_token"] = time.monotonic()
            first_token_seen = True

        # Fallback: if server doesn't emit dispatch events, detect from finish_reason
        if choice.get("finish_reason") == "tool_calls" and "tool_detected" not in t:
            t["tool_detected"] = time.monotonic()
            if tool_thread is None:
                tool_thread = threading.Thread(target=_exec_tool, daemon=True)
                tool_thread.start()

    t["stream_done"] = time.monotonic()

    # Wait for tool to finish (may already be done if dispatch was early)
    if tool_thread is not None:
        tool_thread.join()
    else:
        # No tool call detected at all — run synchronously as fallback
        tool_times["started"] = time.monotonic()
        simulate_tool(tool_latency_s)
        tool_times["finished"] = time.monotonic()

    t["tool_started"] = tool_times.get("started", t["stream_done"])
    t["tool_finished"] = tool_times.get("finished", t["stream_done"])

    resp.close()
    return t


# ---------------------------------------------------------------------------
# Row formatting
# ---------------------------------------------------------------------------

COLUMNS = [
    "run",
    "variant",
    "request_sent_ms",
    "first_token_ms",
    "tool_detected_ms",
    "tool_started_ms",
    "tool_finished_ms",
    "stream_done_ms",
    "total_wall_ms",
]


def _ms(t: dict, key: str) -> str:
    if key not in t:
        return ""
    return f"{(t[key] - t['request_sent']) * 1000:.2f}"


def make_row(run_idx: int, variant: str, t: dict) -> dict:
    total = max(t.get("tool_finished", 0), t.get("stream_done", 0)) - t["request_sent"]
    return {
        "run": str(run_idx),
        "variant": variant,
        "request_sent_ms": "0.00",
        "first_token_ms": _ms(t, "first_token"),
        "tool_detected_ms": _ms(t, "tool_detected"),
        "tool_started_ms": _ms(t, "tool_started"),
        "tool_finished_ms": _ms(t, "tool_finished"),
        "stream_done_ms": _ms(t, "stream_done"),
        "total_wall_ms": f"{total * 1000:.2f}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Harness dispatch experiment")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tool-latency-ms", type=float, default=50)
    parser.add_argument("--output", default="-", help="CSV output path (- for stdout)")
    parser.add_argument("--jsonl", default=None, help="Raw JSONL output path")
    args = parser.parse_args()

    tool_latency_s = args.tool_latency_ms / 1000.0
    rows: list[dict] = []
    raw_records: list[dict] = []
    wall_times: dict[str, list[float]] = {"A_buffered": [], "B_dispatch": []}

    # --- Variant A ---
    print(f"=== Variant A (Buffered) x{args.runs} ===", file=sys.stderr)
    for i in range(args.runs):
        t = run_buffered(args.url, args.model, tool_latency_s)
        row = make_row(i + 1, "A_buffered", t)
        rows.append(row)
        wall = float(row["total_wall_ms"])
        wall_times["A_buffered"].append(wall)
        raw_records.append(
            {
                "run": i + 1,
                "variant": "A_buffered",
                "timestamps": {k: v for k, v in t.items()},
            }
        )
        print(f"  run {i+1:>3d}: {wall:8.2f} ms", file=sys.stderr)

    # --- Variant B ---
    print(f"\n=== Variant B (Dispatch-aware) x{args.runs} ===", file=sys.stderr)
    for i in range(args.runs):
        t = run_dispatch(args.url, args.model, tool_latency_s)
        row = make_row(i + 1, "B_dispatch", t)
        rows.append(row)
        wall = float(row["total_wall_ms"])
        wall_times["B_dispatch"].append(wall)
        raw_records.append(
            {
                "run": i + 1,
                "variant": "B_dispatch",
                "timestamps": {k: v for k, v in t.items()},
            }
        )
        print(f"  run {i+1:>3d}: {wall:8.2f} ms", file=sys.stderr)

    # --- CSV output ---
    out_fh: io.TextIOBase
    if args.output == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(args.output, "w", newline="")

    writer = csv.DictWriter(out_fh, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

    if args.output != "-":
        out_fh.close()

    # --- JSONL output ---
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for rec in raw_records:
                f.write(json.dumps(rec) + "\n")

    # --- Summary ---
    print("\n=== Summary ===", file=sys.stderr)
    for variant, times in wall_times.items():
        if not times:
            continue
        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        print(
            f"  {variant:>15s}: mean={mean:8.2f} ms  stdev={stdev:7.2f} ms  (n={len(times)})",
            file=sys.stderr,
        )

    a_mean = (
        statistics.mean(wall_times["A_buffered"]) if wall_times["A_buffered"] else 0
    )
    b_mean = (
        statistics.mean(wall_times["B_dispatch"]) if wall_times["B_dispatch"] else 0
    )
    if a_mean > 0:
        delta = a_mean - b_mean
        pct = (delta / a_mean) * 100
        print(f"\n  Delta (A - B): {delta:+.2f} ms ({pct:+.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
