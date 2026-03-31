# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""OpenClaw streaming dispatch multi-turn experiment.

Simulates OpenClaw's multi-turn tool-using pattern where a user asks questions
that trigger tool calls across a conversation. Measures:

1. Whether tool_call_dispatch events arrive before finish_reason (streaming dispatch)
2. Total wall time comparison: buffered vs dispatch-aware harness
3. Multi-turn accumulation: how dispatch savings compound over N turns

OpenClaw-specific: uses Anthropic Messages API (not OpenAI format), and the
conversation history grows naturally across turns (like a real chat session).

Two harness variants tested per scenario:
  A (Buffered):  Wait for message_stop, then parse tool_use blocks, then execute
  B (Dispatch):  Start tool execution on content_block_stop for tool_use blocks

Uses simulated tool execution with configurable latency to make overlap visible.

Usage:
    python3 openclaw_dispatch_multiturn.py \\
        --url http://localhost:8000 \\
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
        --tool-latency-ms 50 --runs 5 \\
        --jsonl ../openclaw-experiments/dispatch-multiturn.jsonl
"""

import argparse
import json
import sys
import threading
import time

import requests

# ---------------------------------------------------------------------------
# System prompt and tools (Anthropic Messages format)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI assistant with access to tools. When the user asks a math \
question, always use the calculator tool. When they ask to search, use the \
web_search tool. You can use multiple tools in a single response if needed."""

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Returns the numeric result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '42 * 17 + 3')",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
]

# Multi-turn conversation where each turn triggers tool use
CONVERSATION_TURNS = [
    "What is 42 * 17?",
    "Now compute sqrt(714) to 3 decimal places.",
    "What is 2^10 + 3^7?",
    "Compute the factorial of 12.",
    "What is sin(pi/4) * cos(pi/3)?",
]


def simulate_tool(name: str, latency_s: float) -> str:
    """Simulate tool execution with configurable latency."""
    time.sleep(latency_s)
    results = {
        "calculator": '{"result": 714}',
        "web_search": '{"results": [{"title": "Example", "url": "https://example.com"}]}',
    }
    return results.get(name, '{"result": "ok"}')


# ---------------------------------------------------------------------------
# Anthropic Messages API SSE parsing
# ---------------------------------------------------------------------------


def parse_anthropic_sse(resp):
    """Yield (event_type, parsed_data) from Anthropic SSE stream."""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            # Anthropic format: event line followed by data line
            continue
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                yield ("done", {"done": True})
            else:
                try:
                    data = json.loads(raw)
                    yield (data.get("type", "unknown"), data)
                except json.JSONDecodeError:
                    pass


# ---------------------------------------------------------------------------
# Variant A: Buffered (wait for message_stop)
# ---------------------------------------------------------------------------


def run_buffered_turn(
    url: str,
    model: str,
    messages: list[dict],
    tool_latency_s: float,
) -> dict:
    """Run one turn using buffered strategy. Returns timing dict."""
    body = {
        "model": model,
        "max_tokens": 512,
        "stream": True,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOLS,
    }

    t = {"request_sent": time.monotonic()}

    resp = requests.post(
        f"{url}/v1/messages",
        json=body,
        headers={"Content-Type": "application/json", "x-api-key": "dummy"},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    first_delta_seen = False
    tool_use_blocks = []  # Completed tool_use blocks
    current_tool = None
    content_text = ""

    for etype, data in parse_anthropic_sse(resp):
        if etype == "done":
            break

        now = time.monotonic()

        if etype == "content_block_start":
            cb = data.get("content_block", {})
            if cb.get("type") == "tool_use":
                current_tool = {
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "input_json": "",
                }

        elif etype == "content_block_delta":
            if not first_delta_seen:
                t["first_token"] = now
                first_delta_seen = True

            delta = data.get("delta", {})
            if delta.get("type") == "input_json_delta" and current_tool:
                current_tool["input_json"] += delta.get("partial_json", "")
            elif delta.get("type") == "text_delta":
                content_text += delta.get("text", "")

        elif etype == "content_block_stop":
            if current_tool:
                tool_use_blocks.append(current_tool)
                current_tool = None

        elif etype == "message_stop":
            t["stream_done"] = now

    resp.close()

    if "stream_done" not in t:
        t["stream_done"] = time.monotonic()

    # Execute tools AFTER stream completes (buffered strategy)
    tool_results = []
    t["tools_started"] = time.monotonic()
    for tool in tool_use_blocks:
        result = simulate_tool(tool["name"], tool_latency_s)
        tool_results.append({
            "tool_use_id": tool["id"],
            "name": tool["name"],
            "result": result,
        })
    t["tools_finished"] = time.monotonic()

    return {
        "timestamps": t,
        "tool_count": len(tool_use_blocks),
        "tool_names": [tb["name"] for tb in tool_use_blocks],
        "content_preview": content_text[:100],
        "tool_results": tool_results,
    }


# ---------------------------------------------------------------------------
# Variant B: Dispatch-aware (start tool on content_block_stop)
# ---------------------------------------------------------------------------


def run_dispatch_turn(
    url: str,
    model: str,
    messages: list[dict],
    tool_latency_s: float,
) -> dict:
    """Run one turn using dispatch-aware strategy. Returns timing dict."""
    body = {
        "model": model,
        "max_tokens": 512,
        "stream": True,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOLS,
    }

    t = {"request_sent": time.monotonic()}

    resp = requests.post(
        f"{url}/v1/messages",
        json=body,
        headers={"Content-Type": "application/json", "x-api-key": "dummy"},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    first_delta_seen = False
    current_tool = None
    content_text = ""
    tool_threads = []
    tool_times = {}  # tool_id -> {started, finished}
    tool_results_lock = threading.Lock()
    tool_results = []

    def _exec_tool(tool_id: str, tool_name: str):
        tool_times[tool_id] = {"started": time.monotonic()}
        result = simulate_tool(tool_name, tool_latency_s)
        tool_times[tool_id]["finished"] = time.monotonic()
        with tool_results_lock:
            tool_results.append({
                "tool_use_id": tool_id,
                "name": tool_name,
                "result": result,
            })

    for etype, data in parse_anthropic_sse(resp):
        if etype == "done":
            break

        now = time.monotonic()

        if etype == "content_block_start":
            cb = data.get("content_block", {})
            if cb.get("type") == "tool_use":
                current_tool = {
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "input_json": "",
                }

        elif etype == "content_block_delta":
            if not first_delta_seen:
                t["first_token"] = now
                first_delta_seen = True

            delta = data.get("delta", {})
            if delta.get("type") == "input_json_delta" and current_tool:
                current_tool["input_json"] += delta.get("partial_json", "")
            elif delta.get("type") == "text_delta":
                content_text += delta.get("text", "")

        elif etype == "content_block_stop":
            if current_tool:
                # DISPATCH: start tool execution immediately
                t.setdefault("first_tool_dispatched", now)
                thread = threading.Thread(
                    target=_exec_tool,
                    args=(current_tool["id"], current_tool["name"]),
                    daemon=True,
                )
                tool_threads.append(thread)
                thread.start()
                current_tool = None

        elif etype == "message_stop":
            t["stream_done"] = now

    resp.close()

    if "stream_done" not in t:
        t["stream_done"] = time.monotonic()

    # Wait for all dispatched tools to finish
    for thread in tool_threads:
        thread.join()

    t["tools_finished"] = time.monotonic()

    # Compute earliest tool start
    if tool_times:
        earliest = min(v["started"] for v in tool_times.values())
        latest = max(v["finished"] for v in tool_times.values())
        t["tools_started"] = earliest
        t["tools_finished"] = latest

    return {
        "timestamps": t,
        "tool_count": len(tool_threads),
        "tool_names": [tr["name"] for tr in tool_results],
        "content_preview": content_text[:100],
        "tool_results": tool_results,
    }


# ---------------------------------------------------------------------------
# Row formatting
# ---------------------------------------------------------------------------


def make_row(
    run_idx: int,
    turn_idx: int,
    variant: str,
    result: dict,
) -> dict:
    t = result["timestamps"]
    t0 = t["request_sent"]

    def ms(key):
        return round((t[key] - t0) * 1000, 2) if key in t else None

    total = max(t.get("tools_finished", 0), t.get("stream_done", 0)) - t0

    return {
        "run": run_idx + 1,
        "turn": turn_idx + 1,
        "variant": variant,
        "first_token_ms": ms("first_token"),
        "stream_done_ms": ms("stream_done"),
        "first_tool_dispatched_ms": ms("first_tool_dispatched"),
        "tools_started_ms": ms("tools_started"),
        "tools_finished_ms": ms("tools_finished"),
        "total_wall_ms": round(total * 1000, 2),
        "tool_count": result["tool_count"],
        "tool_names": result["tool_names"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw streaming dispatch multi-turn experiment"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--tool-latency-ms", type=float, default=50)
    parser.add_argument("--jsonl", required=True, help="Output JSONL path")
    args = parser.parse_args()

    tool_latency_s = args.tool_latency_ms / 1000.0
    n_turns = min(args.turns, len(CONVERSATION_TURNS))
    all_rows = []

    for variant_name, run_fn in [
        ("A_buffered", run_buffered_turn),
        ("B_dispatch", run_dispatch_turn),
    ]:
        print(f"\n=== Variant {variant_name} ===", file=sys.stderr)

        for run_idx in range(args.runs):
            print(f"  Run {run_idx+1}/{args.runs}", file=sys.stderr)
            messages = []

            for turn_idx in range(n_turns):
                user_msg = CONVERSATION_TURNS[turn_idx]
                messages.append({"role": "user", "content": user_msg})

                result = run_fn(
                    args.url, args.model, messages, tool_latency_s
                )

                row = make_row(run_idx, turn_idx, variant_name, result)
                all_rows.append(row)

                # Add assistant response + tool results to history for next turn
                assistant_content = result["content_preview"] or "Let me compute that."
                messages.append({"role": "assistant", "content": assistant_content})

                # Add simulated tool results
                for tr in result.get("tool_results", []):
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr["tool_use_id"],
                                "content": tr["result"],
                            }
                        ],
                    })

                print(
                    f"    Turn {turn_idx+1}: wall={row['total_wall_ms']:>8.2f}ms "
                    f"tools={row['tool_count']} "
                    f"dispatch@{row.get('first_tool_dispatched_ms', 'N/A')}",
                    file=sys.stderr,
                )

                time.sleep(0.3)

    # Write JSONL
    with open(args.jsonl, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(all_rows)} records to {args.jsonl}", file=sys.stderr)

    # Summary
    print("\n=== Summary (mean total_wall_ms per turn) ===", file=sys.stderr)
    import statistics

    for variant in ["A_buffered", "B_dispatch"]:
        rows = [r for r in all_rows if r["variant"] == variant]
        walls = [r["total_wall_ms"] for r in rows]
        if walls:
            mean = statistics.mean(walls)
            stdev = statistics.stdev(walls) if len(walls) > 1 else 0
            print(
                f"  {variant:>15s}: mean={mean:>8.2f}ms  stdev={stdev:>7.2f}ms  (n={len(walls)})",
                file=sys.stderr,
            )

    # Per-turn comparison
    print("\n=== Per-turn comparison ===", file=sys.stderr)
    for turn in range(1, n_turns + 1):
        a_rows = [
            r for r in all_rows if r["variant"] == "A_buffered" and r["turn"] == turn
        ]
        b_rows = [
            r for r in all_rows if r["variant"] == "B_dispatch" and r["turn"] == turn
        ]
        a_mean = statistics.mean([r["total_wall_ms"] for r in a_rows]) if a_rows else 0
        b_mean = statistics.mean([r["total_wall_ms"] for r in b_rows]) if b_rows else 0
        delta = a_mean - b_mean
        print(
            f"  Turn {turn}: A={a_mean:>8.2f}ms  B={b_mean:>8.2f}ms  delta={delta:>+8.2f}ms",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
