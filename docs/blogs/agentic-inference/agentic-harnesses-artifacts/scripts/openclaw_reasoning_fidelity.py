# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""OpenClaw reasoning fidelity experiment.

Tests that Dynamo correctly preserves reasoning_content across multi-turn
OpenClaw conversations. In OpenClaw's long-lived chat pattern:
  1. User sends a reasoning-heavy prompt (triggers <think> tags)
  2. Response includes reasoning_content in content_block_delta events
  3. Follow-up turn references the previous reasoning
  4. Tool-using turn follows to verify reasoning + tools coexist

Measures:
  - Whether reasoning_content is present and non-empty
  - Whether reasoning appears BEFORE content (correct ordering)
  - Token counts for reasoning vs content
  - TTFT for turns with and without reasoning

Uses Anthropic Messages API format with streaming.

Usage:
    python3 openclaw_reasoning_fidelity.py \\
        --url http://localhost:8000 \\
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
        --runs 5 \\
        --jsonl ../openclaw-experiments/reasoning-fidelity.jsonl
"""

import argparse
import json
import sys
import time

import requests

# ---------------------------------------------------------------------------
# System prompt and conversation scenarios
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI assistant with reasoning capabilities. When solving complex \
problems, think step by step in your reasoning before providing the answer. \
You have access to a calculator tool for arithmetic.

Always show your work for math problems. For multi-step problems, break down \
each step clearly in your thinking."""

TOOL_DEFS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '42 * 17')",
                }
            },
            "required": ["expression"],
        },
    }
]

# Scenario: reasoning-heavy prompt, then follow-up, then tool use
SCENARIOS = [
    {
        "name": "tsp_then_tool",
        "description": "Reasoning about TSP, then tool-using follow-up",
        "turns": [
            {
                "role": "user",
                "content": (
                    "Think step by step about the traveling salesman problem "
                    "for 5 cities arranged in a pentagon. What is the optimal "
                    "tour length if each side of the pentagon has length 1? "
                    "Consider at least 3 different tours before concluding."
                ),
                "expect_reasoning": True,
                "expect_tool": False,
            },
            {
                "role": "user",
                "content": (
                    "Good analysis. Now use the calculator to compute the exact "
                    "length of the tour that cuts across the pentagon: "
                    "1 + 1 + sqrt(1^2 + 1^2 - 2*1*1*cos(144*pi/180)) + "
                    "sqrt(1^2 + 1^2 - 2*1*1*cos(72*pi/180)) + 1"
                ),
                "expect_reasoning": True,
                "expect_tool": True,
            },
        ],
    },
    {
        "name": "code_review_reason",
        "description": "Reason about code correctness, then explain",
        "turns": [
            {
                "role": "user",
                "content": (
                    "Think carefully: is this Python function correct?\n\n"
                    "```python\n"
                    "def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr)\n"
                    "    while lo < hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        elif arr[mid] > target:\n"
                    "            hi = mid\n"
                    "        else:\n"
                    "            return mid\n"
                    "    return -1\n"
                    "```\n\n"
                    "Analyze edge cases: empty array, single element, target "
                    "not present, target at boundaries."
                ),
                "expect_reasoning": True,
                "expect_tool": False,
            },
            {
                "role": "user",
                "content": (
                    "Use the calculator to verify: if the array has 1000 "
                    "elements, what is the maximum number of comparisons "
                    "binary search makes? Compute ceil(log2(1000))."
                ),
                "expect_reasoning": True,
                "expect_tool": True,
            },
        ],
    },
    {
        "name": "probability_chain",
        "description": "Multi-step probability reasoning with calculation",
        "turns": [
            {
                "role": "user",
                "content": (
                    "Think step by step: A bag has 5 red balls and 3 blue balls. "
                    "You draw 3 balls without replacement. What is the probability "
                    "of getting exactly 2 red balls? Show your combinatorial reasoning."
                ),
                "expect_reasoning": True,
                "expect_tool": False,
            },
            {
                "role": "user",
                "content": (
                    "Now use the calculator to verify your answer. Compute "
                    "C(5,2) * C(3,1) / C(8,3) where C(n,k) = n! / (k! * (n-k)!)"
                ),
                "expect_reasoning": True,
                "expect_tool": True,
            },
        ],
    },
]


def parse_anthropic_stream(resp) -> dict:
    """Parse a streaming Anthropic Messages API response.

    Returns timing, content blocks, reasoning blocks, and tool_use blocks.
    """
    t0 = time.monotonic()
    ttft = None

    reasoning_chunks = []
    content_chunks = []
    tool_use_blocks = []
    event_timeline = []  # (relative_ms, event_type, brief)

    # Track block types in order
    block_order = []  # list of (block_type, block_index)
    current_block_type = None
    current_block_idx = None

    input_tokens = 0
    output_tokens = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        now_ms = (time.monotonic() - t0) * 1000
        etype = event.get("type", "")

        if etype == "content_block_start":
            cb = event.get("content_block", {})
            block_type = cb.get("type", "unknown")
            idx = event.get("index", -1)
            current_block_type = block_type
            current_block_idx = idx
            block_order.append((block_type, idx))
            event_timeline.append((now_ms, "block_start", f"{block_type}[{idx}]"))

            if block_type == "tool_use":
                tool_use_blocks.append({
                    "index": idx,
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "input_json": "",
                    "start_ms": now_ms,
                })

        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")

            if ttft is None:
                ttft = now_ms

            if delta_type == "thinking_delta":
                reasoning_chunks.append(delta.get("thinking", ""))
            elif delta_type == "text_delta":
                content_chunks.append(delta.get("text", ""))
            elif delta_type == "input_json_delta":
                if tool_use_blocks:
                    tool_use_blocks[-1]["input_json"] += delta.get(
                        "partial_json", ""
                    )

        elif etype == "content_block_stop":
            event_timeline.append(
                (now_ms, "block_stop", f"{current_block_type}[{current_block_idx}]")
            )

        elif etype == "message_start":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)

        elif etype == "message_delta":
            usage = event.get("usage", {})
            output_tokens = usage.get("output_tokens", 0)

    resp.close()
    total_ms = (time.monotonic() - t0) * 1000

    reasoning_text = "".join(reasoning_chunks)
    content_text = "".join(content_chunks)

    # Determine block ordering
    block_types_in_order = [bt for bt, _ in block_order]
    reasoning_before_content = False
    if "thinking" in block_types_in_order and "text" in block_types_in_order:
        reasoning_before_content = block_types_in_order.index(
            "thinking"
        ) < block_types_in_order.index("text")

    return {
        "ttft_ms": round(ttft, 2) if ttft else None,
        "total_ms": round(total_ms, 2),
        "reasoning_text": reasoning_text,
        "reasoning_len": len(reasoning_text),
        "content_text": content_text,
        "content_len": len(content_text),
        "has_reasoning": len(reasoning_text) > 0,
        "has_content": len(content_text) > 0,
        "has_tool_use": len(tool_use_blocks) > 0,
        "tool_use_count": len(tool_use_blocks),
        "tool_names": [t["name"] for t in tool_use_blocks],
        "reasoning_before_content": reasoning_before_content,
        "block_order": block_types_in_order,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "event_timeline": event_timeline[:20],  # Truncate for output
    }


def run_scenario(url: str, model: str, scenario: dict, run_idx: int) -> list[dict]:
    """Run a single scenario (multi-turn) and return per-turn results."""
    results = []
    messages = []

    for turn_idx, turn in enumerate(scenario["turns"]):
        messages.append({"role": "user", "content": turn["content"]})

        body = {
            "model": model,
            "max_tokens": 1024,
            "stream": True,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }

        # Include tools if this turn expects tool use
        if turn.get("expect_tool"):
            body["tools"] = TOOL_DEFS

        resp = requests.post(
            f"{url}/v1/messages",
            json=body,
            headers={"Content-Type": "application/json", "x-api-key": "dummy"},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        parsed = parse_anthropic_stream(resp)

        result = {
            "run": run_idx + 1,
            "scenario": scenario["name"],
            "turn": turn_idx + 1,
            "expected_reasoning": turn["expect_reasoning"],
            "expected_tool": turn["expect_tool"],
            "ttft_ms": parsed["ttft_ms"],
            "total_ms": parsed["total_ms"],
            "has_reasoning": parsed["has_reasoning"],
            "reasoning_len": parsed["reasoning_len"],
            "has_content": parsed["has_content"],
            "content_len": parsed["content_len"],
            "has_tool_use": parsed["has_tool_use"],
            "tool_use_count": parsed["tool_use_count"],
            "tool_names": parsed["tool_names"],
            "reasoning_before_content": parsed["reasoning_before_content"],
            "block_order": parsed["block_order"],
            "input_tokens": parsed["input_tokens"],
            "output_tokens": parsed["output_tokens"],
            # Fidelity checks
            "reasoning_fidelity_ok": (
                parsed["has_reasoning"] == turn["expect_reasoning"]
            ),
            "tool_fidelity_ok": parsed["has_tool_use"] == turn["expect_tool"],
            "order_fidelity_ok": (
                parsed["reasoning_before_content"]
                if parsed["has_reasoning"] and parsed["has_content"]
                else True  # No ordering issue if one is missing
            ),
        }
        results.append(result)

        # Build assistant message for conversation history
        assistant_content = parsed["content_text"][:500] if parsed["content_text"] else "I'll analyze this step by step."
        messages.append({"role": "assistant", "content": assistant_content})

        # If there were tool uses, simulate tool results
        if parsed["has_tool_use"]:
            for tool in parsed.get("tool_names", []):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"tool_{turn_idx}",
                            "content": "42.0",  # Simulated result
                        }
                    ],
                })

        ok_marker = "OK" if all([
            result["reasoning_fidelity_ok"],
            result["tool_fidelity_ok"],
            result["order_fidelity_ok"],
        ]) else "FAIL"

        print(
            f"    Turn {turn_idx+1}: TTFT={parsed['ttft_ms']:>8.1f}ms "
            f"reasoning={parsed['has_reasoning']} "
            f"tool={parsed['has_tool_use']} "
            f"order={'R>C' if parsed['reasoning_before_content'] else 'C>R'} "
            f"[{ok_marker}]",
            file=sys.stderr,
        )

        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw reasoning fidelity experiment"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--jsonl", required=True, help="Output JSONL path")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only this scenario (by name). Omit for all.",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["name"] == args.scenario]
        if not scenarios:
            print(
                f"Unknown scenario: {args.scenario}. "
                f"Available: {[s['name'] for s in SCENARIOS]}",
                file=sys.stderr,
            )
            sys.exit(1)

    all_results = []

    for run_idx in range(args.runs):
        for scenario in scenarios:
            print(
                f"Run {run_idx+1}/{args.runs}, Scenario: {scenario['name']}",
                file=sys.stderr,
            )
            results = run_scenario(args.url, args.model, scenario, run_idx)
            all_results.extend(results)
            time.sleep(1)

    # Write JSONL
    with open(args.jsonl, "w") as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(all_results)} records to {args.jsonl}", file=sys.stderr)

    # Print summary
    print("\n=== Fidelity Summary ===", file=sys.stderr)
    total = len(all_results)
    reasoning_ok = sum(1 for r in all_results if r["reasoning_fidelity_ok"])
    tool_ok = sum(1 for r in all_results if r["tool_fidelity_ok"])
    order_ok = sum(1 for r in all_results if r["order_fidelity_ok"])

    print(f"  Reasoning presence: {reasoning_ok}/{total} correct", file=sys.stderr)
    print(f"  Tool use presence:  {tool_ok}/{total} correct", file=sys.stderr)
    print(f"  Block ordering:     {order_ok}/{total} correct", file=sys.stderr)

    # TTFT summary by turn type
    reasoning_turns = [r for r in all_results if r["has_reasoning"]]
    tool_turns = [r for r in all_results if r["has_tool_use"]]

    if reasoning_turns:
        ttfts = [r["ttft_ms"] for r in reasoning_turns if r["ttft_ms"]]
        print(
            f"\n  Reasoning turns TTFT: mean={sum(ttfts)/len(ttfts):.1f}ms (n={len(ttfts)})",
            file=sys.stderr,
        )
    if tool_turns:
        ttfts = [r["ttft_ms"] for r in tool_turns if r["ttft_ms"]]
        print(
            f"  Tool turns TTFT:      mean={sum(ttfts)/len(ttfts):.1f}ms (n={len(ttfts)})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
