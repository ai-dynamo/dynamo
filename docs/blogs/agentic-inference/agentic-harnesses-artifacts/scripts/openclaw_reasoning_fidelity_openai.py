# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""OpenClaw reasoning fidelity experiment (OpenAI format).

Adapted from openclaw_reasoning_fidelity.py to use OpenAI Chat Completions API
instead of Anthropic Messages API (for use with standalone vLLM).

Tests that the model correctly produces reasoning content in multi-turn
conversations. Nemotron-3-Super uses <think>...</think> tags natively.

Measures:
  - Whether reasoning/thinking content is present
  - Whether reasoning appears BEFORE content (correct ordering)
  - Token counts for reasoning vs content
  - TTFT for turns with and without reasoning
  - Tool calling fidelity

Uses OpenAI Chat Completions API with streaming.

Usage:
    python3 openclaw_reasoning_fidelity_openai.py \\
        --url http://localhost:8000 \\
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
        --runs 5 \\
        --jsonl ../openclaw-experiments/reasoning-fidelity.jsonl
"""

import argparse
import json
import re
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
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '42 * 17')",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

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


def parse_openai_stream(resp) -> dict:
    """Parse a streaming OpenAI Chat Completions response.

    Detects <think>...</think> tags in the content for reasoning extraction.
    Also detects tool_calls in the stream.
    """
    t0 = time.monotonic()
    ttft = None

    full_content = ""
    tool_calls = {}  # index -> {id, name, arguments}
    block_order = []  # track what we see: "thinking", "text", "tool_use"

    input_tokens = 0
    output_tokens = 0

    thinking_started = False

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
        choices = event.get("choices", [])
        usage = event.get("usage")

        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            # Detect reasoning content
            if delta.get("reasoning_content"):
                # vLLM with reasoning parser exposes reasoning_content directly
                if not thinking_started:
                    block_order.append("thinking")
                    thinking_started = True
                if ttft is None:
                    ttft = now_ms

            if content:
                if ttft is None:
                    ttft = now_ms
                full_content += content

            # Detect tool calls
            tc_list = delta.get("tool_calls", [])
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "arguments": "",
                    }
                    block_order.append("tool_use")
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls[idx]["arguments"] += fn["arguments"]

        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

    resp.close()
    total_ms = (time.monotonic() - t0) * 1000

    # Parse <think>...</think> tags from content
    reasoning_text = ""
    content_text = full_content
    think_match = re.search(r"<think>(.*?)</think>", full_content, re.DOTALL)
    if think_match:
        reasoning_text = think_match.group(1).strip()
        # Remove think tags from content
        content_text = re.sub(
            r"<think>.*?</think>\s*", "", full_content, flags=re.DOTALL
        ).strip()
        if "thinking" not in block_order:
            block_order.insert(0, "thinking")
        if "text" not in block_order:
            block_order.append("text")
    elif thinking_started:
        # reasoning_content was in delta directly
        reasoning_text = "[via reasoning_content field]"

    if content_text and "text" not in block_order:
        block_order.append("text")

    has_reasoning = len(reasoning_text) > 0 or thinking_started
    has_content = len(content_text) > 0
    has_tool_use = len(tool_calls) > 0

    reasoning_before_content = False
    if "thinking" in block_order and "text" in block_order:
        reasoning_before_content = block_order.index("thinking") < block_order.index(
            "text"
        )

    return {
        "ttft_ms": round(ttft, 2) if ttft else None,
        "total_ms": round(total_ms, 2),
        "reasoning_text": reasoning_text,
        "reasoning_len": len(reasoning_text),
        "content_text": content_text,
        "content_len": len(content_text),
        "has_reasoning": has_reasoning,
        "has_content": has_content,
        "has_tool_use": has_tool_use,
        "tool_use_count": len(tool_calls),
        "tool_names": [tc["name"] for tc in tool_calls.values()],
        "reasoning_before_content": reasoning_before_content,
        "block_order": block_order,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_scenario(url: str, model: str, scenario: dict, run_idx: int) -> list[dict]:
    """Run a single scenario (multi-turn) and return per-turn results."""
    results = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn_idx, turn in enumerate(scenario["turns"]):
        messages.append({"role": "user", "content": turn["content"]})

        body = {
            "model": model,
            "max_tokens": 1024,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": messages,
        }

        # Include tools if this turn expects tool use
        if turn.get("expect_tool"):
            body["tools"] = TOOL_DEFS

        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        parsed = parse_openai_stream(resp)

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
                else True
            ),
        }
        results.append(result)

        # Build assistant message for conversation history
        assistant_content = (
            parsed["content_text"][:500]
            if parsed["content_text"]
            else "I'll analyze this step by step."
        )

        if parsed["has_tool_use"] and parsed["tool_names"]:
            # Assistant message with tool_calls
            tool_call_objs = []
            for i, tn in enumerate(parsed["tool_names"]):
                tool_call_objs.append(
                    {
                        "id": f"call_turn{turn_idx}_{i}",
                        "type": "function",
                        "function": {"name": tn, "arguments": '{"expression": "42"}'},
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content if assistant_content else None,
                    "tool_calls": tool_call_objs,
                }
            )
            # Add tool results
            for tc_obj in tool_call_objs:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_obj["id"],
                        "content": "42.0",
                    }
                )
        else:
            messages.append({"role": "assistant", "content": assistant_content})

        ok_marker = (
            "OK"
            if all(
                [
                    result["reasoning_fidelity_ok"],
                    result["tool_fidelity_ok"],
                    result["order_fidelity_ok"],
                ]
            )
            else "FAIL"
        )

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
        description="OpenClaw reasoning fidelity experiment (OpenAI format)"
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
        if ttfts:
            print(
                f"\n  Reasoning turns TTFT: mean={sum(ttfts)/len(ttfts):.1f}ms (n={len(ttfts)})",
                file=sys.stderr,
            )
    if tool_turns:
        ttfts = [r["ttft_ms"] for r in tool_turns if r["ttft_ms"]]
        if ttfts:
            print(
                f"  Tool turns TTFT:      mean={sum(ttfts)/len(ttfts):.1f}ms (n={len(ttfts)})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
