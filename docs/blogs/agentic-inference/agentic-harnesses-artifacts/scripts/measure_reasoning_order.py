# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Compare TTFT for segmented vs flattened reasoning reconstruction.

Sends the same multi-turn conversation twice in sequence:
1. First request establishes KV cache
2. Second request (with a follow-up user message) measures TTFT

Hypothesis: segmented reasoning preserves the correct token order,
so the second request gets better cache reuse (lower TTFT) when
the reconstruction matches the original generation order.

Usage:
    python3 measure_reasoning_order.py --url http://localhost:8000 --model MODEL --runs N
"""

import argparse
import csv
import json
import sys
import time

import requests

MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
URL = "http://localhost:8000"

SYSTEM = "You have access to a calculator. Use it for all math."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate math",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def make_warmup_request(model: str, reasoning_content) -> dict:
    """First turn: establish the conversation in KV cache."""
    return {
        "model": model,
        "max_tokens": 200,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is sqrt(144), then multiply by 7?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_content,
                "tool_calls": [
                    {
                        "id": "call_01",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"sqrt(144)"}',
                        },
                    },
                    {
                        "id": "call_02",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"12 * 7"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_01", "content": "12"},
            {"role": "tool", "tool_call_id": "call_02", "content": "84"},
            {"role": "user", "content": "Now add 16 to that result."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }


def make_followup_request(
    model: str, reasoning_content, prev_response_reasoning: str
) -> dict:
    """Second turn: extends the conversation — TTFT depends on cache reuse of prefix."""
    return {
        "model": model,
        "max_tokens": 200,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is sqrt(144), then multiply by 7?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_content,
                "tool_calls": [
                    {
                        "id": "call_01",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"sqrt(144)"}',
                        },
                    },
                    {
                        "id": "call_02",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"12 * 7"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_01", "content": "12"},
            {"role": "tool", "tool_call_id": "call_02", "content": "84"},
            {"role": "user", "content": "Now add 16 to that result."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": prev_response_reasoning,
                "tool_calls": [
                    {
                        "id": "call_03",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"84 + 16"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_03", "content": "100"},
            {"role": "user", "content": "Now divide by 5."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    }


# Reasoning content variants
SEGMENTED = [
    "The user wants sqrt(144) first. sqrt(144) = 12. I should use calculator.",
    "Got 12. Now multiply by 7.",
    "",
]

FLATTENED = "The user wants sqrt(144) first. sqrt(144) = 12. I should use calculator.\nGot 12. Now multiply by 7.\n"


def measure_ttft(url: str, body: dict) -> float:
    """Send streaming request and return TTFT in ms."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=30,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            break
        try:
            data = json.loads(raw)
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                ):
                    ttft = (time.monotonic() - t0) * 1000
                    resp.close()
                    return ttft
        except json.JSONDecodeError:
            pass

    resp.close()
    return (time.monotonic() - t0) * 1000


def run_experiment(url: str, model: str, n_runs: int):
    results = []

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}...", file=sys.stderr)

        # Segmented (correct order)
        warmup = make_warmup_request(model, SEGMENTED)
        ttft_warmup_seg = measure_ttft(url, warmup)
        time.sleep(0.3)

        followup = make_followup_request(
            model, SEGMENTED, "We need to compute 84 + 16 = 100. Use calculator.\n"
        )
        ttft_followup_seg = measure_ttft(url, followup)
        time.sleep(0.3)

        # Flattened (incorrect order)
        warmup = make_warmup_request(model, FLATTENED)
        ttft_warmup_flat = measure_ttft(url, warmup)
        time.sleep(0.3)

        followup = make_followup_request(
            model, FLATTENED, "We need to compute 84 + 16 = 100. Use calculator.\n"
        )
        ttft_followup_flat = measure_ttft(url, followup)
        time.sleep(0.5)

        row = {
            "run": i + 1,
            "warmup_segmented_ms": round(ttft_warmup_seg, 2),
            "followup_segmented_ms": round(ttft_followup_seg, 2),
            "warmup_flattened_ms": round(ttft_warmup_flat, 2),
            "followup_flattened_ms": round(ttft_followup_flat, 2),
        }
        results.append(row)
        print(
            f"  seg: {ttft_warmup_seg:.1f}/{ttft_followup_seg:.1f}ms  flat: {ttft_warmup_flat:.1f}/{ttft_followup_flat:.1f}ms",
            file=sys.stderr,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="-")
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args()

    results = run_experiment(args.url, args.model, args.runs)

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    fieldnames = [
        "run",
        "warmup_segmented_ms",
        "followup_segmented_ms",
        "warmup_flattened_ms",
        "followup_flattened_ms",
    ]
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    if args.output != "-":
        out.close()


if __name__ == "__main__":
    main()
