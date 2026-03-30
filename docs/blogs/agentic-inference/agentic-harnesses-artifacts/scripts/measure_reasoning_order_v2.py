# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Compare TTFT for segmented vs flattened reasoning reconstruction (v2).

Loads a multi-turn conversation from a JSON file and constructs two variants:
1. SEGMENTED: reasoning_content kept as an array of strings (correct token order)
2. FLATTENED: reasoning_content concatenated into a single string (incorrect order)

A follow-up user message is appended and TTFT is measured for the model's response.
Runs alternate between variants (A, B, A, B...) to control for transient effects.

Conversation file format:
{
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": ["Let me think...", "2+2 is 4."]
        },
        {"role": "user", "content": "Now multiply by 3."}
    ],
    "followup": "What about dividing by 6?"
}

Usage:
    python3 measure_reasoning_order_v2.py \
        --url http://localhost:8000 \
        --model MODEL \
        --conversation-file conversation.json \
        --runs 10
"""

import argparse
import copy
import csv
import json
import sys
import time

import requests

# Default conversation used when no file is provided
DEFAULT_CONVERSATION = {
    "system": "You have access to a calculator. Use it for all math.",
    "messages": [
        {"role": "user", "content": "What is sqrt(144), then multiply by 7?"},
        {
            "role": "assistant",
            "content": "Let me calculate that step by step.",
            "reasoning_content": [
                "The user wants sqrt(144) first. sqrt(144) = 12. I should use calculator.",
                "Got 12. Now multiply by 7. 12 * 7 = 84.",
            ],
        },
        {"role": "user", "content": "Now add 16 to that result."},
        {
            "role": "assistant",
            "content": "Adding 16 to 84 gives us 100.",
            "reasoning_content": [
                "Previous result was 84. Need to add 16.",
                "84 + 16 = 100.",
            ],
        },
    ],
    "followup": "Now divide by 5.",
}


def flatten_reasoning(messages):
    """Return a deep copy of messages with reasoning_content flattened to a single string."""
    out = copy.deepcopy(messages)
    for msg in out:
        rc = msg.get("reasoning_content")
        if isinstance(rc, list):
            msg["reasoning_content"] = "\n".join(rc)
    return out


def build_body(model, system, messages, followup, max_tokens=200):
    """Build an Anthropic /v1/messages request body."""
    all_messages = list(messages) + [{"role": "user", "content": followup}]
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "system": system,
        "messages": all_messages,
    }


def send_streaming(url, body):
    """Send an Anthropic /v1/messages streaming request.

    Returns (ttft_ms, total_ms) where ttft_ms is the time until the first
    content_block_delta event arrives.
    """
    t0 = time.monotonic()
    ttft = None

    resp = requests.post(
        f"{url}/v1/messages",
        json=body,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "dummy",
        },
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
                if ttft is None and data.get("type") == "content_block_delta":
                    ttft = (time.monotonic() - t0) * 1000
            except json.JSONDecodeError:
                pass

    total = (time.monotonic() - t0) * 1000
    resp.close()

    if ttft is None:
        ttft = total
    return ttft, total


def main():
    parser = argparse.ArgumentParser(
        description="Compare TTFT for segmented vs flattened reasoning (v2)"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--conversation-file",
        default=None,
        help="Path to a JSON file with the conversation. See docstring for format.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measurement runs per variant (default: 10)",
    )
    parser.add_argument("--output", default="-", help="CSV output path (- for stdout)")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL output path")
    args = parser.parse_args()

    if args.conversation_file:
        with open(args.conversation_file) as f:
            conv = json.load(f)
        print(f"Loaded conversation from {args.conversation_file}", file=sys.stderr)
    else:
        conv = DEFAULT_CONVERSATION
        print("Using default (short) conversation", file=sys.stderr)

    system = conv["system"]
    messages_segmented = conv["messages"]
    messages_flattened = flatten_reasoning(conv["messages"])
    followup = conv.get("followup", "Please continue.")

    print(f"System prompt: {len(system)} chars", file=sys.stderr)
    print(f"Messages: {len(messages_segmented)} turns", file=sys.stderr)
    print(f"Follow-up: {followup!r}", file=sys.stderr)

    # Warmup: send each variant once to prime prefix cache
    print("\nWarmup phase...", file=sys.stderr)
    body_seg = build_body(args.model, system, messages_segmented, followup)
    body_flat = build_body(args.model, system, messages_flattened, followup)

    print("  Warming segmented variant...", file=sys.stderr)
    send_streaming(args.url, body_seg)
    time.sleep(0.5)

    print("  Warming flattened variant...", file=sys.stderr)
    send_streaming(args.url, body_flat)
    time.sleep(0.5)

    # Measurement: alternate A, B, A, B...
    results = []
    seg_ttfts = []
    flat_ttfts = []

    print(f"\nMeasuring {args.runs} runs per variant (alternating)...", file=sys.stderr)

    for i in range(args.runs):
        run_num = i + 1

        # Segmented
        ttft_seg, total_seg = send_streaming(args.url, body_seg)
        seg_ttfts.append(ttft_seg)
        time.sleep(0.3)

        # Flattened
        ttft_flat, total_flat = send_streaming(args.url, body_flat)
        flat_ttfts.append(ttft_flat)
        time.sleep(0.3)

        row = {
            "run": run_num,
            "segmented_ttft_ms": round(ttft_seg, 2),
            "segmented_total_ms": round(total_seg, 2),
            "flattened_ttft_ms": round(ttft_flat, 2),
            "flattened_total_ms": round(total_flat, 2),
            "delta_ttft_ms": round(ttft_flat - ttft_seg, 2),
        }
        results.append(row)

        print(
            f"  Run {run_num:2d}: seg={ttft_seg:.1f}ms  flat={ttft_flat:.1f}ms  "
            f"delta={ttft_flat - ttft_seg:+.1f}ms",
            file=sys.stderr,
        )

    # Summary
    seg_avg = sum(seg_ttfts) / len(seg_ttfts)
    flat_avg = sum(flat_ttfts) / len(flat_ttfts)
    print("\nSummary:", file=sys.stderr)
    print(f"  Segmented avg TTFT: {seg_avg:.1f}ms", file=sys.stderr)
    print(f"  Flattened avg TTFT: {flat_avg:.1f}ms", file=sys.stderr)
    print(f"  Delta (flat - seg): {flat_avg - seg_avg:+.1f}ms", file=sys.stderr)

    # Output
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

    fieldnames = [
        "run",
        "segmented_ttft_ms",
        "segmented_total_ms",
        "flattened_ttft_ms",
        "flattened_total_ms",
        "delta_ttft_ms",
    ]
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    if args.output != "-":
        out.close()


if __name__ == "__main__":
    main()
