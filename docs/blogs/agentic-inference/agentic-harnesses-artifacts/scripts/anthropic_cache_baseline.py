# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Measure Anthropic API prompt caching behavior for baseline comparison.

Sends repeated requests with cache_control to measure cache_read_input_tokens.
Requires ANTHROPIC_API_KEY env var.

Two conditions:
1. STABLE: Same system prompt every request → cache hits expected
2. VARYING: Different billing preamble each request → cache misses expected

Usage:
    ANTHROPIC_API_KEY=sk-... python3 anthropic_cache_baseline.py --rounds 3 --sequential 5
"""

import argparse
import csv
import json
import sys
import time
import uuid

try:
    import anthropic
except ImportError:
    print("Install anthropic SDK: pip install anthropic", file=sys.stderr)
    sys.exit(1)


# A realistic Claude Code-style system prompt (abbreviated)
SYSTEM_PROMPT = """You are Claude Code, an interactive CLI tool that helps users with software engineering tasks.

# Environment
- Platform: darwin
- Shell: zsh
- Working directory: /Users/user/work

# Instructions
- Use tools to help the user
- Be concise and helpful
- Follow the user's instructions carefully

# Available Tools
You have access to: Read, Write, Edit, Bash, Glob, Grep, Agent

# Output Style
Keep responses short and direct. Lead with the answer or action, not the reasoning."""


def make_billing_header():
    """Generate a unique billing header like Claude Code does per session."""
    session_id = uuid.uuid4().hex[:12]
    return f"x-anthropic-billing-header: cc_version=0.2.93; cch={session_id};\n"


def send_request(client, system_text, model="claude-haiku-4-5"):
    """Send a non-streaming request with cache_control and return usage."""
    t0 = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=20,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": "What files are in the current directory?"}
        ],
    )
    ttft = (time.monotonic() - t0) * 1000
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0
        )
        or 0,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
        or 0,
        "ttft_ms": round(ttft, 2),
    }


def run_experiment(client, model, n_rounds, n_sequential):
    results = []

    for r in range(n_rounds):
        print(f"Round {r+1}/{n_rounds}...", file=sys.stderr)

        # Condition 1: STABLE (same system prompt)
        print("  Stable prefix...", file=sys.stderr)
        for i in range(n_sequential):
            usage = send_request(client, SYSTEM_PROMPT, model)
            results.append(
                {
                    "round": r + 1,
                    "request_index": i + 1,
                    "condition": "stable",
                    **usage,
                }
            )
            print(
                f"    [{i+1}] cache_read={usage['cache_read_input_tokens']} cache_create={usage['cache_creation_input_tokens']} ttft={usage['ttft_ms']}ms",
                file=sys.stderr,
            )
            time.sleep(0.3)

        time.sleep(1)

        # Condition 2: VARYING (different billing header each time)
        print("  Varying prefix...", file=sys.stderr)
        for i in range(n_sequential):
            system = make_billing_header() + SYSTEM_PROMPT
            usage = send_request(client, system, model)
            results.append(
                {
                    "round": r + 1,
                    "request_index": i + 1,
                    "condition": "varying",
                    **usage,
                }
            )
            print(
                f"    [{i+1}] cache_read={usage['cache_read_input_tokens']} cache_create={usage['cache_creation_input_tokens']} ttft={usage['ttft_ms']}ms",
                file=sys.stderr,
            )
            time.sleep(0.3)

        time.sleep(1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sequential", type=int, default=5)
    parser.add_argument("--output", default="-")
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    results = run_experiment(client, args.model, args.rounds, args.sequential)

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"Raw data written to {args.jsonl}", file=sys.stderr)

    fieldnames = [
        "round",
        "request_index",
        "condition",
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "ttft_ms",
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
