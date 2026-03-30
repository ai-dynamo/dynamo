# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Measure the impact of unstable preambles on prefix cache reuse.

Three conditions:
1. STABLE: Same system prompt every request (simulates stripping enabled)
2. VARYING: Different billing header each request (simulates stripping disabled)
3. BASELINE: No system prompt at all

For each condition, sends N sequential requests with the same user content.
If prefix caching is active, repeated identical prompts should show lower TTFT
after the first request. Varying preambles break this.

Usage:
    python3 measure_prompt_instability.py --url http://localhost:8000 --model MODEL --runs N
"""

import argparse
import csv
import json
import sys
import time
import uuid

import requests

# A realistic Claude Code system prompt (abbreviated but representative)
SYSTEM_PROMPT = """You are Claude Code, an interactive CLI tool that helps users with software engineering tasks.

# Environment
- Platform: darwin
- Shell: zsh

# Instructions
- Use tools to help the user
- Be concise and helpful
- Follow the user's instructions carefully

# Available Tools
You have access to: Read, Write, Edit, Bash, Glob, Grep, Agent"""


def make_billing_header():
    """Generate a unique billing header like Claude Code does per session."""
    session_id = uuid.uuid4().hex[:12]
    return f"x-anthropic-billing-header: cc_version=0.2.93; cch={session_id};\n"


def make_request(
    model,
    system_prompt,
    user_msg="What files are in the current directory?",
    max_tokens=50,
):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }


def measure_ttft_anthropic(url, body):
    """Send streaming Anthropic request, return TTFT in ms."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{url}/v1/messages",
        json=body,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "dummy",
        },
        stream=True,
        timeout=30,
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
                # Look for first content
                if data.get("type") in ("content_block_start", "content_block_delta"):
                    ttft = (time.monotonic() - t0) * 1000
                    resp.close()
                    return ttft
            except json.JSONDecodeError:
                pass

    resp.close()
    return (time.monotonic() - t0) * 1000


def run_condition(url, model, condition, n_sequential):
    """Run n_sequential requests for a given condition and return TTFTs."""
    ttfts = []
    for i in range(n_sequential):
        if condition == "stable":
            system = SYSTEM_PROMPT
        elif condition == "varying":
            system = make_billing_header() + SYSTEM_PROMPT
        elif condition == "stripped":
            # Billing header present but will be stripped by Dynamo
            system = make_billing_header() + SYSTEM_PROMPT
        else:
            system = SYSTEM_PROMPT

        body = make_request(model, system)
        ttft = measure_ttft_anthropic(url, body)
        ttfts.append(ttft)
        time.sleep(0.2)

    return ttfts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    parser.add_argument(
        "--sequential", type=int, default=8, help="Sequential requests per condition"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="Rounds of the full experiment"
    )
    parser.add_argument("--output", default="-")
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args()

    all_results = []

    for r in range(args.rounds):
        print(f"Round {r+1}/{args.rounds}...", file=sys.stderr)

        # Condition 1: Stable (same system prompt, simulates stripping on)
        print("  Stable prefix...", file=sys.stderr)
        stable_ttfts = run_condition(args.url, args.model, "stable", args.sequential)

        time.sleep(1)

        # Condition 2: Varying (different billing header each time, simulates stripping off)
        print("  Varying prefix...", file=sys.stderr)
        varying_ttfts = run_condition(args.url, args.model, "varying", args.sequential)

        time.sleep(1)

        # Condition 3: Stripped (billing header present but stripped by Dynamo)
        print("  Stripped prefix...", file=sys.stderr)
        stripped_ttfts = run_condition(
            args.url, args.model, "stripped", args.sequential
        )

        for i in range(args.sequential):
            row = {
                "round": r + 1,
                "request_index": i + 1,
                "stable_ttft_ms": round(stable_ttfts[i], 2),
                "varying_ttft_ms": round(varying_ttfts[i], 2),
                "stripped_ttft_ms": round(stripped_ttfts[i], 2),
            }
            all_results.append(row)

        print(
            f"  Stable avg: {sum(stable_ttfts)/len(stable_ttfts):.1f}ms",
            file=sys.stderr,
        )
        print(
            f"  Varying avg: {sum(varying_ttfts)/len(varying_ttfts):.1f}ms",
            file=sys.stderr,
        )
        print(
            f"  Stripped avg: {sum(stripped_ttfts)/len(stripped_ttfts):.1f}ms",
            file=sys.stderr,
        )

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for row in all_results:
                f.write(json.dumps(row) + "\n")

    fieldnames = [
        "round",
        "request_index",
        "stable_ttft_ms",
        "varying_ttft_ms",
        "stripped_ttft_ms",
    ]
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)
    if args.output != "-":
        out.close()


if __name__ == "__main__":
    main()
