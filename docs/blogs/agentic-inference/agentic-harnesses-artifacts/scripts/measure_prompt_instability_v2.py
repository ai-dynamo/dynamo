#!/usr/bin/env python3
"""Measure the impact of unstable preambles on prefix cache reuse (v2).

Accepts a large system prompt from file to simulate real-world agentic workloads.

Three conditions:
1. STABLE: Same system prompt every request (simulates stripping enabled)
2. VARYING: Different billing header prepended each request (simulates stripping disabled)
3. STRIPPED: Billing header present but Dynamo strips it before the engine sees it

Includes a warmup phase to prime the prefix cache before measurement begins.

Usage:
    python3 measure_prompt_instability_v2.py \
        --url http://localhost:8000 \
        --model MODEL \
        --system-prompt-file system_prompt.txt \
        --sequential 8 --rounds 3
"""

import argparse
import csv
import json
import sys
import time
import uuid

import requests


DEFAULT_SYSTEM_PROMPT = (
    "You are Claude Code, an interactive CLI tool that helps users with "
    "software engineering tasks.\n\n# Environment\n- Platform: darwin\n"
    "- Shell: zsh\n\n# Instructions\n- Use tools to help the user\n"
    "- Be concise and helpful\n- Follow the user's instructions carefully\n\n"
    "# Available Tools\nYou have access to: Read, Write, Edit, Bash, Glob, Grep, Agent"
)

USER_MESSAGE = "What files are in the current directory?"


def make_billing_header():
    """Generate a unique billing header like Claude Code does per session."""
    session_id = uuid.uuid4().hex[:12]
    return f"x-anthropic-billing-header: cc_version=0.2.93; cch={session_id};\n"


def build_body(model, system_prompt, max_tokens=50):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "system": system_prompt,
        "messages": [{"role": "user", "content": USER_MESSAGE}],
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


def warmup(url, model, system_prompt, n=2):
    """Send warmup requests to prime the prefix cache."""
    print("  Warmup phase...", file=sys.stderr)
    for i in range(n):
        body = build_body(model, system_prompt)
        _, _ = send_streaming(url, body)
        time.sleep(0.3)


def run_condition(url, model, condition, system_prompt, n_sequential):
    """Run n_sequential requests for a condition. Returns list of (ttft, total) tuples."""
    results = []
    for i in range(n_sequential):
        if condition == "stable":
            prompt = system_prompt
        elif condition == "varying":
            prompt = make_billing_header() + system_prompt
        elif condition == "stripped":
            # Billing header present; Dynamo should strip it
            prompt = make_billing_header() + system_prompt
        else:
            prompt = system_prompt

        body = build_body(model, prompt)
        ttft, total = send_streaming(url, body)
        results.append((ttft, total))

        if i == 0:
            print(f"    req 1 (cache-warming): ttft={ttft:.1f}ms", file=sys.stderr)

        time.sleep(0.2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure prefix cache impact of unstable preambles (v2)"
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a text file containing the system prompt. "
             "If not provided, uses a short default prompt.",
    )
    parser.add_argument("--sequential", type=int, default=8,
                        help="Sequential requests per condition (default: 8)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Rounds of the full experiment (default: 3)")
    parser.add_argument("--output", default="-", help="CSV output path (- for stdout)")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL output path")
    args = parser.parse_args()

    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read()
        print(f"Loaded system prompt: {len(system_prompt)} chars", file=sys.stderr)
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        print("Using default (short) system prompt", file=sys.stderr)

    all_results = []

    for r in range(args.rounds):
        print(f"\nRound {r + 1}/{args.rounds}", file=sys.stderr)

        # --- Stable ---
        print("  [stable] warming up...", file=sys.stderr)
        warmup(args.url, args.model, system_prompt)
        print("  [stable] measuring...", file=sys.stderr)
        stable = run_condition(args.url, args.model, "stable", system_prompt, args.sequential)

        time.sleep(1)

        # --- Varying ---
        print("  [varying] warming up...", file=sys.stderr)
        warmup(args.url, args.model, make_billing_header() + system_prompt)
        print("  [varying] measuring...", file=sys.stderr)
        varying = run_condition(args.url, args.model, "varying", system_prompt, args.sequential)

        time.sleep(1)

        # --- Stripped ---
        print("  [stripped] warming up...", file=sys.stderr)
        warmup(args.url, args.model, make_billing_header() + system_prompt)
        print("  [stripped] measuring...", file=sys.stderr)
        stripped = run_condition(args.url, args.model, "stripped", system_prompt, args.sequential)

        # Collect rows
        for i in range(args.sequential):
            row = {
                "round": r + 1,
                "request_index": i + 1,
                "stable_ttft_ms": round(stable[i][0], 2),
                "stable_total_ms": round(stable[i][1], 2),
                "varying_ttft_ms": round(varying[i][0], 2),
                "varying_total_ms": round(varying[i][1], 2),
                "stripped_ttft_ms": round(stripped[i][0], 2),
                "stripped_total_ms": round(stripped[i][1], 2),
            }
            all_results.append(row)

        # Summary
        s_avg = sum(t for t, _ in stable) / len(stable)
        v_avg = sum(t for t, _ in varying) / len(varying)
        x_avg = sum(t for t, _ in stripped) / len(stripped)
        print(f"  Avg TTFT  stable={s_avg:.1f}ms  varying={v_avg:.1f}ms  stripped={x_avg:.1f}ms",
              file=sys.stderr)

    # --- Output ---
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for row in all_results:
                f.write(json.dumps(row) + "\n")

    fieldnames = [
        "round", "request_index",
        "stable_ttft_ms", "stable_total_ms",
        "varying_ttft_ms", "varying_total_ms",
        "stripped_ttft_ms", "stripped_total_ms",
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
