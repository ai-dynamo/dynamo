# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Demonstrate cache breaking from reasoning content mutation in the cities conversation.

Uses the 30-city golden trace with padded tool results to make the conversation
~30K+ tokens. Then compares:
- Exact replay (cache hit — conversation prefix matches)
- Mutated replay (cache miss — reasoning changed in early turns, breaks prefix for all subsequent turns)

The mutation simulates what happens when reasoning reconstruction puts tokens in
the wrong order: the prefix diverges at the mutation point, forcing recomputation
of everything after it.
"""

import copy
import json
import statistics
import sys
import time
import uuid

import requests


def ttft_streaming(url, model, msgs, followup):
    body = {
        "model": model,
        "max_tokens": 200,
        "stream": True,
        "messages": msgs + [followup],
    }
    headers = {"Content-Type": "application/json"}
    t0 = time.monotonic()
    r = requests.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers=headers,
        stream=True,
        timeout=120,
    )
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: ") and line != "data: [DONE]":
            try:
                d = json.loads(line[6:])
                delta = d.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    r.close()
                    return (time.monotonic() - t0) * 1000
            except Exception:
                pass
    r.close()
    return (time.monotonic() - t0) * 1000


def pad_tool_results(messages, pad_chars=1500):
    """Pad tool result content to make conversation larger."""
    out = copy.deepcopy(messages)
    for msg in out:
        if msg.get("role") == "tool":
            original = msg["content"]
            # Pad with realistic-looking content
            padding = (
                "\nAdditional context: This city has a rich cultural heritage "
                "spanning centuries of history. The urban landscape features "
                "a mix of modern architecture and historical landmarks. "
                "The local cuisine reflects diverse cultural influences. "
                "Transportation infrastructure includes metro systems, "
                "bus networks, and cycling paths. The economy is driven by "
                "technology, finance, tourism, and manufacturing sectors. "
                "Educational institutions include several world-renowned "
                "universities and research centers. "
            ) * 3
            msg["content"] = original + padding[:pad_chars]
    return out


def mutate_early_reasoning(messages):
    """Mutate reasoning in the FIRST assistant turn only — breaks prefix for all subsequent turns."""
    out = copy.deepcopy(messages)
    mutated = False
    for msg in out:
        if msg.get("role") == "assistant" and not mutated:
            rc = msg.get("reasoning_content")
            noise = f"[session={uuid.uuid4().hex[:16]}] "
            if isinstance(rc, list) and rc:
                rc[0] = noise + rc[0]
            elif isinstance(rc, str) and rc:
                msg["reasoning_content"] = noise + rc
            mutated = True
    return out


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

    with open("/home/scratch.mkosec_hw/golden-trace-cities-multiturn.json") as f:
        trace = json.load(f)

    messages = trace["messages"]
    # Use a SHORT system prompt so conversation dominates
    messages[0] = {
        "role": "system",
        "content": "You are a travel expert. Rank cities by population when asked.",
    }

    # Pad tool results to make conversation ~30K+ tokens
    messages = pad_tool_results(messages, pad_chars=1500)

    # Estimate token count
    total_chars = sum(len(json.dumps(m)) for m in messages)
    est_tokens = total_chars // 4
    print(
        f"Conversation: {len(messages)} messages, ~{total_chars} chars, ~{est_tokens} estimated tokens",
        file=sys.stderr,
    )

    followup = {
        "role": "user",
        "content": "Now rank all 30 cities by population, largest first.",
    }

    msgs_exact = copy.deepcopy(messages)

    # Warm cache
    print("Warming cache (5 exact requests)...", file=sys.stderr)
    for _ in range(5):
        ttft_streaming(url, model, msgs_exact, followup)
        time.sleep(0.3)

    time.sleep(1)

    # Measure
    print("\nMeasuring 15 runs (alternating exact vs mutated)...", file=sys.stderr)
    exact_times = []
    mutated_times = []

    for i in range(15):
        e = ttft_streaming(url, model, msgs_exact, followup)
        time.sleep(0.2)

        msgs_mut = mutate_early_reasoning(messages)
        m = ttft_streaming(url, model, msgs_mut, followup)
        time.sleep(0.2)

        exact_times.append(e)
        mutated_times.append(m)
        print(
            f"  {i+1}: exact={e:.1f}ms  mutated={m:.1f}ms  delta={m-e:+.1f}ms",
            file=sys.stderr,
        )

    print(
        f"\nExact:   mean={statistics.mean(exact_times):.1f}ms  stdev={statistics.stdev(exact_times):.1f}ms",
        file=sys.stderr,
    )
    print(
        f"Mutated: mean={statistics.mean(mutated_times):.1f}ms  stdev={statistics.stdev(mutated_times):.1f}ms",
        file=sys.stderr,
    )
    delta = statistics.mean(mutated_times) - statistics.mean(exact_times)
    ratio = (
        statistics.mean(mutated_times) / statistics.mean(exact_times)
        if statistics.mean(exact_times) > 0
        else 0
    )
    print(f"Delta:   {delta:+.1f}ms ({ratio:.1f}x)", file=sys.stderr)

    with open("/home/scratch.mkosec_hw/reasoning-cache-cities-padded.jsonl", "w") as f:
        for i in range(15):
            f.write(
                json.dumps(
                    {
                        "run": i + 1,
                        "exact_ms": round(exact_times[i], 1),
                        "mutated_ms": round(mutated_times[i], 1),
                        "delta_ms": round(mutated_times[i] - exact_times[i], 1),
                    }
                )
                + "\n"
            )
    print("Saved.", file=sys.stderr)


if __name__ == "__main__":
    main()
