#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a scaled AgentX chain workload for the real-SGLang cadence probe.

Each trace session is flattened into chains (the main thread and each subagent); a chain is a
sequence of requests sharing prefixes through the session-local hash ids. Every 64-token trace
block becomes SCALE tokens; output lengths are divided by the same factor (ceil, >= 1), so the
token ratios (and KV pressure per lane) match the full-scale trace.
"""
import argparse
import hashlib
import json
import random


def walk(reqs, sid, chains):
    chain = []
    for q in reqs:
        if q.get("type") == "subagent":
            walk(q["requests"], sid, chains)
            continue
        chain.append(q)
    if chain:
        chains.append((sid, chain))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scale-tokens", type=int, default=8, help="tokens per 64-token trace block"
    )
    ap.add_argument("--chains", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    div = 64 // a.scale_tokens
    chains = []
    sha = hashlib.sha256()
    with open(a.trace, "rb") as f:
        for line in f:
            sha.update(line)
            s = json.loads(line)
            walk(s["requests"], s["id"], chains)
    rng = random.Random(a.seed)
    rng.shuffle(chains)
    picked = chains[: a.chains]
    n = 0
    with open(a.out, "w") as f:
        for ci, (sid, chain) in enumerate(picked):
            reqs = []
            for q in chain:
                out = max(1, -(-q["out"] // div))
                reqs.append(
                    {
                        "h": q["hash_ids"],
                        "out": out,
                        "in_full": q["in"],
                        "out_full": q["out"],
                    }
                )
                n += 1
            f.write(json.dumps({"chain": ci, "sid": sid, "reqs": reqs}) + "\n")
    print(
        json.dumps(
            {
                "trace_sha256": sha.hexdigest(),
                "chains": len(picked),
                "requests": n,
                "scale_tokens": a.scale_tokens,
                "out_div": div,
                "seed": a.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
