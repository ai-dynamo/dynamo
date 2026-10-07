#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a real-SGLang run's request log into a Mooncake trace for the SGLang-mode mocker.

Same chains, same turns, same prompt/output lengths; each chain is a session whose first turn
arrives when the real run first sent it and whose later turns follow closed-loop (delay 0).
Hash ids are made globally unique per (session id, trace hash id) so cross-chain sharing
matches the real token construction.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=["chains", "lanes"],
        default="lanes",
        help="lanes: one closed-loop session per driver lane (fixed concurrency); "
        "chains: one session per chain starting at its real first send",
    )
    a = ap.parse_args()
    reqs = [json.loads(line) for line in open(a.requests)]
    reqs = [r for r in reqs if r["err"] is None and r["meta"]]
    chains = {}
    for line in open(a.workload):
        c = json.loads(line)
        chains[c["chain"]] = c
    t0 = min(r["t_send"] for r in reqs)

    def key(r):
        return f"lane-{r['lane']}" if a.mode == "lanes" else f"chain-{r['chain']}"

    if a.mode == "lanes":
        reqs.sort(key=lambda r: (r["lane"], r["t_send"]))
    else:
        reqs.sort(key=lambda r: (r["chain"], r["turn"]))
    prev_key = None
    ids = {}
    n = 0
    with open(a.out, "w") as f:
        for r in reqs:
            c = chains[r["chain"]]
            spec = c["reqs"][r["turn"]]
            hids = [ids.setdefault((c["sid"], h), len(ids)) for h in spec["h"]]
            row = {
                "session_id": key(r),
                "input_length": r["prompt_len"],
                "output_length": r["meta"]["completion_tokens"],
                "hash_ids": hids,
            }
            if prev_key != key(r):
                row["timestamp"] = (r["t_send"] - t0) * 1000.0
            else:
                row["delay"] = 0.0
            prev_key = key(r)
            f.write(json.dumps(row) + "\n")
            n += 1
    print(json.dumps({"rows": n, "unique_hash_ids": len(ids)}))


if __name__ == "__main__":
    main()
