#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze a recorded SGLang KV-event stream against the driver's request log.

Blocks are classified exactly: with page size 1 every block carries one token, so a block's
token path from the root is rebuilt through parent hashes; a block is a PROMPT block when its
path equals a prefix of some sent prompt, otherwise an OUTPUT (decode) block.
"""
import argparse
import json
import random
import statistics as st
import sys
from collections import Counter

import msgpack

VOCAB_LO, VOCAB_HI = 100, 150000
MASK = (1 << 64) - 1


def rh(prev, tok):
    return hash((prev, tok)) & MASK


def decode_event(ev):
    if isinstance(ev, dict):
        t = ev.get("type")
        return t, ev
    t = ev[0]
    if t == "BlockStored":
        keys = [
            "block_hashes",
            "parent_block_hash",
            "token_ids",
            "block_size",
            "lora_id",
            "medium",
            "cache_salt",
            "session_id",
        ]
    elif t == "BlockRemoved":
        keys = ["block_hashes", "medium"]
    else:
        keys = []
    return t, dict(zip(keys, ev[1:]))


def pct(a, q):
    if not a:
        return None
    a = sorted(a)
    return a[min(len(a) - 1, int(q * len(a)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--scale-tokens", type=int, default=8)
    ap.add_argument(
        "--warmup-s",
        type=float,
        default=0.0,
        help="exclude this many seconds after the first send from rate windows",
    )
    a = ap.parse_args()
    reqs = [json.loads(line) for line in open(f"{a.run_dir}/requests.jsonl")]
    ok = [r for r in reqs if r["err"] is None and r["meta"]]
    chains = {}
    for line in open(a.workload):
        c = json.loads(line)
        chains[c["chain"]] = c

    # Prompt-prefix rolling hashes, built per chain incrementally.
    prompt_rh = set()
    blk_cache = {}
    sent = sorted(ok, key=lambda r: (r["chain"], r["turn"]))
    last = {}
    for r in sent:
        c = chains[r["chain"]]
        sid = c["sid"]
        h = c["reqs"][r["turn"]]["h"]
        prev = last.get(r["chain"])  # (hash ids, list of rh per token)
        start_blk = 0
        rhs = []
        if prev:
            ph, prhs = prev
            while start_blk < min(len(ph), len(h)) and ph[start_blk] == h[start_blk]:
                start_blk += 1
            rhs = prhs[: start_blk * a.scale_tokens]
        cur = rhs[-1] if rhs else 0
        for b in range(start_blk, len(h)):
            key = (sid, h[b])
            toks = blk_cache.get(key)
            if toks is None:
                rr = random.Random(f"{sid}:{h[b]}")
                toks = [rr.randrange(VOCAB_LO, VOCAB_HI) for _ in range(a.scale_tokens)]
                blk_cache[key] = toks
            for t in toks:
                cur = rh(cur, t)
                rhs.append(cur)
                prompt_rh.add(cur)
        last[r["chain"]] = (h, rhs)

    # Event stream.
    block_rh = {}
    n_msgs = 0
    stored_sizes, removed_sizes = [], []
    ev_types = Counter()
    stored_prompt = stored_output = stored_unknown_parent = 0
    events_per_msg = []
    t_events = []  # (recv_t, kind, nblocks)
    seq_prev = None
    seq_gaps = 0
    mixed_events = 0
    with open(f"{a.run_dir}/events.msgpack", "rb") as f:
        for recv_t, frames in msgpack.Unpacker(f, raw=False, strict_map_key=False):
            n_msgs += 1
            payload = frames[-1]
            if len(frames) >= 3:
                seq = int.from_bytes(frames[1], "big")
                if seq_prev is not None and seq != seq_prev + 1:
                    seq_gaps += 1
                seq_prev = seq
            batch = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            evs = batch[1] if isinstance(batch, list) else batch["events"]
            events_per_msg.append(len(evs))
            for ev in evs:
                t, d = decode_event(ev)
                ev_types[t] += 1
                if t == "BlockStored":
                    hs = d["block_hashes"]
                    toks = d["token_ids"]
                    stored_sizes.append(len(hs))
                    parent = d["parent_block_hash"]
                    if parent is None:
                        cur = 0
                    elif parent in block_rh:
                        cur = block_rh[parent]
                    else:
                        cur = None
                        stored_unknown_parent += len(hs)
                    kinds = set()
                    for bh, tok in zip(hs, toks):
                        if cur is not None:
                            cur = rh(cur, tok)
                            block_rh[bh] = cur
                            if cur in prompt_rh:
                                stored_prompt += 1
                                kinds.add("p")
                            else:
                                stored_output += 1
                                kinds.add("o")
                    if len(kinds) > 1:
                        mixed_events += 1
                    t_events.append((recv_t, "s", len(hs)))
                elif t == "BlockRemoved":
                    removed_sizes.append(len(d["block_hashes"]))
                    t_events.append((recv_t, "r", len(d["block_hashes"])))
                else:
                    t_events.append((recv_t, "c", 0))

    t0 = min(r["t_send"] for r in ok)
    t1 = max(r["t_done"] for r in ok)
    w0 = t0 + a.warmup_s
    win_reqs = [r for r in ok if r["t_done"] >= w0]
    win_s = t1 - w0
    win_ev = [e for e in t_events if w0 <= e[0] <= t1 + 2]
    win_stored_ev = sum(1 for e in win_ev if e[1] == "s")
    win_removed_ev = sum(1 for e in win_ev if e[1] == "r")
    win_stored_blk = sum(e[2] for e in win_ev if e[1] == "s")
    win_removed_blk = sum(e[2] for e in win_ev if e[1] == "r")
    prompt_tok = sum(r["meta"]["prompt_tokens"] for r in ok)
    cached_tok = sum(r["meta"]["cached_tokens"] or 0 for r in ok)
    out_tok = sum(r["meta"]["completion_tokens"] for r in ok)
    n = len(ok)
    bounds = [1, 8, 64, 1024, 8192, float("inf")]
    hist_e, hist_b = [0] * 6, [0] * 6
    for sz in stored_sizes:
        b = next(i for i, ub in enumerate(bounds) if sz <= ub)
        hist_e[b] += 1
        hist_b[b] += sz
    res = {
        "run": a.run_dir.rstrip("/").split("/")[-1],
        "requests_ok": n,
        "requests_err": len(reqs) - n,
        "wall_s": t1 - t0,
        "req_per_s": n / (t1 - t0),
        "zmq_messages": n_msgs,
        "seq_gaps": seq_gaps,
        "event_types": dict(ev_types),
        "events_per_message": {
            "mean": st.mean(events_per_msg) if events_per_msg else None,
            "p50": pct(events_per_msg, 0.5),
            "p99": pct(events_per_msg, 0.99),
            "max": max(events_per_msg) if events_per_msg else None,
        },
        "stored_blocks_per_event": {
            "mean": st.mean(stored_sizes) if stored_sizes else None,
            "p10": pct(stored_sizes, 0.1),
            "p50": pct(stored_sizes, 0.5),
            "p90": pct(stored_sizes, 0.9),
            "p99": pct(stored_sizes, 0.99),
            "max": max(stored_sizes) if stored_sizes else None,
            "frac_events_1_block": (
                sum(1 for s in stored_sizes if s == 1) / len(stored_sizes)
            )
            if stored_sizes
            else None,
        },
        "removed_blocks_per_event": {
            "mean": st.mean(removed_sizes) if removed_sizes else None,
            "p50": pct(removed_sizes, 0.5),
            "p99": pct(removed_sizes, 0.99),
            "max": max(removed_sizes) if removed_sizes else None,
        },
        "stored_size_hist": {
            "buckets": ["1", "2-8", "9-64", "65-1024", "1025-8192", ">8192"],
            "events": hist_e,
            "blocks": hist_b,
        },
        "totals": {
            "stored_events": len(stored_sizes),
            "stored_blocks": sum(stored_sizes),
            "removed_events": len(removed_sizes),
            "removed_blocks": sum(removed_sizes),
            "stored_prompt_blocks": stored_prompt,
            "stored_output_blocks": stored_output,
            "stored_unclassified_blocks": stored_unknown_parent,
            "mixed_prompt_output_events": mixed_events,
            "prompt_tokens": prompt_tok,
            "cached_tokens": cached_tok,
            "uncached_prompt_tokens": prompt_tok - cached_tok,
            "completion_tokens": out_tok,
        },
        "per_request": {
            "prompt_tokens": prompt_tok / n,
            "cached_tokens": cached_tok / n,
            "uncached_prompt_tokens": (prompt_tok - cached_tok) / n,
            "completion_tokens": out_tok / n,
            "stored_events": len(stored_sizes) / n,
            "stored_blocks": sum(stored_sizes) / n,
            "stored_prompt_blocks": stored_prompt / n,
            "stored_output_blocks": stored_output / n,
            "removed_events": len(removed_sizes) / n,
            "removed_blocks": sum(removed_sizes) / n,
            "lookup_blocks": prompt_tok / n,
        },
        "ratios": {
            "stored_output_blocks_per_completion_token": stored_output / out_tok
            if out_tok
            else None,
            "stored_prompt_blocks_per_uncached_prompt_token": stored_prompt
            / (prompt_tok - cached_tok)
            if prompt_tok > cached_tok
            else None,
            "stored_events_per_completion_token": len(stored_sizes) / out_tok
            if out_tok
            else None,
        },
        "window": {
            "warmup_s": a.warmup_s,
            "seconds": win_s,
            "requests_done": len(win_reqs),
            "req_per_s": len(win_reqs) / win_s if win_s > 0 else None,
            "stored_events_per_s": win_stored_ev / win_s if win_s > 0 else None,
            "removed_events_per_s": win_removed_ev / win_s if win_s > 0 else None,
            "stored_blocks_per_s": win_stored_blk / win_s if win_s > 0 else None,
            "removed_blocks_per_s": win_removed_blk / win_s if win_s > 0 else None,
            "stored_events_per_request": win_stored_ev / len(win_reqs)
            if win_reqs
            else None,
            "events_per_request": (win_stored_ev + win_removed_ev) / len(win_reqs)
            if win_reqs
            else None,
        },
    }
    json.dump(res, sys.stdout, indent=1)
    print()
    with open(f"{a.run_dir}/analysis.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
