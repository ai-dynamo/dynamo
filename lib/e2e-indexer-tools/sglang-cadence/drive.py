#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop chain driver for SGLang /generate (input_ids, streaming).

L lanes; each lane takes the next chain and sends its requests back to back. Prompt tokens are
deterministic per (session id, hash id) so shared trace blocks give shared token prefixes.
Writes one JSON line per request with send / first-token / done wall times and SGLang meta_info.
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp

VOCAB_LO, VOCAB_HI = 100, 150000


class Tokens:
    def __init__(self, scale):
        self.scale = scale
        self.cache = {}

    def block(self, sid, hid):
        key = (sid, hid)
        toks = self.cache.get(key)
        if toks is None:
            r = random.Random(f"{sid}:{hid}")
            toks = [r.randrange(VOCAB_LO, VOCAB_HI) for _ in range(self.scale)]
            self.cache[key] = toks
        return toks

    def prompt(self, sid, hids):
        out = []
        for h in hids:
            out.extend(self.block(sid, h))
        return out


async def one(session, url, ids, max_new):
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": max_new,
            "ignore_eos": True,
            "temperature": 0.0,
        },
        "stream": True,
    }
    t_send = time.time()
    t_first = None
    meta = None
    async with session.post(url + "/generate", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            msg = json.loads(body)
            if t_first is None:
                t_first = time.time()
            meta = msg.get("meta_info", meta)
    return t_send, t_first, time.time(), meta


async def lane(lid, args, queue, tokens, session, out_f, deadline, stats):
    while time.time() < deadline:
        try:
            chain = queue.pop()
        except IndexError:
            return
        sid = chain["sid"]
        for ri, r in enumerate(chain["reqs"]):
            if time.time() >= deadline or (
                args.max_requests and stats["sent"] >= args.max_requests
            ):
                return
            ids = tokens.prompt(sid, r["h"])
            stats["sent"] += 1
            try:
                t_send, t_first, t_done, meta = await one(
                    session, args.url, ids, r["out"]
                )
                err = None
            except Exception as e:  # keep driving; record the failure
                t_send, t_first, t_done, meta, err = (
                    time.time(),
                    None,
                    time.time(),
                    None,
                    repr(e),
                )
            rec = {
                "lane": lid,
                "chain": chain["chain"],
                "turn": ri,
                "prompt_len": len(ids),
                "max_new": r["out"],
                "t_send": t_send,
                "t_first": t_first,
                "t_done": t_done,
                "meta": {
                    k: meta.get(k)
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                        "id",
                    )
                }
                if meta
                else None,
                "err": err,
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--duration-s", type=float, default=1500)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--scale-tokens", type=int, default=8)
    ap.add_argument("--chain-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chains = [json.loads(line) for line in open(args.workload)][args.chain_offset :]
    queue = list(reversed(chains))
    tokens = Tokens(args.scale_tokens)
    stats = {"sent": 0}
    deadline = time.time() + args.duration_s
    timeout = aiohttp.ClientTimeout(total=None)
    t0 = time.time()
    with open(args.out, "w") as out_f:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                *(
                    lane(i, args, queue, tokens, session, out_f, deadline, stats)
                    for i in range(args.lanes)
                )
            )
    print(json.dumps({"sent": stats["sent"], "wall_s": time.time() - t0}))


if __name__ == "__main__":
    asyncio.run(main())
