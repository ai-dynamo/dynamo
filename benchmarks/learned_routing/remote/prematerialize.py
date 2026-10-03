# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize a bundle's CRN replicates in parallel before lr-eval / lr-train plan their tasks.

    python3.12 -S prematerialize.py --root BUNDLE --cells BUNDLE/cells.jsonl --k 0-7 [--workers N]

The harness materializes replicate k of each cell inside the parent process while it plans tasks,
one after another. On a fresh node (empty ``runs/replicates``) that serial phase dominated short
batches (several minutes for AgentX traces). ``materialize_replicate`` is a pure, idempotent
function that writes through a per-PID temporary file and ``os.replace``, so materializing the same
set in parallel first produces byte-identical files, and the harness then finds and reuses them.
Run with ``PYTHONPATH=BUNDLE/site`` (as run.sh does).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _one(root: str, raw: dict, k: int) -> tuple[str, int, float]:
    from learned_routing.cells import Cell, resolve_replicate
    from learned_routing.paths import Layout

    started = time.monotonic()
    cell = Cell(raw=raw, layout=Layout.resolve(root))
    resolve_replicate(cell, k)
    return raw["cell_id"], k, time.monotonic() - started


def parse_k(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--k", required=True, help="replicate indices, e.g. 0-7 or 0,1,8")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument(
        "--shard", help="i/n: only the (cell, k) pairs lr-eval --shard i/n owns"
    )
    args = p.parse_args(argv)
    cells = {}
    for path in args.cells:
        for line in open(path):
            if line.strip():
                raw = json.loads(line)
                if raw.get("trace_format") != "synthetic" and raw.get("trace_files"):
                    cells[raw["cell_id"]] = raw
    # One task per distinct (trace, k): cells sharing a trace share its replicates.
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None

    def owned(cell_id: str, k: int) -> bool:
        # Same owner function as learned_routing.eval_cli.build_tasks.
        if shard is None:
            return True
        digest = hashlib.sha256(f"{cell_id}|{k}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % shard[1] == shard[0]

    seen, work = set(), []
    for raw in cells.values():
        for k in parse_k(args.k):
            if not owned(raw["cell_id"], k):
                continue
            ident = (tuple(raw["trace_files"]), raw.get("arrival_spread_ms"), k)
            if ident not in seen:
                seen.add(ident)
                work.append((raw, k))
    started = time.monotonic()
    slowest = 0.0
    with ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(work) or 1))
    ) as pool:
        futures = [pool.submit(_one, args.root, raw, k) for raw, k in work]
        for future in as_completed(futures):
            slowest = max(slowest, future.result()[2])
    print(
        json.dumps(
            {
                "replicates": len(work),
                "cells": len(cells),
                "wall_s": round(time.monotonic() - started, 2),
                "slowest_s": round(slowest, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
