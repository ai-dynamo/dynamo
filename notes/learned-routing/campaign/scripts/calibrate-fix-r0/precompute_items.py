"""Parallel precompute of scored items (analyze.items_for) for result files; usage: precompute_items.py DIR..."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import calib
import analyze as A


def one(args):
    rec, cell = args
    A.items_for(rec, cell)
    return 1


def main():
    jobs = []
    for d in sys.argv[1:]:
        d = Path(d)
        cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (d / "cells.jsonl").read_text().splitlines()}
        latest = {}
        for r in calib.read_results(d / "results.jsonl"):
            if not r.get("error") and r["cell_id"] in cells:
                latest[(r["cell_id"], r["policy_name"], r["repeat"])] = r
        for r in latest.values():
            c = cells[r["cell_id"]]
            if c["family"] == "agentx":
                continue  # AgentX items carry a measure tag (decide.ax_measure); not re-swept here
            p = A.ITEMS / r["cache_key"][:2] / f"{r['cache_key']}.pkl"
            p0 = A.ITEMS_R0 / r["cache_key"][:2] / f"{r['cache_key']}.pkl"
            if not p.exists() and not p0.exists():
                jobs.append((r, c))
    print("todo", len(jobs), flush=True)
    calib.e0_table()
    with ProcessPoolExecutor(16) as ex:
        n = sum(ex.map(one, jobs, chunksize=4))
    print("done", n, flush=True)


if __name__ == "__main__":
    main()
