"""Warm-up rule check: lanes' first-copy ends vs mean/q90 copy span; window lengths per rule."""
import json, math
from collections import defaultdict
import numpy as np
import analyze as A
import calib

def lane_stats(record):
    rows = calib.load_rows(record)
    plays = defaultdict(lambda: [math.inf, -math.inf, None])
    for r in rows:
        p = plays[r["play_id"]]
        p[0] = min(p[0], r["arrival_time_ms"]); p[1] = max(p[1], r["terminal_time_ms"] or -math.inf); p[2] = r.get("lane_id")
    first, last = {}, defaultdict(float)
    for s, e, lane in plays.values():
        if lane is None: continue
        if lane not in first or s < first[lane][0]: first[lane] = (s, e)
        last[lane] = max(last[lane], e)
    return np.array([e for _, e in first.values()]), min(last.values())

out = defaultdict(list)
for group_dir in ("sweep2/ax-lanes", "step4_r1"):
    base = calib.CAL / group_dir
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (base / "cells.jsonl").read_text().splitlines()} if (base / "cells.jsonl").exists() else {}
    if not cells:
        for f in ("train", "val"):
            for l in (calib.CAL / "final_r1" / f"{f}.jsonl").read_text().splitlines():
                c = json.loads(l); cells[c["cell_id"]] = c
    for r in calib.read_results(base / "results.jsonl"):
        c = cells.get(r["cell_id"])
        if c is None or c["family"] != "agentx" or not r["policy_name"].startswith("default") or r.get("error") and "empty" not in r["error"]:
            continue
        if not r.get("per_request_path"):
            continue
        spans = A.copy_spans(c)
        ends, full_end = lane_stats(r)
        lv = c["load"]["per_worker"]
        for name, warm in (("1.5q90", 1.5 * np.percentile(spans, 90)), ("1.5mean", 1.5 * spans.mean()), ("1.0q90", np.percentile(spans, 90))):
            out[name].append({"cell": r["cell_id"], "N": c["num_workers"], "l": lv, "frac_first_done": float((ends <= warm).mean()), "window_s": (full_end - warm) / 1000})
for name, rows in out.items():
    w = np.array([x["window_s"] for x in rows]); f = np.array([x["frac_first_done"] for x in rows])
    print(f"{name}: runs {len(rows)} window_s min {w.min():.0f} p10 {np.percentile(w,10):.0f} median {np.median(w):.0f} empty {int((w<=0).sum())} | frac lanes past first copy at warm-up: min {f.min():.2f} p10 {np.percentile(f,10):.2f} median {np.median(f):.2f}")
    bad = sorted(rows, key=lambda x: x["window_s"])[:5]
    print("   shortest:", [(x["cell"][-28:], round(x["window_s"])) for x in bad])
