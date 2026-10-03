"""Sweep analysis: per-record scored items (cached), SLA rule, band curves, level interpolation."""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import calib

ITEMS = calib.CAL / "items"
TARGETS = {"L1": 0.95, "L2": 0.85, "L3": 0.65}
ITL_MULT = 1.2
ITL_Q = 95.0


def group_records(group: str, root: Path | None = None) -> tuple[dict, list[dict]]:
    base = root or (calib.CAL / "sweep2" / group)
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (base / "cells.jsonl").read_text().splitlines()}
    recs = [r for r in calib.read_results(base / "results.jsonl") if not r.get("error") and r["cell_id"] in cells]
    # one record per (cell, policy, k): the latest
    latest = {}
    for r in recs:
        latest[(r["cell_id"], r["policy_name"], r["repeat"])] = r
    return cells, list(latest.values())


def items_for(record: dict, cell: dict, measure: dict | None = None) -> dict:
    import hashlib

    tag = "" if measure is None else "-" + hashlib.sha256(json.dumps(measure, sort_keys=True).encode()).hexdigest()[:12]
    path = ITEMS / record["cache_key"][:2] / f"{record['cache_key']}{tag}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    items = calib.scored_items(record, cell, measure)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(items))
    tmp.replace(path)
    return items


def table(group: str, root: Path | None = None, measure_fn=None) -> dict:
    """{(N, load): {policy: [(segment, k, items), ...]}}"""
    cells, recs = group_records(group, root)
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in recs:
        cell = cells[r["cell_id"]]
        measure = None if measure_fn is None else measure_fn(cell, r)
        it = items_for(r, cell, measure)
        key = (cell["num_workers"], cell["load"]["per_worker"])
        pol = "default" if r["policy_name"].startswith("default") else r["policy_name"]
        out[key][pol].append((cell["segment"], r["repeat"], it, r))
    return out


def mean_gf(entries, itl, slow) -> float:
    vals = [calib.gf(it, itl, slow)[0] for _, _, it, _ in entries]
    return float(np.mean(vals)) if vals else math.nan


def mean_gp(entries, itl, slow) -> float:
    vals = [calib.gf(it, itl, slow)[1] for _, _, it, _ in entries]
    return float(np.mean(vals)) if vals else math.nan


def pooled(entries, key: str) -> np.ndarray:
    arr = np.concatenate([it[key] for _, _, it, _ in entries])
    return arr


def itl_threshold(entries) -> float:
    itl = pooled(entries, "itl")
    itl = itl[~np.isnan(itl)]
    return float(ITL_MULT * np.percentile(itl, ITL_Q))


def solve_slowdown(entries, itl: float, target: float = 0.85) -> float:
    """Smallest S (to 1e-4 relative) with mean default good_frac >= target at ITL itl."""
    lo, hi = 1.0, 2.0
    if mean_gf(entries, itl, lo) >= target:
        return lo
    while mean_gf(entries, itl, hi) < target:
        hi *= 2.0
        if hi > 1e5:
            return math.inf
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if mean_gf(entries, itl, mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1 + 1e-5:
            break
    return hi


def drift(entries) -> float:
    """Open loop: median slowdown of the last third of in-window arrivals / the first third."""
    ratios = []
    for _, _, it, _ in entries:
        s = it["slow"]
        n = len(s)
        if n < 30:
            continue
        a, b = s[: n // 3], s[-(n // 3):]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) and len(b):
            ratios.append(float(np.median(b) / np.median(a)))
    return float(np.median(ratios)) if ratios else math.nan


def interp_level(loads, gfs, target) -> float | None:
    """Per-worker load where gf crosses target (log-linear between the bracketing grid points,
    first crossing from light load); None if the grid does not bracket it."""
    pts = sorted(zip(loads, gfs))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= target > y1 or y0 > target >= y1:
            if y0 == y1:
                return x0
            f = (y0 - target) / (y0 - y1)
            return float(math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))))
    return None


# --------------------------------------------------------------------------------------------
# Stationarity, knees, band levels

DRIFT_MAX = 1.15  # open loop: last-third / first-third mean slowdown of in-window arrivals
PLATEAU = 0.95  # closed loop: throughput within 5% of its grid maximum


def drift_mean(entries) -> float:
    """Median over (segment, k) of mean slowdown(last third) / mean slowdown(first third)."""
    ratios = []
    for _, _, it, _ in entries:
        s = it["slow"]
        n = len(s)
        if n < 30:
            continue
        a, b = s[: n // 3], s[-(n // 3):]
        a = a[~np.isnan(a)]
        # incomplete late arrivals count as an unbounded slowdown
        b = np.where(np.isnan(b), 1e6, b)
        if len(a):
            ratios.append(float(np.mean(np.minimum(b, 1e6)) / np.mean(a)))
    return float(np.median(ratios)) if ratios else math.nan


def throughput(entries) -> float:
    """Mean over (segment, k) of completions per second in the measurement window."""
    vals = [it["done"].sum() / it["window_s"] for _, _, it, _ in entries if it["window_s"]]
    return float(np.mean(vals)) if vals else math.nan


def log_interp(x0, y0, x1, y1, y):
    if y0 == y1:
        return x0
    f = (y - y0) / (y1 - y0)
    return float(math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))))


def is_open_group(group: str) -> bool:
    return group.endswith("open")


def open_knee(tab, n) -> dict:
    loads = sorted(k[1] for k in tab if k[0] == n)
    drifts = [drift_mean(tab[(n, v)]["default"]) for v in loads]
    knee, last_ok = None, None
    for v, d, nv, nd in zip(loads, drifts, loads[1:] + [None], drifts[1:] + [None]):
        if not d <= DRIFT_MAX:
            break
        last_ok = v
        if nv is not None and not nd <= DRIFT_MAX:
            knee = log_interp(v, d, nv, min(nd, 1e3), DRIFT_MAX)
            break
    return {"loads": loads, "drift": drifts, "last_stationary": last_ok, "knee": knee}


def closed_knee(tab, n) -> dict:
    loads = sorted(k[1] for k in tab if k[0] == n)
    thr = [throughput(tab[(n, v)]["default"]) for v in loads]
    peak = max(thr)
    knee = None
    for v, t, pv, pt in zip(loads, thr, [None] + loads[:-1], [None] + thr[:-1]):
        if t >= PLATEAU * peak:
            knee = v if pv is None else log_interp(pv, pt, v, t, PLATEAU * peak)
            break
    return {"loads": loads, "throughput": thr, "peak": peak, "knee": knee}


def gf_curve(tab, n, policy, itl, slow):
    loads = sorted(k[1] for k in tab if k[0] == n)
    return loads, [mean_gf(tab[(n, v)].get(policy, []), itl, slow) for v in loads]


def gf_at(tab, n, policy, itl, slow, load) -> float:
    """Default/RR good fraction at an off-grid load (linear in log load between grid points)."""
    loads, g = gf_curve(tab, n, policy, itl, slow)
    if load <= loads[0]:
        return g[0]
    for x0, y0, x1, y1 in zip(loads, g, loads[1:], g[1:]):
        if x0 <= load <= x1:
            f = (math.log(load) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    return g[-1]


def itl_p95_at(tab, n, load) -> float:
    """1.2 x p95 ITL of default, interpolated in log load between grid points."""
    loads = sorted(k[1] for k in tab if k[0] == n)
    vals = [itl_threshold(tab[(n, v)]["default"]) for v in loads]
    if load <= loads[0]:
        return vals[0]
    for x0, y0, x1, y1 in zip(loads, vals, loads[1:], vals[1:]):
        if x0 <= load <= x1:
            f = (math.log(load) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    return vals[-1]


def level_loads(tab, n, itl, slow) -> dict:
    loads, g = gf_curve(tab, n, "default", itl, slow)
    out = {}
    for level, target in TARGETS.items():
        out[level] = interp_level(loads, g, target)
    return out


def solve_sla(tab, n, knee, iters: int = 30) -> dict:
    """Fixed point: default good_frac 0.65 at the knee (L3) with I = 1.2 x p95 ITL at L2."""
    loads = sorted(k[1] for k in tab if k[0] == n)
    lam2 = knee * 0.8
    history = []
    for _ in range(iters):
        itl = itl_p95_at(tab, n, lam2)
        lo, hi = 1.0, 2.0
        while gf_at(tab, n, "default", itl, hi, knee) < TARGETS["L3"]:
            hi *= 2
            if hi > 1e4:
                break
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if gf_at(tab, n, "default", itl, mid, knee) >= TARGETS["L3"]:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1 + 1e-6:
                break
        slow = hi
        levels = level_loads(tab, n, itl, slow)
        history.append({"itl": itl, "slow": slow, "lam2": levels["L2"]})
        if levels["L2"] is None or abs(levels["L2"] - lam2) / lam2 < 1e-4:
            lam2 = levels["L2"] or lam2
            break
        lam2 = levels["L2"]
    return {"itl_ms": itl, "e2e_slowdown": slow, "levels": level_loads(tab, n, itl, slow),
            "history": history}


# --------------------------------------------------------------------------------------------
# AgentX lanes warm-up rule

_span_memo: dict = {}


def copy_spans(cell: dict) -> np.ndarray:
    """Recorded span (ms) of every copy in a lowered cell trace (trace-intrinsic)."""
    meta_path = calib.resolve(cell["derived_meta"])
    key = str(meta_path)
    if key not in _span_memo:
        meta = json.loads(meta_path.read_text())
        manifest = json.loads(Path(meta["base_manifest"]).read_text())["plays"]
        _span_memo[key] = np.array([manifest[c["play"]]["span_ms"] for c in meta["copies"]], dtype=float)
    return _span_memo[key]


def first_copy_end_p90(record: dict) -> float:
    rows = calib.load_rows(record)
    plays: dict = defaultdict(lambda: [math.inf, -math.inf, None])
    for r in rows:
        p = plays[r["play_id"]]
        p[0] = min(p[0], r["arrival_time_ms"])
        p[1] = max(p[1], r["terminal_time_ms"] if r["terminal_time_ms"] is not None else -math.inf)
        p[2] = r.get("lane_id")
    first: dict = {}
    for start, end, lane in plays.values():
        if lane is None:
            continue
        if lane not in first or start < first[lane][0]:
            first[lane] = (start, end)
    ends = [end for _, end in first.values()]
    return float(np.percentile(ends, 90)) if ends else math.nan
