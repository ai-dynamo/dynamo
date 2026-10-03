"""Stationarity knee for open-loop sessions (fixer r0): doubled-history sweep of default@defaults.

For each train segment s0-s3, N and load v (sessions/s/worker), two replays measure the SAME
sessions (those starting in [2W, 2W + 540) s of replay time; W = the cell's lifetime warm-up):
  arm h1: source window [W s, (2W + 600) s), warm-up W   (history W)
  arm h2: source window [0, (2W + 600) s),   warm-up 2W  (history 2W)
hist_drift(v) = median over (segment, k) of mean slowdown(h2) / mean slowdown(h1) over the
in-window arrivals (incomplete requests count as slowdown 1e6). The open-loop knee is where
hist_drift crosses 1.15 (log-linear), i.e. where doubling the history shifts default's mean
slowdown by more than 15%: the same threshold as calibration r0's within-window drift test, which
the ramp-up had inflated and which, once the population is settled, no longer sees relaxation
slower than one window (audits/calibration/fix-r0.md).

usage: hist_sweep.py build | eval | analyze_r1   (HIST_SWEEP_DIR=hist_sweep for round 1)
GROUP: ss-open (base) or a transform tag (think0.5, ...) -- only ss-open feeds the SLA anchor.
"""
import copy, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import calib
import analyze as A
from sweep import session_template, NS

import os

# round 1 (W = one envelope lifetime, 600 s): hist_sweep; round 2 (W = two lifetimes): hist_sweep2
OUT = calib.FIX / os.environ.get("HIST_SWEEP_DIR", "hist_sweep2")
GRID = (0.3, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525) if OUT.name == "hist_sweep" else \
    (0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5)
# Round 2 knee: doubling the history (W -> 2W, same in-window sessions) lowers default@defaults'
# mean good fraction (under the fixed-point iterate's I, S) by more than GF_SHIFT_MAX relative.
GF_SHIFT_MAX = 0.05
SEGS = ["sessions:s0", "sessions:s1", "sessions:s2", "sessions:s3"]
K = 2
DRIFT_MAX = 1.15


def arms_for(template, cid, n, v):
    base = calib.session_open_cell(template, cid, n, v)
    src, spec = calib.session_open_job(template, n, v)  # same spec form as arm_jobs (index hits)
    s = float(base["load"]["value"])
    w = base["measure"]["warmup_ms"] / 1000.0
    end = (2 * w + calib.SESS_WINDOW_S + calib.SESS_TAIL_S) * 1000.0 * s
    out = []
    for arm, t0, warm in (("h1", w * 1000.0 * s, w), ("h2", 0.0, 2 * w)):
        d = calib.derive(src, dict(spec, window=[t0, end]))
        c = copy.deepcopy(base)
        c["cell_id"] = f"{cid}-{arm}"
        calib._attach_trace(c, d)
        c["measure"] = {"basis": "arrival", "warmup_ms": warm * 1000.0, "window_ms": calib.SESS_WINDOW_S * 1000.0}
        c["measure_trace"] = dict(base["measure_trace"], hist_arm=arm, source_window_ms=[t0, end], hist_base=cid)
        out.append(c)
    return out


def arm_jobs(template, n, v):
    src, spec = calib.session_open_job(template, n, v)
    s = float(f"{v * n:.6g}")
    w = calib.session_warmup_s(template)
    end = (2 * w + calib.SESS_WINDOW_S + calib.SESS_TAIL_S) * 1000.0 * s
    return [(src, spec), (src, dict(spec, window=[w * 1000.0 * s, end])), (src, dict(spec, window=[0.0, end]))]


def build():
    jobs = []
    for seg in SEGS:
        t = session_template(seg, "open_speedup")
        for n in NS:
            for v in GRID:
                jobs += arm_jobs(t, n, v)
    t0 = time.time()
    calib.derive_many(jobs, processes=10)
    print("derived", len(jobs), round(time.time() - t0, 1), flush=True)
    cells = []
    for seg in SEGS:
        t = session_template(seg, "open_speedup")
        for n in NS:
            for v in GRID:
                prefix = "cal-hs" if OUT.name == "hist_sweep" else "cal-hs2"
                cells += arms_for(t, f"{prefix}-ss-open-{seg.split(':')[1]}-n{n}-{v}", n, v)
    calib.write_cells(cells, OUT / "cells.jsonl")
    calib.save_index()
    print("cells", len(cells))


def evaluate():
    cells = [json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()]
    summary = calib.evaluate(cells, ["default"], K, OUT / "results.jsonl")
    summary["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(summary), flush=True)
    with (OUT / "eval_log.jsonl").open("a") as h:
        h.write(json.dumps(summary) + "\n")


def mean_slow(items):
    s = np.where(np.isnan(items["slow"]), 1e6, items["slow"])
    return float(np.mean(np.minimum(s, 1e6)))


def tables(I=None, S=None):
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()}
    by = defaultdict(dict)
    for r in calib.read_results(OUT / "results.jsonl"):
        if r.get("error") or r["cell_id"] not in cells:
            continue
        c = cells[r["cell_id"]]
        it = A.items_for(r, c)
        by[(c["measure_trace"]["hist_base"], r["repeat"])][c["measure_trace"]["hist_arm"]] = (c, it)
    rows = defaultdict(list)
    for (base, k), arms in by.items():
        if set(arms) != {"h1", "h2"}:
            continue
        c = arms["h1"][0]
        e = {"seg": c["segment"], "k": k, "ratio": mean_slow(arms["h2"][1]) / mean_slow(arms["h1"][1])}
        if I is not None:
            e["gf_h1"] = calib.gf(arms["h1"][1], I, S)[0]
            e["gf_h2"] = calib.gf(arms["h2"][1], I, S)[0]
        rows[(c["num_workers"], c["load"]["per_worker"])].append(e)
    return rows


def knee(rows, n):
    loads = sorted(v for (m, v) in rows if m == n)
    drifts = [float(np.median([e["ratio"] for e in rows[(n, v)]])) for v in loads]
    kn = None
    for v, d, nv, nd in zip(loads, drifts, loads[1:] + [None], drifts[1:] + [None]):
        if not d <= DRIFT_MAX:
            kn = loads[0] if v == loads[0] else None
            break
        if nv is not None and not nd <= DRIFT_MAX:
            kn = A.log_interp(v, d, nv, min(nd, 1e3), DRIFT_MAX)
            break
    return {"loads": loads, "hist_drift": drifts, "knee": kn}


_ITEMS = None


def _items():
    """{(N, load): [(segment, k, items_h1, items_h2)]} (cached in memory)."""
    global _ITEMS
    if _ITEMS is None:
        cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()}
        by = defaultdict(dict)
        for r in calib.read_results(OUT / "results.jsonl"):
            if r.get("error") or r["cell_id"] not in cells:
                continue
            c = cells[r["cell_id"]]
            by[(c["measure_trace"]["hist_base"], r["repeat"])][c["measure_trace"]["hist_arm"]] = (c, A.items_for(r, c))
        out = defaultdict(list)
        for (base, k), arms in by.items():
            if set(arms) == {"h1", "h2"}:
                c = arms["h1"][0]
                out[(c["num_workers"], c["load"]["per_worker"])].append((c["segment"], k, arms["h1"][1], arms["h2"][1]))
        _ITEMS = out
    return _ITEMS


def gf_shift_curve(n, I, S):
    rows = _items()
    loads = sorted(v for (m, v) in rows if m == n)
    g1 = [float(np.mean([calib.gf(a, I, S)[0] for _, _, a, _ in rows[(n, v)]])) for v in loads]
    g2 = [float(np.mean([calib.gf(b, I, S)[0] for _, _, _, b in rows[(n, v)]])) for v in loads]
    shift = [(1.0 - y / x) if x > 0 else float("inf") for x, y in zip(g1, g2)]
    return loads, g1, g2, shift


def gf_knee(n, I, S):
    """Lowest load where the doubled-history relative drop of default's good fraction crosses
    GF_SHIFT_MAX (linear interpolation in load); None if the grid never crosses it."""
    loads, g1, g2, shift = gf_shift_curve(n, I, S)
    knee = None
    for i, (v, d) in enumerate(zip(loads, shift)):
        if d > GF_SHIFT_MAX:
            if i == 0:
                knee = v
            else:
                v0, d0 = loads[i - 1], shift[i - 1]
                knee = v0 + (GF_SHIFT_MAX - d0) / (d - d0) * (v - v0)
            break
    return {"loads": loads, "gf_h1": g1, "gf_h2": g2, "rel_shift": shift, "knee": knee,
            "rule": f"doubled history (W -> 2W) lowers default's good fraction by > {GF_SHIFT_MAX:.0%}"}


def analyze(I=None, S=None):
    rows = tables(I, S)
    out = {}
    for n in NS:
        kn = knee(rows, n)
        if I is not None:
            kn["gf_h1"] = [float(np.mean([e["gf_h1"] for e in rows[(n, v)]])) for v in kn["loads"]]
            kn["gf_h2"] = [float(np.mean([e["gf_h2"] for e in rows[(n, v)]])) for v in kn["loads"]]
        out[str(n)] = kn
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "eval":
        evaluate()
    elif cmd == "analyze_r1":
        res = analyze(53.3996, 2.4296)
        (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
        for n, kn in res.items():
            print(n, "knee", kn["knee"])
            for v, d, g1, g2 in zip(kn["loads"], kn["hist_drift"], kn["gf_h1"], kn["gf_h2"]):
                print(f"    {v:6} hist_drift {d:.3f} gf_h1 {g1:.3f} gf_h2 {g2:.3f}")
