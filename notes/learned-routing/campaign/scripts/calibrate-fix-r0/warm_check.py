"""Doubled-warm-up check (LR-01 action 5; audit calibration F1) on every open-loop session cell
of the fixer's train/val/noise sets.

For a cell with warm-up W (replay s), speedup s and the 540 s window, three arms measure the SAME
sessions (those starting in [2W, 2W + 540) s of replay time, i.e. [2W s, (2W + 540) s) of the seed):
  arm w1x: source window [W s, (2W + 600) s), warm-up W      (the rule's warm-up)
  arm w2x: source window [0, (2W + 600) s),   warm-up 2W     (doubled)
  arm w180: source window [(2W - 180) s, (2W + 600) s), warm-up 180 s   (calibration r0's rule)
Only the history before the window differs. default@defaults and round_robin, k = 0..3.

usage: warm_check.py build FINAL_DIR | eval [--deadline S] | report
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import calib

import os

# round 1 (levels from the lifetime warm-up + within-window knee): warm_check; final round: warm_check_r2
OUT = calib.FIX / os.environ.get("WARM_CHECK_DIR", "warm_check")
ARMS = ("w1x", "w2x", "w180")
K = 4


def arm_cells(cell):
    src, spec = calib.session_open_job(cell, int(cell["num_workers"]), float(cell["load"]["per_worker"]))
    meta = {"source": {"path": str(src)}, "trace_block_size": 64}
    s = float(cell["load"]["value"])
    w = cell["measure"]["warmup_ms"] / 1000.0
    end = (2 * w + calib.SESS_WINDOW_S + calib.SESS_TAIL_S) * 1000.0 * s
    arms = {"w1x": (w * 1000.0 * s, w), "w2x": (0.0, 2 * w), "w180": ((2 * w - 180.0) * 1000.0 * s, 180.0)}
    out = []
    for arm, (t0, warm) in arms.items():
        sp = dict(spec, window=[t0, end])
        d = calib.derive(Path(meta["source"]["path"]), sp, block_size=meta["trace_block_size"])
        c = copy.deepcopy(cell)
        c["cell_id"] = f"{cell['cell_id']}-chk-{arm}"
        c["split"] = "calib"
        calib._attach_trace(c, d)
        c["measure"] = {"basis": "arrival", "warmup_ms": warm * 1000.0, "window_ms": calib.SESS_WINDOW_S * 1000.0}
        c["measure_trace"] = dict(cell.get("measure_trace", {}), check_arm=arm, source_window_ms=[t0, end],
                                  base_cell=cell["cell_id"])
        c["notes"] = "fixer r0 doubled-warm-up check arm (calibration-only, never a split cell)"
        out.append(c)
    return out


def build(final_dir):
    cells = []
    for f in ("train", "val", "noise"):
        for l in (Path(final_dir) / f"{f}.jsonl").read_text().splitlines():
            c = json.loads(l)
            if c["family"] == "synthetic_sessions" and c["load"]["mode"] == "open_speedup":
                cells.extend(arm_cells(c))
    calib.write_cells(cells, OUT / "cells.jsonl")
    calib.save_index()
    print("cells", len(cells))


def evaluate(opts):
    cells = [json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()]
    summary = calib.evaluate(cells, ["default", "round_robin"], K, OUT / "results.jsonl",
                             deadline_s=float(opts["--deadline"]) if "--deadline" in opts else None)
    summary["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(summary), flush=True)
    with (OUT / "eval_log.jsonl").open("a") as h:
        h.write(json.dumps(summary) + "\n")


def active_profile(record):
    rows = calib.load_rows(record)
    t0 = min(r["arrival_time_ms"] for r in rows)
    span = {}
    for r in rows:
        end = r["terminal_time_ms"] if r["terminal_time_ms"] is not None else float("inf")
        lo, hi = span.get(r["session_id"], (r["arrival_time_ms"], end))
        span[r["session_id"]] = (min(lo, r["arrival_time_ms"]), max(hi, end))
    ws, we = record["window_start_ms"], record["window_end_ms"]
    def active(t):
        return sum(1 for lo, hi in span.values() if lo <= t < hi)
    at_start = active(ws)
    mean_win = float(np.mean([active(t) for t in np.linspace(ws, we, 28)]))
    return at_start, mean_win


def report():
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()}
    recs = {}
    for r in calib.read_results(OUT / "results.jsonl"):
        if r.get("error") or r["cell_id"] not in cells:
            continue
        pol = "default" if r["policy_name"].startswith("default") else "round_robin"
        recs[(r["cell_id"], pol, r["repeat"])] = r
    bases = sorted({c["measure_trace"]["base_cell"] for c in cells.values()})
    out = {"per_cell": {}, "rule": __doc__}
    for b in bases:
        e = {}
        for arm in ARMS:
            cid = f"{b}-chk-{arm}"
            d = [recs.get((cid, "default", k)) for k in range(K)]
            r = [recs.get((cid, "round_robin", k)) for k in range(K)]
            if any(x is None for x in d + r):
                e[arm] = None
                continue
            gf = np.array([x["good_frac_window"] for x in d])
            gfr = np.array([x["good_frac_window"] for x in r])
            ratio = np.array([y["goodput_rps_window"] / x["goodput_rps_window"] for x, y in zip(d, r)])
            prof = [active_profile(x) for x in d[:2]]
            e[arm] = {"gf_default_mean": float(gf.mean()), "gf_default_sd": float(gf.std(ddof=1)),
                      "gf_rr_mean": float(gfr.mean()),
                      "rr_over_default_mean": float(ratio.mean()), "rr_over_default_sd": float(ratio.std(ddof=1)),
                      "window_requests_default_mean": float(np.mean([x["window_requests"] for x in d])),
                      "default_active_sessions_at_window_start_over_window_mean": float(np.mean([a / m for a, m in prof]))}
        ok = all(e.get(a) for a in ARMS)
        if ok:
            def z(a, b2, key):
                sa, sb = e[a][key.replace("mean", "sd")], e[b2][key.replace("mean", "sd")]
                se = float(np.sqrt((sa ** 2 + sb ** 2) / K)) or 1e-12
                return {"delta": e[a][key] - e[b2][key], "se": se, "z": (e[a][key] - e[b2][key]) / se}
            e["w1x_vs_w2x"] = {"gf_default": z("w1x", "w2x", "gf_default_mean"),
                               "rr_over_default": z("w1x", "w2x", "rr_over_default_mean")}
            e["w180_vs_w2x"] = {"gf_default": z("w180", "w2x", "gf_default_mean"),
                                "rr_over_default": z("w180", "w2x", "rr_over_default_mean")}
        out["per_cell"][b] = e
    summ = {}
    for cmp in ("w1x_vs_w2x", "w180_vs_w2x"):
        for key in ("gf_default", "rr_over_default"):
            ds = [v[cmp][key]["delta"] for v in out["per_cell"].values() if cmp in v]
            zs = [v[cmp][key]["z"] for v in out["per_cell"].values() if cmp in v]
            if ds:
                summ[f"{cmp}|{key}"] = {"cells": len(ds), "mean_delta": float(np.mean(ds)),
                                        "se_across_cells": float(np.std(ds, ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else None,
                                        "max_abs_delta": float(np.max(np.abs(ds))), "max_abs_z": float(np.max(np.abs(zs))),
                                        "cells_abs_z_gt_3": int(sum(abs(x) > 3 for x in zs))}
    out["summary"] = summ
    (OUT / "report.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(summ, indent=1))
    for b, e in out["per_cell"].items():
        if all(e.get(a) for a in ARMS):
            print(b, " ".join(f"{a}: gf {e[a]['gf_default_mean']:.3f} rr/def {e[a]['rr_over_default_mean']:.3f} fill {e[a]['default_active_sessions_at_window_start_over_window_mean']:.2f} |" for a in ARMS))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "eval":
        a = sys.argv[2:]
        evaluate({a[i]: a[i + 1] for i in range(0, len(a), 2)})
    else:
        report()
