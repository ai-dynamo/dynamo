"""Step 4 analysis: per-cell goodput of default@defaults and round_robin, noise, rescoring check.

usage: step4_report.py OUT.json CELLS.jsonl... -- RESULTS.jsonl...
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import calib
from learned_routing import noise
from learned_routing.cells import Cell

args = sys.argv[1:]
out_path = Path(args.pop(0))
sep = args.index("--")
cell_files, result_files = args[:sep], args[sep + 1:]
cells = {}
for f in cell_files:
    for line in Path(f).read_text().splitlines():
        c = json.loads(line)
        cells[c["cell_id"]] = c
shas = {cid: Cell(raw=c, layout=calib.LAYOUT).content_sha() for cid, c in cells.items()}
recs = {}
errors = []
for f in result_files:
    for r in calib.read_results(Path(f)):
        cid = r["cell_id"]
        if cid not in cells:
            continue
        if r.get("cell_sha") != shas[cid]:
            continue
        if r.get("error"):
            errors.append({"cell_id": cid, "policy": r["policy_name"], "k": r["repeat"], "error": r["error"][:300]})
            continue
        pol = "default" if r["policy_name"].startswith("default") else r["policy_name"]
        recs[(cid, pol, int(r["repeat"]))] = r

per_cell = {}
ratios_by_cell = {}
check = {"records": 0, "max_abs_diff_good_frac_window": 0.0}
for cid, c in sorted(cells.items()):
    entry = {"family": c["family"], "split": c["split"], "mode": c["load"]["mode"], "level": c["load"]["level"],
             "num_workers": c["num_workers"], "segment": c.get("segment"), "load_value": c["load"]["value"],
             "per_worker": c["load"].get("per_worker")}
    for pol in ("default", "round_robin"):
        rs = [recs[(cid, pol, k)] for k in range(8) if (cid, pol, k) in recs]
        gp = [r["goodput_rps_window"] for r in rs]
        gf = [r["good_frac_window"] for r in rs]
        entry[pol] = {
            "replicates": len(rs),
            "goodput_rps_window_mean": float(np.mean(gp)) if gp else None,
            "goodput_rps_window_sd": float(np.std(gp, ddof=1)) if len(gp) > 1 else None,
            "good_frac_window_mean": float(np.mean(gf)) if gf else None,
            "good_frac_window_sd": float(np.std(gf, ddof=1)) if len(gf) > 1 else None,
            "window_requests_mean": float(np.mean([r["window_requests"] for r in rs])) if rs else None,
            "slowdown_atom_frac_mean": float(np.mean([r["slowdown_atom_frac"] for r in rs if r.get("slowdown_atom_frac") is not None])) if rs else None,
            "ttft_p50_mean": float(np.mean([r["ttft_p50"] for r in rs])) if rs else None,
            "itl_p90_mean": float(np.mean([r["itl_p90"] for r in rs])) if rs else None,
            "replay_wall_s_sum": float(sum(r["wall_s"] for r in rs)),
            "window_below_cap_frac_max": max((r.get("window_below_cap_frac") or 0.0) for r in rs) if rs else None,
        }
        entry[pol]["goodput_cv"] = (entry[pol]["goodput_rps_window_sd"] / entry[pol]["goodput_rps_window_mean"]
                                    if entry[pol]["goodput_rps_window_sd"] is not None and entry[pol]["goodput_rps_window_mean"] else None)
    ks = [k for k in range(8) if (cid, "default", k) in recs and (cid, "round_robin", k) in recs]
    rat = [recs[(cid, "round_robin", k)]["goodput_rps_window"] / recs[(cid, "default", k)]["goodput_rps_window"]
           for k in ks if recs[(cid, "default", k)]["goodput_rps_window"]]
    if len(rat) >= 2:
        ratios_by_cell[cid] = rat
    entry["paired_ratio_rr_over_default"] = {
        "k": ks, "mean": float(np.mean(rat)) if rat else None,
        "sd": float(np.std(rat, ddof=1)) if len(rat) > 1 else None,
    }
    per_cell[cid] = entry
    # rescoring check: offline gf at the cell's SLA == the harness record
    for pol in ("default", "round_robin"):
        for k in range(8):
            r = recs.get((cid, pol, k))
            if r is None:
                continue
            items = calib.scored_items(r, c)
            mine = calib.gf(items, c["sla"]["itl_ms"], c["sla"]["e2e_slowdown"])[0]
            check["records"] += 1
            check["max_abs_diff_good_frac_window"] = max(check["max_abs_diff_good_frac_window"], abs(mine - r["good_frac_window"]))

# pooled paired-ratio sd per (family, N, mode)
groups = defaultdict(dict)
for cid, rat in ratios_by_cell.items():
    e = per_cell[cid]
    groups[(e["family"], e["num_workers"], e["mode"])][cid] = rat
pooled = {f"{fam}|N{n}|{mode}": {"cells": len(g), "pooled_sd": noise.pooled_sd(g)} for (fam, n, mode), g in sorted(groups.items())}

# "RR differs from default beyond noise" on validation (contract rule, Amendment A1)
verdicts = {}
for split in ("train", "val"):
    paired = {}
    segs = {}
    for cid, e in per_cell.items():
        if e["split"] != split or cid not in ratios_by_cell:
            continue
        paired.update({(cid, k): v for k, v in zip(per_cell[cid]["paired_ratio_rr_over_default"]["k"], ratios_by_cell[cid])})
        segs[cid] = e["segment"]
    if paired:
        by_cell = defaultdict(list)
        for (cid, k), v in paired.items():
            by_cell[cid].append(v)
        try:
            from dataclasses import asdict

            v = noise.differs_beyond_noise(dict(by_cell), segs)
            verdicts[split] = {**asdict(v), "differs": v.differs}
        except Exception as exc:  # report, never hide
            verdicts[split] = f"error: {type(exc).__name__}: {exc}"

out = {"per_cell": per_cell, "pooled_paired_ratio_sd": pooled, "rr_vs_default_differs_beyond_noise": verdicts,
       "rescoring_check": check, "errors": errors,
       "replay_wall_s_total": float(sum(r["wall_s"] for r in recs.values()))}
out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps({"cells": len(per_cell), "errors": len(errors), "check": check, "pooled": pooled, "verdicts": verdicts}, indent=1, default=str))
