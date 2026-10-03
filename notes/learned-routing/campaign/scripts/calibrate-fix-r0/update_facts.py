"""Fixer r0: update facts/calibration.json and facts/noise_calibration.json for lr-cells-v5.

usage: update_facts.py STEP4_REPORT.json WARM_CHECK_REPORT.json
Snapshots calibration r0's facts to facts/superseded/calibration-r0/ first (no deletion). Sections
that only the sessions re-calibration changes are replaced; everything else is kept.
"""
from __future__ import annotations

import glob
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import calib
from learned_routing import HARNESS_VERSION
from learned_routing.cache import bindings_build_id

CR = calib.CR
FIX = calib.FIX
WT = Path("<worktree>")
KV_TOKENS = 301808


def snapshot():
    dst = CR / "facts" / "superseded" / "calibration-r0"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("calibration.json", "noise_calibration.json", "test_freeze.json"):
        if not (dst / name).exists():
            shutil.copy2(CR / "facts" / name, dst / name)
    return dst


def pressure(cell):
    meta = json.loads(calib.resolve(cell["derived_meta"]).read_text())
    tab = {float(k): v for k, v in meta["realized"]["distinct_block_tokens_per_trace_window_s"].items()}
    w = 100.0 * float(cell["load"]["value"])
    xs = sorted(tab)
    ext = w > xs[-1] or w < xs[0]
    if w <= xs[0]:
        a, b = xs[0], xs[1]
    elif w >= xs[-1]:
        a, b = xs[-2], xs[-1]
    else:
        a, b = next((a, b) for a, b in zip(xs, xs[1:]) if a <= w <= b)
    la, lb, ya, yb = math.log(a), math.log(b), math.log(tab[a]), math.log(tab[b])
    u = math.exp(ya + (math.log(w) - la) * (yb - ya) / (lb - la))
    return u / (cell["num_workers"] * KV_TOKENS), ext


def results_costs():
    out, n, wall = {}, 0, 0.0
    for path in sorted(glob.glob(str(FIX / "**" / "results.jsonl"), recursive=True)):
        seen, cnt, err, w = set(), 0, 0, 0.0
        for r in calib.read_results(Path(path)):
            key = r.get("cache_key")
            if key in seen or r.get("cached"):
                continue
            seen.add(key)
            cnt += 1
            err += bool(r.get("error"))
            w += float(r.get("wall_s") or 0.0)
        out[str(Path(path).relative_to(CR))] = {"replays": cnt, "errors": err, "replay_wall_s_sum": round(w, 1)}
        n += cnt
        wall += w
    return {"replays_total": n, "replay_wall_s_total": round(wall, 1), "by_results_file": out}


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    warm = json.loads(Path(sys.argv[2]).read_text())
    snap = snapshot()
    old = json.loads((snap / "calibration.json").read_text())
    dec = json.loads((FIX / "decisions.json").read_text())
    curves = json.loads((FIX / "curves.json").read_text())
    warmup_table = json.loads((FIX / "out" / "warmup_table.json").read_text())
    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cells = {}
    for split in ("train", "val", "test"):
        for line in (CR / "cells" / f"{split}.jsonl").read_text().splitlines():
            c = json.loads(line)
            cells[c["cell_id"]] = c
    for line in (FIX / "final_r5" / "noise.jsonl").read_text().splitlines():
        c = json.loads(line)
        cells[c["cell_id"]] = c
    per_cell = report["per_cell"]
    facts = dict(old)
    facts["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    facts["stage"] = "calibration (phase 2), sessions family re-calibrated by calibration fixer r0"
    facts["worktree_head"] = head
    facts["harness_version"] = HARNESS_VERSION
    facts["bindings_build_id"] = bindings_build_id(calib.LAYOUT.cache_dir)["build_id"]
    facts["sla"] = dec["sla"]
    facts["levels_per_worker"] = dec["levels"]
    facts["transform_levels_per_worker"] = dec["transform_levels"]
    facts["curves"] = curves
    facts["curves_file"] = "runs/calibrate-fix-r0/curves.json (r0: runs/calibrate/curves.json)"
    facts["decisions_file"] = "runs/calibrate-fix-r0/decisions.json (r0: runs/calibrate/decisions.json)"
    facts["scripts"] = "runs/calibrate/scripts/ (r0) and runs/calibrate-fix-r0/scripts/ (fixer r0: sessions)"
    atom = defaultdict(list)
    for e in per_cell.values():
        if e["default"]["slowdown_atom_frac_mean"] is not None:
            atom[e["family"]].append(e["default"]["slowdown_atom_frac_mean"])
    facts["slowdown_atom_frac_default_mean_by_family"] = {f: float(np.mean(v)) for f, v in atom.items()}
    facts["step4_reference"] = {
        "policies": ["default@defaults", "round_robin"],
        "replicates": 8,
        "results": "runs/calibrate-fix-r0/step4_r6/results.jsonl (non-session cells are result-cache hits of runs/calibrate/step4_r4)",
        "per_cell": per_cell,
        "rr_vs_default_differs_beyond_noise": report["rr_vs_default_differs_beyond_noise"],
        "rescoring_check": report["rescoring_check"],
        "errors": report["errors"],
    }
    facts["noise"] = {"pooled_paired_ratio_sd_rr_over_default": report["pooled_paired_ratio_sd"],
                      "file": "facts/noise_calibration.json"}
    # rules: only the sessions open-loop window and the sessions knee change
    rules = json.loads(json.dumps(old["rules"]))
    rules["windows"][1] = (
        "Open loop, sessions (fixer r0, audit calibration F1): arrival basis, 540 s window in replay time after a "
        "lifetime-derived warm-up of TWO envelope lifetimes, W = 2 x max(180 s, ceil to 60 s of the p99 of "
        "m x sum(think delays) + 3 x sum(E0) over the train segments' sessions (seeds 0-3, starts in [0, 2000) s) with "
        "the cell's transform applied, m = think_mult); W = 960 s (think 0.25, 0.5), 1200 s (think 1, incl. islu1.5 and "
        "root2), 1320 s (think 1.5), 2160 s (think 4.0). The derived trace keeps sessions starting in "
        "[0, (W + 540 + 60) s x speedup) of the long seed trace (60 s tail of continued arrivals). Calibration r0 used "
        "a fixed 180 s warm-up.")
    rules["sla"].insert(3, (
        "Sessions open-loop knee (fixer r0): with the population settled, the within-window drift test no longer sees "
        "relaxation slower than one window, so the sessions knee is the load at which doubling the warm-up history "
        "(W -> 2W, the same in-window sessions, default@defaults, train segments s0-s3 x 2 replicates) lowers default's "
        "mean good fraction by more than 5% relative (linear interpolation; runs/calibrate-fix-r0/hist_sweep2). The "
        "knee depends on (I, S), so it is part of the fixed point: knee -> L3 anchor = 0.9 x knee -> r0's (I, S) fixed "
        "point -> knee, to 1e-4 relative. Mooncake and AgentX knees and SLAs are unchanged."))
    facts["rules"] = rules
    facts["session_open_warmup"] = {
        "rule_id": "session-lifetime-p99-kappa3-x2-v1",
        "kappa": calib.SESS_WARMUP_KAPPA, "quantile": calib.SESS_WARMUP_QUANTILE, "min_s": calib.SESS_WARMUP_MIN_S,
        "lifetimes": calib.SESS_WARMUP_LIFETIMES,
        "why_two_lifetimes": "with one envelope lifetime (600 s at think 1) default's good fraction at N = 8, 0.45 "
                             "sessions/s/worker was 13% above its 2400 s-history value; with two it is within 1-3% at "
                             "loads up to 0.45 (runs/calibrate-fix-r0/conv_check); round-1 doubled-warm-up check at "
                             "one lifetime: runs/calibrate-fix-r0/warm_check/report.json",
        "by_transform_tag": warmup_table["by_transform_tag"],
        "kappa_basis": "steady-state contended service stretch sum(e2e)/sum(E0) per session: default 1.7-2.0 mean, round robin 2.3-2.7 mean (runs/calibrate-fix-r0/out/late_lifetimes.json, auditor's 900 s warm-up replays)",
        "doubled_warmup_check": {"file": "runs/calibrate-fix-r0/warm_check_r2/report.json", "summary": warm["summary"]},
    }
    # cache pressure of the open-loop session cells (same formula as r0)
    cp = json.loads(json.dumps(old["cache_pressure_open_loop"]))
    for cid in list(cp["per_cell"]):
        c = cells[cid]
        if c["family"] != "synthetic_sessions":
            continue
        p, ext = pressure(c)
        cp["per_cell"][cid] = {"N": c["num_workers"], "extrapolated": ext, "family": c["family"],
                               "level": c["load"]["level"], "pressure": p, "split": c["split"]}
    summ = {}
    for split in ("train", "val", "test"):
        vals = [e["pressure"] for e in cp["per_cell"].values() if e["split"] == split]
        summ[split] = {"cells": len(vals), "min": min(vals), "median": float(np.median(vals)), "max": max(vals),
                       "below_0.5": sum(v < 0.5 for v in vals), "0.5_to_1.5": sum(0.5 <= v <= 1.5 for v in vals),
                       "above_1.5": sum(v > 1.5 for v in vals)}
    cp["summary_by_split"] = summ
    facts["cache_pressure_open_loop"] = cp
    # cost projection (same definitions as r0)
    per_row = defaultdict(list)
    for cid, e in per_cell.items():
        if e["split"] in ("train", "val"):
            per_row[e["family"]].append(e["default"]["replay_wall_s_sum"] / e["default"]["replicates"] / cells[cid]["trace_rows"])
    per_row = {f: float(np.mean(v)) for f, v in per_row.items()}
    per_row["fast25_conversation"] = per_row["mooncake"]
    per_row["fast25_synthetic"] = per_row["mooncake"]
    passes = defaultdict(float)
    train_fam = defaultdict(float)
    for e in per_cell.values():
        for pol in ("default", "round_robin"):
            passes[f"{e['split']}|{pol}"] += e[pol]["replay_wall_s_sum"] / e[pol]["replicates"]
        if e["split"] == "train":
            train_fam[e["family"]] += e["default"]["replay_wall_s_sum"] / e["default"]["replicates"]
    test = [c for c in cells.values() if c["split"] == "test"]
    cost = dict(old["cost_projection"])
    cost.update({
        "basis": old["cost_projection"]["basis"].replace("runs/calibrate/step4_r4", "runs/calibrate-fix-r0/step4_r6 (sessions re-measured; other families are r0's step4_r4 records via the result cache)"),
        "local_capacity": "20 slots; r0 step 4 ran 1,008 replays (11,566 replay-s) in 917.7 s wall; fixer r0 step 4 re-ran the 304 sessions replays (2,506 replay-s) in 212.8 s wall",
        "replay_s_per_policy_per_replicate_pass": {k: round(v, 1) for k, v in sorted(passes.items())},
        "replay_s_per_trace_row_by_family_default": per_row,
        "test_pass_replay_s_per_policy_per_replicate_estimate": round(sum(c["trace_rows"] * per_row[c["family"]] for c in test), 1),
        "train_pass_replay_s_by_family_default": {k: round(v, 1) for k, v in sorted(train_fam.items())},
    })
    facts["cost_projection"] = cost
    costs = dict(old["costs"])
    costs["fixer_r0"] = results_costs()
    facts["costs"] = costs
    lessons = json.loads(json.dumps(old["lessons"]))
    lessons["applied"]["LR-01"] = (
        lessons["applied"]["LR-01"] + "; fixer r0: action 5 (doubled warm-up) run at calibration on every open-loop "
        "session train/val/noise cell (runs/calibrate-fix-r0/warm_check_r2) and used to define the sessions open-loop knee")
    lessons["partially_applied_or_rejected"]["LR-01 (10-minute warm-up)"] = (
        "Mooncake windows keep B2's 4-minute trace warm-up (six disjoint windows; single-turn rows, no population "
        "ramp); sessions open loop now uses a lifetime-derived warm-up of 480-1080 s of replay (fixer r0, audit F1) "
        "instead of 180 s")
    facts["lessons"] = lessons
    summary = dict(old["summary"])
    lv = dec["levels"]["synthetic_sessions"]
    summary["levels_n8_per_worker"] = dict(summary["levels_n8_per_worker"])
    summary["levels_n8_per_worker"]["synthetic_sessions"] = {m: lv[m]["8"] for m in lv}
    summary["sla"] = dict(summary["sla"])
    s = dec["sla"]["synthetic_sessions"]
    summary["sla"]["synthetic_sessions"] = {"e2e_slowdown": s["e2e_slowdown"], "itl_ms": s["itl_ms"],
                                            "itl_only_fail_frac_default_at_L2": s["itl_only_fail_at_l2_grid"]["default"],
                                            "ttft_ms": None}
    gf_min = min(e["default"]["good_frac_window_mean"] for e in per_cell.values() if e["split"] in ("train", "val"))
    summary["step4"] = (
        f"fixer r0: {sum(e[p]['replicates'] for e in per_cell.values() for p in ('default', 'round_robin'))} reference records (63 train+val+noise cells x default@defaults and round_robin x 8 "
        f"CRN replicates; the 44 non-session cells are result-cache hits of r0's step 4), {len(report['errors'])} errors; "
        f"every train/val cell has default good_frac_window >= {gf_min:.3f}")
    test_sha = json.loads((CR / "facts" / "test_freeze.json").read_text())["test_jsonl"]["sha256"]
    summary["test_freeze"] = f"facts/test_freeze.json (cells/test.jsonl sha256 {test_sha}, 60 cells; re-frozen by fixer r0, r0's freeze kept at facts/superseded/calibration-r0/)"
    summary["fixer_r0"] = "audits/calibration/fix-r0.md"
    facts["summary"] = summary
    facts["fix_r0"] = {
        "finding": "audits/calibration/independent-rerun-r0.md F1 (major): sessions open-loop 180 s warm-up shorter than the session-population relaxation",
        "fix": "audits/calibration/fix-r0.md",
        "sessions_sla_r0": {k: old["sla"]["synthetic_sessions"][k] for k in ("itl_ms", "e2e_slowdown", "anchor_knee_n8", "anchor_l3_n8", "l2_n8")},
        "sessions_sla_fix_r0": {k: s[k] for k in ("itl_ms", "e2e_slowdown", "anchor_knee_n8", "anchor_l3_n8", "l2_n8")},
        "sessions_levels_r0": old["levels_per_worker"]["synthetic_sessions"],
        "unchanged": "Mooncake, FAST25 and AgentX SLAs, levels and cells are byte-identical to calibration r0 (lr-cells-v4)",
    }
    (CR / "facts" / "calibration.json").write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")
    noise = {
        "schema": "learned-routing.noise-calibration.v1",
        "written": facts["written"],
        "protocol": "crn-order-v1 / crn-spread-v1 (arrival_spread_ms cells) / crn-order-v1 copy permutation (agentic_mooncake); 8 replicates per cell",
        "reference": "default@defaults (seed k+1); comparison policy round_robin",
        "pooled_paired_ratio_sd_by_family_N_mode": report["pooled_paired_ratio_sd"],
        "per_cell_default_goodput_cv": {cid: e["default"]["goodput_cv"] for cid, e in per_cell.items()},
        "per_cell_paired_ratio": {cid: e["paired_ratio_rr_over_default"] for cid, e in per_cell.items()},
        "rr_vs_default_differs_beyond_noise": report["rr_vs_default_differs_beyond_noise"],
        "note": "noise-* cells are calibration-only cells on train segments at N = 2, 16, 32 (noise.json calibration_todo); they are not split cells. Fixer r0 re-measured the sessions cells (lr-cells-v5); r0's file is kept at facts/superseded/calibration-r0/noise_calibration.json",
    }
    (CR / "facts" / "noise_calibration.json").write_text(json.dumps(noise, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"snapshot": str(snap), "sessions_sla": facts["fix_r0"]["sessions_sla_fix_r0"],
                      "cost_test_estimate": cost["test_pass_replay_s_per_policy_per_replicate_estimate"],
                      "cache_pressure_summary": summ}, indent=1))


if __name__ == "__main__":
    main()
