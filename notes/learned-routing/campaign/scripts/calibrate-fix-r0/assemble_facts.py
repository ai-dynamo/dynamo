"""Write facts/calibration.json and facts/noise_calibration.json from the calibration artifacts.

usage: assemble_facts.py STEP4_REPORT.json
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import calib
from learned_routing import HARNESS_VERSION
from learned_routing.cache import bindings_build_id
from learned_routing.e0 import METHOD as E0_METHOD
from learned_routing.goodput import E2E_REL_TOL

CR = calib.CR
CAL = calib.CAL
WT = Path("<worktree>")


def replay_costs(paths):
    out = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        n = err = 0
        wall = 0.0
        seen = set()
        for r in calib.read_results(p):
            key = r.get("cache_key") or (r["cell_id"], r.get("policy_name"), r.get("repeat"))
            if key in seen or r.get("cached"):
                continue
            seen.add(key)
            n += 1
            err += bool(r.get("error"))
            wall += float(r.get("wall_s") or 0.0)
        out[str(p.relative_to(CR))] = {"replays": n, "errors": err, "replay_wall_s_sum": round(wall, 1)}
    return out


def eval_logs(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    dec = json.loads((CAL / "decisions.json").read_text())
    curves = json.loads((CAL / "curves.json").read_text())
    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    build = bindings_build_id(calib.LAYOUT.cache_dir)
    results = sorted(glob.glob(str(CAL / "**" / "results.jsonl"), recursive=True))
    costs = replay_costs(results)
    total_replays = sum(v["replays"] for v in costs.values())
    total_wall = sum(v["replay_wall_s_sum"] for v in costs.values())
    logs = eval_logs(glob.glob(str(CAL / "**" / "eval_log.jsonl"), recursive=True)
                     + glob.glob(str(CAL / "**" / "step4_log.jsonl"), recursive=True))
    per_cell = report["per_cell"]
    slowdown_atom = defaultdict(list)
    for e in per_cell.values():
        if e["default"]["slowdown_atom_frac_mean"] is not None:
            slowdown_atom[e["family"]].append(e["default"]["slowdown_atom_frac_mean"])
    facts = {
        "schema": "learned-routing.calibration.v1",
        "stage": "calibration (phase 2)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "worktree_head": head,
        "harness_version": HARNESS_VERSION,
        "bindings_build_id": build["build_id"],
        "e0": {"method": E0_METHOD, "e2e_rel_tol": E2E_REL_TOL,
               "how": "AIS chunked estimator of the request alone on an idle worker, no prefix reuse (learned_routing.e0, cached in runs/cache/e0); evaluation only, never seen by policies"},
        "slowdown_atom_frac_default_mean_by_family": {f: sum(v) / len(v) for f, v in slowdown_atom.items()},
        "sla": dec["sla"],
        "levels_per_worker": dec["levels"],
        "transform_levels_per_worker": dec.get("transform_levels", {}),
        "agentx_warmup": dec["agentx_warmup"],
        "rules": RULES,
        "step4_reference": {
            "policies": ["default@defaults", "round_robin"],
            "replicates": 8,
            "per_cell": per_cell,
            "rr_vs_default_differs_beyond_noise": report["rr_vs_default_differs_beyond_noise"],
            "rescoring_check": report["rescoring_check"],
            "errors": report["errors"],
        },
        "noise": {"pooled_paired_ratio_sd_rr_over_default": report["pooled_paired_ratio_sd"],
                  "file": "facts/noise_calibration.json"},
        "costs": {"replays_total": total_replays, "replay_wall_s_total": round(total_wall, 1),
                  "by_results_file": costs, "eval_runs": logs},
        "compute": {"where": "local host (24 cores, slot pool of 20)", "remote": None,
                    "why_no_cpu_cluster": "the whole calibration (sweeps + reference) ran locally well under the 6 h threshold; no CPU cluster allocation was made, so facts/remote.json was not written"},
        "curves_file": "runs/calibrate/curves.json",
        "curves": curves,
        "decisions_file": "runs/calibrate/decisions.json",
        "scripts": "runs/calibrate/scripts/",
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
        "note": "noise-* cells are calibration-only cells on train segments at N = 2, 16, 32 (noise.json calibration_todo); they are not split cells",
    }
    (CR / "facts" / "noise_calibration.json").write_text(json.dumps(noise, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"replays_total": total_replays, "replay_wall_s_total": round(total_wall, 1)}))


RULES = {
    "sla": [
        "Amendment A2 binds: no TTFT SLO (ttft_ms null); a request is good iff mean ITL <= I (skipped for OSL <= 1) and e2e <= S x E0(ISL, OSL) x (1 + 1e-6).",
        "One (I, S) per family from default@defaults on train segments at N = 8 in the family's primary mode (Mooncake: open loop; sessions: open loop; AgentX: lanes).",
        "Fixed point: L3 anchor = 0.9 x knee (open loop: the load where mean in-window slowdown drift, last third / first third of in-window arrivals, crosses 1.15; lanes: completions/s reaches 0.95 x its grid maximum); S makes default's good fraction 0.65 at the anchor; I = 1.2 x p95 of default's per-request mean ITL at L2 (the load where good fraction is 0.85 under (I, S)); iterate to convergence.",
        "Held-out FAST25 families (conversation, synthetic) have no train cells and inherit Mooncake's SLA.",
        "The same (I, S) applies to every split and every load mode of the family.",
    ],
    "levels": [
        "L1, L2, L3 = the per-worker load at which default@defaults' mean good fraction (over train segments x 2 CRN replicates) crosses 0.95, 0.85, 0.65 under the family SLA (log-linear interpolation between sweep points).",
        "Knee-matched per N (LR-12): levels are calibrated separately at each N in {2, 4, 6, 8, 16, 32} on train segments only; a test cell's load = per-worker level(family, mode, level, N) x N.",
        "Transformed cells use the same band rule calibrated on train segments with that transform applied at the cell's N (no test trace is replayed); held-out FAST25 families use Mooncake's per-worker levels in Mooncake's units.",
        "Units: Mooncake-format open loop = offered input tokens per second per worker over the untransformed base window (speedup = u x N / TR_base; transforms reuse the base window's TR); sessions open loop = sessions/s/worker (generator emits 1 session/s, speedup = rho x N, think delays pre-multiplied by the speedup so think time is load-invariant); closed loop = concurrent sessions per worker; AgentX = lanes per worker. Closed loop and lanes round N x per-worker to an integer.",
    ],
    "windows": [
        "Open loop, Mooncake-format windows: arrival basis, warmup_ms = measure_trace.warmup_ms / speedup, window_ms = measure_trace.window_ms / speedup.",
        "Open loop, sessions: arrival basis, 180 s warm-up and 540 s window in replay time; the derived trace keeps sessions starting in [0, (180 + 540 + 60) s x speedup) of the long seed trace (60 s tail of continued arrivals).",
        "Closed loop: completion basis, end full_occupancy; Mooncake-format windows keep B2's identity warm-up (warmup_trace_ms); sessions use the first 2C sessions as identity warm-up and 10C measured sessions (C = concurrency).",
        "AgentX lanes (lowered agentic_mooncake): completion basis, end full_occupancy, warmup_ms = 1.0 x p90 recorded span over the cell's copies; copies per lane = smallest of (8, 10, 12, 16, 24) whose least-loaded lane (recorded spans, every replicate k < 16) holds warm-up + 2 x mean copy span.",
    ],
    "replicates": [
        "Mooncake, FAST25 conversation and sessions cells carry arrival_spread_ms (3053 / 3000 / 1000 ms, each trace's logging quantum): protocol crn-spread-v1 re-draws each session's first arrival uniformly inside its slot per replicate. FAST25 synthetic (not quantized) keeps crn-order-v1.",
        "AgentX lowered cells: replicate k permutes the recycled copies (crn-order-v1 on agentic_mooncake); copies are drawn from the cell's own B2 play subset.",
    ],
    "levels_band_targets": {"L1": 0.95, "L2": 0.85, "L3": 0.65},
}


if __name__ == "__main__":
    main()
