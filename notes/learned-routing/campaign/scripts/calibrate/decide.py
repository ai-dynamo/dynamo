"""Assemble calibration decisions from the v2 sweep: SLAs, per-N level table, curves, flags.

Writes CAL/decisions.json (input of build_final.py) and CAL/curves.json (facts/calibration.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import analyze as A
import calib

AX_WARMUP_MULT = 1.0
KNEE_FRACTION = 0.9  # L3 anchor: 10% headroom below the N = 8 stationarity / throughput knee
GROUPS = {
    "mooncake": {"primary": "mc-open", "modes": {"open_speedup": "mc-open", "closed_concurrency": "mc-closed"}},
    "synthetic_sessions": {"primary": "ss-open", "modes": {"open_speedup": "ss-open", "closed_concurrency": "ss-closed"}},
    "agentx": {"primary": "ax-lanes", "modes": {"agentic_lanes": "ax-lanes"}},
}
UNITS = {
    "mc-open": "input_tokens_per_s_per_worker",
    "mc-closed": "sessions_per_worker",
    "ss-open": "sessions_per_s_per_worker",
    "ss-closed": "sessions_per_worker",
    "ax-lanes": "lanes_per_worker",
}


def ax_measure(cell, record):
    q90 = float(np.percentile(A.copy_spans(cell), 90))
    return {"basis": "completion", "end": "full_occupancy",
            "warmup_ms": float(round(AX_WARMUP_MULT * q90 / 1000.0) * 1000.0)}


def table(group):
    tab = A.table(group, measure_fn=ax_measure if group == "ax-lanes" else None)
    # keep only (N, load) points that both policies reached (an in-flight sweep is partial)
    return {k: v for k, v in tab.items() if v.get("default") and v.get("round_robin")}


def knee(tab, group, n):
    return A.open_knee(tab, n) if group.endswith("open") else A.closed_knee(tab, n)


def itl_only_fail(entries, itl):
    vals = [1.0 - calib.gf(it, itl, None)[0] for _, _, it, _ in entries]
    return float(np.mean(vals))


def curve_rows(tab, group, n, itl, slow):
    rows = []
    kn = knee(tab, group, n)
    for v in kn["loads"]:
        e = tab[(n, v)]
        d, r = e["default"], e.get("round_robin", [])
        sd = A.pooled(d, "slow"); sd = sd[~np.isnan(sd)]
        rows.append({
            "load": v,
            "replicates_default": len(d),
            "replicates_rr": len(r),
            "gf_default": A.mean_gf(d, itl, slow),
            "gf_rr": A.mean_gf(r, itl, slow) if r else None,
            "goodput_default": A.mean_gp(d, itl, slow),
            "goodput_rr": A.mean_gp(r, itl, slow) if r else None,
            "gf_sd_default": float(np.std([calib.gf(it, itl, slow)[0] for _, _, it, _ in d], ddof=1)) if len(d) > 1 else None,
            "itl_1p2_p95_default": A.itl_threshold(d),
            "slowdown_p50_default": float(np.median(sd)),
            "slowdown_p85_default": float(np.percentile(sd, 85)),
            "drift_default": A.drift_mean(d),
            "throughput_default": A.throughput(d),
            "throughput_rr": A.throughput(r) if r else None,
        })
    return rows, kn


def tx_tables():
    """Transform sweeps: {(family, tag, mode, N): table} with the AgentX warm-up rule applied."""
    from collections import defaultdict as dd

    root = calib.CAL / "sweep2" / "tx"
    cells, recs = A.group_records("tx", root)
    out = dd(lambda: dd(lambda: dd(list)))
    for r in recs:
        c = cells[r["cell_id"]]
        measure = ax_measure(c, r) if c["family"] == "agentx" else None
        it = A.items_for(r, c, measure)
        combo = (c["family"], c["transform_tag"], c["load"]["mode"], c["num_workers"])
        pol = "default" if r["policy_name"].startswith("default") else r["policy_name"]
        out[combo][(c["num_workers"], c["load"]["per_worker"])][pol].append((c["segment"], r["repeat"], it, r))
    return {k: {kk: vv for kk, vv in v.items() if vv.get("default") and vv.get("round_robin")}
            for k, v in out.items()}


def tx_levels(decisions, curves):
    decisions["transform_levels"] = {}
    curves["transforms"] = {}
    for (fam, tag, mode, n), tab in sorted(tx_tables().items()):
        sla = decisions["sla"][fam]
        group = {"open_speedup": "x-open", "closed_concurrency": "x-closed", "agentic_lanes": "x-lanes"}[mode]
        lv = A.level_loads(tab, n, sla["itl_ms"], sla["e2e_slowdown"])
        rows, kn = curve_rows(tab, group, n, sla["itl_ms"], sla["e2e_slowdown"])
        key = f"{fam}|{tag}|{mode}|{n}"
        decisions["transform_levels"][key] = {lvl: (None if x is None else float(f"{x:.6g}")) for lvl, x in lv.items()}
        curves["transforms"][key] = {
            "knee": kn["knee"],
            "levels": {lvl: (None if x is None else {
                "per_worker": float(f"{x:.6g}"),
                "gf_default_interp": A.gf_at(tab, n, "default", sla["itl_ms"], sla["e2e_slowdown"], x),
                "gf_rr_interp": A.gf_at(tab, n, "round_robin", sla["itl_ms"], sla["e2e_slowdown"], x),
                "load_over_knee": (x / kn["knee"]) if kn["knee"] else None}) for lvl, x in lv.items()},
            "curve": rows,
        }


def main():
    decisions = {"rule_id": "lr-calib-v1", "calibration_id": "calibration-r0",
                 "sla": {}, "levels": {}, "agentx_warmup": {"multiplier": AX_WARMUP_MULT}}
    curves = {}
    for fam, spec in GROUPS.items():
        prim = spec["primary"]
        tab = table(prim)
        kn8 = knee(tab, prim, 8)
        sla = A.solve_sla(tab, 8, KNEE_FRACTION * kn8["knee"])
        itl, slow = sla["itl_ms"], sla["e2e_slowdown"]
        l2 = sla["levels"]["L2"]
        decisions["sla"][fam] = {
            "itl_ms": float(f"{itl:.6g}"),
            "e2e_slowdown": float(f"{slow:.6g}"),
            "anchor_mode": prim,
            "anchor_knee_n8": kn8["knee"],
            "anchor_l3_n8": KNEE_FRACTION * kn8["knee"],
            "knee_fraction": KNEE_FRACTION,
            "l2_n8": l2,
            "itl_only_fail_at_l2_grid": None,
            "fixed_point_history": sla["history"],
        }
        itl_r, slow_r = decisions["sla"][fam]["itl_ms"], decisions["sla"][fam]["e2e_slowdown"]
        decisions["levels"][fam] = {}
        curves[fam] = {}
        for mode, group in spec["modes"].items():
            t = tab if group == prim else table(group)
            decisions["levels"][fam][mode] = {}
            curves[fam][mode] = {"group": group, "unit": UNITS[group], "by_n": {}}
            for n in sorted({k[0] for k in t}):
                lv = A.level_loads(t, n, itl_r, slow_r)
                rows, kn = curve_rows(t, group, n, itl_r, slow_r)
                info = {}
                for level, x in lv.items():
                    if x is None:
                        info[level] = None
                        continue
                    k = kn["knee"]
                    info[level] = {
                        "per_worker": float(f"{x:.6g}"),
                        "gf_default_interp": A.gf_at(t, n, "default", itl_r, slow_r, x),
                        "gf_rr_interp": A.gf_at(t, n, "round_robin", itl_r, slow_r, x),
                        "load_over_knee": (x / k) if k else None,
                        "beyond_knee": bool(k is not None and x > k * (1 + 1e-9)),
                    }
                decisions["levels"][fam][mode][str(n)] = {
                    lvl: (None if v is None else v["per_worker"]) for lvl, v in info.items()
                }
                curves[fam][mode]["by_n"][str(n)] = {
                    "knee": kn["knee"],
                    "knee_rule": ("open: mean slowdown drift (last/first third of in-window arrivals) "
                                  f"crosses {A.DRIFT_MAX}") if group.endswith("open") else
                                 f"closed: completions/s reaches {A.PLATEAU} x the grid maximum",
                    "levels": info,
                    "curve": rows,
                }
        # ITL binding at L2 (N=8): nearest grid load to L2.
        t = tab
        loads = sorted(k[1] for k in t if k[0] == 8)
        near = min(loads, key=lambda v: abs(math.log(v) - math.log(l2))) if l2 else None
        if near is not None:
            decisions["sla"][fam]["itl_only_fail_at_l2_grid"] = {
                "grid_load": near,
                "default": itl_only_fail(t[(8, near)]["default"], itl_r),
                "round_robin": itl_only_fail(t[(8, near)]["round_robin"], itl_r),
            }
    if (calib.CAL / "sweep2" / "tx" / "results.jsonl").exists():
        tx_levels(decisions, curves)
    decisions["sla"]["fast25_conversation"] = dict(decisions["sla"]["mooncake"], inherited_from="mooncake")
    decisions["sla"]["fast25_synthetic"] = dict(decisions["sla"]["mooncake"], inherited_from="mooncake")
    (calib.CAL / "decisions.json").write_text(json.dumps(decisions, indent=1, sort_keys=True))
    (calib.CAL / "curves.json").write_text(json.dumps(curves, indent=1, sort_keys=True))
    print(json.dumps({f: decisions["sla"][f] | {"fixed_point_history": None} for f in decisions["sla"]}, indent=1))
    print(json.dumps(decisions["levels"], indent=1))


if __name__ == "__main__":
    main()
