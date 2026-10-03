"""Print sweep curves: per (N, load): default/RR at a given SLA rule."""
import json, sys, math
import numpy as np
import analyze as A

group = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
tab = A.table(group)
Ns = sorted({k[0] for k in tab})
print(group)
for n in Ns:
    loads = sorted(k[1] for k in tab if k[0] == n)
    for v in loads:
        e = tab[(n, v)]
        d, r = e.get("default", []), e.get("round_robin", [])
        if not d:
            continue
        I = A.itl_threshold(d)
        S = A.solve_slowdown(d, I)
        itl_only = A.mean_gf(d, I, None)
        g_r = A.mean_gf(r, I, S) if r else float("nan")
        gp_d, gp_r = A.mean_gp(d, I, S), (A.mean_gp(r, I, S) if r else float("nan"))
        sd = A.pooled(d, "slow"); sd = sd[~np.isnan(sd)]
        sr = A.pooled(r, "slow") if r else np.array([np.nan]); sr = sr[~np.isnan(sr)]
        tt = A.pooled(d, "ttft"); tt = tt[~np.isnan(tt)]
        print(f"N{n:<3d} load {v:<8g} reps {len(d)}/{len(r)}  I={I:7.1f} itl_only_gf={itl_only:.3f} S={S:6.2f} | RR gf={g_r:.3f} gp ratio RR/def={gp_r/gp_d if gp_d else float('nan'):.3f} | def sd50 {np.median(sd):5.2f} sd85 {np.percentile(sd,85):6.2f} RR sd85 {np.percentile(sr,85) if len(sr) else float('nan'):7.2f} ttft50 {np.median(tt):7.0f} drift {A.drift(d):5.2f}")
