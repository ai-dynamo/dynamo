"""Sweep v2 curves: per N, per load: default/RR stats, drift, throughput, knee and SLA at fixed I/S."""
import sys, math
import numpy as np
import analyze as A

group = sys.argv[1]
I = float(sys.argv[2]) if len(sys.argv) > 2 else None
S = float(sys.argv[3]) if len(sys.argv) > 3 else None
tab = A.table(group)
open_loop = group.endswith("open")
for n in sorted({k[0] for k in tab}):
    kn = A.open_knee(tab, n) if open_loop else A.closed_knee(tab, n)
    print(f"--- {group} N{n} knee={kn['knee']} last_stationary={kn.get('last_stationary')} peak_thr={kn.get('peak')}")
    for v in kn["loads"]:
        e = tab[(n, v)]
        d, r = e["default"], e.get("round_robin", [])
        sd = A.pooled(d, "slow"); sd = sd[~np.isnan(sd)]
        sr = A.pooled(r, "slow"); sr = sr[~np.isnan(sr)]
        tt = A.pooled(d, "ttft"); tt = tt[~np.isnan(tt)]
        itl = A.itl_threshold(d)
        extra = ""
        if I is not None:
            extra = f" gf@I,S def {A.mean_gf(d, I, S):.3f} RR {A.mean_gf(r, I, S):.3f}"
        print(f"  load {v:<8g} n/rep {len(sd)//max(len(d),1):5d} 1.2p95itl {itl:6.1f} sd50 {np.median(sd):5.2f} sd85 {np.percentile(sd,85):6.2f} RRsd85 {np.percentile(sr,85):7.2f} ttft50 {np.median(tt):7.0f} drift {A.drift_mean(d):6.2f} thr {A.throughput(d):6.2f}/{A.throughput(r):6.2f}{extra}")
