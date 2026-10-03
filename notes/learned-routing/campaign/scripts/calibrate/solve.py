"""SLA fixed point at N=8 (primary mode) and band levels at every N under that SLA."""
import sys, json, math
import numpy as np
import analyze as A

group = sys.argv[1]
drift_max = float(sys.argv[2]) if len(sys.argv) > 2 else 1.15
A.DRIFT_MAX = drift_max
tab = A.table(group)
open_loop = group.endswith("open")
kn = A.open_knee(tab, 8) if open_loop else A.closed_knee(tab, 8)
if len(sys.argv) > 4:
    I, S = float(sys.argv[3]), float(sys.argv[4])
    print(f"{group} given SLA I={I} S={S}; N8 knee={kn['knee']}")
else:
    sla = A.solve_sla(tab, 8, kn["knee"])
    I, S = sla["itl_ms"], sla["e2e_slowdown"]
    print(f"{group} drift_max={drift_max} N8 knee={kn['knee']:.1f} -> I={I:.2f} S={S:.4f} levels={sla['levels']} iters={len(sla['history'])}")
for n in sorted({k[0] for k in tab}):
    k = A.open_knee(tab, n) if open_loop else A.closed_knee(tab, n)
    lv = A.level_loads(tab, n, I, S)
    row = []
    for L in ("L1", "L2", "L3"):
        x = lv[L]
        if x is None:
            row.append(f"{L}=None"); continue
        d = A.gf_at(tab, n, "default", I, S, x); r = A.gf_at(tab, n, "round_robin", I, S, x)
        row.append(f"{L}={x:9.4g} def {d:.3f} RR {r:.3f} D {d-r:+.3f}")
    print(f"  N{n:<3d} knee {k['knee'] if k['knee'] else float('nan'):8.1f} | " + " | ".join(row))
