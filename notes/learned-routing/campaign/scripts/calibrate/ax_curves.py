"""AgentX lanes curves with the warm-up rule (multiplier x p90 copy span), knee and SLA."""
import sys, math, json
import numpy as np
import analyze as A

MULT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5


def measure_fn(cell, record):
    q90 = float(np.percentile(A.copy_spans(cell), 90))
    return {"basis": "completion", "end": "full_occupancy", "warmup_ms": float(round(MULT * q90 / 1000.0) * 1000.0)}


tab = A.table("ax-lanes", measure_fn=measure_fn)
kn8 = A.closed_knee(tab, 8)
sla = A.solve_sla(tab, 8, kn8["knee"])
I, S = sla["itl_ms"], sla["e2e_slowdown"]
print(f"ax-lanes mult={MULT} N8 knee={kn8['knee']} -> I={I:.3f} S={S:.4f} levels={sla['levels']}")
for n in sorted({k[0] for k in tab}):
    kn = A.closed_knee(tab, n)
    print(f"--- N{n} knee={kn['knee']} peak_thr={kn['peak']:.4f}")
    for v in kn["loads"]:
        e = tab[(n, v)]
        d, r = e["default"], e.get("round_robin", [])
        sd = A.pooled(d, "slow"); sd = sd[~np.isnan(sd)]
        sr = A.pooled(r, "slow"); sr = sr[~np.isnan(sr)]
        print(f"  l {v:<4g} n/rep {len(sd)//max(len(d),1):6d} 1.2p95itl {A.itl_threshold(d):6.1f} sd50 {np.median(sd):5.2f} sd85 {np.percentile(sd,85):5.2f} RRsd85 {np.percentile(sr,85):5.2f} thr {A.throughput(d):.4f}/{A.throughput(r):.4f} gf def {A.mean_gf(d, I, S):.3f} RR {A.mean_gf(r, I, S):.3f} gp def {A.mean_gp(d, I, S):.4f} RR {A.mean_gp(r, I, S):.4f} win {np.mean([it['window_s'] for _,_,it,_ in d]):.0f}s")
    lv = A.level_loads(tab, n, I, S)
    print("   levels", {k: (None if x is None else round(x, 3)) for k, x in lv.items()},
          {k: (None if x is None else round(A.gf_at(tab, n, 'round_robin', I, S, x), 3)) for k, x in lv.items()})
