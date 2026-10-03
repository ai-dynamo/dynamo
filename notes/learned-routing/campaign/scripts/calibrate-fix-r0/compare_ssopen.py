"""Old (180 s warm-up, sweep2) vs new (lifetime warm-up, sweep3) ss-open curves at r0's sessions SLA."""
import json, sys
import numpy as np
import analyze as A
import calib

I0, S0 = 40.4165, 1.90627
old = A.table("ss-open", root=calib.CAL / "sweep2" / "ss-open")
new = A.table("ss-open", root=calib.FIX / "sweep3" / "ss-open")
out = {}
for n in (2, 4, 6, 8, 16, 32):
    ko, kn = A.open_knee(old, n), A.open_knee(new, n)
    rows = []
    for v in sorted(k[1] for k in new if k[0] == n):
        e_o, e_n = old.get((n, v)), new[(n, v)]
        rows.append({"load": v,
                     "gf_def_old": A.mean_gf(e_o["default"], I0, S0) if e_o else None,
                     "gf_def_new": A.mean_gf(e_n["default"], I0, S0),
                     "gf_rr_old": A.mean_gf(e_o["round_robin"], I0, S0) if e_o else None,
                     "gf_rr_new": A.mean_gf(e_n["round_robin"], I0, S0),
                     "drift_old": A.drift_mean(e_o["default"]) if e_o else None,
                     "drift_new": A.drift_mean(e_n["default"])})
    out[n] = {"knee_old": ko["knee"], "knee_new": kn["knee"], "rows": rows,
              "levels_old_at_r0_sla": A.level_loads(old, n, I0, S0), "levels_new_at_r0_sla": A.level_loads(new, n, I0, S0)}
    print(f"N={n} knee old {ko['knee']} new {kn['knee']} | levels@r0 SLA old {A.level_loads(old, n, I0, S0)} new {A.level_loads(new, n, I0, S0)}")
    for r in rows:
        print("   ", " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
(calib.FIX / "out" / "ssopen_old_vs_new.json").write_text(json.dumps(out, indent=1))
