"""ss-open default/RR good fraction by warm-up rule (r0 180 s, round 1 one lifetime, round 2 two
lifetimes) at the final sessions SLA and at r0's SLA. Writes ../out/ssopen_by_warmup.json"""
import json
import analyze as A
import calib

dec = json.loads((calib.FIX / "decisions.json").read_text())["sla"]["synthetic_sessions"]
SLAS = {"final": (dec["itl_ms"], dec["e2e_slowdown"]), "r0": (40.4165, 1.90627)}
roots = {"r0_180s": calib.CAL / "sweep2" / "ss-open", "round1_600s": calib.FIX / "sweep3" / "ss-open",
         "round2_1200s": calib.FIX / "sweep4" / "ss-open"}
tabs = {k: A.table("ss-open", root=v) for k, v in roots.items()}
out = {}
for sla_name, (I, S) in SLAS.items():
    for n in (4, 8, 32):
        for v in (0.3, 0.35, 0.4, 0.45, 0.5):
            row = {}
            for k, t in tabs.items():
                e = t.get((n, v))
                if e:
                    row[k] = {"gf_default": A.mean_gf(e["default"], I, S), "gf_rr": A.mean_gf(e["round_robin"], I, S),
                              "drift_default": A.drift_mean(e["default"])}
            out[f"{sla_name}|N{n}|{v}"] = row
            print(sla_name, n, v, " ".join(f"{k}: def {x['gf_default']:.3f} rr {x['gf_rr']:.3f}" for k, x in row.items()))
(calib.FIX / "out" / "ssopen_by_warmup.json").write_text(json.dumps(out, indent=1))
