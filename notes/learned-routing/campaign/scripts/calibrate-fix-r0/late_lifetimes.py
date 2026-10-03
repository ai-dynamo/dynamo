"""Contended session lifetimes near steady state: sessions starting in [900, 1300) s of the
auditor's 900 s warm-up replays (EV/longwarm), default and RR, all k. Writes ../out/late_lifetimes.json."""
import gzip, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "<campaign-root>/runs/calibrate/scripts")
import calib
EV = calib.CR / "runs/audit-calibration/independent-rerun-r0/longwarm"
e0 = calib.e0_table()
out = {}
for l in open(EV / "results.jsonl"):
    r = json.loads(l)
    rows = [json.loads(x) for x in gzip.open(r["per_request_path"], "rt")]
    t0 = min(x["arrival_time_ms"] for x in rows)
    span, se0, se2e, turns, delays = {}, defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(float)
    for x in rows:
        s = x["session_id"]
        end = x["terminal_time_ms"] if x["terminal_time_ms"] is not None else float("inf")
        lo, hi = span.get(s, (x["arrival_time_ms"], end))
        span[s] = (min(lo, x["arrival_time_ms"]), max(hi, end))
        se0[s] += e0(x["input_length"], x["output_length"])
        se2e[s] += x["e2e_latency_ms"] if x["e2e_latency_ms"] is not None else float("inf")
    late = [s for s, (lo, hi) in span.items() if 900e3 <= lo - t0 < 1300e3]
    dur = np.array([(span[s][1] - span[s][0]) / 1e3 for s in late])
    st = np.array([se2e[s] / se0[s] for s in late])
    key = f"{r['cell_id'].replace('-audit-warm900','')}|{r['policy_name'][:12]}|k{r['repeat']}"
    out[key] = {"sessions": len(late), "dur_p90": float(np.percentile(dur, 90)), "dur_p99": float(np.percentile(dur, 99)),
                "dur_max": float(dur.max()), "stretch_mean": float(np.mean(st)), "stretch_p50": float(np.median(st)),
                "stretch_p90": float(np.percentile(st, 90)), "good_frac_window": r["good_frac_window"]}
e0.persist()
(calib.CR / "runs/calibrate-fix-r0/out/late_lifetimes.json").write_text(json.dumps(out, indent=1, sort_keys=True))
for k, v in sorted(out.items()):
    print(k, {a: round(b, 2) for a, b in v.items()})
