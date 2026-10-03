"""Per-2-minute profile (arrival time bins) of good fraction, active sessions and in-flight requests
for chosen check-arm cells (default and RR, k=0). usage: profile.py RESULTS CELLS cell_id..."""
import json, sys
import numpy as np
import calib

res, cellsf = sys.argv[1], sys.argv[2]
want = set(sys.argv[3:])
cells = {json.loads(l)["cell_id"]: json.loads(l) for l in open(cellsf)}
e0 = calib.e0_table()
for r in calib.read_results(calib.Path(res)):
    if r["cell_id"] not in want or r["repeat"] != 0 or r.get("error"):
        continue
    c = cells[r["cell_id"]]
    I, S = c["sla"]["itl_ms"], c["sla"]["e2e_slowdown"]
    rows = calib.load_rows(r)
    t0 = min(x["arrival_time_ms"] for x in rows)
    span = {}
    for x in rows:
        end = x["terminal_time_ms"] if x["terminal_time_ms"] is not None else 1e18
        lo, hi = span.get(x["session_id"], (x["arrival_time_ms"], end))
        span[x["session_id"]] = (min(lo, x["arrival_time_ms"]), max(hi, end))
    B = 120e3
    out = []
    for b in range(0, int((r["window_end_ms"] - t0) // B) + 1):
        lo, hi = t0 + b * B, t0 + (b + 1) * B
        sel = [x for x in rows if lo <= x["arrival_time_ms"] < hi]
        if not sel:
            continue
        g = 0
        for x in sel:
            if x["e2e_latency_ms"] is None or x["terminal_status"] != "completed":
                continue
            osl = x["output_length"]
            if osl > 1 and (x["e2e_latency_ms"] - x["ttft_ms"]) / (osl - 1) > I:
                continue
            if x["e2e_latency_ms"] > S * e0(x["input_length"], osl) * (1 + 1e-6):
                continue
            g += 1
        mid = (lo + hi) / 2
        act = sum(1 for a, z in span.values() if a <= mid < z)
        out.append(f"{int(b*B/1000):5d}s gf {g/len(sel):.2f} act {act:4d}")
    print(r["cell_id"], r["policy_name"][:7], f"window [{(r['window_start_ms']-t0)/1e3:.0f},{(r['window_end_ms']-t0)/1e3:.0f}]", "gfw", round(r["good_frac_window"], 3))
    print("    " + " | ".join(out))
e0.persist()
