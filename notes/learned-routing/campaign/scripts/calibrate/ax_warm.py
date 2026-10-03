"""AgentX lanes: p90 first-copy end vs p90 recorded copy span (warm-up rule input)."""
import json, sys
from collections import defaultdict
import numpy as np
import analyze as A
import calib

cells, recs = A.group_records("ax-lanes")
rows = defaultdict(list)
for r in recs:
    if not r["policy_name"].startswith("default"):
        continue
    c = cells[r["cell_id"]]
    spans = A.copy_spans(c)
    q90 = float(np.percentile(spans, 90))
    f90 = A.first_copy_end_p90(r)
    rows[(c["num_workers"], c["load"]["per_worker"])].append((f90, q90, f90 / q90, c["segment"], r["repeat"], r["full_occupancy_end_ms"]))
out = {}
for (n, l), vals in sorted(rows.items()):
    ratios = [v[2] for v in vals]
    print(f"N{n:<3d} l{l:<3g} f90 {np.mean([v[0] for v in vals])/1000:8.0f}s q90 {np.mean([v[1] for v in vals])/1000:7.0f}s ratio mean {np.mean(ratios):.3f} max {np.max(ratios):.3f}  full_end {np.mean([v[5] for v in vals])/1000:8.0f}s")
    out[f"{n}-{l}"] = {"ratio_mean": float(np.mean(ratios)), "ratio_max": float(np.max(ratios)), "f90_s": [v[0]/1000 for v in vals], "q90_s": [v[1]/1000 for v in vals], "full_end_s": [v[5]/1000 for v in vals]}
(calib.CAL / "ax_warmup_ratios.json").write_text(json.dumps(out, indent=1))
