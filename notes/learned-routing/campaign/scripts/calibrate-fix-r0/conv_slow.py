import json
from collections import defaultdict
import numpy as np
import calib, analyze as A
from conv_check import OUT
cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()}
g = defaultdict(dict)
for r in calib.read_results(OUT / "results.jsonl"):
    c = cells[r["cell_id"]]
    it = A.items_for(r, c)
    s = np.where(np.isnan(it["slow"]), 1e6, it["slow"])
    g[(c["num_workers"], c["load"]["per_worker"], c["segment"], r["repeat"])][c["measure_trace"]["conv_history_mult"]] = float(np.mean(s))
agg = defaultdict(lambda: defaultdict(list))
for (n, v, seg, k), m in g.items():
    for a, b in ((1, 2), (2, 4), (1, 4), (2, 3), (3, 4)):
        agg[(n, v)][f"{b}/{a}"].append(m[b] / m[a])
for (n, v), d in sorted(agg.items()):
    print(f"N{n} {v}: " + " ".join(f"{k} med {np.median(x):.3f} mean {np.mean(x):.3f}" for k, x in d.items()))
