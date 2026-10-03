"""Print per-(cell, policy) summaries of a results file with offline rescoring."""
import json, sys
from collections import defaultdict
import numpy as np
import calib

path = calib.Path(sys.argv[1])
cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (path.parent / "cells.jsonl").read_text().splitlines()}
recs = [r for r in calib.read_results(path) if not r.get("error")]
by = defaultdict(list)
for r in recs:
    by[(r["cell_id"], r["policy_name"])].append(r)

def q(a, p):
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")

print(f"{'cell':40s} {'pol':8s} {'n':>5s} {'done':>5s} {'itl50':>6s} {'itl95':>6s} {'sd50':>6s} {'sd85':>6s} {'sd95':>7s} {'ttft50':>7s} {'gf40/3':>6s} {'gpw':>6s} {'reuse':>5s}")
for (cid, pol), rs in sorted(by.items()):
    items = [calib.scored_items(r, cells[cid]) for r in rs]
    done = np.concatenate([i["done"] for i in items]); itl = np.concatenate([i["itl"] for i in items])
    sd = np.concatenate([i["slow"] for i in items]); tt = np.concatenate([i["ttft"] for i in items])
    g = [calib.gf(i, 40.0, 3.0) for i in items]
    print(f"{cid:40s} {pol[:8]:8s} {len(done):5d} {done.mean():5.3f} {q(itl,50):6.1f} {q(itl,95):6.1f} {q(sd,50):6.2f} {q(sd,85):6.2f} {q(sd,95):7.2f} {q(tt,50):7.0f} {np.mean([x[0] for x in g]):6.3f} {np.mean([x[1] for x in g]):6.3f} {np.mean([r['prefix_reuse'] or 0 for r in rs]):5.3f}")
