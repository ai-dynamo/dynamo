"""Divergence D = gf_def - gf_RR over (S, load) at fixed I, for one group and N."""
import sys
import numpy as np
import analyze as A

group, n, I = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
Ss = [float(x) for x in sys.argv[4].split(",")]
tab = A.table(group)
loads = sorted(k[1] for k in tab if k[0] == n)
print(f"{group} N{n} I={I}: rows S, cols load; cell = gf_def/gf_RR (D)")
print("S\\load " + " ".join(f"{v:>14g}" for v in loads))
for S in Ss:
    cells = []
    for v in loads:
        e = tab[(n, v)]
        d = A.mean_gf(e["default"], I, S); r = A.mean_gf(e["round_robin"], I, S)
        cells.append(f"{d:.2f}/{r:.2f}({d-r:+.2f})")
    print(f"{S:<7g} " + " ".join(f"{c:>14s}" for c in cells))
