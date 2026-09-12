#!/usr/bin/env python3
"""Per-phase memory peaks for the replenishment cycle (kimi_replenish.sh).

Usage: analyze_replenish.py OUT_DIR [--ceiling-mib N]
Reads OUT_DIR/dev_mem.csv + OUT_DIR/phases.csv.
"""
import csv
import sys

O = sys.argv[1]
ceiling = 183359
if "--ceiling-mib" in sys.argv:
    ceiling = float(sys.argv[sys.argv.index("--ceiling-mib") + 1])

dev = {}
for r in csv.DictReader(open(f"{O}/dev_mem.csv")):
    dev.setdefault(int(r["gpu"]), []).append((float(r["ts"]), float(r["mem_mib"])))
ph = {r["label"]: float(r["epoch"]) for r in csv.DictReader(open(f"{O}/phases.csv"))}


def peak(a, b):
    return max(max((m for t, m in dev[g] if a <= t <= b), default=0) for g in dev)


def val(lbl):
    t = ph[lbl]
    return max(min(dev[g], key=lambda x: abs(x[0] - t))[1] for g in dev)


rows = [
    ("colocated_1 (e0 active + e1 shadow)", val("colocated_1")),
    ("failover-1 promotion peak [kill..promote]", peak(ph["failover1_kill"], ph["e1_promoted"])),
    ("REPLENISH bring-up peak [launch..standby]", peak(ph["replenish_launch"], ph["colocated_2"])),
    ("colocated_2 (e1 active + eC shadow) REPLENISHED", val("colocated_2")),
    ("failover-2 promotion peak [kill..promote]", peak(ph["failover2_kill"], ph["eC_promoted"])),
]
for name, v in rows:
    print(f"  {name:50s}: {v:7.0f} MiB")
w = max(max(m for _, m in dev[g]) for g in dev)
print(f"  {'WORST-GPU overall peak':50s}: {w:7.0f} MiB  (headroom {ceiling - w:.0f})")
