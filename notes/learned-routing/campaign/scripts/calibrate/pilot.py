"""Pilot sweep: N=8, one train segment per family x mode, k=0, default + round_robin."""
import json, sys
from pathlib import Path
import calib

OUT = calib.CAL / "pilot"
cells = []
mc = calib.base_template("mooncake:w0", mode="open_speedup")
for u in (3000, 4000, 5000, 6500, 8000, 10000, 12500, 15000):
    cells.append(calib.open_cell(mc, f"cal-pilot-mc-w0-n8-open-u{u}", 8, u))
mcc = calib.base_template("mooncake:w0", mode="closed_concurrency")
for c in (1, 2, 3, 4, 6, 8, 12):
    cells.append(calib.closed_cell(mcc, f"cal-pilot-mc-w0-n8-closed-c{c}", 8, c))
ss = calib.base_template("sessions:s0", mode="open_speedup")
for u in (4000, 6000, 8000, 10000, 13000, 16000, 20000):
    cells.append(calib.open_cell(ss, f"cal-pilot-ss-s0-n8-open-u{u}", 8, u))
ssc = calib.base_template("sessions:s0", mode="closed_concurrency")
for c in (2, 4, 6, 8, 12, 16, 24):
    cells.append(calib.closed_cell(ssc, f"cal-pilot-ss-s0-n8-closed-c{c}", 8, c))
if "--agentx" in sys.argv:
    ax_t = calib.base_template("agentx:A1", mode="agentic_lanes")
    for l in (3, 4, 5, 6, 7, 8):
        cells.append(calib.lanes_cell(ax_t, f"cal-pilot-ax-A1-n8-lanes-l{l}", 8, l, 0.0))
calib.write_cells(cells, OUT / "cells.jsonl")
summary = calib.evaluate(cells, ["default", "round_robin"], 1, OUT / "results.jsonl",
                         deadline_s=float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else None)
print(json.dumps(summary))
