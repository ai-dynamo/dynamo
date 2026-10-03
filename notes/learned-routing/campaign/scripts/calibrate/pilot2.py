"""Pilot 2: heavier sessions loads and AgentX lanes at N=8 (k=0, default + round_robin)."""
import json, sys
import calib

OUT = calib.CAL / "pilot2"
cells = []
ss = calib.base_template("sessions:s0", mode="open_speedup")
for u in (28000, 40000, 56000, 80000, 112000):
    cells.append(calib.open_cell(ss, f"cal-pilot-ss-s0-n8-open-u{u}", 8, u))
ssc = calib.base_template("sessions:s0", mode="closed_concurrency")
for c in (32, 48, 64, 96, 128):
    cells.append(calib.closed_cell(ssc, f"cal-pilot-ss-s0-n8-closed-c{c}", 8, c))
ax_t = calib.base_template("agentx:A1", mode="agentic_lanes")
for l in (3, 4, 5, 6, 7, 8):
    cells.append(calib.lanes_cell(ax_t, f"cal-pilot-ax-A1-n8-lanes-l{l}", 8, l, 0.0))
calib.write_cells(cells, OUT / "cells.jsonl")
summary = calib.evaluate(cells, ["default", "round_robin"], 1, OUT / "results.jsonl",
                         deadline_s=float(sys.argv[1]))
print(json.dumps(summary))
