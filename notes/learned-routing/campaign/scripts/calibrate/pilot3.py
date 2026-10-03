"""Pilot 3: think-invariant sessions (long seed-0 trace) at N=8, k=0, default + round_robin."""
import json, sys, time
import calib

OUT = calib.CAL / "pilot3"
cells = []
t = time.time()
ss = calib.base_template("sessions:s0", mode="open_speedup")
for rho in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    cells.append(calib.session_open_cell(ss, f"cal-pilot3-ss-s0-n8-open-r{rho}", 8, rho))
ssc = calib.base_template("sessions:s0", mode="closed_concurrency")
for c in (8, 16, 32, 48, 64, 96):
    cells.append(calib.session_closed_cell(ssc, f"cal-pilot3-ss-s0-n8-closed-c{c}", 8, c))
print("derive s", round(time.time() - t, 1), flush=True)
calib.write_cells(cells, OUT / "cells.jsonl")
summary = calib.evaluate(cells, ["default", "round_robin"], 1, OUT / "results.jsonl",
                         deadline_s=float(sys.argv[1]))
print(json.dumps(summary))
