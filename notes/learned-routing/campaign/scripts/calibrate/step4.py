"""Step 4: evaluate frozen train/val cells (and noise cells) with default@defaults and round_robin.

usage: step4.py OUT_DIR CELLS.jsonl [CELLS.jsonl ...] [--repeats K] [--slots S]
"""
import json, sys, time
from pathlib import Path
import calib

args = sys.argv[1:]
opts = {}
files = []
i = 0
while i < len(args):
    if args[i].startswith("--"):
        opts[args[i]] = args[i + 1]; i += 2
    else:
        files.append(args[i]); i += 1
out_dir = Path(files.pop(0))
cells = [json.loads(l) for f in files for l in Path(f).read_text().splitlines() if l.strip()]
t0 = time.time()
summary = calib.evaluate(cells, ["default", "round_robin"], int(opts.get("--repeats", 8)),
                         out_dir / "results.jsonl", slots=int(opts.get("--slots", 20)))
summary.update(files=files, cells=len(cells), wall_s_total=round(time.time() - t0, 1),
               started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
print(json.dumps(summary), flush=True)
with (out_dir / "step4_log.jsonl").open("a") as h:
    h.write(json.dumps(summary) + "\n")
