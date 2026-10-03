"""Load sweep on train segments (stage 2 step 1).

usage: sweep.py build            # derive/generate traces, write sweep/<group>/cells.jsonl
       sweep.py eval GROUP [--n N,..] [--slots S] [--deadline SEC] [--repeats K]
Groups: mc-open mc-closed ss-open ss-closed ax-lanes. Policies: default, round_robin.
"""
import copy, json, sys, time
from pathlib import Path
import calib

NS = (2, 4, 6, 8, 16, 32)
GRID = {
    "mc-open": (3000, 4000, 5500, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 16000),
    "mc-closed": (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40),
    "ss-open": (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7),
    "ss-closed": (16, 24, 32, 40, 48, 56, 64),
    "ax-lanes": (2, 2.5, 3, 4, 5, 6, 7),
}
# v2: every Mooncake-format cell carries arrival_spread_ms (crn-spread-v1); v1 results without
# the spread are kept under sweep/<group>/ for comparison, v2 under sweep2/<group>/.
SEGMENTS = {
    "mc": ["mooncake:w0", "mooncake:w2", "mooncake:w4"],
    "ss": ["sessions:s0", "sessions:s1", "sessions:s2", "sessions:s3"],
    "ax": ["agentx:A1", "agentx:A2", "agentx:A3"],
}
OUT_R0 = calib.CAL / "sweep2"
# fixer r0: only ss-open is re-swept (lifetime-derived warm-up); other groups stay in sweep2.
# round 1 (one envelope lifetime) wrote sweep3; round 2 (two lifetimes) writes sweep4 for ss-open
OUT_R1 = calib.FIX / "sweep3"
OUT = calib.FIX / "sweep4"
RESWEPT = ("ss-open", "ss-closed")
# Finer points where the corrected sessions levels now fall (the r0 grids stepped 0.05 / 8 across
# the band; audit F3). ss-closed cells do not depend on the warm-up rule: existing points keep
# their r0 cell ids and content, so they are result-cache hits.
FINE = {"ss-open": (0.375, 0.425, 0.475, 0.525), "ss-closed": (28, 36, 44, 52)}


def group_root(group: str):
    if group == "ss-open":
        return OUT / group
    if group == "ss-closed":
        return OUT_R1 / group  # closed cells do not depend on the warm-up rule
    return OUT_R0 / group


def session_template(segment: str, mode: str) -> dict:
    t = copy.deepcopy(calib.base_template("sessions:s0", mode=mode))
    t["segment"] = segment
    t["cell_id"] = f"template-{segment}-{mode}"
    return t


def templates(group: str) -> dict:
    fam, mode = group.split("-")
    out = {}
    for seg in SEGMENTS[fam]:
        if fam == "mc":
            out[seg] = calib.base_template(seg, mode="open_speedup" if mode == "open" else "closed_concurrency")
        elif fam == "ss":
            out[seg] = session_template(seg, "open_speedup" if mode == "open" else "closed_concurrency")
        else:
            out[seg] = calib.base_template(seg, mode="agentic_lanes")
    return out


def grid(group: str, n: int):
    return tuple(sorted(set(_grid_r0(group, n)) | set(FINE.get(group, ()))))


def _grid_r0(group: str, n: int):
    # AgentX lanes at N >= 16 cost 2-4 min per replay: sweep only around the expected band.
    if group == "ax-lanes" and n >= 16:
        return (3, 3.5, 4, 5, 6)
    if group == "mc-open" and n >= 16:
        # the N = 16/32 knees sit above the N <= 8 grid
        return GRID[group] + (18000, 20000, 22000, 24000)
    return GRID[group]


def cell_id(group, seg, n, v):
    prefix = "cal-sw4" if group == "ss-open" else "cal-sw"
    return f"{prefix}-{group}-{seg.split(':')[1]}-n{n}-{v}"


def build(group: str) -> list[dict]:
    temps = templates(group)
    if group.startswith("ss-"):
        jobs = []
        for seg, t in temps.items():
            for n in NS:
                for v in grid(group, n):
                    jobs.append(calib.session_open_job(t, n, v) if group == "ss-open"
                                else calib.session_closed_job(t, n, v))
        t0 = time.time()
        calib.derive_many(jobs, processes=8)
        print(group, "derived", len(jobs), round(time.time() - t0, 1), flush=True)
    cells = []
    for seg, t in temps.items():
        for n in NS:
            for v in grid(group, n):
                cid = cell_id(group, seg, n, v)
                if group == "mc-open":
                    cells.append(calib.open_cell(t, cid, n, v))
                elif group == "mc-closed":
                    cells.append(calib.closed_cell(t, cid, n, v))
                elif group == "ss-open":
                    cells.append(calib.session_open_cell(t, cid, n, v))
                elif group == "ss-closed":
                    cells.append(calib.session_closed_cell(t, cid, n, v))
                else:
                    cells.append(calib.lanes_cell(t, cid, n, v, 0.0))
    calib.write_cells(cells, OUT / group / "cells.jsonl")
    print(group, "cells", len(cells), flush=True)
    return cells


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        for group in (sys.argv[2:] or ("ss-open",)):
            if group != "ss-open":
                raise SystemExit("fixer r0 round 2 re-sweeps only ss-open (ss-closed: sweep3)")
            build(group)
        return
    group = sys.argv[2]
    args = sys.argv[3:]
    opt = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
    root = group_root(group)
    cells = [json.loads(l) for l in (root / "cells.jsonl").read_text().splitlines()]
    if "--n" in opt:
        keep = {int(x) for x in opt["--n"].split(",")}
        cells = [c for c in cells if c["num_workers"] in keep]
    summary = calib.evaluate(
        cells, ["default", "round_robin"], int(opt.get("--repeats", 2)),
        root / "results.jsonl",
        deadline_s=float(opt["--deadline"]) if "--deadline" in opt else None,
        slots=int(opt.get("--slots", 20)),
    )
    summary["group"] = group
    summary["n"] = opt.get("--n")
    print(json.dumps(summary), flush=True)
    with (OUT / "eval_log.jsonl").open("a") as h:
        h.write(json.dumps({**summary, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


if __name__ == "__main__":
    main()
