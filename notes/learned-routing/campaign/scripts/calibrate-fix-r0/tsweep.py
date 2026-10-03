"""Transform sweeps (stage 2): per (family, transform tag, mode, N) used by any cell, sweep the
load on TRAIN segments with that transform applied (default + round_robin, K = 2).

usage: tsweep.py build | eval [--slots S] [--deadline SEC]
"""
import copy, json, sys, time
from collections import defaultdict
from pathlib import Path
import calib
from sweep import session_template

OUT_R0 = calib.CAL / "sweep2" / "tx"
# fixer r0: only open-loop session transforms are re-swept (lifetime-derived warm-up).
# round 1 wrote sweep3/tx; round 2 (two-lifetime warm-up) writes sweep4/tx
OUT_R1 = calib.FIX / "sweep3" / "tx"
OUT = calib.FIX / "sweep4" / "tx"


def reswept(family: str, mode: str) -> bool:
    # open loop: lifetime-derived warm-up; closed loop: unchanged cells (cache hits) plus finer points
    return family == "synthetic_sessions"


FINE = {("synthetic_sessions", "open_speedup"): (0.425, 0.45, 0.475),
        ("synthetic_sessions", "closed_concurrency"): (36, 44, 52, 56)}
MOONCAKE_SOURCE = calib.CR / "traces" / "mooncake" / "mooncake_trace.jsonl"
TRAIN_SEGMENTS = {
    "mooncake": ["mooncake:w0", "mooncake:w2", "mooncake:w4"],
    "synthetic_sessions": ["sessions:s0", "sessions:s1", "sessions:s2", "sessions:s3"],
    "agentx": ["agentx:A1", "agentx:A2", "agentx:A3"],
}
GRIDS = {
    ("mooncake", "open_speedup"): (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000),
    ("mooncake", "closed_concurrency"): (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24),
    ("synthetic_sessions", "open_speedup"): (0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3),
    ("synthetic_sessions", "closed_concurrency"): (8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 100),
    ("agentx", "agentic_lanes"): (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 13, 16),
}
TRANSFORM_KEYS = ("isl_unique_mult", "isl_prefix_mult", "osl_mult", "prefix_root_mult", "think_mult")


def combos():
    out = {}
    for split in calib.SPLITS:
        for c in calib.candidates(split):
            if c["transform_tag"] == "base":
                continue
            key = (c["family"], c["transform_tag"], c["load"]["mode"], c["num_workers"])
            if not reswept(c["family"], c["load"]["mode"]):
                continue
            out.setdefault(key, {k: c["transform"][k] for k in TRANSFORM_KEYS})
    return out


def sess_grid(fam, tag, mode):
    g = GRIDS[(fam, mode)]
    if fam == "synthetic_sessions" and tag.startswith("think") and float(tag[5:]) >= 2:
        g = g + ((1.6, 2.0, 2.5) if mode == "open_speedup" else (128, 160))
    return tuple(sorted(set(g) | set(FINE.get((fam, mode), ()))))


def lanes_grid(tag, n):
    g = GRIDS[("agentx", "agentic_lanes")]
    if tag.startswith("think") and float(tag[5:]) >= 2:
        return tuple(x for x in g if x >= 3)
    return tuple(x for x in g if x <= 6)


def build():
    cells = []
    session_jobs = []
    plan = []
    for (fam, tag, mode, n), tvals in sorted(combos().items()):
        for seg in TRAIN_SEGMENTS[fam]:
            if fam == "mooncake":
                base = calib.base_template(seg, mode=mode if mode == "open_speedup" else "closed_concurrency")
                meta = json.loads(calib.resolve(base["derived_meta"]).read_text())
                spec = dict(meta["spec"])
                spec.update(tvals)
                d = calib.derive(MOONCAKE_SOURCE, spec, block_size=512)
                t = copy.deepcopy(base)
                calib._attach_trace(t, d)
                t["transform"] = dict(t["transform"], **tvals)
                t["transform_tag"] = tag
                rate = calib.token_rate(base)["tokens_per_s"]
                plan.append((fam, tag, mode, n, seg, t, rate))
            elif fam == "synthetic_sessions":
                t = session_template(seg, mode)
                t["transform"] = dict(t["transform"], **tvals)
                t["transform_tag"] = tag
                for v in sess_grid(fam, tag, mode):
                    session_jobs.append(calib.session_open_job(t, n, v) if mode == "open_speedup"
                                        else calib.session_closed_job(t, n, v))
                plan.append((fam, tag, mode, n, seg, t, None))
            else:
                t = copy.deepcopy(calib.base_template(seg, mode="agentic_lanes"))
                t["transform"] = dict(t["transform"], **tvals)
                t["transform_tag"] = tag
                plan.append((fam, tag, mode, n, seg, t, None))
    calib.derive_many(session_jobs, processes=8)
    for fam, tag, mode, n, seg, t, rate in plan:
        grid = lanes_grid(tag, n) if fam == "agentx" else sess_grid(fam, tag, mode)
        for v in grid:
            prefix = "cal-tx4" if mode == "open_speedup" else "cal-tx"
            cid = f"{prefix}-{fam[:2]}-{tag}-{seg.split(':')[1]}-n{n}-{mode.split('_')[0]}-{v}"
            if fam == "mooncake" and mode == "open_speedup":
                cells.append(calib.open_cell(t, cid, n, v, rate=rate))
            elif fam == "mooncake":
                cells.append(calib.closed_cell(t, cid, n, v))
            elif mode == "open_speedup":
                cells.append(calib.session_open_cell(t, cid, n, v))
            elif mode == "closed_concurrency":
                cells.append(calib.session_closed_cell(t, cid, n, v))
            else:
                cells.append(calib.lanes_cell(t, cid, n, v, 0.0))
            cells[-1]["tx_combo"] = None  # placeholder removed below
            cells[-1].pop("tx_combo")
    calib.write_cells(cells, OUT / "cells.jsonl")
    calib.save_index()
    print("cells", len(cells), "combos", len(combos()))


def evaluate(opts):
    cells = [json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()]
    if "--family" in opts:
        cells = [c for c in cells if c["family"] == opts["--family"]]
    summary = calib.evaluate(cells, ["default", "round_robin"], 2, OUT / "results.jsonl",
                             deadline_s=float(opts["--deadline"]) if "--deadline" in opts else None,
                             slots=int(opts.get("--slots", 20)))
    print(json.dumps(summary), flush=True)
    with (OUT / "eval_log.jsonl").open("a") as h:
        h.write(json.dumps({**summary, "family": opts.get("--family"), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        a = sys.argv[2:]
        evaluate({a[i]: a[i + 1] for i in range(0, len(a), 2)})
