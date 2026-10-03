"""Warm-up convergence (fixer r0): default@defaults' window metrics vs history length at the band
loads. For W0 = the lifetime warm-up (600 s at think 1), arms with history W0, 2W0, 3W0, 4W0 measure
the SAME sessions (those starting in [4W0, 4W0 + 540) s of replay time). Train segments s0-s3, K = 2.

usage: conv_check.py build | eval | analyze
"""
import copy, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import calib
import analyze as A
from sweep import session_template

OUT = calib.FIX / "conv_check"
LOADS = {4: (0.425, 0.45, 0.475), 8: (0.4, 0.425, 0.45, 0.475), 32: (0.4, 0.425, 0.45)}
SEGS = ["sessions:s0", "sessions:s1", "sessions:s2", "sessions:s3"]
MULTS = (1, 2, 3, 4)
K = 2


def plan():
    for seg in SEGS:
        t = session_template(seg, "open_speedup")
        for n, loads in LOADS.items():
            for v in loads:
                src, spec = calib.session_open_job(t, n, v)
                s = float(f"{v * n:.6g}")
                w0 = calib.session_warmup_s(t)
                end = (4 * w0 + calib.SESS_WINDOW_S + calib.SESS_TAIL_S) * 1000.0 * s
                for m in MULTS:
                    yield seg, t, n, v, m, src, dict(spec, window=[(4 - m) * w0 * 1000.0 * s, end]), w0


def build():
    jobs = [(src, spec) for *_, src, spec, _ in plan()]
    calib.derive_many(jobs, processes=10)
    cells = []
    for seg, t, n, v, m, src, spec, w0 in plan():
        base = calib.session_open_cell(t, f"cal-cv-{seg.split(':')[1]}-n{n}-{v}", n, v)
        d = calib.derive(src, spec)
        c = copy.deepcopy(base)
        c["cell_id"] = f"cal-cv-{seg.split(':')[1]}-n{n}-{v}-h{m}"
        calib._attach_trace(c, d)
        c["measure"] = {"basis": "arrival", "warmup_ms": m * w0 * 1000.0, "window_ms": calib.SESS_WINDOW_S * 1000.0}
        c["measure_trace"] = dict(base["measure_trace"], conv_history_mult=m, source_window_ms=spec["window"])
        cells.append(c)
    calib.write_cells(cells, OUT / "cells.jsonl")
    calib.save_index()
    print("cells", len(cells))


def evaluate():
    cells = [json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()]
    summary = calib.evaluate(cells, ["default"], K, OUT / "results.jsonl")
    summary["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(summary), flush=True)
    with (OUT / "eval_log.jsonl").open("a") as h:
        h.write(json.dumps(summary) + "\n")


def analyze(I, S):
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in (OUT / "cells.jsonl").read_text().splitlines()}
    g = defaultdict(lambda: defaultdict(list))
    for r in calib.read_results(OUT / "results.jsonl"):
        if r.get("error") or r["cell_id"] not in cells:
            continue
        c = cells[r["cell_id"]]
        it = A.items_for(r, c)
        g[(c["num_workers"], c["load"]["per_worker"])][c["measure_trace"]["conv_history_mult"]].append(
            (c["segment"], r["repeat"], calib.gf(it, I, S)[0]))
    out = {}
    for (n, v), arms in sorted(g.items()):
        e = {}
        ref = {(s, k): x for s, k, x in arms[4]}
        for m in MULTS:
            vals = {(s, k): x for s, k, x in arms[m]}
            d = [vals[key] - ref[key] for key in ref if key in vals]
            e[f"h{m}"] = {"gf_mean": float(np.mean(list(vals.values()))),
                          "delta_vs_h4_mean": float(np.mean(d)), "delta_vs_h4_se": float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                          "rel_delta_vs_h4": float(np.mean(list(vals.values())) / np.mean(list(ref.values())) - 1.0)}
        out[f"N{n}|{v}"] = e
        print(f"N{n} load {v}: " + " | ".join(f"h{m} gf {e[f'h{m}']['gf_mean']:.3f} (d {e[f'h{m}']['delta_vs_h4_mean']:+.3f}, rel {e[f'h{m}']['rel_delta_vs_h4']:+.1%})" for m in MULTS))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "eval":
        evaluate()
    else:
        I, S = float(sys.argv[2]), float(sys.argv[3])
        res = analyze(I, S)
        (OUT / f"analysis_I{I}_S{S}.json").write_text(json.dumps(res, indent=1))
