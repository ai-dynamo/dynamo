"""F1 diagnosis: session lifetimes, trace-intrinsic (uncontended) vs measured under replay.

Trace-intrinsic lifetime of session i at think multiplier m (replay time):
    D0_i(m) = m * sum(delay_ij) + sum_j E0(ISL_ij, OSL_ij)
Measured (replay) lifetime: last terminal - first arrival over the session's rows.
Writes ../out/lifetimes.json.
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "<campaign-root>/runs/calibrate/scripts")
import calib

CR = calib.CR
OUT = CR / "runs" / "calibrate-fix-r0" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def sessions_of(path, limit=None):
    sess = defaultdict(lambda: {"delay": 0.0, "pairs": []})
    order = []
    for line in open(path):
        r = json.loads(line)
        s = r["session_id"]
        if s not in sess:
            order.append(s)
            if limit and len(order) > limit:
                break
        e = sess[s]
        e["delay"] += float(r.get("delay", 0.0))
        e["pairs"].append((r["input_length"], r["output_length"]))
    return {s: sess[s] for s in order[:limit]} if limit else dict(sess)


def main():
    e0 = calib.e0_table()
    res = {"intrinsic": {}, "measured": {}}
    # 1) trace-intrinsic, train seeds 0..3, first 4000 sessions each (base transform)
    for seed in range(4):
        ss = sessions_of(calib.long_session_source(seed), limit=4000)
        think = np.array([v["delay"] for v in ss.values()]) / 1000.0
        serv = np.array([sum(e0(i, o) for i, o in v["pairs"]) for v in ss.values()]) / 1000.0
        turns = np.array([len(v["pairs"]) for v in ss.values()])
        e0.persist()
        for m in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
            for kappa in (1.0, 2.0, 3.0):
                d = m * think + kappa * serv
                res["intrinsic"][f"s{seed}|m{m}|k{kappa}"] = {
                    "mean": float(d.mean()), "p50": float(np.percentile(d, 50)),
                    "p90": float(np.percentile(d, 90)), "p99": float(np.percentile(d, 99)),
                    "p999": float(np.percentile(d, 99.9)), "max": float(d.max()),
                }
        res["intrinsic"][f"s{seed}|parts"] = {"think_mean_s": float(think.mean()), "serv_e0_mean_s": float(serv.mean()),
                                               "turns_mean": float(turns.mean()), "sessions": len(ss)}
    # 2) measured contended lifetimes in step4_r4 (default + RR, k0..3), sessions open cells
    cells = {}
    for f in ("train", "val", "noise"):
        for l in open(CR / "runs/calibrate/final_r4" / f"{f}.jsonl"):
            c = json.loads(l); cells[c["cell_id"]] = c
    for l in open(CR / "runs/calibrate/step4_r4/results.jsonl"):
        r = json.loads(l)
        c = cells.get(r["cell_id"])
        if c is None or c["family"] != "synthetic_sessions" or c["load"]["mode"] != "open_speedup" or r["repeat"] > 1:
            continue
        rows = [json.loads(x) for x in gzip.open(r["per_request_path"], "rt")]
        t0 = min(x["arrival_time_ms"] for x in rows)
        span = {}
        serv_e0 = defaultdict(float)
        serv = defaultdict(float)
        for x in rows:
            s = x["session_id"]
            end = x["terminal_time_ms"] if x["terminal_time_ms"] is not None else float("inf")
            lo, hi = span.get(s, (x["arrival_time_ms"], end))
            span[s] = (min(lo, x["arrival_time_ms"]), max(hi, end))
            serv_e0[s] += e0(x["input_length"], x["output_length"])
            if x["e2e_latency_ms"] is not None:
                serv[s] += x["e2e_latency_ms"]
        # only sessions that start before 300 s (complete history inside the replay)
        early = [s for s, (lo, hi) in span.items() if lo - t0 < 300e3]
        dur = np.array([(span[s][1] - span[s][0]) / 1000 for s in early])
        stretch = np.array([serv[s] / serv_e0[s] for s in early if serv_e0[s] > 0])
        key = f"{r['cell_id']}|{r['policy_name']}|k{r['repeat']}"
        res["measured"][key] = {
            "think_mult": c["transform"].get("think_mult", 1.0), "level": c["load"]["level"],
            "N": c["num_workers"],
            "dur_p50": float(np.percentile(dur, 50)), "dur_p90": float(np.percentile(dur, 90)),
            "dur_p99": float(np.percentile(dur, 99)), "dur_max": float(dur.max()),
            "service_stretch_mean": float(np.mean(stretch)), "service_stretch_p90": float(np.percentile(stretch, 90)),
        }
    e0.persist()
    (OUT / "lifetimes.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    for k, v in res["intrinsic"].items():
        if "|k" in k and ("m1.0" in k or "m4.0" in k or "m0.25" in k):
            print(k, {a: round(b, 1) for a, b in v.items()})
        elif "parts" in k:
            print(k, v)
    for k, v in sorted(res["measured"].items()):
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
