#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 report generator: bench.py output dir -> plots + markdown report.

Works for ANY bench.py run (local or slurm, any sweep shape, any set of
transports/modes). Aggregates across trials and subscribers (mean), emits:
  - plot_<topic>.png : latency vs p, faceted by s, one line per transport/mode
  - report.md        : run metadata + per-topic latency tables + clock/loopback caveat
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--metric", default="p50", choices=["p50", "p90", "p95", "p99", "mean"])
    a = ap.parse_args()

    rows = json.load(open(f"{a.indir}/summary.json"))
    meta = json.load(open(f"{a.indir}/meta.json")) if os.path.exists(f"{a.indir}/meta.json") else {}

    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # topic->mode->(p,s)->[us]
    drops = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # throughput events/s
    for r in rows:
        if r.get("error") or not r.get("latency_ns"):
            continue
        mode = r.get("mode", r.get("transport", "?"))
        key = (r.get("n_workers"), r.get("n_subs"))
        g[r["topic"]][mode][key].append(r["latency_ns"][a.metric] / 1000.0)
        drops[r["topic"]][mode][key].append(r.get("drop_rate", 0.0))
        w = r.get("window_secs") or 1
        thr[r["topic"]][mode][key].append(r.get("received", 0) / w)

    topics = sorted(g)
    ps = sorted({p for t in g.values() for m in t.values() for (p, _) in m})
    ss = sorted({s for t in g.values() for m in t.values() for (_, s) in m})
    modes = sorted({m for t in g.values() for m in t})

    def cell(t, m, p, s):
        v = g[t][m].get((p, s))
        return mean(v) if v else None

    def cell_thr(t, m, p, s):
        v = thr[t][m].get((p, s))
        return mean(v) if v else None

    def cell_drop(t, m, p, s):
        v = drops[t][m].get((p, s))
        return max(v) if v else None  # worst-case loss across trials/subs

    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for t in topics:
            n = max(1, len(ss))
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
            for ax, s in zip(axes[0], ss):
                for m in modes:
                    pts = [(p, cell(t, m, p, s)) for p in ps if cell(t, m, p, s) is not None]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, marker="o", label=m)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("p (workers)")
                ax.set_title(f"{t}  s={s}")
                ax.grid(True, which="both", alpha=0.3)
            axes[0][0].set_ylabel(f"{a.metric} latency (us)")
            axes[0][0].legend(fontsize=8)
            out = f"{a.indir}/plot_{t}.png"
            fig.suptitle(f"{t} — {a.metric} latency vs p ({meta.get('launcher','?')}, clock={meta.get('clock','?')})")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            figs.append(t)
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; markdown only)")

    L = ["# DIS-2172 event-plane benchmark report", ""]
    L.append(f"- **launcher**: {meta.get('launcher','?')}  |  **nodes**: "
             f"{len(meta.get('nodes',[])) or meta.get('nnodes','?')}  |  **clock**: {meta.get('clock','?')}")
    L.append(f"- **modes**: {modes}  |  **p (workers)**: {ps}  |  **s (subscribers)**: {ss}")
    L.append(f"- **topics**: {topics}  |  **trials**: {meta.get('trials','?')}  |  **metric**: {a.metric}")
    L.append("")
    if meta.get("clock") == "realtime" or meta.get("launcher") == "slurm":
        L.append("> ⚠️ **multi-node / CLOCK_REALTIME** — one-way latency validity depends on PTP/NTP "
                 "clock sync across nodes. Record the sync source + measured offset next to these numbers.")
    else:
        L.append("> ⚠️ **single-host loopback** — traffic goes over `lo` (no NIC). Measures the transport "
                 "*software path*, NOT the hardware/NIC cost of brokerless O(p×s) fan-out. Multi-node "
                 "(real NIC + PTP) still required before a default-transport decision (DYN-2941).")
    L.append("")
    hdr = "| mode \\ p | " + " | ".join(str(p) for p in ps) + " |"
    sep = "|" + "---|" * (len(ps) + 1)
    for t in topics:
        # latency: valid on single-host (monotonic); N/A cross-node without PTP
        L.append(f"## {t} — {a.metric} latency (µs)"
                 + ("  _(single-host only; cross-node N/A without PTP)_"
                    if meta.get("clock") != "monotonic" else ""))
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cells = [f"{cell(t,m,p,s):.0f}" if cell(t, m, p, s) is not None else "-" for p in ps]
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cells) + " |")
        L.append("")
        # throughput (events/s) — primary metric without PTP (subscriber-local, no cross-node clock)
        L.append(f"### {t} — throughput (events/s) ⟵ no-PTP primary metric")
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cs = [f"{cell_thr(t,m,p,s):.0f}" if cell_thr(t, m, p, s) is not None else "-" for p in ps]
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cs) + " |")
        L.append("")
        # drop_rate (seq-gap loss) — saturation / back-pressure signal (clock-free)
        L.append(f"### {t} — drop_rate (seq-gap loss)")
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cs = [f"{cell_drop(t,m,p,s):.3f}" if cell_drop(t, m, p, s) is not None else "-" for p in ps]
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cs) + " |")
        L.append("")
        if t in figs:
            L.append(f"![{t}](plot_{t}.png)")
            L.append("")

    open(f"{a.indir}/report.md", "w").write("\n".join(L))
    print(f"wrote {a.indir}/report.md" + (f" + {len(figs)} plots" if figs else ""))


if __name__ == "__main__":
    main()
