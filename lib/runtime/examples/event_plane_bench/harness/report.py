#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 TRANSPORT-LAYER report generator: bench.py output dir -> markdown.

STOCK-TRANSPORT / counting-only variant: there is NO latency in the data (the
subscriber carries no `published_at_ns` instrumentation), so this report emits
ONLY the clock-free transport metrics:
  - throughput (events/s)  : received / window_secs, mean over trials+subs
  - drop_rate (seq-gap)    : worst-case intra-window loss across trials/subs
  - resource cost          : established-TCP peak (O(p×s) ZMQ mesh vs O(p+s) NATS
                             to-broker) + per-class CPU/RSS/fd peaks

Works for ANY bench.py run (local or slurm, any sweep shape, any set of
transports). Writes report.md (+ optional throughput plots if matplotlib is
present).
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    a = ap.parse_args()

    rows = json.load(open(f"{a.indir}/summary.json"))
    meta = json.load(open(f"{a.indir}/meta.json")) if os.path.exists(f"{a.indir}/meta.json") else {}

    drops = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # events/s
    # resource summaries are topic-agnostic per cell: keep one per (mode,p,s).
    res = defaultdict(dict)  # mode->(p,s)->res summary (last seen)
    for r in rows:
        # Count-only gating: a row is valid if it ran (no error) and reported a
        # received count (NOT latency, which no longer exists).
        if r.get("error") or "received" not in r:
            continue
        mode = r.get("mode", r.get("transport", "?"))
        key = (r.get("n_workers"), r.get("n_subs"))
        w = r.get("window_secs") or 1
        thr[r["topic"]][mode][key].append(r.get("received", 0) / w)
        drops[r["topic"]][mode][key].append(r.get("drop_rate", 0.0))
        if r.get("res"):
            res[mode][key] = r["res"]

    topics = sorted(thr)
    ps = sorted({p for t in thr.values() for m in t.values() for (p, _) in m})
    ss = sorted({s for t in thr.values() for m in t.values() for (_, s) in m})
    modes = sorted({m for t in thr.values() for m in t})

    def cell_thr(t, m, p, s):
        v = thr[t][m].get((p, s))
        return mean(v) if v else None

    def cell_thr_spread(t, m, p, s):
        v = thr[t][m].get((p, s))
        if not v:
            return None
        return (min(v), max(v))

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
                    pts = [(p, cell_thr(t, m, p, s)) for p in ps if cell_thr(t, m, p, s) is not None]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, marker="o", label=m)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("p (workers)")
                ax.set_title(f"{t}  s={s}")
                ax.grid(True, which="both", alpha=0.3)
            axes[0][0].set_ylabel("throughput (events/s)")
            axes[0][0].legend(fontsize=8)
            out = f"{a.indir}/plot_{t}.png"
            fig.suptitle(f"{t} — throughput vs p ({meta.get('launcher','?')})")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            figs.append(t)
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; markdown only)")

    L = ["# DIS-2172 event-plane TRANSPORT-LAYER report (stock transport, count-only)", ""]
    L.append(f"- **launcher**: {meta.get('launcher','?')}  |  **nodes**: "
             f"{len(meta.get('nodes',[])) or meta.get('nnodes','?')}  |  **clock**: {meta.get('clock','?')}")
    L.append(f"- **modes**: {modes}  |  **p (workers)**: {ps}  |  **s (subscribers)**: {ss}")
    L.append(f"- **topics**: {topics}  |  **trials**: {meta.get('trials','?')}")
    L.append("")
    L.append("> **Stock transport, no instrumentation.** The subscriber counts only; the "
             "`EventEnvelope` publish path is un-perturbed (no `published_at_ns`). Metrics are "
             "clock-free (throughput + seq-gap loss + resource footprint), so they are valid "
             "across nodes WITHOUT PTP/NTP latency sync.")
    L.append("")
    hdr = "| mode \\ p | " + " | ".join(str(p) for p in ps) + " |"
    sep = "|" + "---|" * (len(ps) + 1)
    for t in topics:
        # throughput (events/s) — primary transport metric (subscriber-local count)
        L.append(f"## {t} — throughput (events/s, mean over trials×subs)")
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cs = [f"{cell_thr(t,m,p,s):.0f}" if cell_thr(t, m, p, s) is not None else "-" for p in ps]
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cs) + " |")
        L.append("")
        # throughput spread (min..max across trials) — show run-to-run stability
        L.append(f"### {t} — throughput spread (min..max events/s across trials×subs)")
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cs = []
                for p in ps:
                    sp = cell_thr_spread(t, m, p, s)
                    cs.append(f"{sp[0]:.0f}..{sp[1]:.0f}" if sp else "-")
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cs) + " |")
        L.append("")
        # drop_rate (seq-gap loss) — saturation / back-pressure signal (clock-free)
        L.append(f"### {t} — drop_rate (worst-case seq-gap loss across trials×subs)")
        L.append(hdr); L.append(sep)
        for s in ss:
            for m in modes:
                cs = [f"{cell_drop(t,m,p,s):.3f}" if cell_drop(t, m, p, s) is not None else "-" for p in ps]
                L.append(f"| **{m}** (s={s}) | " + " | ".join(cs) + " |")
        L.append("")
        if t in figs:
            L.append(f"![{t}](plot_{t}.png)")
            L.append("")

    # Resource cost: established-TCP peak (O(p×s) vs O(p+s)) + per-class CPU/RSS/fd.
    if any(res[m] for m in res):
        L.append("## resource cost (peak during measure window)")
        L.append("")
        L.append("Established-TCP peak quantifies the connection-count blow-up: the brokerless "
                 "ZMQ direct mesh is O(p×s) (every publisher dials every subscriber) while NATS "
                 "is O(p+s) (each side holds ONE connection to the broker, which itself burns "
                 "CPU/RSS).")
        L.append("")
        rhdr = ("| mode | p | s | est_tcp node_peak | est_tcp total_peak | "
                "sub fds_peak | sub rss_mb | nats rss_mb | nats cpu% |")
        rsep = "|" + "---|" * 9
        L.append(rhdr); L.append(rsep)
        for m in modes:
            for (p, s) in sorted(res[m]):
                rs = res[m][(p, s)]
                procs = rs.get("procs", {})
                subp = procs.get("sub", {})
                natsp = procs.get("nats", {})
                L.append("| {m} | {p} | {s} | {ntp} | {ttp} | {sfd} | {srss} | {nrss} | {ncpu} |".format(
                    m=m, p=p, s=s,
                    ntp=rs.get("established_tcp_node_peak", "-"),
                    ttp=rs.get("established_tcp_total_peak", "-"),
                    sfd=subp.get("fds_peak", "-"),
                    srss=subp.get("rss_mb_peak", "-"),
                    nrss=natsp.get("rss_mb_peak", "-"),
                    ncpu=natsp.get("cpu_pct_peak", "-"),
                ))
        L.append("")

    open(f"{a.indir}/report.md", "w").write("\n".join(L))
    print(f"wrote {a.indir}/report.md" + (f" + {len(figs)} plots" if figs else ""))


if __name__ == "__main__":
    main()
