#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 (p × s) latency heatmaps: NATS vs ZMQ.

Reads a sweep summary.json that covers a p×s grid (workers × subs) and renders,
for a given topic:
  * NATS p50 heatmap        (rows = s subscribers, cols = p publishers)
  * ZMQ  p50 heatmap
  * ZMQ − NATS diff heatmap (blue = ZMQ faster, red = ZMQ slower)  <- the decision map
Cells annotated with the value; the ZMQ panel also annotates p×s connection count.

Multiple rows at the same (transport, p, s) — extra subscribers and trials —
are averaged.
"""
import argparse
import json
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topic", default="kv-events")
    a = ap.parse_args()

    rows = json.load(open(a.summary))
    g = defaultdict(lambda: defaultdict(list))  # transport -> (p,s) -> [p50_us]
    for r in rows:
        if r.get("error") or not r.get("latency_ns") or r.get("topic") != a.topic:
            continue
        g[r["transport"]][(r["n_workers"], r["n_subs"])].append(r["latency_ns"]["p50"] / 1000.0)

    ps = sorted({p for tr in g.values() for (p, _s) in tr})
    ss = sorted({s for tr in g.values() for (_p, s) in tr})

    def mat(tr):
        M = np.full((len(ss), len(ps)), np.nan)
        for i, s in enumerate(ss):
            for j, p in enumerate(ps):
                v = g.get(tr, {}).get((p, s))
                if v:
                    M[i, j] = mean(v)
        return M

    nats, zmq = mat("Nats"), mat("Zmq")
    diff = zmq - nats

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        ("NATS p50 (µs)", nats, "viridis", False),
        ("ZMQ p50 (µs)", zmq, "viridis", True),
        ("ZMQ − NATS p50 (µs)  [blue=ZMQ faster]", diff, "RdBu_r", False),
    ]
    vmax_diff = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
    for ax, (title, M, cmap, annot_conn) in zip(axes, panels):
        kw = dict(origin="lower", aspect="auto", cmap=cmap)
        if cmap == "RdBu_r":
            kw.update(vmin=-vmax_diff, vmax=vmax_diff)
        im = ax.imshow(M, **kw)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels(ps)
        ax.set_yticks(range(len(ss)))
        ax.set_yticklabels(ss)
        ax.set_xlabel("p  (workers / publishers)")
        ax.set_ylabel("s  (subscribers / router replicas)")
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                txt = f"{M[i, j]:.0f}"
                if annot_conn:
                    txt += f"\n({ps[j]*ss[i]} conn)"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if cmap == "viridis" else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"DIS-2172 event-plane p50 latency over (p × s) — topic={a.topic}", fontsize=12)
    fig.tight_layout()
    out = f"{a.outdir}/grid_heatmap_{a.topic}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Supplementary: latency vs connection-count (p*s for ZMQ, p+s for NATS).
    fig2, ax = plt.subplots(figsize=(8, 5))
    for tr, conn_fn, mk in [("Zmq", lambda p, s: p * s, "o"), ("Nats", lambda p, s: p + s, "s")]:
        pts = sorted(((conn_fn(p, s), mean(v)) for (p, s), v in g.get(tr, {}).items()))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=mk, linestyle="-", label=f"{tr} (conn={'p×s' if tr=='Zmq' else 'p+s'})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("connections (ZMQ: p×s   |   NATS: p+s)")
    ax.set_ylabel("p50 latency (µs)")
    ax.set_title(f"DIS-2172 latency vs connection count — topic={a.topic}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    out2 = f"{a.outdir}/grid_latency_vs_conn_{a.topic}.png"
    fig2.savefig(out2, dpi=120)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
