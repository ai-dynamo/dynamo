#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 plotting / table builder.

Reads a sweep summary.json and produces, against a chosen x-axis (--xkey,
default n_workers; use n_subs for the M-sweep):
  - latency_vs_<xkey>_<topic>.png : p50/p99 publish->deliver latency, NATS vs ZMQ.
    (For ZMQ direct mode #workers == #ZMQ PUB ports/topic.)
  - fanout_vs_<xkey>.png : established-TCP count + drop_rate (brokerless N*M evidence).
  - summary_table.csv : every cell, traceable to config.

Multiple rows sharing the same (topic, transport, xkey) — e.g. the M subscribers
in an M-sweep cell — are averaged. CSV always written (stdlib only); PNGs only if
matplotlib is importable.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def _us(ns):
    return round(ns / 1000.0, 3) if isinstance(ns, (int, float)) else ""


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0


def write_csv(rows, out):
    cols = ["transport", "topic", "n_workers", "n_subs", "sub_idx", "received",
            "measured", "gaps", "drop_rate", "zero_ts_events", "tcp_conns_est",
            "lat_min_us", "lat_mean_us", "lat_p50_us", "lat_p90_us",
            "lat_p95_us", "lat_p99_us", "lat_max_us", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            lat = r.get("latency_ns", {})
            w.writerow({
                "transport": r.get("transport"), "topic": r.get("topic"),
                "n_workers": r.get("n_workers"), "n_subs": r.get("n_subs"),
                "sub_idx": r.get("sub_idx"), "received": r.get("received"),
                "measured": r.get("measured"), "gaps": r.get("gaps"),
                "drop_rate": r.get("drop_rate"), "zero_ts_events": r.get("zero_ts_events"),
                "tcp_conns_est": r.get("tcp_conns_est"),
                "lat_min_us": _us(lat.get("min")), "lat_mean_us": _us(lat.get("mean")),
                "lat_p50_us": _us(lat.get("p50")), "lat_p90_us": _us(lat.get("p90")),
                "lat_p95_us": _us(lat.get("p95")), "lat_p99_us": _us(lat.get("p99")),
                "lat_max_us": _us(lat.get("max")), "error": r.get("error", ""),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/main_sweep/summary.json")
    ap.add_argument("--outdir", default="results/main_sweep")
    ap.add_argument("--xkey", default="n_workers", choices=["n_workers", "n_subs"])
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    rows = load(a.summary)
    write_csv(rows, f"{a.outdir}/summary_table.csv")
    print(f"wrote {a.outdir}/summary_table.csv ({len(rows)} rows)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); CSV only.")
        return

    xkey = a.xkey
    xlabel = ("#workers (= #ZMQ PUB ports/topic in direct mode)" if xkey == "n_workers"
              else "#subscribers M (router replicas) — ZMQ holds N×M direct connections")

    # (topic, transport) -> { xval: [rows] }
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error") or not r.get("latency_ns"):
            continue
        groups[(r["topic"], r["transport"])][r[xkey]].append(r)

    topics = sorted({t for (t, _tr) in groups})
    for topic in topics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for transport in sorted({tr for (t, tr) in groups if t == topic}):
            d = groups[(topic, transport)]
            xs = sorted(d)
            p50 = [_us(_mean([rr["latency_ns"]["p50"] for rr in d[x]])) for x in xs]
            p99 = [_us(_mean([rr["latency_ns"]["p99"] for rr in d[x]])) for x in xs]
            ax.plot(xs, p50, marker="o", label=f"{transport} p50")
            ax.plot(xs, p99, marker="^", linestyle="--", label=f"{transport} p99")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("publish->deliver latency (µs)")
        ax.set_title(f"DIS-2172 event-plane latency vs {xkey} — topic={topic}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = f"{a.outdir}/latency_vs_{xkey}_{topic}.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")

    ftopic = "kv-events" if any(t == "kv-events" for (t, _tr) in groups) else topics[0]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    for transport in sorted({tr for (t, tr) in groups if t == ftopic}):
        d = groups[(ftopic, transport)]
        xs = sorted(d)
        conns = [_mean([rr.get("tcp_conns_est") for rr in d[x]]) for x in xs]
        drops = [_mean([rr.get("drop_rate", 0) for rr in d[x]]) for x in xs]
        ax1.plot(xs, conns, marker="o", label=f"{transport} est-TCP-conns")
        ax2.plot(xs, drops, marker="x", linestyle=":", label=f"{transport} drop_rate")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("established TCP connections (proxy)")
    ax2.set_ylabel("drop_rate (seq gaps)")
    ax1.set_title(f"DIS-2172 brokerless fan-out & drops vs {xkey} — topic={ftopic}")
    ax1.grid(True, which="both", alpha=0.3)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")
    fig.tight_layout()
    out = f"{a.outdir}/fanout_vs_{xkey}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
