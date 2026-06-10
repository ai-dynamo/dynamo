#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 DYNAMIC-BEHAVIOR / RECOVERY post-processor.

Consumes a bench.py dynamic run (summary.json + per-sub *.buckets.jsonl timelines
+ per-cell *_inject.json) and computes, PER SUBSCRIBER and per (scenario,
transport):

  - B'          : the EMPIRICALLY-MEASURED post-disturbance steady-state mean
                  throughput (events/s). NOT a (N-1)/N formula — measured from a
                  flat trailing stretch of the timeline (low CoV).
  - recovery_s  : t0 -> the first time the sub's throughput reaches AND HOLDS
                  >=90% of B' for a stable window (default 2.0s). t0 = the node0
                  CLOCK_MONOTONIC instant the orchestrator issued the kill/spawn.
  - loss        : count of seq-gap events in [t0, recovery] attributable to the
                  disturbance (raw + baseline-adjusted), PER SUBSCRIBER.

CLOCK ALIGNMENT (PTP-immune): both t0 and the bucket timestamps live on node0.
Each buckets file's header carries `anchor_unix_ns` (a CLOCK_REALTIME reading
taken back-to-back with the sub's monotonic reference), so bucket realtime =
anchor_unix_ns + t_mono_ns. The inject file carries `t0_real_ns` (orchestrator
CLOCK_REALTIME). rel(bucket) = bucket_realtime_ns - t0_real_ns. Because both
realtime reads happen on the SAME node0 kernel clock, this is a difference of two
node0 instants — no cross-node comparison, valid without PTP/NTP.

For scenario=slow there is NO discrete t0; instead we report the slow sub's
drop_rate vs the FAST subs' drop_rate (the isolation test: does ZMQ backpressure
bleed onto the fast subs, while NATS isolates it?).

Usage:  python recovery.py --in <bench-out-dir> [--bprime-window 4]
                           [--recover-frac 0.90] [--stable-window 2.0]
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def load_buckets(path):
    """Return (anchor_unix_ns, [(t_mono_ns, received, gaps), ...]) or (None, [])."""
    anchor = None
    rows = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                o = json.loads(ln)
                if o.get("dis2172_buckets") == "header":
                    anchor = o.get("anchor_unix_ns")
                    continue
                rows.append((o["t_mono_ns"], o["received"], o["gaps"]))
    except Exception:
        return None, []
    return anchor, rows


def series_rel(anchor_unix_ns, rows, t0_real_ns, bucket_ms):
    """Return [(rel_s, thr_eps, gaps)] with rel_s = seconds relative to t0."""
    bsec = bucket_ms / 1000.0
    out = []
    for (t_mono_ns, recv, gaps) in rows:
        real_ns = anchor_unix_ns + t_mono_ns
        rel_s = (real_ns - t0_real_ns) / 1e9
        out.append((rel_s, recv / bsec if bsec > 0 else 0.0, gaps))
    return out


def measure_bprime(series, bprime_window):
    """Empirical post-disturbance steady-state mean throughput.

    Take the trailing `bprime_window` seconds of the timeline; B' = mean throughput
    there. Flag unstable if the coefficient of variation exceeds 0.15.
    """
    if not series:
        return None, None, True
    end = series[-1][0]
    tail = [thr for (rel, thr, _g) in series if rel >= end - bprime_window]
    if len(tail) < 3:
        tail = [thr for (_r, thr, _g) in series[-max(3, len(series) // 4):]]
    if not tail:
        return None, None, True
    m = mean(tail)
    cov = (pstdev(tail) / m) if m > 0 else 1.0
    return m, cov, (cov > 0.15)


def find_recovery(series, bprime, recover_frac, stable_window):
    """First rel_s>=0 where thr stays >= recover_frac*B' for stable_window secs."""
    if not bprime or bprime <= 0:
        return None
    thresh = recover_frac * bprime
    post = [(rel, thr) for (rel, thr, _g) in series if rel >= 0.0]
    if not post:
        return None
    n = len(post)
    for i in range(n):
        start_rel = post[i][0]
        ok = True
        j = i
        # require coverage of the full [start_rel, start_rel+stable_window] span
        while j < n and post[j][0] <= start_rel + stable_window:
            if post[j][1] < thresh:
                ok = False
                break
            j += 1
        covered = (j >= n and (post[-1][0] >= start_rel + stable_window - 1e-9)) or \
                  (j < n)
        if ok and covered:
            # also require we actually saw enough buckets across the window
            span_buckets = j - i
            if span_buckets >= 1:
                return max(0.0, start_rel)
    return None


def dip_depth(series, bprime):
    """Minimum post-t0 throughput as a fraction of B' (1.0 = no dip)."""
    if not bprime or bprime <= 0:
        return None
    post = [thr for (rel, thr, _g) in series if rel >= 0.0]
    if not post:
        return None
    return round(min(post) / bprime, 3)


def loss_in(series, lo, hi):
    """Sum of gaps in rel_s in [lo, hi)."""
    return sum(g for (rel, _t, g) in series if lo <= rel < hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--bprime-window", type=float, default=4.0,
                    help="trailing seconds used to measure B' (steady state)")
    ap.add_argument("--recover-frac", type=float, default=0.90)
    ap.add_argument("--stable-window", type=float, default=2.0,
                    help="seconds throughput must hold >=frac*B' to count recovered")
    ap.add_argument("--baseline-window", type=float, default=4.0,
                    help="pre-t0 seconds used for baseline gap rate")
    a = ap.parse_args()

    summ = json.load(open(f"{a.indir}/summary.json"))
    meta = json.load(open(f"{a.indir}/meta.json")) if os.path.exists(f"{a.indir}/meta.json") else {}
    bucket_ms = meta.get("bucket_ms", 100)

    # Group rows by (scenario, transport, p, s, trial). Each row is one sub.
    per_sub = []  # computed records
    slow_rows = []  # scenario=slow rows for isolation table

    for r in summ:
        if r.get("error"):
            continue
        scen = r.get("scenario", "none")
        if scen in ("none", ""):
            continue
        bpath = r.get("buckets_out")
        if not bpath or not os.path.exists(bpath):
            continue
        anchor, rows = load_buckets(bpath)
        if anchor is None or not rows:
            continue

        if scen == "slow":
            # No t0; use absolute timeline. drop_rate isolation only.
            slow_rows.append(r)
            continue

        t0_real = r.get("t0_real_ns")
        if t0_real is None:
            # fall back to the cell inject file
            inj = f"{a.indir}/{r['mode']}_p{r['n_workers']}_s{r['n_subs']}_t{r['trial']}_inject.json"
            if os.path.exists(inj):
                t0_real = json.load(open(inj)).get("t0_real_ns")
        if t0_real is None:
            continue

        series = series_rel(anchor, rows, t0_real, bucket_ms)
        bprime, cov, unstable = measure_bprime(series, a.bprime_window)
        rec = find_recovery(series, bprime, a.recover_frac, a.stable_window)
        dip = dip_depth(series, bprime)
        # loss attributable to the disturbance, per sub.
        base_rate = loss_in(series, -a.baseline_window, 0.0) / max(a.baseline_window, 1e-9)
        if rec is not None:
            raw_loss = loss_in(series, 0.0, rec)
            adj_loss = max(0.0, raw_loss - base_rate * rec)
        else:
            # never recovered within the window: count loss over the whole post-t0
            raw_loss = loss_in(series, 0.0, series[-1][0] if series else 0.0)
            adj_loss = raw_loss
        per_sub.append({
            "scenario": scen, "transport": r["mode"], "p": r["n_workers"],
            "s": r["n_subs"], "trial": r["trial"], "sub_idx": r.get("sub_idx"),
            "bprime_eps": round(bprime, 1) if bprime else None,
            "bprime_cov": round(cov, 3) if cov is not None else None,
            "unstable_bprime": unstable,
            "recovery_s": round(rec, 3) if rec is not None else None,
            "dip_depth_frac": dip,
            "loss_raw": int(raw_loss), "loss_adj": int(adj_loss),
            "res": r.get("res", {}),
        })

    # --- Aggregate + print -------------------------------------------------- #
    print(f"# DIS-2172 Dynamic-Behavior / Recovery report  (in={a.indir})")
    print(f"# bucket_ms={bucket_ms} recover_frac={a.recover_frac} "
          f"stable_window={a.stable_window}s bprime_window={a.bprime_window}s "
          f"etcd_lease_ttl={meta.get('etcd_lease_ttl')}\n")

    # Per-scenario, per-transport recovery + loss (per-sub, NOT aggregated away;
    # then median/min/max across the disturbed subs).
    by = defaultdict(list)
    for rec in per_sub:
        by[(rec["scenario"], rec["transport"])].append(rec)

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    print("## Recovery + loss (per scenario x transport)\n")
    print("| scenario | transport | n_sub_obs | recov_med_s | recov_min_s | recov_max_s | "
          "B'_eps_med | dip_med | loss_adj_med | loss_adj_max | no_recover |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (scen, tr) in sorted(by):
        recs = by[(scen, tr)]
        rvals = [r["recovery_s"] for r in recs]
        no_rec = sum(1 for r in recs if r["recovery_s"] is None)
        print(f"| {scen} | {tr} | {len(recs)} | "
              f"{med(rvals)} | "
              f"{min([x for x in rvals if x is not None], default=None)} | "
              f"{max([x for x in rvals if x is not None], default=None)} | "
              f"{med([r['bprime_eps'] for r in recs])} | "
              f"{med([r['dip_depth_frac'] for r in recs])} | "
              f"{med([r['loss_adj'] for r in recs])} | "
              f"{max([r['loss_adj'] for r in recs], default=None)} | "
              f"{no_rec} |")

    # sub_restart: separate the restarted sub (sub_idx 0) from the bystanders to
    # show a restart only disturbs the restarted sub.
    restart = [r for r in per_sub if r["scenario"] == "sub_restart"]
    if restart:
        print("\n## sub_restart: restarted sub (sub0) vs bystanders\n")
        print("| transport | which | recov_med_s | dip_med | loss_adj_med |")
        print("|---|---|---|---|---|")
        for tr in sorted({r["transport"] for r in restart}):
            for label, pred in (("restarted(sub0)", lambda r: r["sub_idx"] == 0),
                                ("bystanders", lambda r: r["sub_idx"] != 0)):
                g = [r for r in restart if r["transport"] == tr and pred(r)]
                if not g:
                    continue
                print(f"| {tr} | {label} | {med([r['recovery_s'] for r in g])} | "
                      f"{med([r['dip_depth_frac'] for r in g])} | "
                      f"{med([r['loss_adj'] for r in g])} |")

    # ZMQ established-TCP conn-count overlay (mesh rebuild evidence).
    print("\n## established-TCP peak (mesh-rebuild cost, per scenario x transport)\n")
    print("| scenario | transport | est_tcp_node_peak_med | est_tcp_total_peak_med |")
    print("|---|---|---|---|")
    for (scen, tr) in sorted(by):
        recs = by[(scen, tr)]
        nodep = [r["res"].get("established_tcp_node_peak") for r in recs if r.get("res")]
        totp = [r["res"].get("established_tcp_total_peak") for r in recs if r.get("res")]
        print(f"| {scen} | {tr} | {med(nodep)} | {med(totp)} |")

    # slow scenario: drop_rate isolation table (slow sub vs fast subs).
    if slow_rows:
        print("\n## slow-subscriber backpressure: isolation (drop_rate)\n")
        print("| transport | slow_sub_drop | fast_subs_drop_med | fast_subs_thr_med_eps |")
        print("|---|---|---|---|")
        wsec = meta.get("duration", 20)
        bytr = defaultdict(list)
        for r in slow_rows:
            bytr[r["mode"]].append(r)
        for tr in sorted(bytr):
            g = bytr[tr]
            slow = [r for r in g if r.get("slow_sleep_ms")]
            fast = [r for r in g if not r.get("slow_sleep_ms")]
            slow_drop = med([r.get("drop_rate") for r in slow])
            fast_drop = med([r.get("drop_rate") for r in fast])
            fast_thr = med([(r.get("received", 0) / (r.get("window_secs") or wsec))
                            for r in fast])
            print(f"| {tr} | {slow_drop} | {fast_drop} | "
                  f"{round(fast_thr, 1) if fast_thr is not None else None} |")

    # machine-readable dump
    with open(f"{a.indir}/recovery.json", "w") as f:
        json.dump({"per_sub": per_sub, "slow_rows": [
            {k: v for k, v in r.items() if k != "res"} for r in slow_rows]}, f, indent=2)
    print(f"\n[recovery] wrote {a.indir}/recovery.json ({len(per_sub)} sub records, "
          f"{len(slow_rows)} slow rows)")


if __name__ == "__main__":
    main()
