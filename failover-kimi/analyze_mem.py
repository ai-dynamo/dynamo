#!/usr/bin/env python3
"""Shadow memory-overhead + concurrent bring-up peak from a kimi_failover run.

Consumes the 1 Hz all-GPU sampler (dev_mem.csv: ts,gpu,mem_mib) and the
epoch-stamped phase boundaries (phases.csv: epoch,label). Reports, per GPU:
  - active_resting     : mem with only the ACTIVE engine up (resting)
  - colocated_resting  : mem with ACTIVE + resting SHADOW
  - shadow_overhead    : colocated_resting - active_resting  (Q1)
  - bringup_peak       : max mem during [shadow_launch, shadow_standby] (Q2 —
                         the concurrent bring-up peak, shadow init w/ live active)
  - overall_peak       : max mem over the whole run
and headroom vs the card ceiling.

Usage: analyze_mem.py DEV_MEM_CSV PHASES_CSV [--ceiling-mib N]
"""
import csv
import sys


def load_dev(path):
    g = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                gpu = int(row["gpu"])
                g.setdefault(gpu, []).append((float(row["ts"]), float(row["mem_mib"])))
            except (ValueError, KeyError):
                continue
    for gpu in g:
        g[gpu].sort()
    return g


def load_phases(path):
    ph = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                ph[row["label"]] = float(row["epoch"])  # last occurrence wins
    except FileNotFoundError:
        pass
    return ph


def val_at(series, t):
    # nearest sample to t
    return min(series, key=lambda x: abs(x[0] - t))[1]


def peak_in(series, t0, t1):
    vals = [m for (ts, m) in series if t0 <= ts <= t1]
    return max(vals) if vals else float("nan")


def main():
    dev_csv, phases_csv = sys.argv[1], sys.argv[2]
    ceiling = None
    if "--ceiling-mib" in sys.argv:
        ceiling = float(sys.argv[sys.argv.index("--ceiling-mib") + 1])
    dev = load_dev(dev_csv)
    ph = load_phases(phases_csv)

    p_active = ph.get("active_resting", ph.get("active_registered"))
    p_coloc = ph.get("colocated_resting", ph.get("shadow_standby"))
    p_sl = ph.get("shadow_launch")
    p_ss = ph.get("shadow_standby")
    miss = [n for n, v in [("active", p_active), ("coloc", p_coloc),
                            ("shadow_launch", p_sl), ("shadow_standby", p_ss)] if v is None]
    if miss:
        print(f"WARN missing phase markers: {miss} (labels present: {sorted(ph)})")

    hdr = f"{'gpu':>3} {'active':>9} {'coloc':>9} {'sh_over':>8} {'bringup_pk':>11} {'overall_pk':>11}"
    if ceiling:
        hdr += f" {'headroom':>9}"
    print(hdr)
    worst_over = worst_peak = 0.0
    for gpu in sorted(dev):
        s = dev[gpu]
        a = val_at(s, p_active) if p_active else float("nan")
        c = val_at(s, p_coloc) if p_coloc else float("nan")
        over = c - a
        bpk = peak_in(s, p_sl, p_ss) if (p_sl and p_ss) else float("nan")
        opk = max(m for _, m in s)
        worst_over = max(worst_over, over if over == over else 0)
        worst_peak = max(worst_peak, bpk if bpk == bpk else 0)
        line = f"{gpu:>3} {a:>9.0f} {c:>9.0f} {over:>8.0f} {bpk:>11.0f} {opk:>11.0f}"
        if ceiling:
            line += f" {ceiling - opk:>9.0f}"
        print(line)
    print(f"\nWORST-GPU  shadow_overhead={worst_over:.0f} MiB  "
          f"concurrent_bringup_peak={worst_peak:.0f} MiB", end="")
    if ceiling:
        print(f"  headroom_at_peak={ceiling - worst_peak:.0f} MiB (ceiling {ceiling:.0f})")
        print(f"VERDICT: concurrent shadow bring-up "
              f"{'FITS' if worst_peak < ceiling else 'EXCEEDS'} the ceiling")
    else:
        print()


if __name__ == "__main__":
    main()
