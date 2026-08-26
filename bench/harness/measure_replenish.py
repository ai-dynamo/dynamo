#!/usr/bin/env python3
"""Windowed serving metrics around failover and during shadow replenishment.

Reads a snaptrack-style run bundle and prints:

  promotion_s     last failover_state waking->active window after the kill
  replenish_s     kill until the killed engine is standby again (RO import + graphs)
  pre / during-replenish / post   TTFT p50, ITL p50, output tok/s

The replenishment window is the one that answers "how much contention does
bringing the shadow back impose on the survivor". It starts when the promoted
engine is active and ends when the restarted engine logs
``failover_state engine=N -> standby``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cutover import parse_ts, waking_window  # noqa: E402


def _ts(line):
    t = parse_ts(line)
    if t is None:
        return None
    return t.replace(tzinfo=datetime.timezone.utc).timestamp()


def kill_unix(bundle: pathlib.Path) -> float:
    return int(open(bundle / "harness/kill.txt").read().split()[0]) / 1e9


def replenish_end(logs, t0: float, killed_engine: str | None) -> float | None:
    """Last '-> standby' after t0, preferring the killed engine's log."""
    best = None
    for f in logs:
        for line in open(f, errors="ignore"):
            if "failover_state" not in line or "-> standby" not in line:
                continue
            ts = _ts(line)
            if ts is None or ts < t0:
                continue
            if best is None or ts > best[0]:
                best = (ts, f.stem)
    if best is None:
        return None
    return best[0]


def load_records(bundle: pathlib.Path):
    rows = []
    p = bundle / "aiperf/profile_export.jsonl"
    if not p.exists():
        return rows
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        md, mx = j.get("metadata", {}), j.get("metrics", {})
        start = md.get("request_start_ns") or md.get("start_ns")
        if start is None:
            continue
        start_s = start / 1e9 if start > 1e12 else start
        err = bool(j.get("error") or mx.get("error_request_count"))
        ttft = mx.get("time_to_first_token_ns") or mx.get("ttft_ns")
        itl = mx.get("inter_token_latency_ns") or mx.get("itl_ns")
        out = mx.get("output_token_count") or mx.get("output_tokens") or 0
        end = md.get("request_end_ns") or md.get("end_ns")
        end_s = (end / 1e9 if end and end > 1e12 else end) if end else None
        rows.append(
            {
                "start": start_s,
                "end": end_s,
                "err": err,
                "ttft_ms": (ttft / 1e6) if ttft else None,
                "itl_ms": (itl / 1e6) if itl else None,
                "out": out,
            }
        )
    return rows


def window_stats(rows, lo: float, hi: float) -> dict:
    sel = [r for r in rows if lo <= r["start"] < hi]
    ok = [r for r in sel if not r["err"]]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    itls = [r["itl_ms"] for r in ok if r["itl_ms"] is not None]
    toks = sum(r["out"] for r in ok)
    span = max(hi - lo, 1e-6)
    return {
        "n": len(sel),
        "ok": len(ok),
        "err": len(sel) - len(ok),
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
        "itl_p50_ms": statistics.median(itls) if itls else None,
        "output_tok_s": toks / span,
    }


def fmt(stats: dict) -> str:
    def n(x, d=1):
        return "n/a" if x is None else f"{x:.{d}f}"

    return (
        f"n={stats['n']} ok={stats['ok']} err={stats['err']}  "
        f"TTFT_p50={n(stats['ttft_p50_ms'])}ms  "
        f"ITL_p50={n(stats['itl_p50_ms'], 2)}ms  "
        f"out={n(stats['output_tok_s'])} tok/s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    args = ap.parse_args()
    bundle = pathlib.Path(args.bundle)
    t0 = kill_unix(bundle)
    logs = sorted((bundle / "logs").glob("engine-*.log"))
    ww = waking_window(logs)
    promo = ww[0] if ww else None
    # Approximate promoted-active time as kill + promotion.
    active_at = t0 + promo if promo is not None else t0
    repl = replenish_end(logs, t0, None)
    rows = load_records(bundle)
    if not rows:
        print("no aiperf records")
        sys.exit(1)

    pre_lo, pre_hi = t0 - 300, t0
    promo_hi = active_at
    repl_hi = repl if repl is not None else active_at + 300
    post_lo, post_hi = repl_hi, repl_hi + 300

    print(f"kill_unix={t0:.3f}")
    print(f"promotion_s={promo if promo is not None else 'n/a'}")
    print(
        f"replenish_end_s_after_kill="
        f"{(repl - t0) if repl is not None else 'n/a'}"
    )
    print(f"pre-fault          {fmt(window_stats(rows, pre_lo, pre_hi))}")
    print(f"promotion window   {fmt(window_stats(rows, t0, promo_hi))}")
    print(f"replenishment      {fmt(window_stats(rows, active_at, repl_hi))}")
    print(f"post-replenish     {fmt(window_stats(rows, post_lo, post_hi))}")


if __name__ == "__main__":
    main()
