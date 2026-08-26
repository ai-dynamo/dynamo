#!/usr/bin/env python3
"""TTFT and per-user decode rate around a failover, from aiperf records.

Per-user decode rate is the metric that answers "what did a user experience".
System throughput can hold steady while every individual stream slows down, and
only the per-user view separates those. TTFT answers the complementary
question -- whether new requests were being admitted promptly.

Both are plotted against time from the SIGKILL, with a windowed median over the
raw scatter. Medians rather than means because the failure window produces
extreme outliers that would drag a mean around without describing anything a
user sees.
"""
import argparse, datetime, json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def is_nofault(d):
    """KILL=0 writes 'none none' where the kill mode and targets would be.

    Worth detecting rather than ignoring: on a smoke run the vertical marker is
    not a fault and the x-axis is not measured from one, and a chart that says
    SIGKILL when nothing was killed invites exactly the wrong reading.
    """
    try:
        return open(pathlib.Path(d) / "harness/kill.txt").read().split()[2] == "none"
    except Exception:
        return False


def load(d):
    """Classify every record and anchor it to when the request STARTED.

    Anchoring on request_end_ns puts a measurement in the wrong place. TTFT is
    fixed within a second of the request being issued, so a request that started
    64s before the kill and was cut off 1.6s after it has a TTFT measured well
    before the fault -- plotting that point after the fault reads as though the
    promoted engine served it. Start time is where the measurement belongs.

    Three outcomes, not two:

    completed  ran to its own end while one engine served it throughout.
    truncated  in flight at the instant of the kill. aiperf records these as
               successes because partial output exists, but they are wreckage:
               on this run all twelve ended at the same timestamp with output
               lengths proportional to how long each had been decoding -- 3,615
               tokens for one running 64s, 217 for one running 2s. Counting them
               as completions is what made a 47s outage look like 4.2s.
    errored    no output at all; the frontend rejected the request outright.
    """
    d = pathlib.Path(d)
    t0 = int(open(d / "harness/kill.txt").read().split()[0]) / 1e9
    killed = open(d / "harness/kill.txt").read().split()[2] != "none"
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from cutover import waking_window, parse_ts
    resumed = None
    if killed:
        logs = sorted((d / "logs").glob("engine-*.log"))
        ww = waking_window(logs)
        if ww:
            for f in logs:
                if f.stem != ww[1]:
                    continue
                for line in open(f, errors="ignore"):
                    if "failover_state" in line and "-> active" in line:
                        ts = parse_ts(line)
                        if ts:
                            resumed = ts.replace(tzinfo=datetime.timezone.utc).timestamp() - t0
    done, trunc, bad = [], [], []
    for line in open(d / "aiperf/profile_export.jsonl"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        md, mx = j.get("metadata", {}), j.get("metrics", {})
        if md.get("benchmark_phase") != "profiling":
            continue
        st = md.get("request_start_ns", 0) / 1e9 - t0
        en = md.get("request_end_ns", 0) / 1e9 - t0
        if j.get("error") or not mx.get("output_sequence_length"):
            bad.append(st)
            continue
        g = lambda k: (mx.get(k) or {}).get("value")
        rec = {"t": st, "end": en, "ttft": g("time_to_first_token"),
               "tpu": g("output_token_throughput_per_user"),
               "itl": g("inter_token_latency"),
               "osl": g("output_sequence_length")}
        # Truncated means "ended before service resumed", not "straddled the
        # kill". SIGKILL is not instantaneous: requests admitted in the moment
        # after it get a fraction of a second from the dying engine and end with
        # a handful of tokens, and a straddle test classifies those as clean
        # completions.
        # Bounded on BOTH sides. Without the en > 0 test this also swallows every
        # normal completion from before the fault, because they too ended before
        # service resumed -- which reported zero pre-fault completions on a run
        # that had hundreds.
        cut = killed and resumed is not None and 0 < en < resumed
        (trunc if cut else done).append(rec)
    return done, trunc, bad


WINDOW_S = 30.0


def windowed_stat(xs, ys, width=30.0, stat="median"):
    """Median, not mean, over a 30s window.

    Median because this distribution has a long, sparse tail that a mean would
    chase: TTFT p50 is ~890ms against a p90 of 12s, and cold-prefill requests
    after a promotion reach 56s. A handful of those drags a windowed mean far
    above anything a typical request experienced, so the line stops describing
    the workload and starts tracking its outliers.

    30s rather than 5s because at 5s each point held only a few requests, so the
    line jittered hard enough to obscure the trend it exists to show. Throughput
    is still summed, not averaged -- for capacity questions the total is the
    statistic, and no robustness argument applies.
    """
    if not xs:
        return np.array([]), np.array([])
    lo, hi = min(xs), max(xs)
    edges = np.arange(lo, hi + width, width)
    idx = np.digitize(xs, edges) - 1
    ys = np.asarray(ys, dtype=float)
    cx, cy = [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= 3:
            cx.append(edges[i] + width / 2)
            cy.append(float(np.median(ys[m]) if stat == "median" else np.mean(ys[m])))
    return np.asarray(cx), np.asarray(cy)


def _break_gaps(cx, cy, width):
    """Insert NaNs across gaps so the line does not interpolate over absent data."""
    if not len(cx):
        return cx, cy
    cut = np.flatnonzero(np.diff(cx) > 4 * width)
    cxb, cyb = cx.astype(float).copy(), cy.astype(float).copy()
    for k in reversed(cut):
        cxb = np.insert(cxb, k + 1, cxb[k] + width)
        cyb = np.insert(cyb, k + 1, np.nan)
    return cxb, cyb


def panel(ax, ok, trunc, bad, key, ylabel, sla=None, log=False, first_fresh=None):
    pts = [(r["t"], r[key]) for r in ok if r.get(key)]
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=6, alpha=.25, color="#1f77b4",
                   label=f"completed ({len(pts)})", rasterized=True)
        # Truncated requests are drawn, but separately and never in the median.
        # They are real measurements of a request the fault destroyed, so hiding
        # them loses information, while folding them into the median would let
        # the wreckage of the old engine set the reported client experience.
        tp = [(r["t"], r[key]) for r in trunc if r.get(key)]
        if tp:
            ax.scatter([q[0] for q in tp], [q[1] for q in tp], s=44, marker="x",
                       color="#ff7f0e", linewidths=1.4, zorder=4,
                       label=f"truncated by the fault ({len(tp)})")
        # Median and mean together. Where they diverge, the window contains
        # outliers the median is deliberately ignoring -- after a promotion the
        # mean runs far above the median because a handful of cold-prefill
        # requests take tens of seconds. Showing both makes that visible instead
        # of asking the reader to trust one summary.
        cx, cy = windowed_stat(xs, ys, WINDOW_S, "median")
        if len(cx):
            a, b = _break_gaps(cx, cy, WINDOW_S)
            ax.plot(a, b, color="#d62728", lw=2, zorder=5,
                    label=f"{WINDOW_S:.0f}s median")
        mx_, my_ = windowed_stat(xs, ys, WINDOW_S, "mean")
        if len(mx_):
            a, b = _break_gaps(mx_, my_, WINDOW_S)
            ax.plot(a, b, color="#9467bd", lw=1.6, ls="--", zorder=4,
                    label=f"{WINDOW_S:.0f}s mean")

    if bad:
        lo, hi = min(bad), max(bad)
        # Rejected requests have no latency to plot, so they are shown as rug
        # ticks along the bottom -- present and countable without pretending to
        # a y-value they never had.
        # Pinned to the axis floor in axes coordinates. Reading get_ylim() here
        # samples the limit before set_ylim() runs later in this function, which
        # left the ticks floating mid-plot as if they carried a real value.
        ax.scatter(bad, [0.012] * len(bad), marker="|", s=60, color="#d62728",
                   alpha=.55, zorder=4, label=f"rejected ({len(bad)})",
                   transform=ax.get_xaxis_transform())
        ax.axvspan(lo, hi, color="#d62728", alpha=.12, zorder=0)
        ax.annotate(f"rejecting {hi - lo:.1f}s", xy=((lo + hi) / 2, ax.get_ylim()[1]),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="#a00")
    # The rejection window is not the outage a user experiences, and showing only
    # it is what made these charts look wrong: the band said 6.4s while the plot
    # plainly had no completions for 47s. The engine starts accepting as soon as
    # the promotion finishes, but it comes back with a cold prefix cache -- 29.9%
    # against 84% before the fault -- so every session re-prefills its whole
    # accumulated context before anything completes. Marking the first completion
    # of a request issued after the kill shows that second, larger cost.
    if first_fresh is not None:
        ax.axvspan(0, first_fresh, color="#ff7f0e", alpha=.07, zorder=0)
        ax.axvline(first_fresh, color="#ff7f0e", ls="-.", lw=1.6, zorder=2)
        ax.annotate(f"first post-kill completion {first_fresh:.0f}s",
                    xy=(first_fresh, ax.get_ylim()[1]), xytext=(6, -30),
                    textcoords="offset points", fontsize=9, color="#b35900")
    if sla:
        ax.axhline(sla, color="gray", ls=":", alpha=.7, label=f"SLA {sla:g}")
    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.3, which="both" if log else "major")
    ax.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("bundle")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    name = a.name or pathlib.Path(a.bundle).name
    nofault = is_nofault(a.bundle)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from cutover import recovery as _recovery
    _rc = None if nofault else _recovery(pathlib.Path(a.bundle))
    first_fresh = _rc[0] if _rc else None
    ok, trunc, bad = load(a.bundle)
    if not ok:
        print(f"{name}: no successful records"); return 1

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    panel(axes[0], ok, trunc, bad, "ttft", "TTFT (ms)", log=True, first_fresh=first_fresh)
    axes[0].set_title(f"{name} — client experience" + ("" if not nofault else " (control: no fault injected)"))
    panel(axes[1], ok, trunc, bad, "tpu", "output tokens/s/user", sla=None, first_fresh=first_fresh)
    axes[1].set_xlabel("seconds from T0, by request START time (no fault injected)" if nofault
                       else "seconds from SIGKILL, by request START time")
    fig.tight_layout(); fig.savefig(out / f"client-{name}.png", dpi=140); plt.close(fig)

    pre = [r for r in ok if r["t"] < 0]
    post = [r for r in ok if r["t"] > (max(bad) if bad else 0)]
    def med(rs, k):
        v = [r[k] for r in rs if r.get(k)]
        return float(np.median(v)) if v else float("nan")
    print(f"  {name}")
    print(f"    completed before fault: {len(pre):>5}   after recovery: {len(post)}")
    print(f"    TTFT   median  before {med(pre,'ttft'):7.1f} ms   after {med(post,'ttft'):7.1f} ms")
    print(f"    tok/s/user     before {med(pre,'tpu'):7.1f}      after {med(post,'tpu'):7.1f}")
    print(f"    ITL    median  before {med(pre,'itl'):7.2f} ms   after {med(post,'itl'):7.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
