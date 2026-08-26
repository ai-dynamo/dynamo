#!/usr/bin/env python3
"""Report promotion cutover for a run bundle, on either failover runtime.

The two runtimes account for promotion differently and neither can be assumed
present, so this reads whichever evidence a bundle actually has:

  fork A  -- "[Shadow] checkpoint_restore took Xs" + "wake_up(...) took Ys",
             emitted by instrumentation we added and compiled into that image.
  LDP     -- the failover state machine: "failover_state engine=N -> waking"
             through "-> active", plus a Prometheus gauge
             failover_last_state_duration_seconds{state="waking"} that the
             runtime documents as the wake/switch time.

Client outage is computed for both and is the number to compare across
runtimes, because it is measured entirely at the client and so cannot be
skewed by either runtime's choice of what to instrument.
"""
import json, pathlib, re, sys, datetime

TS = re.compile(r"(\d{4}-\d{2}-\d{2}T[\d:.]+)Z")


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_ts(line):
    m = TS.search(strip_ansi(line))
    if not m:
        return None
    t = m.group(1)
    if "." in t:
        head, frac = t.split(".")
        t = head + "." + (frac + "000000")[:6]
    return datetime.datetime.fromisoformat(t)


def forka(logs):
    r = w = None
    for f in logs:
        for line in open(f, errors="ignore"):
            if "checkpoint_restore took" in line:
                r = float(re.search(r"took ([\d.]+)s", line).group(1))
            elif "wake_up(" in line and "took" in line:
                w = float(re.search(r"took ([\d.]+)s", line).group(1))
    return (r, w) if (r is not None or w is not None) else None


def waking_window(logs):
    """Wake window from the state machine: -> waking through -> active.

    Both runtimes emit this, so it is the one number directly comparable across
    them -- and on fork A it is verifiably the whole promotion: the window
    contains checkpoint_restore plus wake_up with ~0.3s to spare.

    Every engine's log is scanned and the LAST completed window wins. Taking the
    first match reads the standby's own start-up promotion from earlier in the
    pod's life rather than the promotion this run triggered, which understated a
    37.2s cutover as 8.1s.
    """
    best = None
    for f in logs:
        start = None
        for line in open(f, errors="ignore"):
            if "failover_state" not in line:
                continue
            if "-> waking" in line:
                start = parse_ts(line)
            elif "-> active" in line and start:
                end = parse_ts(line)
                if end and (best is None or end > best[1]):
                    best = (start, end, f.stem)
                start = None
    if not best:
        return None
    return (best[1] - best[0]).total_seconds(), best[2]


def metric_waking(bundle):
    out = {}
    for f in sorted((bundle / "metrics").glob("engine-*.prom")):
        for line in open(f, errors="ignore"):
            if "last_state_duration_seconds" in line and 'state="waking"' in line:
                out[f.stem] = float(line.rsplit(None, 1)[1])
    return out


def outage(bundle):
    """Longest gap between consecutive successful completions.

    This is wider than the window in which requests were failing, because after
    the engine is back the queued prefills still have to run before anything
    completes. Both are real; this one is quoted because it is what a client
    actually experiences and needs no server-side signal at all.
    """
    p = bundle / "aiperf/profile_export.jsonl"
    if not p.exists():
        return None
    ends = []
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("metrics", {}).get("output_sequence_length"):
            ends.append(j["metadata"]["request_end_ns"] / 1e9)
    if len(ends) < 2:
        return None
    ends.sort()
    return max(b - a for a, b in zip(ends, ends[1:]))



def kill_anchored(bundle):
    """Client outage measured from the kill: last completion before, first after.

    This is the number to quote, not the largest gap anywhere in the run. The
    largest gap can sit somewhere unrelated -- a ramp, a slow tail -- and on
    sparse runs it is pure noise.

    It also needs a completion count beside it. Some early fork A runs used
    OSL=20000 and finished only 64 requests in the whole run, exactly one per
    lane, so every completion landed in one burst and the gap between them read
    as 0.0s while the engine had in fact been away for thirteen seconds. A run
    with fewer completions than about twice the concurrency cannot measure this
    at all, and saying so is better than reporting the artefact.
    """
    p = bundle / "aiperf/profile_export.jsonl"
    kf = bundle / "harness/kill.txt"
    if not p.exists() or not kf.exists():
        return None
    try:
        kill = int(open(kf).read().split()[0]) / 1e9
    except Exception:
        return None
    ends = []
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("metrics", {}).get("output_sequence_length"):
            ends.append(j["metadata"]["request_end_ns"] / 1e9)
    if len(ends) < 2:
        return None
    before = [t for t in ends if t < kill]
    after = [t for t in ends if t >= kill]
    if not before or not after:
        return None
    return (min(after) - max(before), len(ends))


def recovery(bundle):
    """Time from the kill to the first completion of a request issued after it.

    The obvious measure -- first completion after the kill -- is wrong here, and
    wrong in a flattering direction. Killing the engine terminates every stream
    in flight, and aiperf records those as successes because partial output
    exists: on the 200k trace, twelve requests all "completed" at the same
    instant 1.6s after the kill, with output lengths of 217 to 635 tokens against
    a trace median of 786. Measuring to those reports a 4.2s outage for a
    failover whose first real completion came at 47.6s.

    Restricting to requests that STARTED after the kill removes the artefact and
    measures what a user actually waits for. That number includes far more than
    the promotion: the promoted engine restores weights but not KV, so its prefix
    cache starts cold -- 29.9% against 84% before the fault -- and every session
    must re-prefill its whole accumulated context before anything can complete.
    """
    p = bundle / "aiperf/profile_export.jsonl"
    kf = bundle / "harness/kill.txt"
    if not p.exists() or not kf.exists():
        return None
    try:
        kill = int(open(kf).read().split()[0]) / 1e9
    except Exception:
        return None
    # Service resumes when the promoted engine reaches "active"; anything that
    # ended before that was cut off, whenever it started.
    #
    # An earlier rule -- started before the kill, ended after it -- missed
    # requests issued in the moment between SIGKILL and the engine actually
    # dying. On the c=24 run two such requests were admitted 0.28s and 1.07s
    # after the kill, served for a fraction of a second by the dying engine, and
    # ended at 1.80s with 31 and 70 output tokens against a trace median of 786.
    # Counting those as recovery reported the first post-kill completion at 1.8s
    # for a failover whose promotion alone took 25.9s.
    resumed = None
    logs = sorted((bundle / "logs").glob("engine-*.log"))
    ww = waking_window(logs)
    if ww:
        for f in logs:
            if f.stem != ww[1]:
                continue
            for line in open(f, errors="ignore"):
                if "failover_state" in line and "-> active" in line:
                    ts = parse_ts(line)
                    if ts:
                        resumed = ts.replace(tzinfo=datetime.timezone.utc).timestamp() - kill
    fresh = []
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if not j.get("metrics", {}).get("output_sequence_length"):
            continue
        m = j["metadata"]
        end = m["request_end_ns"] / 1e9 - kill
        if m["request_start_ns"] / 1e9 <= kill:
            continue
        if resumed is not None and end < resumed:
            continue          # truncated by the fault, not a recovery completion
        fresh.append(end)
    if not fresh:
        return None
    return (min(fresh), len(fresh))

def main():
    for arg in sys.argv[1:]:
        b = pathlib.Path(arg)
        logs = sorted((b / "logs").glob("engine-*.log"))
        print(f"\n{b.name}")
        fa = forka(logs)
        if fa:
            r, w = fa
            tot = sum(x for x in fa if x is not None)
            print(f"  fork A     checkpoint_restore={r}s  wake_up={w}s  total={tot:.3f}s")
        ww = waking_window(logs)
        if ww is not None:
            print(f"  both       waking->active ({ww[1]})  = {ww[0]:.3f}s")
        for k, v in metric_waking(b).items():
            print(f"  LDP        waking ({k}, metric)     = {v:.3f}s")
        ko = kill_anchored(b)
        if ko:
            print(f"  client     outage from kill          = {ko[0]:.1f}s  ({ko[1]} completions)")
        else:
            print("  client     outage from kill          = n/a (too few completions)")
        rc = recovery(b)
        if rc:
            print(f"  client     first POST-KILL completion = {rc[0]:.1f}s  ({rc[1]} such requests)")


if __name__ == "__main__":
    main()
