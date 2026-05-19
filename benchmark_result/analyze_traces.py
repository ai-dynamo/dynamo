#!/usr/bin/env python3
"""Parse [TTFT-TRACE] events from frontend + worker logs and render a per-request
waterfall plus aggregate percentile stats.

Usage:
  analyze_traces.py [--dir DIR] [--requests N] [--request-id RID] [--width W]
                    [--top-by {ttft,tokenize,prefill_wait,prefill_engine,
                               kv_transfer,decode_engine}]

Defaults read frontend.log, ctx_worker.log, gen_worker.log from --dir
(which defaults to the script's directory).

The frontend tags some lines with two request_id values (trace UUID + HTTP-span
UUID) and some lines with only the span UUID. We build a span->trace map from
the dual-tagged lines and remap orphans.
"""

import argparse
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TRACE_RE = re.compile(r"\[TTFT-TRACE\]\s+(.*)$")
KV_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')
RID_RE = re.compile(r'request_id=("[^"]*"|\S+)')


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_kv_first(rest: str) -> dict:
    """Return dict of key=value, keeping the FIRST occurrence on duplicates."""
    out = {}
    for k, v in KV_RE.findall(rest):
        if k in out:
            continue
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        out[k] = v
    return out


def extract_request_ids(rest: str) -> list:
    out = []
    for v in RID_RE.findall(rest):
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        out.append(v)
    return out


@dataclass
class Event:
    source: str  # 'frontend' | 'ctx' | 'gen'
    stage: str
    epoch_ms: Optional[int]
    fields: dict = field(default_factory=dict)


@dataclass
class RequestTrace:
    request_id: str
    frontend: list = field(default_factory=list)
    ctx: list = field(default_factory=list)
    gen: list = field(default_factory=list)


def read_raw_events(path: str, source: str) -> list:
    out = []
    if not os.path.exists(path):
        print(f"warning: {path} not found, skipping", file=sys.stderr)
        return out
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "TTFT-TRACE" not in line:
                continue
            clean = strip_ansi(line)
            m = TRACE_RE.search(clean)
            if not m:
                continue
            rest = m.group(1)
            rids = extract_request_ids(rest)
            fields = parse_kv_first(rest)
            stage = fields.get("stage")
            if not stage:
                continue
            epoch_ms = None
            if "epoch_ms" in fields:
                try:
                    epoch_ms = int(fields["epoch_ms"])
                except ValueError:
                    pass
            out.append((rids, source, stage, epoch_ms, fields))
    return out


def build_traces(raw_events: list) -> dict:
    """First pass: span->trace map from lines tagged with both.
    Second pass: bucket every event by trace_id (remapping orphans)."""
    span_to_trace = {}
    for rids, *_ in raw_events:
        if len(rids) >= 2:
            # frontend: first is trace UUID, second is HTTP span UUID
            span_to_trace[rids[-1]] = rids[0]

    traces: dict = {}
    for rids, source, stage, epoch_ms, fields in raw_events:
        if not rids:
            continue
        rid = rids[0]
        if len(rids) == 1 and rid in span_to_trace:
            rid = span_to_trace[rid]
        tr = traces.get(rid)
        if tr is None:
            tr = RequestTrace(request_id=rid)
            traces[rid] = tr
        getattr(tr, source).append(
            Event(source=source, stage=stage, epoch_ms=epoch_ms, fields=fields)
        )
    return traces


FRONTEND_ORDER = [
    "http_handler_entered",
    "preprocessor_entered",
    "tokenize_done",
    "prefill_router_entry",
    "prefill_worker_resolved",
    "prefill_dispatch",
    "prefill_rpc_send",
    "kv_push_router_worker_selected",  # prefill
    "kv_push_router_dispatch",  # prefill
    "kv_push_router_first_response",  # prefill
    "kv_push_router_first_token",  # prefill
    "prefill_rpc_first_response",
    "prefill_returned_to_router",
    "decode_dispatch_to_next_operator",
    "kv_push_router_worker_selected",  # decode (second occurrence)
    "kv_push_router_dispatch",  # decode
    "kv_push_router_first_response",  # decode
    "kv_push_router_first_token",  # decode
    "first_chunk_detokenized",
    "first_sse_chunk_emitted",
]

WORKER_ORDER = [
    "worker_rpc_entry",
    "handler_received",
    "prep_breakdown",
    "pre_engine_generate_async",
    "post_engine_generate_async_call",
    "engine_first_response",
    "first_chunk_yielded",
]

# Transitions for percentile aggregation.
# (label, source, from_stage, to_stage, phase_filter_for_b_or_None)
TRANSITIONS_EPOCH = [
    (
        "http→preprocessor",
        "frontend",
        "http_handler_entered",
        "preprocessor_entered",
        None,
    ),
    (
        "preprocessor→prefill_router",
        "frontend",
        "preprocessor_entered",
        "prefill_router_entry",
        None,
    ),
    (
        "prefill_router_resolve",
        "frontend",
        "prefill_router_entry",
        "prefill_worker_resolved",
        None,
    ),
    (
        "resolve→rpc_send",
        "frontend",
        "prefill_worker_resolved",
        "prefill_rpc_send",
        None,
    ),
    (
        "prefill_rpc_roundtrip",
        "frontend",
        "prefill_rpc_send",
        "prefill_rpc_first_response",
        None,
    ),
    (
        "prefill_return→decode_dispatch",
        "frontend",
        "prefill_returned_to_router",
        "decode_dispatch_to_next_operator",
        None,
    ),
    (
        "decode_dispatch→decode_first_resp",
        "frontend",
        "decode_dispatch_to_next_operator",
        "kv_push_router_first_response",
        "decode",
    ),
    (
        "decode_first_resp→first_token",
        "frontend",
        "kv_push_router_first_response",
        "kv_push_router_first_token",
        "decode",
    ),
    (
        "first_token→detokenize",
        "frontend",
        "kv_push_router_first_token",
        "first_chunk_detokenized",
        "decode",
    ),
    (
        "detokenize→sse_emit",
        "frontend",
        "first_chunk_detokenized",
        "first_sse_chunk_emitted",
        None,
    ),
    (
        "TOTAL TTFT (http→sse)",
        "frontend",
        "http_handler_entered",
        "first_sse_chunk_emitted",
        None,
    ),
    ("ctx: rpc→handler", "ctx", "worker_rpc_entry", "handler_received", None),
    (
        "ctx: handler→pre_engine",
        "ctx",
        "handler_received",
        "pre_engine_generate_async",
        None,
    ),
    (
        "ctx: pre_engine→first_response",
        "ctx",
        "pre_engine_generate_async",
        "engine_first_response",
        None,
    ),
    ("ctx: total handler", "ctx", "worker_rpc_entry", "engine_first_response", None),
    ("gen: rpc→handler", "gen", "worker_rpc_entry", "handler_received", None),
    (
        "gen: handler→pre_engine",
        "gen",
        "handler_received",
        "pre_engine_generate_async",
        None,
    ),
    (
        "gen: pre_engine→first_response",
        "gen",
        "pre_engine_generate_async",
        "engine_first_response",
        None,
    ),
    ("gen: total handler", "gen", "worker_rpc_entry", "engine_first_response", None),
]

# Field-based aggregations (no epoch_ms math).
# (label, source, stage, field, phase_filter_or_None)
FIELD_AGGS = [
    ("tokenize_ms (field)", "frontend", "tokenize_done", "tokenize_ms", None),
    (
        "ttft_ms (field, decode)",
        "frontend",
        "kv_push_router_first_token",
        "ttft_ms",
        "decode",
    ),
    (
        "prefill_wait_ms (field)",
        "frontend",
        "kv_push_router_first_token",
        "prefill_wait_ms",
        "decode",
    ),
    (
        "prefill_time_ms (field)",
        "frontend",
        "kv_push_router_first_token",
        "prefill_time_ms",
        "decode",
    ),
    (
        "kv_transfer_estimated_ms (upper)",
        "frontend",
        "kv_push_router_first_token",
        "kv_transfer_estimated_ms",
        "decode",
    ),
    ("ctx: engine_ms (field)", "ctx", "engine_first_response", "engine_ms", None),
    (
        "ctx: submit_ms (field)",
        "ctx",
        "post_engine_generate_async_call",
        "submit_ms",
        None,
    ),
    ("ctx: prep_ms (field)", "ctx", "pre_engine_generate_async", "prep_ms", None),
    ("gen: engine_ms (field)", "gen", "engine_first_response", "engine_ms", None),
    (
        "gen: submit_ms (field)",
        "gen",
        "post_engine_generate_async_call",
        "submit_ms",
        None,
    ),
    ("gen: prep_ms (field)", "gen", "pre_engine_generate_async", "prep_ms", None),
]


def find_event(events, stage, phase=None):
    for ev in events:
        if ev.stage != stage:
            continue
        if phase is not None:
            if ev.fields.get("phase", "").lower() != phase:
                continue
        return ev
    return None


def compute_transition_ms(trace, source, a, b, phase):
    events = getattr(trace, source)
    # Only stages that legitimately repeat (kv_push_router_*) need phase
    # filtering. Others don't carry a phase= field.
    a_phase = phase if a.startswith("kv_push") else None
    b_phase = phase if b.startswith("kv_push") else None
    a_ev = find_event(events, a, a_phase)
    b_ev = find_event(events, b, b_phase)
    if not a_ev or not b_ev or a_ev.epoch_ms is None or b_ev.epoch_ms is None:
        return None
    delta = b_ev.epoch_ms - a_ev.epoch_ms
    if delta < 0:
        return None
    return float(delta)


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def fmt_ms(ms):
    if ms is None or ms != ms:
        return "   --   "
    if abs(ms) < 1:
        return f"{ms:8.3f}"
    if abs(ms) < 1000:
        return f"{ms:8.2f}"
    return f"{ms:8.0f}"


def render_waterfall(trace, width=120):
    """Single timeline anchored at earliest event.

    Cross-process clock skew between frontend / ctx / gen is a known caveat
    (see project memory). Worker timestamps are NOT adjusted.
    """
    events = []
    for src in ("frontend", "ctx", "gen"):
        for ev in getattr(trace, src):
            if ev.epoch_ms is not None:
                events.append(ev)
    if not events:
        return f"(no epoch_ms events for {trace.request_id})"

    t0 = min(ev.epoch_ms for ev in events)
    t_end = max(ev.epoch_ms for ev in events)
    span = max(1, t_end - t0)
    src_order = {"frontend": 0, "ctx": 1, "gen": 2}

    def sort_key(ev):
        order_list = FRONTEND_ORDER if ev.source == "frontend" else WORKER_ORDER
        try:
            idx = order_list.index(ev.stage)
        except ValueError:
            idx = 999
        return (ev.epoch_ms, src_order[ev.source], idx)

    events.sort(key=sort_key)

    bar_w = max(20, width - 60)
    out = []
    out.append(f"Request: {trace.request_id}")
    out.append(f"Window: {span} ms  (t0 epoch_ms = {t0})")
    out.append("")
    header = f"{'src':<4} {'+ms':>6} {'Δms':>6}  {'bar':<{bar_w}}  stage"
    out.append(header)
    out.append("-" * len(header))

    prev_t = t0
    for ev in events:
        rel = ev.epoch_ms - t0
        delta = ev.epoch_ms - prev_t
        prev_t = ev.epoch_ms
        pos = int(rel / span * (bar_w - 1)) if span else 0
        bar = [" "] * bar_w
        bar[pos] = {"frontend": "F", "ctx": "C", "gen": "G"}[ev.source]
        bar_str = "".join(bar)
        phase = ev.fields.get("phase", "")
        suffix = f" [{phase}]" if phase else ""
        extras = []
        for k in (
            "tokenize_ms",
            "engine_ms",
            "ttft_ms",
            "prefill_wait_ms",
            "prefill_time_ms",
            "kv_transfer_estimated_ms",
            "resolve_ms",
            "select_ms",
            "prep_ms",
            "submit_ms",
            "num_tokens",
            "normalize_ms",
            "setup_disagg_ms",
            "prepare_input_ms",
            "sampling_trace_ms",
        ):
            if k in ev.fields:
                extras.append(f"{k}={ev.fields[k]}")
        extras_str = "  " + " ".join(extras) if extras else ""
        out.append(
            f"{ev.source[:3]:<4} {rel:>6d} {delta:>6d}  {bar_str}  {ev.stage}{suffix}{extras_str}"
        )

        # Inline next worker no-epoch stages (post_engine_generate_async_call,
        # prep_breakdown) right after their predecessor for visibility.
        if ev.source in ("ctx", "gen"):
            for follow in getattr(trace, ev.source):
                if follow.epoch_ms is not None:
                    continue
                # Render between pre_engine_generate_async and engine_first_response.
                if (
                    ev.stage == "pre_engine_generate_async"
                    and follow.stage == "post_engine_generate_async_call"
                ):
                    sub = " ".join(
                        f"{k}={v}"
                        for k, v in follow.fields.items()
                        if k not in ("stage", "request_id", "mode")
                    )
                    out.append(
                        f"{ev.source[:3]:<4} {'·':>6} {'·':>6}  {' ' * bar_w}  └─ {follow.stage}  {sub}"
                    )
                if ev.stage == "handler_received" and follow.stage == "prep_breakdown":
                    sub = " ".join(
                        f"{k}={v}"
                        for k, v in follow.fields.items()
                        if k not in ("stage", "request_id", "mode")
                    )
                    out.append(
                        f"{ev.source[:3]:<4} {'·':>6} {'·':>6}  {' ' * bar_w}  └─ {follow.stage}  {sub}"
                    )
    return "\n".join(out)


def render_aggregates(traces):
    out = []
    n_total = len(traces)
    out.append(f"Aggregate stage timings across {n_total} requests (ms)")
    out.append("")
    header = (
        f"{'transition':<40} {'n':>6} {'p50':>9} {'p90':>9} "
        f"{'p99':>9} {'max':>9} {'mean':>9}"
    )
    out.append(header)
    out.append("-" * len(header))
    for label, source, a, b, phase in TRANSITIONS_EPOCH:
        values = []
        for tr in traces.values():
            v = compute_transition_ms(tr, source, a, b, phase)
            if v is not None:
                values.append(v)
        if not values:
            out.append(f"{label:<40} {0:>6}    (no data)")
            continue
        out.append(
            f"{label:<40} {len(values):>6} "
            f"{fmt_ms(statistics.median(values)):>9} "
            f"{fmt_ms(percentile(values, 90)):>9} "
            f"{fmt_ms(percentile(values, 99)):>9} "
            f"{fmt_ms(max(values)):>9} "
            f"{fmt_ms(statistics.mean(values)):>9}"
        )

    out.append("")
    out.append("Field-based aggregates (read directly from trace fields):")
    out.append("")
    out.append(header)
    out.append("-" * len(header))
    for label, source, stage, field_name, phase in FIELD_AGGS:
        values = []
        for tr in traces.values():
            ev = find_event(getattr(tr, source), stage, phase)
            if not ev:
                continue
            v = ev.fields.get(field_name)
            if v is None:
                continue
            try:
                values.append(float(v))
            except ValueError:
                continue
        if not values:
            out.append(f"{label:<40} {0:>6}    (no data)")
            continue
        out.append(
            f"{label:<40} {len(values):>6} "
            f"{fmt_ms(statistics.median(values)):>9} "
            f"{fmt_ms(percentile(values, 90)):>9} "
            f"{fmt_ms(percentile(values, 99)):>9} "
            f"{fmt_ms(max(values)):>9} "
            f"{fmt_ms(statistics.mean(values)):>9}"
        )
    return "\n".join(out)


def pick_top_requests(traces, by, n):
    scored = []
    for rid, tr in traces.items():
        ev = find_event(tr.frontend, "kv_push_router_first_token", phase="decode")
        if not ev:
            continue
        try:
            if by == "ttft":
                score = float(ev.fields.get("ttft_ms", "nan"))
            elif by == "prefill_wait":
                score = float(ev.fields.get("prefill_wait_ms", "nan"))
            elif by == "kv_transfer":
                score = float(ev.fields.get("kv_transfer_estimated_ms", "nan"))
            elif by == "tokenize":
                tk = find_event(tr.frontend, "tokenize_done")
                score = (
                    float(tk.fields.get("tokenize_ms", "nan")) if tk else float("nan")
                )
            elif by == "prefill_engine":
                ce = find_event(tr.ctx, "engine_first_response")
                score = float(ce.fields.get("engine_ms", "nan")) if ce else float("nan")
            elif by == "decode_engine":
                ge = find_event(tr.gen, "engine_first_response")
                score = float(ge.fields.get("engine_ms", "nan")) if ge else float("nan")
            else:
                score = float("nan")
        except ValueError:
            score = float("nan")
        if score == score:
            scored.append((score, rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:n]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="directory containing frontend.log / ctx_worker.log / gen_worker.log",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=3,
        help="how many per-request waterfalls to render",
    )
    p.add_argument(
        "--request-id",
        action="append",
        default=[],
        help="render waterfall for specific request_id (repeatable)",
    )
    p.add_argument("--width", type=int, default=140, help="ASCII width for the bar")
    p.add_argument(
        "--top-by",
        choices=[
            "ttft",
            "tokenize",
            "prefill_wait",
            "prefill_engine",
            "kv_transfer",
            "decode_engine",
        ],
        default="ttft",
        help="metric used to pick the slowest N requests",
    )
    p.add_argument("--no-aggregate", action="store_true")
    args = p.parse_args()

    raw = []
    raw += read_raw_events(os.path.join(args.dir, "frontend.log"), "frontend")
    raw += read_raw_events(os.path.join(args.dir, "ctx_worker.log"), "ctx")
    raw += read_raw_events(os.path.join(args.dir, "gen_worker.log"), "gen")
    traces = build_traces(raw)
    print(
        f"Parsed {len(raw)} trace events into {len(traces)} unique request_ids "
        f"(dir={args.dir})"
    )
    print()

    if not args.no_aggregate:
        print(render_aggregates(traces))
        print()

    rids = list(args.request_id)
    if args.requests and not rids:
        rids = pick_top_requests(traces, args.top_by, args.requests)
        print(f"Top {len(rids)} requests by {args.top_by}:")
        for rid in rids:
            print(f"  {rid}")
        print()

    for rid in rids:
        tr = traces.get(rid)
        if not tr:
            print(f"!! request_id not found: {rid}")
            continue
        print(render_waterfall(tr, width=args.width))
        print()


if __name__ == "__main__":
    main()
