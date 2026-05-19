#!/usr/bin/env python3
"""Render an nsys-style Gantt chart of a single request's [TTFT-TRACE] timeline.

Three swimlanes (Frontend, Prefill Worker, Decode Worker). Each consecutive
pair of stage events on a swimlane becomes a colored bar. Hover shows the
from→to stage names plus duration. Output is a standalone HTML file.

Usage:
  plot_trace.py [--dir DIR] [--request-id RID] [--top-by METRIC]
                [--output PATH]

If --request-id is omitted, picks the slowest request by --top-by (default:
ttft).
"""

import argparse
import os
import sys

# Reuse the parser from analyze_traces.py to avoid duplication.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotly.graph_objects as go  # noqa: E402
from analyze_traces import (  # noqa: E402
    FRONTEND_ORDER,
    WORKER_ORDER,
    build_traces,
    pick_top_requests,
    read_raw_events,
)

# Stage→category mapping for color coding.
CATEGORY = {
    # Frontend
    "http_handler_entered": "http",
    "preprocessor_entered": "preprocess",
    "tokenize_done": "preprocess",
    "prefill_router_entry": "router",
    "prefill_worker_resolved": "router",
    "prefill_dispatch": "router",
    "prefill_rpc_send": "rpc",
    "kv_push_router_worker_selected": "router",
    "kv_push_router_dispatch": "router",
    "kv_push_router_first_response": "rpc",
    "kv_push_router_first_token": "rpc",
    "prefill_rpc_first_response": "rpc",
    "prefill_returned_to_router": "router",
    "decode_dispatch_to_next_operator": "router",
    "first_chunk_detokenized": "preprocess",
    "first_sse_chunk_emitted": "http",
    # Worker
    "worker_rpc_entry": "rpc",
    "handler_received": "handler",
    "prep_breakdown": "handler",
    "pre_engine_generate_async": "handler",
    "post_engine_generate_async_call": "handler",
    "engine_first_response": "engine",
    "first_chunk_yielded": "engine",
}

CATEGORY_COLOR = {
    "http": "#5B8FF9",
    "preprocess": "#5AD8A6",
    "router": "#F6BD16",
    "rpc": "#E8684A",
    "handler": "#9270CA",
    "engine": "#E76C5E",
    "other": "#9AA0A6",
}

LANE_LABEL = {
    "frontend": "Frontend (Rust)",
    "rpc": "RPC (cross-process)*",
    "ctx": "Prefill Worker (ctx)",
    "gen": "Decode Worker (gen)",
}

LANE_ORDER = ["frontend", "rpc", "ctx", "gen"]

# Phase definitions matching ttft_breakdown.md. Each entry:
#   (phase_num, label, display_lane,
#    start_source, start_stage, start_phase,
#    end_source,   end_stage,   end_phase,
#    category)
# `display_lane` is the y-row where the bar is drawn. For cross-process hops
# (5, 9, 11) we draw on the "rpc" lane between the two endpoints' lanes.
# Phases 13 + 14 are merged because `post_engine_generate_async_call` has no
# epoch_ms; submit_ms is surfaced in hover instead.
PHASE_DEFS = [
    (
        1,
        "1. HTTP entry → preprocessor",
        "frontend",
        "frontend",
        "http_handler_entered",
        None,
        "frontend",
        "preprocessor_entered",
        None,
        "http",
    ),
    (
        2,
        "2. Tokenize",
        "frontend",
        "frontend",
        "preprocessor_entered",
        None,
        "frontend",
        "tokenize_done",
        None,
        "preprocess",
    ),
    (
        3,
        "3. tokenize_done → prefill_router",
        "frontend",
        "frontend",
        "tokenize_done",
        None,
        "frontend",
        "prefill_router_entry",
        None,
        "router",
    ),
    (
        4,
        "4. Prefill router (resolve+dispatch+rpc_send)",
        "frontend",
        "frontend",
        "prefill_router_entry",
        None,
        "frontend",
        "prefill_rpc_send",
        None,
        "router",
    ),
    (
        5,
        "5. RPC out (frontend → ctx)*",
        "rpc",
        "frontend",
        "prefill_rpc_send",
        None,
        "ctx",
        "worker_rpc_entry",
        None,
        "rpc",
    ),
    (
        6,
        "6. ctx Python handler dispatch",
        "ctx",
        "ctx",
        "worker_rpc_entry",
        None,
        "ctx",
        "handler_received",
        None,
        "handler",
    ),
    (
        7,
        "7. ctx prep + engine submit",
        "ctx",
        "ctx",
        "handler_received",
        None,
        "ctx",
        "pre_engine_generate_async",
        None,
        "handler",
    ),
    (
        8,
        "8. ctx prefill engine",
        "ctx",
        "ctx",
        "pre_engine_generate_async",
        None,
        "ctx",
        "engine_first_response",
        None,
        "engine",
    ),
    (
        9,
        "9. Return path (ctx → frontend)*",
        "rpc",
        "ctx",
        "engine_first_response",
        None,
        "frontend",
        "prefill_rpc_first_response",
        None,
        "rpc",
    ),
    (
        10,
        "10. Prefill → decode handoff",
        "frontend",
        "frontend",
        "prefill_rpc_first_response",
        None,
        "frontend",
        "decode_dispatch_to_next_operator",
        None,
        "router",
    ),
    (
        11,
        "11. RPC out (frontend → gen)*",
        "rpc",
        "frontend",
        "decode_dispatch_to_next_operator",
        None,
        "gen",
        "worker_rpc_entry",
        None,
        "rpc",
    ),
    (
        12,
        "12. gen Python handler + prep",
        "gen",
        "gen",
        "worker_rpc_entry",
        None,
        "gen",
        "pre_engine_generate_async",
        None,
        "handler",
    ),
    (
        14,
        "13–14. gen engine (submit + KV xfer + 1st decode)",
        "gen",
        "gen",
        "pre_engine_generate_async",
        None,
        "gen",
        "engine_first_response",
        None,
        "engine",
    ),
    (
        15,
        "15. Detokenize + SSE emit",
        "frontend",
        "frontend",
        "first_chunk_detokenized",
        None,
        "frontend",
        "first_sse_chunk_emitted",
        None,
        "preprocess",
    ),
]


def _find_ev(trace, source, stage, phase=None):
    """Return the first event matching (source, stage, optional phase=) with
    a non-null epoch_ms. Phase filter only applies to kv_push_router_* stages
    (those that legitimately repeat); other stages ignore the filter."""
    for ev in getattr(trace, source):
        if ev.stage != stage:
            continue
        if phase is not None and ev.fields.get("phase", "").lower() != phase:
            continue
        if ev.epoch_ms is None:
            continue
        return ev
    return None


def ordered_events(trace, source):
    """Return events with epoch_ms for a source, ordered (epoch_ms, stage_idx).
    For frontend, also distinguishes kv_push_router_* by phase so the two
    occurrences sort distinctly."""
    order_list = FRONTEND_ORDER if source == "frontend" else WORKER_ORDER
    evs = []
    for ev in getattr(trace, source):
        if ev.epoch_ms is None:
            continue
        # For frontend kv_push_router_* events with phase=decode, prefer later
        # FRONTEND_ORDER index (the decode-phase positions).
        try:
            idx = order_list.index(ev.stage)
        except ValueError:
            idx = 999
        if (
            source == "frontend"
            and ev.stage.startswith("kv_push_router")
            and ev.fields.get("phase", "").lower() == "decode"
        ):
            # Find the *last* occurrence in FRONTEND_ORDER (the decode-phase slot).
            for i in range(len(order_list) - 1, -1, -1):
                if order_list[i] == ev.stage:
                    idx = i
                    break
        # decode_dispatch_to_next_operator anchors decode phase; same for prefill_rpc_first_response
        evs.append((ev.epoch_ms, idx, ev))
    evs.sort(key=lambda t: (t[0], t[1]))
    return [e[2] for e in evs]


def build_bars(trace):
    """Build one bar per PHASE_DEFS entry (15-phase breakdown). Cross-process
    bars (5/9/11) land on the dedicated 'rpc' swimlane between their endpoint
    lanes. Returns (bars, point markers, total span ms)."""
    bars = []
    points = []
    t0 = None
    for src in ("frontend", "ctx", "gen"):
        for ev in getattr(trace, src):
            if ev.epoch_ms is not None:
                t0 = ev.epoch_ms if t0 is None else min(t0, ev.epoch_ms)
    if t0 is None:
        return bars, points, 0

    for (
        num,
        label,
        lane_key,
        s_src,
        s_stage,
        s_phase,
        e_src,
        e_stage,
        e_phase,
        cat,
    ) in PHASE_DEFS:
        a = _find_ev(trace, s_src, s_stage, s_phase)
        b = _find_ev(trace, e_src, e_stage, e_phase)
        if a is None or b is None:
            continue
        duration = b.epoch_ms - a.epoch_ms
        extras = []
        for ev, prefix in ((a, "start"), (b, "end")):
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
                "phase",
                "worker_id",
            ):
                if k in ev.fields:
                    extras.append(f"{prefix}.{k}={ev.fields[k]}")
        bars.append(
            dict(
                phase_num=num,
                label=label,
                lane=LANE_LABEL[lane_key],
                x_start=(a.epoch_ms - t0),
                x_end=(b.epoch_ms - t0),
                color=CATEGORY_COLOR[cat],
                stage_from=f"{s_src}:{s_stage}" + (f"[{s_phase}]" if s_phase else ""),
                stage_to=f"{e_src}:{e_stage}" + (f"[{e_phase}]" if e_phase else ""),
                duration_ms=duration,
                fields=" ".join(extras),
                category=cat,
            )
        )

    for src in ("frontend", "ctx", "gen"):
        evs = ordered_events(trace, src)
        for ev in evs:
            points.append(
                dict(
                    lane=LANE_LABEL[src],
                    x=(ev.epoch_ms - t0),
                    stage=ev.stage,
                    phase=ev.fields.get("phase", ""),
                    fields=" ".join(
                        f"{k}={v}"
                        for k, v in ev.fields.items()
                        if k
                        not in (
                            "stage",
                            "request_id",
                            "method",
                            "uri",
                            "version",
                            "model",
                        )
                    ),
                )
            )
    span_ms = max(
        (p["x"] for p in points),
        default=max((b["x_end"] for b in bars), default=1),
    )
    return bars, points, span_ms


def render_html(trace, out_path, title_extra=""):
    bars, points, span = build_bars(trace)
    if not bars:
        print(f"!! no epoch_ms events for {trace.request_id}")
        return

    lane_order = [LANE_LABEL[s] for s in LANE_ORDER]

    fig = go.Figure()

    # One trace per category so the legend collapses related stages. Bars are
    # drawn sorted by phase number so legend ordering is stable.
    bars_sorted = sorted(bars, key=lambda b: b["phase_num"])
    by_cat = {}
    for b in bars_sorted:
        by_cat.setdefault(b["category"], []).append(b)

    for cat, items in by_cat.items():
        widths, ys, hovers, colors, texts = [], [], [], [], []
        for b in items:
            raw_w = b["x_end"] - b["x_start"]
            # Clock skew on cross-process bars can produce a tiny negative
            # width; clip to a sliver so the bar is still visible.
            widths.append(max(raw_w, span * 0.0008))
            ys.append(b["lane"])
            hovers.append(
                f"<b>{b['label']}</b><br>"
                f"{b['stage_from']} → {b['stage_to']}<br>"
                f"duration: {b['duration_ms']} ms<br>"
                f"window: [{b['x_start']}, {b['x_end']}] ms<br>"
                f"{b['fields']}"
            )
            texts.append(f"{b['label']} ({b['duration_ms']} ms)")
            colors.append(b["color"])
        fig.add_trace(
            go.Bar(
                x=widths,
                y=ys,
                base=[b["x_start"] for b in items],
                orientation="h",
                marker=dict(
                    color=colors, line=dict(color="rgba(0,0,0,0.4)", width=0.3)
                ),
                name=cat,
                text=texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=10, color="black"),
                cliponaxis=False,
                hovertext=hovers,
                hoverinfo="text",
                showlegend=True,
            )
        )

    # Event markers (small triangles) for each stage transition.
    fig.add_trace(
        go.Scatter(
            x=[p["x"] for p in points],
            y=[p["lane"] for p in points],
            mode="markers",
            marker=dict(
                symbol="line-ns", size=14, color="rgba(0,0,0,0.6)", line=dict(width=1.2)
            ),
            text=[
                f"{p['stage']}" + (f" [{p['phase']}]" if p["phase"] else "")
                for p in points
            ],
            hovertext=[
                f"<b>{p['stage']}</b>"
                + (f" [phase={p['phase']}]" if p["phase"] else "")
                + f"<br>+{p['x']} ms<br>{p['fields']}"
                for p in points
            ],
            hoverinfo="text",
            name="events",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"TTFT trace — request {trace.request_id}{title_extra}",
        barmode="overlay",
        bargap=0.4,
        xaxis=dict(title="time (ms, relative to t0)", showgrid=True, zeroline=True),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(lane_order)),
            automargin=True,
        ),
        height=520,
        margin=dict(l=180, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")

    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(
        f"wrote {out_path}  (window={span} ms, bars={len(bars)}, events={len(points)})"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--request-id", default=None)
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
    )
    p.add_argument(
        "--output",
        default=None,
        help="output HTML path (default: trace_<rid>.html in --dir)",
    )
    args = p.parse_args()

    raw = []
    raw += read_raw_events(os.path.join(args.dir, "frontend.log"), "frontend")
    raw += read_raw_events(os.path.join(args.dir, "ctx_worker.log"), "ctx")
    raw += read_raw_events(os.path.join(args.dir, "gen_worker.log"), "gen")
    traces = build_traces(raw)
    print(f"loaded {len(traces)} requests from {args.dir}")

    rid = args.request_id
    title_extra = ""
    if not rid:
        picks = pick_top_requests(traces, args.top_by, 1)
        if not picks:
            print("no candidate requests found")
            return
        rid = picks[0]
        title_extra = f" (slowest by {args.top_by})"

    tr = traces.get(rid)
    if not tr:
        print(f"request_id not found: {rid}")
        return

    out = args.output or os.path.join(args.dir, f"trace_{rid[:8]}.html")
    render_html(tr, out, title_extra=title_extra)


if __name__ == "__main__":
    main()
