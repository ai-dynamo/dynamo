#!/usr/bin/env python3
"""Phase 8 (concurrency=48) TTFT breakdown analyzer.

Filters parsed traces to requests whose HTTP entry epoch_ms falls inside the
Phase 8 measurement window from benchmark.log, then prints:
  - Aggregate TTFT percentiles
  - p50 / p95 / p99 representative request waterfalls
  - Stage-by-stage breakdown for each percentile (markdown table)
  - Cross-cut + Dynamo overhead summary
"""

import os
import statistics
import sys

# Reuse the parser from the sibling analyze_traces.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analyze_traces import (  # noqa: E402
    build_traces,
    find_event,
    percentile,
    read_raw_events,
)

DIR = os.path.dirname(os.path.abspath(__file__))

# Phase 8 (concurrency=48), measurement window:
#   17:13 = ramp start of Phase 0 in PDT (UTC-7)
#   Phase 8 starts at 18:22:49 PDT and stopped at 18:41:53 PDT
#   Measurement begins +107 s after Phase 8 ramp start.
FROM_EPOCH_MS = 1779326676000  # 2026-05-21 01:24:36 UTC (18:24:36 PDT)
TO_EPOCH_MS = 1779327713000  # 2026-05-21 01:41:53 UTC (18:41:53 PDT)


def get_event_field(tr, source, stage, field_name, phase=None):
    ev = find_event(getattr(tr, source), stage, phase)
    if not ev:
        return None
    v = ev.fields.get(field_name)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def get_event_epoch(tr, source, stage, phase=None):
    ev = find_event(getattr(tr, source), stage, phase)
    if not ev or ev.epoch_ms is None:
        return None
    return ev.epoch_ms


def diff(a, b):
    if a is None or b is None:
        return None
    return b - a


def fmt_ms(ms):
    if ms is None:
        return "—"
    if ms < 1:
        return "<1 ms"
    return f"{int(round(ms))} ms"


def pct(part, total):
    if part is None or total is None or total == 0:
        return "—"
    return f"{100.0 * part / total:.1f}%"


def stage_table(tr):
    """Return list of (label, duration_ms, kind) rows + TTFT total."""
    f = "frontend"
    c = "ctx"
    g = "gen"

    e_http = get_event_epoch(tr, f, "http_handler_entered")
    e_pre = get_event_epoch(tr, f, "preprocessor_entered")
    e_tok = get_event_epoch(tr, f, "tokenize_done")
    e_router = get_event_epoch(tr, f, "prefill_router_entry")
    e_rpc_send = get_event_epoch(tr, f, "prefill_rpc_send")
    e_rpc_first = get_event_epoch(tr, f, "prefill_rpc_first_response")
    e_prefill_ret = get_event_epoch(tr, f, "prefill_returned_to_router")
    e_decode_dispatch = get_event_epoch(tr, f, "decode_dispatch_to_next_operator")
    e_first_chunk = get_event_epoch(tr, f, "first_chunk_detokenized")
    e_sse = get_event_epoch(tr, f, "first_sse_chunk_emitted")

    e_ctx_rpc = get_event_epoch(tr, c, "worker_rpc_entry")
    e_ctx_handler = get_event_epoch(tr, c, "handler_received")
    e_ctx_pre_engine = get_event_epoch(tr, c, "pre_engine_generate_async")
    e_ctx_first_resp = get_event_epoch(tr, c, "engine_first_response")

    e_gen_rpc = get_event_epoch(tr, g, "worker_rpc_entry")
    e_gen_pre_engine = get_event_epoch(tr, g, "pre_engine_generate_async")
    e_gen_first_resp = get_event_epoch(tr, g, "engine_first_response")

    tokenize_ms = get_event_field(tr, f, "tokenize_done", "tokenize_ms")
    ctx_engine_ms = get_event_field(tr, c, "engine_first_response", "engine_ms")
    gen_engine_ms = get_event_field(tr, g, "engine_first_response", "engine_ms")
    num_tokens = get_event_field(tr, f, "tokenize_done", "num_tokens")

    rows = []
    rows.append(("HTTP entry → preprocessor", diff(e_http, e_pre), "Dynamo"))
    rows.append(
        (
            "Tokenize",
            tokenize_ms if tokenize_ms is not None else diff(e_pre, e_tok),
            "Inherent",
        )
    )
    rows.append(
        ("`tokenize_done` → `prefill_router_entry`", diff(e_tok, e_router), "Dynamo")
    )
    rows.append(
        (
            "Prefill router (resolve + select + dispatch + rpc_send)",
            diff(e_router, e_rpc_send),
            "Dynamo",
        )
    )
    rows.append(("RPC out (frontend → ctx)*", diff(e_rpc_send, e_ctx_rpc), "Dynamo"))
    rows.append(
        ("ctx Python handler dispatch", diff(e_ctx_rpc, e_ctx_handler), "Dynamo")
    )
    rows.append(
        ("ctx prep + engine submit", diff(e_ctx_handler, e_ctx_pre_engine), "Dynamo")
    )
    rows.append(
        (
            "ctx prefill engine",
            ctx_engine_ms
            if ctx_engine_ms is not None
            else diff(e_ctx_pre_engine, e_ctx_first_resp),
            "Inherent",
        )
    )
    rows.append(
        ("Return path (ctx → frontend)*", diff(e_ctx_first_resp, e_rpc_first), "Dynamo")
    )
    rows.append(
        ("Prefill → decode handoff", diff(e_prefill_ret, e_decode_dispatch), "Dynamo")
    )
    rows.append(
        ("RPC out (frontend → gen)*", diff(e_decode_dispatch, e_gen_rpc), "Dynamo")
    )
    rows.append(
        ("gen Python handler + prep", diff(e_gen_rpc, e_gen_pre_engine), "Dynamo")
    )
    rows.append(
        ("gen engine submit", diff(e_gen_pre_engine, e_gen_pre_engine), "Dynamo")
    )  # ~0
    rows.append(
        (
            "gen engine (KV xfer + 1st decode forward)",
            gen_engine_ms
            if gen_engine_ms is not None
            else diff(e_gen_pre_engine, e_gen_first_resp),
            "Inherent",
        )
    )
    rows.append(("Detokenize + SSE emit", diff(e_first_chunk, e_sse), "Dynamo"))

    ttft = diff(e_http, e_sse)
    return rows, ttft, num_tokens, tokenize_ms, ctx_engine_ms, gen_engine_ms


def render_breakdown_md(label, rid, tr):
    rows, ttft, num_tokens, tok_ms, ctx_eng, gen_eng = stage_table(tr)
    out = []
    nt = int(num_tokens) if num_tokens is not None else None
    nt_str = f"{nt:,}" if nt is not None else "?"
    out.append(f"## {label} — `{rid}` · {nt_str} tokens · TTFT {fmt_ms(ttft)}")
    out.append("")
    out.append("| # | Phase | Duration | % TTFT | Kind |")
    out.append("|---|---|---|---|---|")
    dynamo_total = 0.0
    for i, (name, dur, kind) in enumerate(rows, 1):
        out.append(f"| {i} | {name} | {fmt_ms(dur)} | {pct(dur, ttft)} | {kind} |")
        if kind == "Dynamo" and dur is not None:
            dynamo_total += dur
    out.append(
        f"| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **{fmt_ms(dynamo_total)}** | **{pct(dynamo_total, ttft)}** | |"
    )
    out.append("")
    return "\n".join(out), {
        "ttft": ttft,
        "num_tokens": num_tokens,
        "tok_ms": tok_ms,
        "ctx_eng": ctx_eng,
        "gen_eng": gen_eng,
        "dynamo": dynamo_total,
    }


def main():
    raw = []
    raw += read_raw_events(os.path.join(DIR, "frontend.log"), "frontend")
    raw += read_raw_events(os.path.join(DIR, "ctx_worker.log"), "ctx")
    raw += read_raw_events(os.path.join(DIR, "gen_worker.log"), "gen")
    traces = build_traces(raw)
    print(f"Parsed {len(raw)} events into {len(traces)} unique traces", file=sys.stderr)

    # Filter to Phase 8 measurement window using HTTP entry epoch_ms.
    keep = {}
    for rid, tr in traces.items():
        e_http = get_event_epoch(tr, "frontend", "http_handler_entered")
        if e_http is None:
            continue
        if FROM_EPOCH_MS <= e_http <= TO_EPOCH_MS:
            keep[rid] = tr
    print(f"After Phase 8 window filter: {len(keep)} requests", file=sys.stderr)

    # Compute TTFT for every kept request that has a full path.
    ttfts = []
    for rid, tr in keep.items():
        e_http = get_event_epoch(tr, "frontend", "http_handler_entered")
        e_sse = get_event_epoch(tr, "frontend", "first_sse_chunk_emitted")
        if e_http is None or e_sse is None:
            continue
        ttfts.append((e_sse - e_http, rid))

    ttfts.sort()
    n = len(ttfts)
    print(f"Requests with complete TTFT span: {n}", file=sys.stderr)

    def pick(p):
        idx = int(round((n - 1) * p / 100))
        return ttfts[idx]

    ttft_values = [t for t, _ in ttfts]
    p50_v = percentile(ttft_values, 50)
    p90_v = percentile(ttft_values, 90)
    p95_v = percentile(ttft_values, 95)
    p99_v = percentile(ttft_values, 99)
    max_v = ttft_values[-1] if ttft_values else 0

    p50_t, p50_rid = pick(50)
    p95_t, p95_rid = pick(95)
    p99_t, p99_rid = pick(99)

    print("\n=== Aggregate TTFT (Phase 8, concurrency=48) ===", file=sys.stderr)
    print(
        f"  n={n}  p50={p50_v:.0f}  p90={p90_v:.0f}  p95={p95_v:.0f}  p99={p99_v:.0f}  max={max_v:.0f}",
        file=sys.stderr,
    )
    print(f"  picked p50: {p50_rid} (TTFT {p50_t:.0f} ms)", file=sys.stderr)
    print(f"  picked p95: {p95_rid} (TTFT {p95_t:.0f} ms)", file=sys.stderr)
    print(f"  picked p99: {p99_rid} (TTFT {p99_t:.0f} ms)", file=sys.stderr)

    out = []
    out.append("# TTFT breakdown — p50 / p95 / p99")
    out.append("")
    out.append(
        "Benchmark: `openai/gpt-oss-120b` · disagg (1 ctx + 1 gen) · KV router · 48 users · Phase 8 (measurement window)"
    )
    out.append(
        "Source logs: `1ctx1gen_20260520/frontend.log`, `ctx_worker.log`, `gen_worker.log`"
    )
    out.append("Phase 8 window: 18:24:36 → 18:41:53 PDT (measurement only)")
    out.append(f"Total requests parsed (full TTFT span, in window): {n:,}")
    out.append(
        f"Aggregate p50 / p90 / p95 / p99 / max TTFT (ms): {p50_v:.0f} / {p90_v:.0f} / {p95_v:.0f} / {p99_v:.0f} / {max_v:.0f}"
    )
    out.append("")
    out.append("**Inherent costs** (any inference server / disagg setup pays these):")
    out.append("- Tokenize")
    out.append("- ctx `engine_ms` (prefill forward pass)")
    out.append(
        "- gen `engine_ms` (bundles NIXL KV transfer wait + first decode forward — not separable without TRT-LLM internal hooks)"
    )
    out.append("")
    out.append(
        "**Pure Dynamo overhead** = TTFT − (tokenize + ctx prefill engine + gen engine)."
    )
    out.append(
        "This bucket contains: HTTP entry, preprocessor handoffs, KV router, RPC plane, Python handler dispatch, return path, decode dispatch, detokenize, SSE emit."
    )
    out.append("")
    out.append(
        "\\* Rows marked with `*` are cross-process — cross-host clock skew may shift a few ms in either direction."
    )
    out.append("")
    out.append("---")
    out.append("")

    md_p50, m50 = render_breakdown_md("p50", p50_rid, keep[p50_rid])
    md_p95, m95 = render_breakdown_md("p95", p95_rid, keep[p95_rid])
    md_p99, m99 = render_breakdown_md("p99", p99_rid, keep[p99_rid])

    out.append(md_p50)
    out.append("---")
    out.append("")
    out.append(md_p95)
    out.append("---")
    out.append("")
    out.append(md_p99)
    out.append("---")
    out.append("")

    out.append("## Cross-cut")
    out.append("")
    out.append("| | p50 | p95 | p99 |")
    out.append("|---|---|---|---|")

    def fmtcell(v):
        if v is None:
            return "—"
        return f"{int(v):,}" if v >= 100 else f"{v:.0f}"

    def fmt_dur(d, t):
        if d is None or t is None or t == 0:
            return "—"
        return f"{int(round(d))} ms ({100.0*d/t:.1f}%)"

    out.append(
        f"| Input tokens | {fmtcell(m50['num_tokens'])} | {fmtcell(m95['num_tokens'])} | {fmtcell(m99['num_tokens'])} |"
    )
    out.append(
        f"| TTFT | {fmt_ms(m50['ttft'])} | {fmt_ms(m95['ttft'])} | {fmt_ms(m99['ttft'])} |"
    )
    out.append(
        f"| ctx prefill engine | {fmt_dur(m50['ctx_eng'], m50['ttft'])} | "
        f"{fmt_dur(m95['ctx_eng'], m95['ttft'])} | {fmt_dur(m99['ctx_eng'], m99['ttft'])} |"
    )

    def tok_plus_gen(m):
        v = 0.0
        if m["tok_ms"] is not None:
            v += m["tok_ms"]
        if m["gen_eng"] is not None:
            v += m["gen_eng"]
        return v

    out.append(
        f"| Tokenize + gen engine | {fmt_dur(tok_plus_gen(m50), m50['ttft'])} | "
        f"{fmt_dur(tok_plus_gen(m95), m95['ttft'])} | {fmt_dur(tok_plus_gen(m99), m99['ttft'])} |"
    )
    out.append(
        f"| **Pure Dynamo overhead** | **{fmt_dur(m50['dynamo'], m50['ttft'])}** | "
        f"**{fmt_dur(m95['dynamo'], m95['ttft'])}** | **{fmt_dur(m99['dynamo'], m99['ttft'])}** |"
    )
    out.append("")

    # Distribution of the tokenize_done → prefill_router_entry gap.
    gaps = []
    for tr in keep.values():
        a = get_event_epoch(tr, "frontend", "tokenize_done")
        b = get_event_epoch(tr, "frontend", "prefill_router_entry")
        if a is not None and b is not None and b >= a:
            gaps.append(b - a)
    if gaps:
        gaps.sort()
        out.append("## Tokenize → prefill_router gap distribution")
        out.append("")
        out.append("| metric | gap |")
        out.append("|---|---|")
        for label, p in (
            ("p50", 50),
            ("p75", 75),
            ("p90", 90),
            ("p95", 95),
            ("p99", 99),
            ("p99.9", 99.9),
        ):
            out.append(f"| {label} | {int(round(percentile(gaps, p)))} ms |")
        out.append(f"| max | {gaps[-1]} ms |")
        out.append(f"| mean | {statistics.mean(gaps):.2f} ms |")
        out.append(f"| n | {len(gaps):,} |")
        out.append("")

    # Aggregate engine ms percentile spread for context.
    ctx_engines = [
        v
        for v in (
            get_event_field(tr, "ctx", "engine_first_response", "engine_ms")
            for tr in keep.values()
        )
        if v is not None
    ]
    gen_engines = [
        v
        for v in (
            get_event_field(tr, "gen", "engine_first_response", "engine_ms")
            for tr in keep.values()
        )
        if v is not None
    ]
    tokenizes = [
        v
        for v in (
            get_event_field(tr, "frontend", "tokenize_done", "tokenize_ms")
            for tr in keep.values()
        )
        if v is not None
    ]
    if ctx_engines:
        out.append("## Aggregate engine_ms across all phase-8 requests")
        out.append("")
        out.append("| stage | p50 | p90 | p95 | p99 | max |")
        out.append("|---|---|---|---|---|---|")
        for name, vals in (
            ("ctx engine", ctx_engines),
            ("gen engine", gen_engines),
            ("tokenize", tokenizes),
        ):
            if not vals:
                continue
            out.append(
                f"| {name} | {percentile(vals,50):.0f} ms | {percentile(vals,90):.0f} ms | "
                f"{percentile(vals,95):.0f} ms | {percentile(vals,99):.0f} ms | {max(vals):.0f} ms |"
            )
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
