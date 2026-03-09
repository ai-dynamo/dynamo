#!/usr/bin/env python3
"""Regenerate cumulative-reads-writes chart using Dynamo Dark theme."""

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import plotly.graph_objects as go

# Add flash-indexer tools to path for the Dynamo theme
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs/blogs/flash-indexer/tools"))
from plotly_dynamo import dynamo_template, load_tokens

CAPTURE = Path.home() / "memory/claude-code-cache-blog/captures/exp_b_subagent.jsonl"
OUTPUT = Path(__file__).resolve().parent / "cumulative-reads-writes.png"


@dataclass
class Call:
    index: int
    t0: float
    cache_read: int
    cache_write: int
    input_tokens: int


def load_calls(path: Path) -> list[Call]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    reqs, resps = {}, {}
    for r in records:
        ci = r["call_index"]
        if r["direction"] == "request":
            reqs[ci] = r
        else:
            resps[ci] = r

    calls = []
    for ci in sorted(set(reqs) & set(resps)):
        req = reqs[ci]
        resp = resps[ci]
        u = resp.get("usage", {})

        if req.get("max_tokens") == 1:
            continue

        calls.append(Call(
            index=ci,
            t0=req.get("timestamp", ci),
            cache_read=u.get("cache_read_input_tokens", u.get("cache_creation_input_tokens", 0)),
            cache_write=u.get("cache_creation_input_tokens", 0),
            input_tokens=u.get("input_tokens", 0),
        ))
    return calls


def fmt_tokens(val: int) -> str:
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return str(val)


def main():
    tokens = load_tokens()
    series_colors = tokens["colors"]["chart_series"]

    calls = load_calls(CAPTURE)
    calls.sort(key=lambda c: c.t0)

    cum_read = []
    cum_write = []
    cum_input = []
    r, w, i = 0, 0, 0
    for c in calls:
        r += c.cache_read
        w += c.cache_write
        i += c.input_tokens
        cum_read.append(r)
        cum_write.append(w)
        cum_input.append(i)

    seq = list(range(len(calls)))
    total_r = cum_read[-1]
    total_w = cum_write[-1]
    ratio = total_r / total_w if total_w > 0 else 0

    # Colors from Dynamo chart_series
    color_reads = series_colors[0]   # #76b900 Dynamo green
    color_writes = series_colors[1]  # #0071c5 CPU Blue
    color_uncached = series_colors[2]  # #fac200 Fluorite

    fig = go.Figure(layout=go.Layout(template=dynamo_template))

    # Cache reads (filled area)
    fig.add_trace(go.Scatter(
        x=seq, y=cum_read,
        mode="lines",
        name="Cache reads (cumulative)",
        line=dict(color=color_reads, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color_reads[1:3], 16)},{int(color_reads[3:5], 16)},{int(color_reads[5:7], 16)},0.25)",
    ))

    # Cache writes (filled area)
    fig.add_trace(go.Scatter(
        x=seq, y=cum_write,
        mode="lines",
        name="Cache writes (cumulative)",
        line=dict(color=color_writes, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color_writes[1:3], 16)},{int(color_writes[3:5], 16)},{int(color_writes[5:7], 16)},0.25)",
    ))

    # Uncached input (dashed)
    fig.add_trace(go.Scatter(
        x=seq, y=cum_input,
        mode="lines",
        name="Uncached input (cumulative)",
        line=dict(color=color_uncached, width=1.5, dash="dash"),
        opacity=0.7,
    ))

    # Annotation box with totals
    fig.add_annotation(
        x=0.97, y=0.95,
        xref="paper", yref="paper",
        text=(
            f"Total reads: {fmt_tokens(total_r)}<br>"
            f"Total writes: {fmt_tokens(total_w)}<br>"
            f"Read/Write ratio: {ratio:.1f}x"
        ),
        showarrow=False,
        font=dict(
            family=tokens["typography"]["font_family_mono"],
            size=11,
            color=tokens["colors"]["text"]["primary"],
        ),
        align="right",
        xanchor="right",
        yanchor="top",
        bordercolor=tokens["colors"]["border"]["subtle"],
        borderwidth=1,
        bgcolor="rgba(26,26,26,0.85)",
        borderpad=6,
    )

    fig.update_layout(
        title=dict(text="CUMULATIVE CACHE READS VS WRITES"),
        xaxis_title="API call sequence",
        yaxis_title="Cumulative tokens",
        yaxis=dict(
            tickvals=[0, 200_000, 400_000, 600_000, 800_000],
            ticktext=["0", "200K", "400K", "600K", "800K"],
        ),
        legend=dict(
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
        ),
        width=1100,
        height=500,
        margin=dict(b=60),
    )

    fig.write_image(str(OUTPUT), scale=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
