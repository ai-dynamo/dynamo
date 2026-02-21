"""Generate the KV event density heatmap (Figure 1) for the Flash Indexer blog post.

Renders a diverging heatmap showing Store vs Remove event density across
16 workers over time. Uses real Mooncake FAST'25 trace data when available,
otherwise falls back to synthetic Gamma/Zipf event patterns.

Prerequisites:
    pip3 install plotly kaleido numpy pyyaml

Usage:
    python3 gen_heatmap.py                    # synthetic data
    python3 gen_heatmap.py --real-data PATH   # real Mooncake trace
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import dynamo_template, load_tokens  # noqa: E402

TOKENS = load_tokens()
C = TOKENS["colors"]
T = TOKENS["typography"]

BG = C["background"]["primary"]
SURFACE = C["background"]["surface"]
BORDER = C["border"]["subtle"]
TXT = C["text"]["primary"]
TXT2 = C["text"]["secondary"]

STORE = "#76b900"
REMOVE = "#fac200"

SANS = T["font_family"]
MONO = T["font_family_mono"]

OUT = Path(__file__).resolve().parent.parent / "images"


def generate_kv_events(
    num_workers: int = 16,
    duration_ms: float = 100.0,
    rps_per_worker: float = 35.0,
    block_size: int = 16,
    cache_hit_ratio: float = 0.30,
    cache_capacity_blocks: int = 24,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Synthetic KV Store/Remove events.

    Arrivals: Gamma(alpha=1.5) -- bursty.
    Prompt lengths: Zipf(a=1.3) * 384 + 64 -- heavy-tailed.
    Low cache capacity forces frequent eviction sweeps.
    """
    rng = np.random.default_rng(seed)
    gamma_alpha = 1.5

    all_stores: list[np.ndarray] = []
    all_removes: list[np.ndarray] = []

    for _ in range(num_workers):
        mean_iat_ms = 1000.0 / rps_per_worker
        gamma_beta = gamma_alpha / mean_iat_ms

        stores, removes = [], []
        t = rng.exponential(mean_iat_ms / 3)
        occupied = 0

        while t < duration_ms:
            prompt_tokens = int(rng.zipf(1.3) * 384 + 64)
            prompt_tokens = min(prompt_tokens, 8192)
            total_blocks = max(1, prompt_tokens // block_size)
            new_blocks = max(1, int(total_blocks * (1 - cache_hit_ratio)))

            if occupied > cache_capacity_blocks:
                evict_n = occupied - int(cache_capacity_blocks * 0.6)
                evict_n = max(4, evict_n)
                evict_t = t + rng.uniform(0, 0.06)
                removes.extend(evict_t + rng.uniform(0, 0.06, size=evict_n))
                occupied -= evict_n

            store_t = t + rng.uniform(0.12, 0.25)
            stores.extend(store_t + rng.uniform(0, 0.08, size=new_blocks))
            occupied += new_blocks

            t += rng.gamma(gamma_alpha, 1.0 / gamma_beta)

        all_stores.append(np.sort(stores))
        all_removes.append(np.sort(removes))

    return all_stores, all_removes


def save(fig: go.Figure, name: str, w: int = 1200, h: int = 600) -> None:
    """Write figure as both PNG (3x) and SVG."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    svg = OUT / f"{name}.svg"
    fig.write_image(str(png), width=w, height=h, scale=3)
    fig.write_image(str(svg), width=w, height=h)
    print(f"  {png.name}  ({w}x{h})")
    print(f"  {svg.name}  ({w}x{h})")


def _ax(**kw) -> dict:
    """Clean axis defaults with minimal grid."""
    base = {
        "zeroline": False,
        "showgrid": True,
        "gridcolor": BORDER,
        "gridwidth": 0.3,
        "linecolor": BORDER,
        "linewidth": 0.5,
        "tickfont": {"family": SANS, "size": 12, "color": TXT2},
        "title_font": {"family": SANS, "size": 14, "color": TXT2},
    }
    base.update(kw)
    return base


def load_real_events(
    path: Path,
    t_start_s: float = 5.0,
    t_end_s: float = 10.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Load real KV events from JSON, return binned matrices for the time window.

    Returns (num_workers, s_mat, r_mat, bins, bin_width, total_stores, total_removes).
    Times are converted to ms relative to t_start.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else p.open
    with opener(p, "rt") as f:
        data = json.load(f)

    t_start_us = t_start_s * 1e6
    t_end_us = t_end_s * 1e6
    duration_ms = (t_end_s - t_start_s) * 1000

    worker_ids = sorted(int(k) for k in data)
    n = len(worker_ids)

    bw = 10.0
    bins = np.arange(0, duration_ms + bw, bw)
    nb = len(bins) - 1

    s_mat = np.zeros((n, nb))
    r_mat = np.zeros((n, nb))
    total_stores = 0
    total_removes = 0

    for idx, wid in enumerate(worker_ids):
        events = data[str(wid)]
        for ev in events:
            ts_us = ev["timestamp_us"]
            if ts_us < t_start_us or ts_us > t_end_us:
                continue
            ts_ms = (ts_us - t_start_us) / 1000.0
            bin_idx = min(int(ts_ms / bw), nb - 1)
            d = ev["event"]["data"]
            if "stored" in d:
                s_mat[idx, bin_idx] += 1
                total_stores += 1
            elif "removed" in d:
                r_mat[idx, bin_idx] += 1
                total_removes += 1

    return n, s_mat, r_mat, bins, bw, total_stores, total_removes


def make_heatmap(
    stores: list[np.ndarray],
    removes: list[np.ndarray],
    real_data_path: Path | None = None,
) -> None:
    if real_data_path is not None:
        n, s_mat, r_mat, bins, bw, ts_count, tr_count = load_real_events(real_data_path)
        nb = len(bins) - 1
        duration_ms = bins[-1]
        total_events = ts_count + tr_count
        subtitle = f"Mooncake trace (5%) \u00b7 16 Mocker workers \u00b7 2048 GPU blocks/worker \u00b7 {total_events:,} events in 5.0 s"
    else:
        n = len(stores)
        bw = 0.5
        duration_ms = 100.0
        bins = np.arange(0, duration_ms + bw, bw)
        nb = len(bins) - 1
        s_mat = np.zeros((n, nb))
        r_mat = np.zeros((n, nb))
        for i in range(n):
            if len(stores[i]):
                s_mat[i] = np.histogram(stores[i], bins)[0]
            if len(removes[i]):
                r_mat[i] = np.histogram(removes[i], bins)[0]
        ts_count = sum(len(s) for s in stores)
        tr_count = sum(len(r) for r in removes)
        subtitle = f"16 workers \u00b7 TP1 \u00b7 block_size 16 \u00b7 35 RPS/worker \u00b7 30% cache hit \u00b7 {(ts_count + tr_count) // 1000}K events in 100 ms"

    combined = s_mat - r_mat
    total = combined.sum(axis=0)

    disp_pw = np.where(s_mat >= r_mat, s_mat, -r_mat).astype(float)
    disp_pw = np.where((s_mat == 0) & (r_mat == 0), 0.0, disp_pw)

    s_total = s_mat.sum(axis=0)
    r_total = r_mat.sum(axis=0)
    disp_tot = np.where(s_total >= r_total, s_total, -r_total).astype(float)
    disp_tot = np.where((s_total == 0) & (r_total == 0), 0.0, disp_tot)

    pw_zmax = 10.0

    colorscale = [
        [0.0, REMOVE],
        [0.15, "#c89a00"],
        [0.5, BG],
        [0.85, "#5aaa00"],
        [1.0, STORE],
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.05,
    )
    fig.update_layout(template=dynamo_template)

    tick_vals = [-10, -5, 0, 5, 10]
    tick_text = ["\u221210", "\u22125", "0", "5", "10"]

    fig.add_trace(
        go.Heatmap(
            z=disp_pw,
            x0=0,
            dx=bw,
            y0=0,
            dy=1,
            colorscale=colorscale,
            zmid=0,
            zmin=-pw_zmax,
            zmax=pw_zmax,
            colorbar={
                "title": {"text": ""},
                "tickfont": {"family": SANS, "size": 12, "color": TXT2},
                "thickness": 12,
                "len": 0.62,
                "y": 0.62,
                "x": 1.027,
                "tickvals": tick_vals,
                "ticktext": tick_text,
            },
            customdata=combined,
            hovertemplate="W%{y} \u00b7 t=%{x:.0f}ms<br>net %{customdata:.0f}<extra></extra>",
            xgap=0.5,
            ygap=0.5,
        ),
        row=1,
        col=1,
    )

    tot_zmax = 100.0
    tick_vals_tot = [-100, 0, 100]
    tick_text_tot = ["\u2212100", "0", "100"]

    fig.add_trace(
        go.Heatmap(
            z=[disp_tot],
            x0=0,
            dx=bw,
            colorscale=colorscale,
            zmid=0,
            zmin=-tot_zmax,
            zmax=tot_zmax,
            colorbar={
                "title": {"text": ""},
                "tickfont": {"family": SANS, "size": 12, "color": TXT2},
                "thickness": 12,
                "len": 0.16,
                "y": 0.095,
                "x": 1.027,
                "tickvals": tick_vals_tot,
                "ticktext": tick_text_tot,
            },
            customdata=[total],
            hovertemplate="t=%{x:.0f}ms<br>\u03a3 net %{customdata:.0f}<extra></extra>",
            xgap=0.5,
            showscale=True,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={"text": ""},
        margin={"l": 65, "r": 120, "t": 110, "b": 55},
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )

    fig.add_annotation(
        text="KV Events",
        xref="paper",
        yref="paper",
        x=1.015,
        y=0.62,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="KV Events",
        xref="paper",
        yref="paper",
        x=1.015,
        y=0.095,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="Store",
        xref="paper",
        yref="paper",
        x=1.06,
        y=0.935,
        xanchor="center",
        yanchor="bottom",
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="Remove",
        xref="paper",
        yref="paper",
        x=1.06,
        y=0.305,
        xanchor="center",
        yanchor="top",
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="Store",
        xref="paper",
        yref="paper",
        x=1.06,
        y=0.18,
        xanchor="center",
        yanchor="bottom",
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="Remove",
        xref="paper",
        yref="paper",
        x=1.06,
        y=0.005,
        xanchor="center",
        yanchor="top",
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text="KV CACHE EVENT DENSITY",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.16,
        xanchor="left",
        yanchor="bottom",
        font={"family": SANS, "size": 18, "color": TXT},
        showarrow=False,
    )
    fig.add_annotation(
        text=subtitle,
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.09,
        xanchor="left",
        yanchor="bottom",
        font={"family": SANS, "size": 12, "color": TXT2},
        showarrow=False,
    )
    fig.add_annotation(
        text=(
            f"<span style='color:{STORE}'>\u25a0</span> Store (prefill)   "
            f"<span style='color:{REMOVE}'>\u25a0</span> Remove (eviction)"
        ),
        xref="paper",
        yref="paper",
        x=1.0131,
        y=1.09,
        xanchor="right",
        yanchor="bottom",
        font={"family": SANS, "size": 13, "color": TXT2},
        showarrow=False,
    )

    fig.update_xaxes(row=1, col=1, **_ax(range=[0, duration_ms]))
    fig.update_xaxes(row=2, col=1, **_ax(title="Time (ms)", range=[0, duration_ms]))
    fig.update_yaxes(
        row=1,
        col=1,
        **_ax(title="Worker", tickvals=list(range(0, 16, 5)) + [15], showgrid=False),
    )
    fig.update_yaxes(
        row=2,
        col=1,
        **_ax(
            title="\u03a3 Workers",
            showgrid=False,
            tickvals=[],
            showticklabels=False,
            showline=True,
            linecolor=BORDER,
            linewidth=0.5,
        ),
    )

    save(fig, "fig-1-kv-event-density", w=950, h=580)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data",
        type=Path,
        default=None,
        help="Path to kv_events JSON or .json.gz (Mooncake trace). Uses synthetic data if omitted.",
    )
    args = parser.parse_args()

    print("Generating synthetic KV events (Gamma arrivals \u00d7 Zipf bursts)...")
    stores, removes = generate_kv_events()

    ts = sum(len(s) for s in stores)
    tr = sum(len(r) for r in removes)
    print(f"  {ts:,} Store events, {tr:,} Remove events over 100ms\n")

    print("Rendering heatmap...")
    if args.real_data and args.real_data.exists():
        print(f"  (using real event data from {args.real_data})")
        make_heatmap(stores, removes, real_data_path=args.real_data)
    else:
        if args.real_data:
            print(f"  WARNING: {args.real_data} not found, using synthetic data")
        make_heatmap(stores, removes)
    print(f"\nDone \u2192 {OUT}")


if __name__ == "__main__":
    main()
