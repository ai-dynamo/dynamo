# Plotting Craft

Chart-specific rules. These are independent from the diagram rules in [aesthetic.md](aesthetic.md). A chart is a different beast from a diagram, even though both inherit the same palette and typography.

The mandatory render-and-critique loop in [SKILL.md](SKILL.md) walks the chart anti-pattern bank below before claiming done.

## Rule 1: Tick Marks — Sparse and Informative, Always Round

The single most common way to make a chart look amateurish is to let Plotly auto-pick tick locations.

**Rules:**

- Linear axes: ticks at multiples of 1, 2, 5, 10, 25, 50, 100, 250, 500, 1k, 2.5k, 5k, 10k, ... Pick whichever multiplier gives 4-7 ticks across the data range.
- Log axes: ticks at powers of ten only (1, 10, 100, 1k, ...). Optional minor ticks at 2 and 5 within each decade.
- Time axes: snap to clean intervals. 1m / 5m / 15m / 1h / 6h / 1d / 1wk. Never "every 47 seconds".
- **Aim for 4-7 major ticks per axis.** More than 7 is clutter. Fewer than 4 makes the axis hard to read.

**Forbidden:** ticks like 13, 27.5, 191.6, 0.0007. Readers cannot mentally interpolate between non-round numbers. If you see them in the output, the chart is broken.

### Canonical Snippet

The pattern from [`gen_throughput.py`](../../digest/flash-indexer/tools/gen_throughput.py) for log-scale ticks with SI suffix formatting:

```python
fig.update_layout(
    xaxis=dict(
        title="Offered Throughput (block ops/s)",
        type="log",
        range=[math.log10(1e5), math.log10(2e9)],
        tickformat=".2s",   # SI suffix: 1.0K, 1.0M, 1.0G
    ),
    yaxis=dict(
        title="Achieved Throughput (block ops/s)",
        type="log",
        range=[math.log10(1e5), math.log10(5e8)],
        tickformat=".2s",
    ),
)
```

For linear axes that need explicit tick placement (overrides Plotly's auto-picker):

```python
fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=[0, 100, 200, 300, 400, 500],
        ticktext=["0", "100", "200", "300", "400", "500"],
    ),
)
```

For time axes:

```python
fig.update_xaxes(
    tickmode="array",
    tickvals=pd.date_range(start, end, freq="15min"),
    tickformat="%H:%M",
)
```

If the auto-picked ticks happen to be round, you may keep them — but check explicitly. Do not assume.

## Rule 2: Axis Hygiene

- **Units in the axis title.** "Offered Throughput (block ops/s)", not "Offered Throughput". Latency is "Latency (ms)" or "Latency (us)", not "Latency".
- **Title Case for axis labels.** "Simulated Time (ms)", "Queue Depth", "Tok/s/User" — not "simulated time (ms)" or "QUEUE DEPTH". Title Case reads as labeling without shouting. Applies to subplot panel titles too.
- **SI suffixes for magnitudes.** `tickformat=".2s"` produces `1.2K`, `3.4M`, `5.6G`. Never `1.234e+06` or `0.000123`. Never raw `1234567`.
- **Y-axis starts at zero** unless the dynamic range is small (e.g., comparing values between 95% and 100%). If the y-axis is broken (does not start at zero), call attention to it: a broken-axis indicator, an explicit note in the title, or both.
- **One y-axis per chart.** No dual-axis charts. Reading two scales overlaid on the same plot is cognitive overload. If two scales are needed, use two stacked subplots (small multiples) sharing the x-axis.
- **No axis lines that are not necessary** to read the data. The Dynamo theme template already removes them; do not re-add.

## Rule 3: Direct Labeling Over Legends

When you have ≤ 5 series, label each line or bar at its end with the series name (and ideally its terminal value). Legends only when you genuinely have 6+ series.

The corpus pattern from [`gen_throughput.py`](../../digest/flash-indexer/tools/gen_throughput.py) uses `peak_annotations` to place the peak value next to each line at the same x-coordinate, with the series name in a small label below.

```python
peak_annotations.append(
    dict(
        x=label_x,
        y=peak_y,
        text=f"<b>{fmt_si(peak_y)}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        yshift=6,
        font=dict(family=font_mono, size=12, color=series_color),
    )
)
```

Why direct labels beat legends:

- Reader's eye does not have to jump from the chart to the corner and back.
- The label sits next to the visual it identifies — Gestalt proximity.
- Numeric labels do double duty: they identify the series AND give the takeaway value.

## Rule 4: Sorting

Categorical bars sort by **value** (descending) or by **domain meaning** (e.g., backend version order, release order, stage order). Never alphabetical by accident.

When the takeaway is "X wins by 40x", the winner is leftmost (or topmost), the loser is rightmost (or bottommost). Sort by value.

When the takeaway is "performance regressed across versions", sort by version order: v0.8.0, v0.9.0, v1.0.0, v1.1.0. Time order is the meaning.

If you cannot articulate why the bars are in this order, they are in the wrong order.

## Rule 5: Series Ceiling

**Five series maximum on a single chart.** Beyond that, split into small multiples (one chart per category, shared axes) or facet by the most-distinguishing dimension.

If you cannot tell the chart's story without a sixth series, the chart is two charts. Make two.

## Rule 6: Annotate the Punchline

Every chart has one number, one moment, one inflection that is the point. **Mark it directly.**

- Labeled marker on the peak.
- Horizontal threshold line at the SLA target.
- Callout arrow to the regression point.
- "40x" annotation between two bars, with a curly brace bracket.

If the takeaway requires reading the caption to find on the chart, mark it on the chart instead. The reader should see the takeaway *before* they read the caption.

The corpus example: `gen_throughput.py` adds a `40x` bracket annotation between the peak of the winner and the peak of the loser. That annotation IS the takeaway, rendered visually.

### Leader Lines for Cluster Labels

When multiple series have data points at the same x value (e.g., concurrency sweeps across hardware/DynoSim/AIC), label *the cluster*, not each point. A floating label like `c=64` near three nearly-coincident markers reads as "this group is c=64" only with a **leader line** connecting the label to the cluster centroid:

```python
fig.add_annotation(
    x=cluster_centroid_x, y=cluster_centroid_y,
    text=f"c={c}",
    showarrow=True, arrowhead=0, arrowwidth=0.8, arrowcolor=text_muted,
    ax=0, ay=-40,  # label sits 40px above the centroid
    xanchor="center", yanchor="bottom",
    font=dict(family=mono, size=14, color=text_secondary),
)
```

Compute the centroid as the mean of the series' x and y at that group. The leader line is a plain thin line (`arrowhead=0`) so it reads as labeling, not as data.

### Open-Space Punch Lines

When a chart has visible empty space in the plot area (e.g., a Pareto curve hugs the bottom-left, leaving the top-right empty), use that space for **one punch-line annotation** in muted secondary text — a single declarative sentence that telegraphs the takeaway. Place it in data coords so it stays anchored if the figure resizes:

```python
fig.add_annotation(
    x=72, y=1080, xref="x", yref="y",
    text="What GPUs sample in days, DynoSim sweeps in minutes.",
    showarrow=False,
    font=dict(family=sans, size=18, color=text_muted, weight=400),
)
```

Rules: one punch line per figure, never two. Set in `text.muted` so it doesn't fight the title. Place it in data coords (not paper coords) so it visually belongs to the chart, not the figure frame.

### Bracket Annotations for Deltas

When the takeaway is "metric A minus metric B equals improvement X", draw a **bracket spanning the two values** with the delta label centered on the bracket. A bracket is unambiguously a comparison; a floating "saved 140 ms" label is not.

**Horizontal bracket** (delta between two chips along the x-axis): the two-chip + green-bracket pattern in `gen_fig_5_decision_cascade.py` is the reference.

**Vertical bracket** (delta between two series at the same x): draw three connected line segments — top foot, spine, bottom foot — feet pointing toward the data points. Use a single `go.Scatter` trace with four vertices so it renders as one polyline. The label sits to the side of the spine.

```python
# Vertical bracket marking the AIC ↔ DynoSim/HW TTFT gap at c=64.
BRACKET_X_SPINE = 56
BRACKET_X_FOOT  = 64
AIC_Y, HW_Y    = 167.8, 220.4
fig.add_trace(
    go.Scatter(
        x=[BRACKET_X_FOOT, BRACKET_X_SPINE, BRACKET_X_SPINE, BRACKET_X_FOOT],
        y=[AIC_Y,          AIC_Y,           HW_Y,            HW_Y],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.85)", width=2.0),
        showlegend=False, hoverinfo="skip",
    ),
    row=2, col=2,
)
fig.add_annotation(
    x=BRACKET_X_SPINE, y=(AIC_Y + HW_Y) / 2,
    xref="x4", yref="y4",
    xanchor="right", yanchor="middle", xshift=-6,
    text="~50 ms", showarrow=False,
    font=dict(family="Arial, Helvetica, sans-serif",
              size=11, color="#ffffff", weight=400),
)
```

The bracket spine spans the *actual* delta values — readers can verify the gap by eye. The prose callout box (see "Tufte Callout Blocks" below) explains *why* the gap is there; the bracket marks *where* and *how big*.

### Connector Hygiene

- **Arrowheads must be visible.** If a triangle marker representing an arrowhead is placed exactly at a box's edge, half of it sits inside the box and disappears. Offset the marker outward by half its size so the tip kisses the edge and the body sits in clear space.
- **No overlapping markers and vertical lines.** A vertical "decision" line and an "arrival" triangle at the same x will visually merge. Remove one or offset.

### Axis Range Padding for Above-Plot Annotations

When phase tags, callouts, or kicker labels need to sit *above* the data, extend the y-axis range to give them room — don't crowd them against the top edge. Phase tags at `y = box_top + 35` will clip if `yaxis.range` only extends 30 units above the box top.

Rule of thumb: for every annotation type that sits above the data, add ~20 units of y-range padding above its top edge. Same on the bottom for below-data annotations (feedback loops, captions, footnotes).

Worked example from `gen_fig_6_tuning_loop.py`: boxes top out at y=106, phase tags sit at y=141, label texts extend ~10 more units, so `yaxis.range[1] = 165` (160 minimum + 5 buffer).

### Caption-to-Thing Centering

When a caption / label / bracket spans a sub-region of the chart, **center it on the thing it captions**, not on the figure's horizontal midline — *unless* the caption applies to the whole figure.

- Bracket label between two TTFT chips → center on the midpoint of the two chips (the bracket's center).
- Calibration label for a feedback loop that spans boxes 2-5 of a 5-box pipeline → center on the figure midline (it's the figure's caption, not the loop's caption).
- Per-cluster label (`c=64`) → center on the cluster centroid.

The decision rule: if removing this label would also remove information about the thing it captions, center on the thing. If removing it would only remove a high-level summary, center on the figure.

### Z-Order / Layering for Overlay Annotations

When a chart has both data marks (lines, scatter points, shaded regions) and overlay annotations (peak markers, frontier highlights, callout diamonds), the annotations must render **above** the data. Plotly's default trace order is insertion order, so:

- Add the data traces first.
- Add the overlay annotations (peak markers, highlights) last.
- Set `marker.line.color` and `marker.line.width` on overlay markers to a thin outline contrasting with the data (e.g., a white 1px ring on a colored diamond) so the marker is unambiguously a foreground element, not a member of the cluster.

For shapes (annotations vs. data), use `layer="above"` on shapes and `xref="x"` (data coords) so the shape stays anchored to the data when the figure resizes.

### Stacked Subplots vs Single Overlay

A common temptation when comparing two scenarios is to stack them as two subplots sharing the x-axis. **Don't** if the takeaway is the *delta between* the two scenarios.

- Two scenarios that share an x-axis and a y-axis (e.g., "queue depth without scale-up" and "queue depth with scale-up") → **single overlay**. The reader's eye moves between curves at the same x; the delta is immediate.
- Two scenarios that share an x-axis but have different y-axes (e.g., throughput in one panel, latency in another) → **stacked subplots**.
- Two scenarios that share neither → two separate figures.

If you find yourself adding a bracket annotation that spans across stacked subplots to point at the delta, the subplots should have been overlaid. Reach for the single-axis chart and use a within-plot bracket instead.

## Rule 7: Markers and Lines

| Data shape | Mode |
|---|---|
| Sparse data (≤ 30 points per series) | `mode="lines+markers"` |
| Dense data (> 30 points per series) | `mode="lines"` (no markers) |
| Pure scatter, no time/order | `mode="markers"` |

**Never `lines+markers` on dense data.** The markers smear into a thick band and add no information.

## Rule 8: Variance and Uncertainty

For benchmark comparisons, **show error bars or shaded variance bands**. Bare points without uncertainty on a benchmark chart imply false precision and are misleading.

```python
go.Scatter(
    x=offered, y=achieved,
    error_y=dict(type="data", array=stddev, visible=True, thickness=1, width=4),
)
```

Or as a shaded band using two `Scatter` traces with `fill="tonexty"`:

```python
go.Scatter(x=x, y=y_upper, mode="lines", line=dict(width=0), showlegend=False)
go.Scatter(x=x, y=y_lower, mode="lines", line=dict(width=0), fill="tonexty",
           fillcolor="rgba(118,185,0,0.15)", showlegend=False)
go.Scatter(x=x, y=y_mean, mode="lines", line=dict(color="#76b900", width=2.5))
```

If you do not have stddev or replicate runs, run more replicates before publishing.

## Rule 9: Aspect Ratio

- **Wider than tall** (e.g., 16:9, 16:10, 4:3) — emphasizes trend over time. Use for time series, throughput sweeps, latency over duration.
- **Squarer or taller** — emphasizes comparison of categories. Use for bar charts of N backends side by side.
- **Pick deliberately.** Default Plotly aspect ratios (700x500) are not always right.

The flash-indexer chart uses 775x650 because the y-range and x-range cover similar orders of magnitude on log axes; a square-ish ratio keeps the diagonal `y=x` reference line visually unambiguous.

## Rule 10: Title Carries the Takeaway

The chart title is the headline of the figure. It states what the chart proves, not what the chart shows.

- Bad: "GPU Scaling"
- Good: "Throughput Plateaus at 4 GPUs"

- Bad: "Latency Distribution"
- Good: "P99 Latency Doubles Above 100 QPS"

- Bad: "Indexer Comparison"
- Good: "Concurrent Positional Indexer Wins by 40x"

If you cannot write a declarative one-line title, you do not yet know what the chart is for. Stop and figure it out before you ship.

The flash-indexer canonical title is `"ACHIEVED VS. OFFERED THROUGHPUT  (HIGHER IS BETTER)"` — note the all-caps, the parenthetical that orients the reader's interpretation, and the comparison framing that telegraphs the takeaway.

## Rule 11: Title-Subtitle Stack

When the title carries the takeaway, the **subtitle carries the qualifier** — the model, the units, the configuration, the MAPE range. Two-line stack, both left-aligned.

**Typography spec:**

| Role | Family | Size | Weight | Case | Color |
|---|---|---|---|---|---|
| Title | `Helvetica Neue, Helvetica, Arial, sans-serif` | 42 (hero) / 36 (body) | 300 | title case | `text.primary` |
| Subtitle | `Helvetica Neue, Helvetica, Arial, sans-serif` | 22 (hero) / 17 (body) | 300 | sentence case | `text.muted` |

This is the **display / hero** title treatment: the Helvetica set, light weight, title case — the generous headline voice (distinct from the compact Arial 700 ALL-CAPS chart title in Rule 10). Lead the family stack with Helvetica Neue and set `weight=300`; on systems without Helvetica Neue it falls back through Helvetica to Arial, the same visual sans.

**Vertical spacing.** Subtitle top sits **~5 px below the title's bottom edge**. Earlier guidance said 15-20 px below the baseline; in practice that read as a gap, not a stack. Plotly's title bbox is taller than the raw font size (weight-300 with internal padding adds ~8px below the rendered descender), so 5 px below the rendered bottom is what reads as "one block, two lines" on dark backgrounds.

**Title and subtitle casing.** The hero title is **title case** (a short labeled headline like "DynoSim: Simulating the Pareto Frontier") or **sentence case** (a full-sentence verdict like "KV-aware routing lifts the throughput frontier") — never uppercase, which is the compact chart title's treatment.

The subtitle is sentence case, in muted text. Proper nouns (DynoSim, Planner, KVBM, Router, NVIDIA) stay capitalized in every position. The subtitle carries the qualifier the title leaves out: config, units, model, or the numeric result.

**Horizontal alignment.** Subtitle's left edge sits at the figure's `x=0.02` mark, matching the title's `x=0.02`. This is the **alignment trap**:

- Plotly's `title.x` uses **container coords** (0 to 1 across the full figure).
- Plotly's `annotation.x` with `xref="paper"` uses **plot-area coords** (0 to 1 across the plot, *not* the figure).

If you set both to `x=0.02`, they will NOT align — the subtitle will be indented inward by the left margin's worth of paper-space.

**Conversion formula:**

```
paper_x = (title_x_container * figure_width - margin_l) / plot_width
```

Worked example (`width=1240`, `margin_l=80`, plot width = 1240 - 80 - 40 = 1120, title at container `x=0.02`):

```
paper_x = (0.02 * 1240 - 80) / 1120
        = (24.8 - 80) / 1120
        = -55.2 / 1120
        = -0.049
```

So the subtitle annotation gets `x=-0.049, xref="paper"` to line up under the title's left edge. Re-derive per figure when `margin_l` or `width` changes.

**Vertical position formula.** Same pattern — title uses container coords, annotation uses paper coords. To place the subtitle top ~5 px below the title bottom:

```
title_top_px      = (1 - title_y_container) * figure_height
title_bottom_px   = title_top_px + (title_font_size * 0.80)   # empirical bbox, weight 300
subtitle_top_px   = title_bottom_px + 5
paper_y           = 1 + (margin_t - subtitle_top_px) / plot_height
```

The `* 0.80` constant comes from measuring the rendered Helvetica Neue Light title at 42 px; the visible bbox is shorter than the line-height of 42 * 1.2 = 50px because Plotly anchors `yanchor="top"` near the cap-height line rather than the EM-box top. Tune empirically per figure and font weight: render at 2x, measure the gap, adjust.

Worked example (`height=620`, `margin_t=130`, plot height = 420, title at container `y=0.96`, title font 42):

```
title_top      = 0.04 * 620 = 24.8
title_bottom   = 24.8 + 33.6 = 58.4
subtitle_top   = 58.4 + 5    = 63.4
paper_y        = 1 + (130 - 63.4) / 420 = 1.158
```

Use `xref="paper"`, `yref="paper"`, `xanchor="left"`, `yanchor="top"` on the annotation.

**Reference implementation:** a hero figure's generator script should compute these paper coords in code comments so the formula stays auditable.

### Subtitle Content

The subtitle is the qualifier, not a paraphrase of the title. Aim for **8-12 words**. Anything longer reads as a caption that escaped the prose.

**Pattern: `<setup> — <takeaway>`** (em-dash separates the constants from the result).

| Subtitle | Why it works |
|---|---|
| `TP=4, 1k/1k — DynoSim within 5–9% MAPE of real GPUs.` | Setup: config. Takeaway: fidelity number with units. |
| `Planner scales up at 200 ms; 140 ms of TTFT saved.` | Two clauses, both numeric. No hedging. |
| `Engine cores, Router, Planner — one simulated clock, one harness.` | Setup: what's inside. Takeaway: the invariant. |
| `Sweep in sim, verify on the cluster, calibrate from telemetry.` | Three-beat structure, parallel verbs, no jargon. |
| `Two paths to the frontier: DynoSim 5–9% MAPE, AIC 5–11%.` | Reframes a comparison without picking a loser. |

**What to avoid:**

- **Verbose subtitles.** If the subtitle wraps to a second line at the figure's width, it's too long. Cut.
- **Hedging.** "Approximately", "roughly", "about" — say the number or don't say the number. Ranges are fine (`5-9%`); ambiguity is not.
- **Negative framing of a co-product.** "AIC drifts on TTFT" → "Two paths to the frontier" with both MAPE numbers. State the comparison neutrally and let the reader judge.
- **Repeating the title.** If the title is "140 ms of TTFT saved", the subtitle should not be "TTFT improves by 140 ms". The subtitle adds the *qualifier* (when, where, under what config), not the same sentence rephrased.
- **No subtitle at all** on a chart that needs context. If the title is "On the Pareto frontier", the subtitle has to say *which* Pareto, on *what hardware*, at *what config*.

### Subtitles in Hand-Crafted SVG Figures

For SVG-pathway figures, the typography rules are identical (Helvetica-set subtitle, 17–22 px, weight 300, `text.muted`, 8-12 words), but the positioning math differs because SVG uses pixel coords directly and `<text>` defaults to a baseline-anchored render.

```python
text(
    0.02 * W, 92,          # x=0.02 of figure width; y=92 px from top
    "Engine cores, Router, Planner — one simulated clock, one harness.",
    family="Helvetica Neue, Helvetica, Arial, sans-serif",
    size=18, weight="300", color=TEXT_MUTED, anchor="start",
)
```

With `dominant-baseline="middle"` on the SVG text element, the y coordinate is the *vertical center* of the text. For a 42 px title centered at `y=60` and an 18 px subtitle centered ~30 px below it: subtitle `y` ≈ 90-95.

Tune empirically per figure: render at 2x, measure the gap, adjust. The rule is the visible gap (15-20 px between title baseline and subtitle cap-top), not the y coordinate.

## Chart Anti-Pattern Bank

Each entry: **stupid version → corrected version**.

### Axis and Scale

- **Truncated y-axis** to exaggerate differences (Tufte's "lie factor"). → Y-axis starts at zero unless you explicitly call out a broken axis.
- **Auto-picked ticks** that land on 13, 27.5, 191.6. → Explicit `tickmode="array"` with `tickvals` at round multiples.
- **Scientific notation** (`1.234e+06`) on tick labels. → SI suffixes (`1.2M`) via `tickformat=".2s"` or `".1s"`.
- **Three-decimal precision** when one carries the signal (e.g., 1.234567s → 1.23s). → Round to the precision the reader can act on.
- **Comparing incomparable units** on the same axis (e.g., requests/sec next to ms/request). → Two stacked subplots sharing the x-axis. Never one axis with two units.

### Encoding and Decoration

- **Pie charts.** → Bar chart sorted by value. Pie charts are unreadable for anything beyond two slices.
- **3D bars, exploded slices, gradient-filled bars, drop shadows on bars.** → Flat 2D solid muted fills.
- **Rainbow categorical palettes.** → The corpus `chart_series` order: green, CPU blue, fluorite, emerald, gray, amethyst, amber, coral.
- **Double-encoded magnitude** (height AND color saturation both encoding the same value). → Encode magnitude with one channel. Use the other channel for an independent dimension or omit it.

### Annotation and Legend

- **Auto-generated legends that repeat axis titles.** → Either the legend or the axis title carries the information; not both.
- **Legends with > 5 entries** when direct labels would fit. → Direct labels at line ends.
- **Statistical decoration** (p-values, R², fit equations) inside the figure. → Those go in the prose. The figure is for the visual takeaway.

### Composition

- **Six or more series on a single chart.** → Small multiples or facets.
- **Lines + markers on dense data.** → Lines only.
- **Markers without lines on time series.** → Lines + markers (sparse) or lines only (dense).
- **Bare points on benchmark comparisons** (no error bars, no variance band). → Show uncertainty. If you do not have it, run more replicates.

### Information Density

- **Gridlines at every minor tick.** → Gridlines only at major ticks, low contrast (`#3a3a3a`), or no gridlines at all.
- **Chart borders, plot-area borders, redundant axis lines.** → The Dynamo template already removes them. Do not re-add.
- **Captions baked into the chart** with long explanatory sentences. → Caption lives below the figure in prose. The chart carries only its title, axis titles, series labels, and one annotated takeaway.

### Storytelling

- **Figure whose takeaway is "look how complex this is".** → Kill the figure or simplify until the takeaway is positive ("X scales linearly to N=64").
- **Figure that contradicts the prose.** → Fix one of them; never ship the disagreement.
- **Multiple figures saying the same thing.** → Pick the strongest, kill the rest.
- **Figure whose takeaway requires reading the caption to understand.** → Mark the takeaway directly on the chart with an annotation.

## Reference

The canonical example file is [`gen_throughput.py`](../../digest/flash-indexer/tools/gen_throughput.py). It demonstrates:

- Log-scale axes with SI tick formatting
- Direct labeling at peak values
- Annotated bracket showing the 40x improvement
- Series ordered by performance (winner first, loser last)
- Diagonal reference line (`y=x`) to anchor the reader's interpretation
- Custom title with parenthetical orientation
- Two-trace scatter with `lines+markers` because the data is sparse (~10 points per series)
