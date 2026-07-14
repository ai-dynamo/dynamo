# Dynamo Blog Figures — Design Language

Source of truth for the visual system used in Dynamo blog figures. Everything here is quotable — the page-audit skill, the blog-figures skill, and any future automation cite this file verbatim.

Two families share a single dark aesthetic. Pick one family per blog post; never mix. Within a family, the values below are not suggestions.

## Background and Surfaces

| Token | Hex | Use |
|---|---|---|
| Ground | `#000000` | Canvas background. No exceptions, no `#0a0a0a` "soft black". |
| Card surface | `#0f0f0f` | Container fills (cards, panels, plot areas). |
| Card surface, secondary | `#0a0a0a` | Inset/nested surfaces where a subtler step than `#0f0f0f` is needed. |
| Card border, hairline | `#2a2a2a` | 1 px border on cards, panels, frames. |
| Card border, accent | `#74b711` or `#76b900` | 1.5–2 px border on the single accented element only. |

Rounded corners are never used. `border-radius: 0` everywhere.

## Accent Colors

The accent palette is intentionally narrow. Each role gets one consistent color across all figures in the same blog.

| Color | Hex | Role |
|---|---|---|
| Dynamo green | `#76b900` | The "winning" or "after" element. The accent. One per figure, ideally one role across the whole blog. |
| Dynamo green, light | `#9ed649` | Second tier of green when contrast inside the accent is required (e.g., "best of the winners"). Used sparingly. |
| Coral | `#b04040` | The "before" / "loser" / "bottleneck" element. Used only when a figure explicitly needs a second semantic accent. |
| Yellow | `#fac200` | Measured / baseline data point (e.g., real GPU measurements in a sim-vs-real comparison). Semantic only — never decorative. |
| Blue | `#0071c5` | Reference series in cost-latency / Pareto charts. Semantic only. |
| Purple | `#a960e8` | Production / feedback / human-in-the-loop role. Semantic only. |

If a figure uses more than two accent colors, the design is overloaded. Drop colors back to greys + green and let the labels carry the rest.

## Greyscale Ramp

Greys are the workhorse. Used for non-accented content, secondary fills, and the structural skeleton.

| Token | Hex | Use |
|---|---|---|
| White | `#ffffff` | Primary text on cards, in-bar numeric labels on dark fills. |
| Light text | `#cdcdcd` | Subtitle text, secondary labels. |
| Muted text | `#8c8c8c` | Axis ticks, footer captions, sub-meta. |
| Dim text | `#767676` | Mocker subtitle color; tertiary labels. |
| Grey fill, light | `#9a9a9a` | Lightest non-accent bar fill. |
| Grey fill, mid | `#5a5a5a` | Mid-tone bar fill. |
| Grey fill, dark | `#3a3a3a` | Darkest non-accent bar fill, gridlines at higher opacity. |
| Grey divider | `#1a1a1a` | Gridlines, axis lines, faint dividers. |

Bar charts that need to distinguish multiple non-accented series use 2–3 shades from this ramp. Pick from `#3a3a3a`, `#5a5a5a`, `#7a7a7a`, `#9a9a9a`. Do not invent intermediate values.

## Two Type Families

The Dynamo Dark aesthetic ships in two families. They share the palette above but differ in typography scale and headline style. Pick one per blog and hold the line.

### Family 1 — flash-indexer (compact data-dashboard scale)

| Role | Spec |
|---|---|
| Title | 18 px, ALL CAPS, weight 700, `letter-spacing: 0.08em`, `text-transform: uppercase`, color `#ffffff` |
| Subtitle | 12 px, weight 400, color `#cdcdcd` |
| Body sans | `'NVIDIA Sans', Arial, Helvetica, sans-serif`, weight 400, sizes 11–15 px |
| Mono | `'Roboto Mono', 'SF Mono', Menlo, Consolas, monospace`, weight 400, sizes 10–13 px |
| Axis labels | 10–11 px mono, color `#8c8c8c` |

Use this family when the figure is a chart with many marks, dense ticks, a heatmap, or embeds inline at column width.

### Family 2 — Digital Twin / DynoSim (display-scale headline)

| Role | Spec |
|---|---|
| Title | 42 px (full hero) or 36 px (body figure), `'Helvetica Neue'` weight 300, sentence-case, top-left of canvas |
| Subtitle | 22 px (full hero) or 17 px (body figure), `'Helvetica Neue'` weight 300, color `#767676`, em-dash + descriptive clause + takeaway |
| Body sans | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif`, weight 300 by default |
| Mono | `'Geist Mono', 'JetBrains Mono', 'Roboto Mono', 'SF Mono', Menlo, Consolas, monospace`, weight 400–500 |
| Section / panel labels | 12–13 px, weight 600, letter-spacing `0.08em`, `text-transform: uppercase` |
| Section sub-labels | 11–12 px mono, color `#8c8c8c` |
| In-bar numeric labels | 11–12 px mono, weight 500, color `#ffffff` on dark fills, `#000000` on green fills |
| Callout-card labels | weight 600 (only place weight ≥ 500 appears outside mono) |
| Italic editorial accent | `'Iowan Old Style', Georgia, serif`, weight 300, italic — for one-line takeaway captions only |

Use this family when the figure is a hero, section anchor, or embeds standalone at full page width. Default for new Dynamo blogs.

## Canvas and Layout

| Role | Canvas size | Title position |
|---|---|---|
| Hero / standalone | 1600 × 720–900 px | `x = 50, y = 60` |
| Body figure (Digital Twin) | 1280 × 680–1080 px | `x = 40, y = 58` |
| Body chart (flash-indexer) | 1024–1240 × variable | `x = 24–40, y = 24–40` |
| Inline mini-chart | 680 × 360 px | `x = 24, y = 32` |

Title is always top-left, never centered, never floating. The subtitle sits one line height below the title with a consistent `y` offset across all figures in the blog.

Within one blog, every figure picks one canvas width per role and holds it. A blog with three body figures at 1280 px and one at 1180 px reads as broken.

## Bars, Lanes, and Numeric Labels

Bar charts and Gantts share one placement rule per visual category. Mix-and-match across figures in the same blog reads as inconsistent immediately, even when each individual chart "works."

| Visual category | Placement rule |
|---|---|
| Bar / segment value | Always INSIDE the bar, centered. Drop the label if the bar is too narrow to fit the digits (typically `< 28 px` wide for `X.Xs` in 11 px mono). |
| Lane total (sum of a Gantt row, group summary) | Always OUTSIDE the lane to the right, in mono. Anchor `x` is shared across all rows in the chart. |
| Overflow value (bar extends past axis) | Chevron + mono label outside the axis on the right. Optional italic sub-line for context. |
| Speedup callout (`18×`, `21×`) | Dedicated right-edge column. Vertically aligned with each row's center. |

Bars stack horizontally (Gantt style) without rounded corners or drop shadows. Bar height: 18 px for grouped bars in a scoreboard, 36–48 px for single-row Gantt lanes.

## Legend Conventions

| Convention | Spec |
|---|---|
| Position | Bottom-center of canvas |
| Layout | Single row if it fits; two centered rows otherwise. Never three rows. |
| Swatch | 12 × 12 px filled rectangle, no border, hairline gap from label |
| Label | Sans, 12–13 px, color `#cdcdcd` |
| Spacing | 18–22 px between legend items |

Legends only appear when direct-labeling is impractical (6+ series, dense heatmaps). For ≤ 5 series, direct-label and drop the legend.

## Arrows and Connections

| Convention | Spec |
|---|---|
| Stroke | 1.5–2 px solid for primary flow; 1 px dashed `3 3` for "observed" / "dimmed" flows |
| Color | `#7a7a7a` for neutral flows; `#74b711` for the accented flow; `#b04040` for the "bottleneck" or "loser" flow |
| Arrowhead | Filled triangle, 8–10 px length, color matching the stroke |
| Endpoints | Tail starts at the exact right edge of the source card (`source.x + source.width`); tip lands at the exact left edge of the target card (`target.x`, minus the arrowhead length) |
| Path | Orthogonal-only (horizontal + vertical) when paths cross. No diagonals in dense layouts. |

Multi-source / multi-target connections compute a shared meeting line: `meet_x = (sources_right + targets_left) / 2`. Never eyeball arrow bend points.

## Cards and Containers

| Convention | Spec |
|---|---|
| Fill | `#0f0f0f` for primary cards, `#000000` for inset content boxes |
| Border | 1 px `#2a2a2a` hairline by default; 1.5–2 px `#74b711` for the single accent card |
| Padding | 16–24 px interior padding before content |
| Header pattern | Title in 14–16 px Geist 500 at top-left of card; optional state-tag (`WARM`, `LIVE`, `CAPTURED`) as 11–12 px mono caps at top-right |
| Internal divider | 1 px `#2a2a2a` hairline between header and body |

The accented card additionally gets a subtle interior tint (e.g., `#0f1607` for a green-tinted dark). Never combine the green border AND the green-tinted interior with anything else green inside — the eye stops being able to find the single accent.

## What This Design Language Forbids

These are anti-patterns. Every one of them has shown up in a real figure draft. Naming them here makes them easier to refuse.

- **3D bars, isometric perspective, drop shadows, gradient fills.** The only allowed gradient is a soft green glow on a single accent element, and even that is a last resort.
- **Rounded corners.** `border-radius: 0` everywhere.
- **More than two accent colors per figure.** If the figure has coral + green + yellow + blue all carrying meaning, the design is overloaded. Reduce.
- **Inconsistent label placement across the family.** Inside-some, outside-others, above-some in the same blog reads as broken. One rule per visual category.
- **Made-up numbers.** Memory sizes, latency figures, throughput numbers, percentages — every digit comes from a source of truth (blog body, data file, benchmark log). Never plausible-sounding fabrications.
- **Eyeballed geometry.** Arrow tails that don't land on card edges, "right-aligned" columns at ragged `x` positions, legends drifting off center. Compute every important coordinate from named constants or another element's known coordinate.
- **Star-shaped layouts with bent arrows.** Refactor to a 3-column grid (sources | hub | targets) with straight horizontal arrows, or use an auto-layout engine (D2, dynamo-svg).
- **BEFORE panels that just look like a faded AFTER.** Contrast figures must show the actual pre-state machinery, not the post-state with one cell highlighted.
- **Titles that name the chart type instead of the takeaway.** "Performance Overview" is dead air. "Concurrent indexer wins by 40×" is a title.

## Self-Check Before Shipping a Figure

Walk this list explicitly, item by item, before declaring done. No silent "looks fine."

1. **Ground is `#000000`.** Background is pure black, not transparent, not soft.
2. **Family is consistent.** Title size, body font, and headline style match the rest of the blog's figures.
3. **Single accent.** Green appears only on the one item that's "winning"; other figures in the blog use green for the same role.
4. **Numbers are real.** Every digit traces back to a source of truth.
5. **Geometry is computed.** Arrow endpoints, card edges, column alignments, legend centers all derive from named constants — not eyeballed.
6. **Label placement is uniform across the family.** Bars, lane totals, overflow values, callouts each follow one rule across all figures in the blog.
7. **Title carries the takeaway.** The title is a declarative sentence about what the figure shows, not a category name.

A figure that fails any one of these gets cut, not shipped.
