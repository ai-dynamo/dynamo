# Dynamo Dark — Design Language

Source of truth for the visual system used in Dynamo Digest figures. Everything here is quotable — the blog-figures skill and any automation cite this file verbatim.

The machine-readable source of truth is [`design_tokens.yaml`](../../digest/flash-indexer/tools/design_tokens.yaml) (`meta.name: dynamo-dark`, consumed by the D2 theme and the Plotly template). This file is its prose companion. When a value here disagrees with the tokens, the tokens win — fix this file.

**One family.** Every Dynamo figure uses a single aesthetic called **Dynamo Dark**. There is no second family. Figures scale in size — inline chart, body figure, full-width hero — but never switch fonts, weights, or palette. ("DynoSim" and "flash-indexer" are the names of blog posts that use this aesthetic, not separate families; "Mocker" is the Dynamo load-simulator component, not a figure family.)

## Background and Surfaces

| Token | Hex | Use |
|---|---|---|
| `background.primary` | `#000000` | Canvas background. No exceptions, no `#0a0a0a` "soft black". |
| `background.surface` | `#1a1a1a` | Container fill, level 1 — cards, panels, plot areas. |
| `background.surface_alt` | `#2a2a2a` | Container fill, level 2 — nested surfaces. |
| `background.elevated` | `#3a3a3a` | Container fill, level 3 — the most-raised surface. |

Rounded corners are never used. `border-radius: 0` everywhere. Inset content boxes drop to `#000000`.

## Borders

| Token | Hex | Use |
|---|---|---|
| `border.frame` | `#76b900` | Outer frame / single accent border (Dynamo green), 1.5 px. |
| `border.container` | `#008564` | Container borders (Emerald), 1 px. |
| `border.subtle` | `#3a3a3a` | Hairline borders, grid lines, separators, 1 px. |

## Accent Colors

Color encodes role, never decoration. Each role gets one consistent color across all figures in the same blog.

| Token | Hex | Role |
|---|---|---|
| `dynamo_green` | `#76b900` | Primary accent / GPU / data / the "winning" or "after" element. One per figure. |
| `cpu_blue` | `#0071c5` | CPU, compute, control paths, reference series in cost-latency / Pareto charts. |
| `fluorite` | `#fac200` | Data flow, NVLink, highlights, measured / baseline data points. |
| `emerald` | `#008564` | Storage, databases, caches, pipelines. |
| `garnet` | `#890c58` | NIC, network hardware. |
| `amethyst` | `#5d1682` | Services, APIs, middleware, production / feedback / human-in-the-loop role. |
| `amber` | `#c08050` | Queues, events, messaging. |
| `coral` | `#b04040` | Critical paths, errors, the "before" / "loser" / bottleneck element. |
| `olive` | `#909040` | Load balancers, infrastructure. |

If a figure uses more than two accent colors carrying meaning, the design is overloaded. Drop back to greys + green and let the labels carry the rest.

## Muted Fills (desaturated for dark backgrounds)

D2 component fills and low-emphasis surfaces. Pair each with its accent stroke.

| Token | Hex | For |
|---|---|---|
| `fills.green` | `#3a5a00` | Data components. |
| `fills.blue` | `#0f1e30` | CPU, compute. |
| `fills.purple` | `#1a1428` | Services, APIs. |
| `fills.teal` | `#142025` | Storage, databases. |
| `fills.warm` | `#201810` | Queues, events. |
| `fills.wine` | `#2a1520` | NIC, network. |
| `fills.signal` | `#1e1a14` | Flags, signals, state. |
| `fills.red` | `#2a1010` | Critical, alerts. |
| `fills.neutral` | `#1a1a1a` | Generic, utility. |

## Text

| Token | Hex | Use |
|---|---|---|
| `text.primary` | `#ffffff` | Main text; in-bar numeric labels on dark fills. |
| `text.secondary` | `#cdcdcd` | Secondary labels, legend labels. |
| `text.medium` | `#8c8c8c` | Axis ticks, gridline labels, medium-emphasis text. |
| `text.muted` | `#767676` | Footer captions, sub-meta, subtitles (4.6:1 AA min). |

In-bar numeric labels are `#ffffff` on dark fills and `#000000` on green fills.

## Chart Colors

**Series (line / bar / scatter strokes), in order:** `#76b900`, `#0071c5`, `#fac200`, `#008564`, `#8c8c8c`, `#5d1682`, `#c08050`, `#b04040`.

Series 1 is always the primary (Dynamo green) — the thing the reader should look at first. Reorder data so the most important series is first.

**Fills (bar / histogram fills, brighter than D2 component fills), in order:** `#4a7500`, `#0a4a80`, `#9a7800`, `#005a40`, `#555555`, `#3a1050`, `#7a5030`, `#702828`.

For non-accent bars that need to distinguish 2–3 series, pull greys from `#8c8c8c` (`text.medium`), `#555555` (chart fill grey), and `#3a3a3a` (`border.subtle`). Do not invent intermediate greys.

## Typography — One Family

One Dynamo Dark family: the sans (`Helvetica Neue, Helvetica, Arial, sans-serif`, with Helvetica Neue leading for display titles) plus the aligned mono. Never a third. The title has two treatments that differ only by figure scale — Helvetica Neue and Arial are the same visual sans, so both read as one family.

### Display / Hero Title

Hero and section-anchor figures use a light display title in the Helvetica set, title case — the generous, quiet headline voice.

| Role | Family | Size (px) | Weight | Case | Color |
|---|---|---|---|---|---|
| Title | `Helvetica Neue, Helvetica, Arial, sans-serif` | 42 (hero) / 36 (body) | 300 | title case | `#ffffff` |
| Subtitle | `Helvetica Neue, Helvetica, Arial, sans-serif` | 22 (hero) / 17 (body) | 300 | sentence case | `#767676` |

Title is top-left, never uppercase, never a display serif. The subtitle sits one line below with an em-dash pattern: `<hardware / config / model> — <takeaway>`. A short labeled headline stays title case; a full-sentence verdict headline may drop to sentence case.

### Compact / Chart Title

Dense dashboards, heatmaps, and inline charts use the compact title from `design_tokens.yaml` — tight ALL-CAPS for a data-dashboard read.

| Role | Family | Size (px) | Weight | Transform | Letter-spacing |
|---|---|---|---|---|---|
| Title | `Arial, Helvetica, sans-serif` | 18 | 700 | uppercase | 0.08em |

### Shared Roles (both scales)

| Role | Family | Size (px) | Weight | Transform | Letter-spacing |
|---|---|---|---|---|---|
| Heading | `Arial, Helvetica, sans-serif` | 14 | 600 | uppercase | 0.05em |
| Label | `Arial, Helvetica, sans-serif` | 12 | 400 | none | 0 |
| Annotation | `Arial, Helvetica, sans-serif` | 10 | 400 | none | 0 |
| Code / ticks / numbers | `'Roboto Mono', 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace` | 10–13 | 400 | none | 0 |

## Canvas and Layout

| Role | Canvas size | Title position |
|---|---|---|
| Hero / standalone | 1600 × 720–900 px | `x = 50, y = 60` |
| Body figure | 1280 × 680–1080 px | `x = 40, y = 58` |
| Body chart (compact) | 1024–1240 × variable | `x = 24–40, y = 24–40` |
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
| Color | `#8c8c8c` (`text.medium`) for neutral flows; `#76b900` for the accented flow; `#b04040` for the "bottleneck" or "loser" flow |
| Arrowhead | Filled triangle, 8–10 px length, color matching the stroke |
| Endpoints | Tail starts at the exact right edge of the source card (`source.x + source.width`); tip lands at the exact left edge of the target card (`target.x`, minus the arrowhead length) |
| Path | Orthogonal-only (horizontal + vertical) when paths cross. No diagonals in dense layouts. |

Multi-source / multi-target connections compute a shared meeting line: `meet_x = (sources_right + targets_left) / 2`. Never eyeball arrow bend points.

## Cards and Containers

| Convention | Spec |
|---|---|
| Fill | `#1a1a1a` (`background.surface`) for primary cards; `#000000` for inset content boxes |
| Border | 1 px `#3a3a3a` (`border.subtle`) hairline by default; 1.5 px `#76b900` (`border.frame`) for the single accent card |
| Padding | 16 px interior padding before content (`spacing.padding`) |
| Header pattern | Title in 14 px heading style at top-left of card; optional state-tag (`WARM`, `LIVE`, `CAPTURED`) as 11–12 px mono caps at top-right |
| Internal divider | 1 px `#3a3a3a` hairline between header and body |

The single accent card gets EITHER a `#76b900` border OR a subtle green-tinted interior derived from `fills.green` (`#3a5a00`) at low opacity — never both, and never with anything else green inside. When both fire, the eye stops being able to find the single accent.

## What This Design Language Forbids

These are anti-patterns. Every one of them has shown up in a real figure draft. Naming them here makes them easier to refuse.

- **3D bars, isometric perspective, drop shadows, gradient fills.** The only allowed gradient is a soft green glow on a single accent element, and even that is a last resort.
- **Rounded corners.** `border-radius: 0` everywhere.
- **A third type family.** One family — Dynamo Dark: the Helvetica/Arial sans (display titles use its Helvetica Neue Light weight) + `Roboto Mono`. Do not introduce a display serif, Geist, or any font outside that set.
- **More than two accent colors per figure.** If the figure has coral + green + yellow + blue all carrying meaning, the design is overloaded. Reduce.
- **Raw hex outside the tokens.** Every color comes from `design_tokens.yaml`. If you reach for a new hex, the role does not yet exist in the system; add it canonically or reuse an existing role.
- **Inconsistent label placement across the family.** Inside-some, outside-others, above-some in the same blog reads as broken. One rule per visual category.
- **Made-up numbers.** Memory sizes, latency figures, throughput numbers, percentages — every digit comes from a source of truth (blog body, data file, benchmark log). Never plausible-sounding fabrications.
- **Eyeballed geometry.** Arrow tails that don't land on card edges, "right-aligned" columns at ragged `x` positions, legends drifting off center. Compute every important coordinate from named constants or another element's known coordinate.
- **Star-shaped layouts with bent arrows.** Refactor to a 3-column grid (sources | hub | targets) with straight horizontal arrows, or use an auto-layout engine (D2, dynamo-svg).
- **BEFORE panels that just look like a faded AFTER.** Contrast figures must show the actual pre-state machinery, not the post-state with one cell highlighted.
- **Titles that name the chart type instead of the takeaway.** "Performance Overview" is dead air. "Concurrent indexer wins by 40×" is a title.

## Self-Check Before Shipping a Figure

Walk this list explicitly, item by item, before declaring done. No silent "looks fine."

1. **Ground is `#000000`.** Background is pure black, not transparent, not soft.
2. **Type treatment matches the scale.** Hero titles are the Helvetica set, weight 300, title case; compact chart titles are Arial 700 uppercase; body sans and mono are the token stacks. No display serif, no Geist, no third family.
3. **Single accent.** Green appears only on the one item that's "winning"; other figures in the blog use green for the same role.
4. **Numbers are real.** Every digit traces back to a source of truth.
5. **Geometry is computed.** Arrow endpoints, card edges, column alignments, legend centers all derive from named constants — not eyeballed.
6. **Label placement is uniform across the family.** Bars, lane totals, overflow values, callouts each follow one rule across all figures in the blog.
7. **Every color is a token.** No hex in the figure that isn't in `design_tokens.yaml`.
8. **Title carries the takeaway.** The title is a declarative statement about what the figure shows, not a category name.

A figure that fails any one of these gets cut, not shipped.

Items 1, 2, and 7 (pure-black ground, one type family, every color a token) plus WCAG contrast are checked mechanically by [`tools/lint_figures.py`](tools/lint_figures.py); it also computes the measured half of the 0–100 rating in [RATINGS.md](RATINGS.md). The remaining checks need eyes on the rendered figure.
