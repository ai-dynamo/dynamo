# Dynamo Dark Aesthetic Reference

Self-contained reference for palette, typography, semantic color, D2 classes, and diagram anti-patterns. Mirrors [`design_tokens.yaml`](../flash-indexer/tools/design_tokens.yaml) and [`theme.d2`](../flash-indexer/tools/theme.d2) verbatim so this file can be read without leaving the skill.

When tokens drift in the canonical source, update this file too.

## Color Palette

### Brand Primary

| Token | Hex | Use |
|---|---|---|
| `dynamo_green` | `#76b900` | Primary brand and accent. The single most important item per figure. |
| `rich_black` | `#000000` | Outermost background. |
| `white` | `#ffffff` | Light text. |

### Backgrounds

| Token | Hex | Use |
|---|---|---|
| `background.primary` | `#000000` | Outermost canvas. |
| `background.surface` | `#1a1a1a` | Container fill, level 1. |
| `background.surface_alt` | `#2a2a2a` | Container fill, level 2. |
| `background.elevated` | `#3a3a3a` | Container fill, level 3. |

### Borders

| Token | Hex | Use |
|---|---|---|
| `border.frame` | `#76b900` | Outer frame accent (Dynamo green). |
| `border.container` | `#008564` | Container borders (Emerald). |
| `border.subtle` | `#3a3a3a` | Grid lines, separators. |

### Accents (muted for dark theme)

| Token | Hex | Semantic Role |
|---|---|---|
| `dynamo_green` | `#76b900` | Data, GPU, key actions |
| `cpu_blue` | `#0071c5` | CPU, compute, control paths |
| `fluorite` | `#fac200` | Data flow, NVLink, highlights |
| `emerald` | `#008564` | Storage, databases, caches, pipelines |
| `garnet` | `#890c58` | NIC, network hardware |
| `amethyst` | `#5d1682` | Services, APIs, middleware |
| `amber` | `#c08050` | Queues, events, messaging |
| `coral` | `#b04040` | Critical paths, errors, alerts |
| `olive` | `#909040` | Load balancers, infrastructure |

### Muted Fills (desaturated for dark backgrounds)

| Token | Hex | For |
|---|---|---|
| `green` | `#3a5a00` | Data components |
| `blue` | `#0f1e30` | CPU, compute |
| `purple` | `#1a1428` | Services, APIs |
| `teal` | `#142025` | Storage, databases |
| `warm` | `#201810` | Queues, events |
| `wine` | `#2a1520` | NIC, network |
| `signal` | `#1e1a14` | Flags, signals, state |
| `red` | `#2a1010` | Critical, alerts |
| `neutral` | `#1a1a1a` | Generic, utility |

### Text

| Token | Hex | Use |
|---|---|---|
| `text.primary` | `#ffffff` | Main text. |
| `text.secondary` | `#cdcdcd` | Secondary text. |
| `text.muted` | `#767676` | Muted text (4.6:1 AA min). |
| `text.medium` | `#8c8c8c` | Medium gray for ticks, axis. |

### Chart Series (line/bar/scatter strokes)

In order: `#76b900`, `#0071c5`, `#fac200`, `#008564`, `#8c8c8c`, `#5d1682`, `#c08050`, `#b04040`.

Use the order. Series 1 is always the primary (Dynamo green) — the thing you want the reader to look at first. Reorder data so the most important series is first.

### Chart Fills (bar/histogram fills, brighter than D2 fills)

In order: `#4a7500`, `#0a4a80`, `#9a7800`, `#005a40`, `#555555`, `#3a1050`, `#7a5030`, `#702828`.

## Typography

Two canonical families. **Pick one per blog and hold the line.** Mixing families inside one blog reads as two unrelated projects stapled together.

### Family 1 — Flash-Indexer (compact data-dashboard scale)

| Role | Family | Size (px) | Weight | Transform | Letter-Spacing |
|---|---|---|---|---|---|
| Title | `'NVIDIA Sans', Arial, Helvetica, sans-serif` | 18 | 700 | uppercase | 0.08em |
| Heading | `'NVIDIA Sans', Arial, Helvetica, sans-serif` | 14 | 600 | uppercase | 0.05em |
| Label | `'NVIDIA Sans', Arial, Helvetica, sans-serif` | 12 | 400 | none | 0 |
| Annotation | `'NVIDIA Sans', Arial, Helvetica, sans-serif` | 10 | 400 | none | 0 |
| Code / ticks / numbers | `'Roboto Mono', 'SF Mono', Menlo, Consolas, monospace` | 10-12 | 400 | none | 0 |

Use for inline body-prose charts, dense data dashboards, heatmaps, and any figure that needs to pack many marks per inch. Canonical exemplars at [docs/digest/flash-indexer/images/](../flash-indexer/images/).

### Family 2 — Mocker (display-scale headline figures)

| Role | Family | Size (px) | Weight | Transform | Color |
|---|---|---|---|---|---|
| Title | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif` | 42 | 300 | none (sentence case) | `#ffffff` |
| Subtitle | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif` | 22 | 300 | none | `#767676` |
| Label | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif` | 13 | 300 | none | `#cdcdcd` |
| Annotation | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif` | 11 | 300 | none | `#cdcdcd` |
| Code / ticks / numbers | `'Geist Mono', 'JetBrains Mono', 'Roboto Mono', 'SF Mono', Menlo, Consolas, monospace` | 11-13 | 400 | none | varies |
| Callout-card label (in floating annotation boxes) | `Geist, Inter, 'Helvetica Neue', Arial, sans-serif` | 13-14 | 700 | none | `#ffffff` |

Sentence-case titles only (capitalize first word + proper nouns). The takeaway clause lives in the subtitle after an em-dash: "Hardware spec / config / model — narrative takeaway with the punchline." Bold is reserved for callout-card labels.

Use for hero figures, headline artifacts, and any figure that anchors a full page width. Canonical exemplars at [docs/digest/dynosim/](../dynosim/).

**Family rules.** Two families per figure, never three. The body sans + the aligned mono. Within one blog, every figure stays in the same family.

## Borders and Spacing

| Property | Value |
|---|---|
| Frame width | 1.5 px |
| Container width | 1 px |
| Frame style | solid |
| Container style | solid |
| Control flow style | dashed |
| Border radius | 0 (always) |
| Container padding | 16 px |
| Inner gap | 8 px |

## Semantic Color Map

Color encodes role, never aesthetics. Pick the role first; the color follows.

| Role | Color (accent) | Fill (D2) |
|---|---|---|
| GPU, accelerator | `#76b900` Dynamo green | `#151515` with `#4a8c00` stroke |
| CPU, compute, control plane | `#0071c5` CPU blue | `#0f1e30` with `#3d7ab5` stroke |
| Data flow, NVLink, highlight edge | `#fac200` Fluorite | edge stroke only |
| Storage, database, cache | `#008564` Emerald | `#142025` with `#3a7a70` or `#50a090` stroke |
| Service, API, middleware | `#5d1682` Amethyst | `#1a1428` with `#7650a0` or `#8060b0` stroke |
| NIC, network hardware | `#890c58` Garnet | `#2a1520` with `#7a3050` stroke |
| Queue, event, messaging | `#c08050` Amber | `#201810` with `#a06040` or `#c08050` stroke |
| Critical path, error, alert | `#b04040` Coral | `#2a1010` with `#b04040` stroke |
| Load balancer, infra utility | `#909040` Olive | `#1a1a10` with `#909040` stroke |

**Iron rules.**

- **Never reuse a color for two roles in the same figure.** If you paint GPU green AND paint "winning result" green, the reader cannot decode anything. Pick one role per color.
- **Selective accent.** Dynamo green marks the *single* primary item. If everything is green, nothing is.
- **Same role, same color across all figures in a blog.** GPU is green in figure 1, figure 5, and figure 9. Consistency is the source of comprehension.
- **Never paint two stages of a pipeline with NV green and emerald.** They read as the same green at first glance. When a pipeline diagram has more than three colored stages and one is already NV green (`#76b900`), reach for `cpu_blue` (`#0071c5`) as the fourth distinct accent — never emerald.

### Local Token Overrides

The canonical `design_tokens.yaml` is designed for D2 stroke-on-fill where amethyst's purple shows against a `#1a1428` muted fill. Plotly figures that use amethyst as a foreground stroke or marker on a `#000000` plot background often need it brightened — the token value `#5d1682` reads as near-black at small sizes.

When this happens, override locally **at the top of the figure script** (do NOT edit the canonical token):

```python
# Local override: token amethyst (#5d1682) is too dim on black; brighter
# purple keeps the feedback loop visually in-step with the forward pipeline.
AMETHYST = "#a960e8"
```

Document the override in a comment with the reason. The override scope is per-script. Never copy the override into a different blog's `design_tokens.yaml` — that's drift.

The same pattern applies to other accents if you find them too dim against black at small sizes (garnet, olive). The fix is always: keep the role mapping, brighten the value, document the reason in a code comment.

### Phase Markers (Pipeline / Process Diagrams)

When a horizontal pipeline diagram needs to communicate "when in time" each stage runs (pre-deployment, runtime, feedback), add small uppercase **phase tags** above the relevant boxes in the stage's accent color:

- Font: Helvetica Neue Light, 11 pt, weight bold (`<b>...</b>`)
- Case: ALL CAPS (`text.upper()`)
- Color: same as the box's accent (binds the tag visually to its stage)
- Position: ~30-40 px above the box top edge

Phase tags are **selective** — tag the stages that anchor the temporal story (e.g., the verification step and the feedback step), not every stage. Tagging every box is clutter; tagging only the inflection points reads as labeling.

Reference implementation: `gen_fig_6_tuning_loop.py` tags `INITIAL VERIFICATION` (CPU blue) over the Cluster A/B Verify box and `FEEDBACK LOOP` (amethyst) over the Deployed + Telemetry box.

### Squared Feedback Loops (Wired-Return Paths)

When a process diagram has a return path from a downstream stage back to an upstream stage (calibration loop, retry loop, replay edge), draw it as a **squared polyline with two right angles** — not a curved bezier:

```
M x_start,y_box_bottom L x_start,y_loop L x_end,y_loop L x_end,y_box_bottom
```

Three segments: down from the source box, left across the figure, up into the target box. The two corners are right angles.

Why squared, not curved: curves read as "this is illustrative". Squared right angles read as "this is a wire" — the same visual semantics as the orthogonal connectors in architecture diagrams. The Dynamo aesthetic is rectangular; bezier loops break the visual language.

**Dashed stroke** distinguishes the return path from the forward arrows (which are solid). **Match the target box's accent color** (the stage receiving the feedback) so the wire visually belongs to that role.

Reference: the replay edge in `gen_fig_2_architecture.py` (NV green, solid because it's the primary data path) and the calibration loop in `gen_fig_6_tuning_loop.py` (amethyst, dashed because it's a control/feedback path).

## D2 Class Catalog

The classes from [`theme.d2`](../flash-indexer/tools/theme.d2). Pick the class that matches the role; do not invent inline styles.

### Hardware (outlined, muted accent)

| Class | When to use |
|---|---|
| `gpu` | Any GPU or accelerator block. |
| `cpu` | CPU, compute, control plane node. |
| `nic` | Network interface card, network hardware. |

### Logical Components (selective accent fills)

| Class | When to use |
|---|---|
| `data_component` | A data-bearing component (cache, index, table). |
| `flag_component` | State, signal, or flag block. |
| `depth_0`, `depth_1`, `depth_2`, `depth_3` | Tree levels — desaturating green gradient as depth increases. |

### Software / Service Components

| Class | When to use |
|---|---|
| `service` | A service or microservice. |
| `api` | An API endpoint or facade. |
| `database` | A database. |
| `cache` | A cache layer. |
| `queue` | A queue. |
| `event` | An event source/sink. |

### Infrastructure / Deployment

| Class | When to use |
|---|---|
| `container_runtime` | Container runtime layer. |
| `load_balancer` | Load balancer. |
| `endpoint` | A network endpoint. |

### Status / Semantic

| Class | When to use |
|---|---|
| `critical` | Critical-path or error block. |
| `success` | Success state block (use sparingly). |
| `neutral` | Generic neutral block. |

### Containers (subtle, receding)

| Class | When to use |
|---|---|
| `outer_frame` | Top-level frame. |
| `container_l1` | Outer zone. |
| `container_l2` | Mid zone. |
| `container_l3` | Inner zone. |

### Text Elements

| Class | When to use |
|---|---|
| `section_title` | Section header text inside the diagram. |
| `annotation` | Italic annotation text (e.g., "lazy load"). |

### Connection Types

| Class | Stroke | When to use |
|---|---|---|
| `data_flow` | Fluorite, 2 px solid | Primary data movement. |
| `control_flow` | CPU blue, 1 px dashed | Control or signal path. |
| `poll_flow` | Green, 1 px dashed, 0.5 opacity | Polling or pull-based data. |
| `nvlink` | Fluorite, 2 px solid | NVLink edge. |
| `rdma_flow` | Garnet, 2 px solid | RDMA over network. |
| `api_call` | Amethyst, 1 px solid | API call edge. |
| `event_flow` | Amber, 1 px dashed | Event-driven edge. |
| `pipeline` | Emerald, 2 px solid | Pipeline edge. |
| `dependency` | Gray, 1 px dashed | Soft dependency. |
| `critical_path` | Coral, 2 px solid | Critical path. |
| `traversal` | White, 1 px dashed | Tree or graph traversal. |

## Diagram Anti-Pattern Bank

Each entry: **stupid version → corrected version**.

### Shape and Geometry

- **Rounded corners** (`border-radius > 0`). → All shapes have `border-radius: 0`. The aesthetic is rectangular.
- **Diagonal connectors** that swoop from one node to another. → 90-degree connectors only. Diagonals signal that your layout is fighting the structure; rearrange the layout, do not bend the wires.
- **Multiple arrow styles for the same relation** (e.g., one solid green arrow and one solid yellow arrow both meaning "data flow"). → One stroke style per relation type. Pick the right `data_flow` / `control_flow` / `event_flow` class and stick to it.
- **Arrows that flow in arbitrary directions.** → Arrows follow data flow or causality. If a reader cannot guess what an arrow means without a legend, the arrow is wrong.

### Color and Encoding

- **Five-plus colors when two or three carry the meaning.** → Use the minimum number of colors needed to encode role distinctions. Extra colors add cognitive load without information.
- **Reusing a color for two different roles** (green = GPU AND green = "winning result"). → One color per role per figure. If the takeaway needs a "winner" highlight, use a different visual (border, glow, outline) on the existing role color.
- **Double encoding.** Color + shape + position all carrying the same distinction. → Pick one encoding. The others should carry independent information.
- **Raw hex outside the palette.** → Every color in the figure must come from `design_tokens.yaml`. If you reach for a new hex, the role does not yet exist in the system; either add it canonically or reuse an existing role.
- **Rainbow categorical palettes** (the default Plotly or Vega palette). → Use the corpus `chart_series` order. Series 1 is the primary, series 2-3 are the comparisons, series 4+ are background context.

### Typography

- **Mixed font families inside one figure.** → Two families maximum: one body sans + one aligned mono, from the same family (flash-indexer OR mocker — see typography section). Never a third.
- **Mixed families across figures in the same blog.** → Pick the family at the blog level, not the figure level. One blog renders in one family, end to end.
- **Comic Sans, Impact, or any decorative font.** → Only the two canonical families.
- **Title that hedges or asks a question.** ("Performance Overview", "How Fast Is It?") → Title is the takeaway in declarative form. ("Concurrent Indexer Wins by 40x" for flash-indexer ALL-CAPS, or "Snapshot restore collapses cold start" sentence-case + em-dash subtitle for mocker.)
- **ALL CAPS title in a mocker-family blog, or sentence-case 42px in a flash-indexer blog.** → Cross-family contamination. Each family has its own headline grammar; do not borrow.

### Visual Effects

- **Drop shadows.** → No drop shadows. The dark background already provides depth via fill darkness.
- **Gradients on shapes** (linear-gradient backgrounds, radial fills). → Solid muted fills only. The single allowed gradient is the green glow on a single accent (used sparingly).
- **3D effects** (extruded boxes, isometric, "depth" shading). → 2D only. 3D is decoration, not information.

### Composition

- **Empty container zones.** A zone box that contains nothing. → If a zone holds no components, delete the zone. Decoration is clutter.
- **Unlabeled components.** A box with no label. → Components without labels are not architecture, they are decoration. Label everything or delete the unlabeled boxes.
- **Icons not in the icon library.** Importing random Heroicons, Material Icons, or emoji. → Use only the established icon set (or none). Same icon, same meaning, every figure.
- **Watermarks, logos, dates ("Q1 2026") inside the figure.** → No corporate decoration baked in. Dates only when temporal context is the point of the figure.
- **Captions baked into the figure.** Long sentences inside the figure explaining what it shows. → Captions live below the figure in prose. The figure carries one short title, one one-line takeaway, and labels.
- **Transparent backgrounds.** → Always opaque dark. Transparency makes figures invisible on light themes that may render around the embed (Slack, Confluence light mode).
- **Raster icons inside vector figures.** A 32x32 PNG inserted into an SVG. → SVG icons only inside SVG figures. PNGs inside HTML→PNG sources are fine; they're rasterized at export.

### Reference

When in doubt, open one of:

**Flash-indexer family** (compact data-dashboard scale, 18 px ALL-CAPS title):
- [docs/digest/flash-indexer/images/](../flash-indexer/images/) — six figures across all four implemented pathways.
- [docs/digest/agentic-inference/two-gates.svg](../agentic-inference/two-gates.svg) — exemplar hand-crafted SVG.
- [docs/digest/agentic-inference/protocol-stack.svg](../agentic-inference/protocol-stack.svg) — exemplar layered SVG with custom dividers.

**Mocker family** (display-scale headline figures, 42 px sentence-case title + em-dash subtitle):
- [docs/digest/dynosim/](../dynosim/) — nine DynoSim figures (hero + benchmark sweeps + planner studies + KV router comparison). `fig-1-hero-config-space.svg` is the canonical hero pattern; `fig-5-planner-load-interval.svg` is the canonical single-chart pattern.
