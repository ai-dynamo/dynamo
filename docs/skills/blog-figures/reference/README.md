<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Figure-Generator Reference

A browsable catalog of every reusable figure **generator** in this skill and its
digest exemplars. The machine-readable source is
[`figure-manifest.yaml`](figure-manifest.yaml); this page is the human index.

## What It Is

Each generator is a self-contained Python (or HTML+CSS) script that renders one
Dynamo Dark figure. The manifest records, per generator: its **kind** (visual
grammar), **renderer**, **title treatment**, **outputs**, whether its data is
real or illustrative, its measured **lint score**, and a one-line **use-when**.

## How to Pull From It

1. Decide the visual grammar you need and find it in the `kinds:` block of the
   manifest (for example `chart.timeline-anatomy` or `diagram.decision-cascade`).
2. Scan the table below (or `use_when` in the manifest) for the closest exemplar
   by kind.
3. Copy that generator as your starting template, then follow the skill's
   bootstrap step in [`../SKILL.md`](../SKILL.md): copy `design_tokens.yaml` and
   `plotly_dynamo.py` from the canonical reference blog into your own `tools/`
   (never fork the tokens), swap in your source-of-truth data, and re-render.
4. Prefer a `lint_score: 100.0` exemplar as the template. The ModelExpress
   cold-start set is the newest end-to-end 100/100 reference.

Filter the manifest by kind with `yq`:

```bash
yq '.generators[] | select(.kind == "chart.bar") | .generator' \
  docs/skills/blog-figures/reference/figure-manifest.yaml
```

## Kinds Vocabulary

`kind` is a controlled vocabulary (four families). Add a key to `kinds:` in the
manifest before introducing a new value.

- **`chart.*`** data figures: `chart.bar`, `chart.stacked-bar`,
  `chart.timeline-anatomy`, `chart.line`, `chart.small-multiples`,
  `chart.heatmap`.
- **`diagram.*`** node/edge layouts: `diagram.sequence`,
  `diagram.architecture`, `diagram.loop`, `diagram.decision-cascade`.
- **`hero.*`** full-width, display-title compositions: `hero.flow`,
  `hero.scatter`.
- **`cards.*`** HTML+CSS layouts: `cards.comparison`.

## Catalog

### ModelExpress Cold-Start Set (newest 100/100 exemplars)

`drafts/modelexpress-coldstart/`

| Kind | Title | Renderer | Generator | Lint |
|---|---|---|---|---|
| `diagram.sequence` | Cold Start: The Data Plane Bypasses the Metadata Server | plotly | `fig-1/gen_fig_1_coldstart_sequence.py` | 100 |
| `diagram.loop` | ModelExpress Closes the RL Training Loop | plotly | `fig-2/gen_fig_2_rl_loop.py` | 100 |
| `diagram.architecture` | ModelExpress Splits Coordination from Weight Transfer | plotly | `fig-3/gen_fig_3_modelexpress.py` | 100 |
| `chart.stacked-bar` | Cold Start Time With P2P RDMA and Kernel Artifacts | plotly | `fig-4/gen_fig_4_coldstart.py` | 100 |
| `chart.bar` | NIXL Registration Time: VMM Arena Wins by 10.3x | plotly | `fig-5/gen_fig_5_nixl_registration.py` | 100 |
| `chart.timeline-anatomy` | Cold Start Anatomy: P2P Weights, Cold JIT Caches | plotly | `fig-6/gen_fig_6_coldstart_anatomy.py` | 100 |
| `hero.flow` | The Fast Path to Warm GPUs | plotly | `hero/tools/gen_hero.py` | 100 |

### Skill Example Generators (pattern exemplars)

`docs/skills/blog-figures/examples/`

| Kind | Title | Renderer | Generator | Lint |
|---|---|---|---|---|
| `diagram.architecture` | Dynamo Serving Stack: One Simulated Timeline | plotly | `gen_fig_2_architecture.py` | 100 |
| `diagram.decision-cascade` | Prefix-Aware Routing: The Decision Cascade | plotly | `gen_fig_5_decision_cascade.py` | 100 |
| `diagram.loop` | Sweep, Verify, Calibrate: The Tuning Loop | plotly | `gen_fig_6_tuning_loop.py` | 100 |
| `chart.small-multiples` | Round-Robin vs KV Router: Concurrency Sweep and Pareto Curve | plotly | `gen_fig_concurrency_sweep.py` | 100 |
| `chart.bar` | Indexer Throughput by Backend | plotly | `gen_fig_throughput_bars.py` | 100 |
| `cards.comparison` | HTML+CSS Comparison Cards | html-css | `gen_fig_cards.py` | 100 |

### Published Digest Generators

`docs/digest/<slug>/tools/`

| Kind | Title | Renderer | Generator | Lint |
|---|---|---|---|---|
| `hero.scatter` | DynoSim: Simulating the Pareto Frontier | plotly | `dynosim/tools/gen_hero.py` | 100 |
| `chart.line` | Achieved vs. Offered Throughput | plotly | `flash-indexer/tools/gen_throughput.py` | 97 |
| `chart.heatmap` | KV Cache Event Density | plotly | `flash-indexer/tools/gen_heatmap.py` | 85 |

The two flash-indexer generators predate the 100/100 exemplars and carry
raw-hex lint flags (off-token series and colorscale colors); see `lint_flags`
in the manifest. They still render and pass the ship threshold (>= 85), but for
a fresh figure prefer a 100/100 exemplar of the same kind.

## Scoring

`lint_score` is `overall_measured` from the skill's linter, captured per
generator against its own sibling `design_tokens.yaml`:

```bash
python3 docs/skills/blog-figures/tools/lint_figures.py <generator.py> \
  --tokens <sibling>/design_tokens.yaml --score
```

The measured score covers palette, typography, contrast, palette variety, and
label structure only. The judged dimensions (data-ink, single-accent semantics,
composition, before/after honesty, takeaway) still need eyes on the rendered
PNG. See [`../RATINGS.md`](../RATINGS.md).

## Excluded

Non-figure helpers are intentionally left out of the manifest: `plotly_dynamo.py`
(shared Plotly template), `design_tokens.yaml` (the tokens themselves),
`build.sh` (render driver), and `tools/lint_figures.py` (the linter/scorer).
