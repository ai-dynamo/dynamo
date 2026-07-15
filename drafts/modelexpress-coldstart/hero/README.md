# hero — ModelExpress Cold Start

Wide (16:9) headline hero for the ModelExpress / cold-start figure set, in the
**Dynamo Dark** aesthetic with the display / hero title treatment.

**Takeaway:** ModelExpress delivers model weights peer-to-peer over RDMA, so the
weight-load phase collapses from a 70 s cold object-store pull (coral, the slow
baseline) to an 11 s RDMA transfer (the single green accent / fast path) — a
6.4x faster weight load, 59 s reclaimed per cold start.

## Figure Inventory

| File | Description |
|---|---|
| `images/hero-modelexpress-coldstart.png` | Title lockup over a weight-transfer collapse motif: baseline 70 s (coral) vs ModelExpress 11 s (green), dashed green finish line + green delta bracket (1600×900 @2x) |
| `images/hero-modelexpress-coldstart.svg` | Vector source of the same hero |

## Type & Pathway

- **Type:** hero — title lockup + a single supporting motif (not a dense chart).
- **Pathway:** Python + Plotly via `plotly_dynamo`, rendered with kaleido at hero width (1600 px).
- **Title treatment:** display / hero (Helvetica Neue Light, weight 300, title case) + muted Helvetica subtitle.

## Color Convention (shared with the figure set)

- **green** — the fast RDMA data-plane weight-transfer path (what ModelExpress accelerates); the single accent.
- **coral** — the slow baseline / cold object-store pull.

Green appears only on the fast path (bar, 11 s number, finish line, delta bracket, punch line).

## Data (source of truth)

Model-load durations from the ModelExpress cold-start figure set (fig-4 / fig-6):

| Path | Model load | Role → color |
|---|---|---|
| Baseline (cold pull from object store, no P2P) | 70 s | slow baseline → coral |
| ModelExpress (P2P RDMA weight transfer) | 11 s | **MX win → green (single accent)** |

`6.4x` and `59 s` are derived (`70 / 11`, `70 − 11`). Scoped to the
weight-transfer phase only — cold JIT-cache warmup is a separate cold-start cost.

## Reproduce

```bash
./build.sh                 # bootstraps .venv, renders PNG + SVG, lints sources
```

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install plotly==6.9.0 kaleido==1.3.0 PyYAML==6.0.3
.venv/bin/python tools/gen_hero.py
```

## Design System

`tools/design_tokens.yaml` and `tools/plotly_dynamo.py` are copied verbatim from
the canonical blog-figures skill; do not fork them. Colors, surfaces, borders,
and the mono font all come from the tokens.

`.venv/` and `__pycache__/` are build artifacts, not figure source; `build.sh`
recreates the venv on demand.
