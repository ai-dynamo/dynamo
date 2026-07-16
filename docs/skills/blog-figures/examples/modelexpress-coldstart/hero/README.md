# hero — ModelExpress Cold Start

Wide (16:9) headline hero for the ModelExpress / cold-start figure set, in the
**Dynamo Dark** aesthetic with the display / hero title treatment.

This is the canonical hero — a blend of two explored concepts (see
`../hero-concepts/`): the **flow diagram** of concept B is the base, and the
**editorial stat treatment** of concept C supplies the annotations.

**Takeaway:** ModelExpress streams model weights peer-to-peer over a single
glowing green RDMA lane (Source GPU → New GPU), bypassing the long, dim coral
detour down to a cold object store — an 11 s model load vs a 70 s baseline, a
6.4x faster model load / 59 s reclaimed per cold start.

## Figure Inventory

| File | Description |
|---|---|
| `images/hero-modelexpress-coldstart.png` | Fast-path flow diagram: glowing green RDMA spine Source GPU → New GPU ("Model load via MX P2P — 11s"), a dim dashed coral detour from a cold Object Store ("Cold object-store pull — 70s"), a cpu_blue Metadata Store on recessive control wires, and a giant light-weight green **6.4×** win-stat (1600×900 @2x) |
| `images/hero-modelexpress-coldstart.svg` | Vector source of the same hero |

## Type & Pathway

- **Type:** hero — a low-node-count flow diagram with one clear "aha", not a dense chart.
- **Pathway:** Python + Plotly via `plotly_dynamo`, rendered with kaleido at hero width (1600 px).
- **Title treatment:** display / hero (Helvetica Neue Light, weight 300, title case) + muted Helvetica subtitle. The giant numeral uses the same light display face.

## Color Convention (shared with the figure set)

- **green** — the fast RDMA data-plane weight-transfer path / the win ONLY: the spine, its `11s` label, and the `6.4×` stat. Boxes are never green.
- **coral** — the slow baseline / cold object-store pull: the dashed detour and the Object Store box.
- **cpu_blue** — the control-plane Metadata Store (coordination only).

The baseline path label is a light token on black (WCAG AA-safe); its coral role
is carried by the dashed detour line it rides and the coral Object Store below.

## Data (source of truth)

Model-load durations from the ModelExpress cold-start figure set (fig-4 / fig-6):

| Path | Model load | Role → color |
|---|---|---|
| Baseline (cold pull from object store, no P2P) | 70 s | slow baseline → coral |
| ModelExpress (P2P RDMA weight transfer) | 11 s | **MX win → green (single accent)** |

`6.4×` and `59 s` are derived (`70 / 11`, `70 − 11`). Scoped to the
weight-transfer phase only — cold JIT-cache warmup is a separate cold-start cost.

## Reproduce

```bash
./build.sh                 # reuses ../hero-concepts/.venv, renders PNG + SVG, lints sources
```

Or manually (reuse the shared concepts venv; do not create one inside `hero/`):

```bash
../hero-concepts/.venv/bin/python tools/gen_hero.py
```

## Design System

`tools/design_tokens.yaml` and `tools/plotly_dynamo.py` are copied verbatim from
the canonical blog-figures skill; do not fork them. Colors, surfaces, borders,
and the mono font all come from the tokens.

The build venv lives in the sibling `../hero-concepts/.venv` (shared across the
hero + its concepts), never inside `hero/`, so the hero folder can be
lint-scanned recursively without tripping on third-party package hex literals.
