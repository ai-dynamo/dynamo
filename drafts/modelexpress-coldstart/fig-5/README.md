# fig-5 — NIXL Registration Time Scoreboard

Dynamo Dark remake of the ModelExpress / cold-start figure comparing NIXL memory
registration time across three registration strategies.

## Figure Inventory

| File | Description |
|---|---|
| `images/fig-5-nixl-registration.png` | Compact horizontal-bar scoreboard: NIXL registration time, per-tensor default vs pool vs VMM arena, single green accent on the winner. |
| `images/fig-5-nixl-registration.svg` | Vector source (same figure) for embedding. |

## Data

Measured values transcribed from the reference figure (not invented):

| Strategy | Env flag | Time (s) | Speedup vs baseline |
|---|---|---|---|
| Per-tensor registration | (default) | 8.16 | 1× |
| Pool registration | `MX_POOL_REG=1` | 1.14 | 7.1× |
| VMM arena | `MX_VMM_ARENA=1` | 0.79 | 10.3× |

Speedups are computed in-generator as `baseline / time`.

## Pathway

Python + Plotly via the canonical `plotly_dynamo` template, rendered to PNG/SVG
with kaleido. Real benchmark data → the chart pathway.

## Restyle Decisions (reference → Dynamo Dark)

- **Title** rewritten from a category name to the takeaway, in the compact Arial
  700 ALL-CAPS chart-title treatment.
- **Palette** reduced to two semantic accents + grey: green for the winning
  strategy (VMM arena), coral for the slow baseline (loser role), token grey for
  the middle strategy. The reference's third saturated color is dropped.
- **Legend dropped** in favor of direct-labelled rows (config name in the row's
  role color + env-var flag in mono) — ≤ 5 series, so direct labels win.
- **Speedup callouts** moved to a dedicated right-edge column, sharing one anchor
  x; the winner's callout is the largest and the only green one.
- **Subtle full-scale track** behind each bar so the winner reads as a small
  fraction of the baseline extent.

## Reproduce

```bash
# uses the local .venv if present, else system python3
./build.sh
# or:
.venv/bin/python gen_fig_5_nixl_registration.py
```

Prerequisites: `plotly`, `kaleido`, `pyyaml` (installed in `.venv`).

## Design System

`design_tokens.yaml` and `plotly_dynamo.py` are copied verbatim from the
canonical blog-figures reference — do not fork the values.
