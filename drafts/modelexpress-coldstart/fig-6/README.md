# fig-6 — Cold Start Anatomy

Hero-width breakdown of a DeepSeek-V4-Pro / vLLM cold start, restyled from the
reference light-theme figure into the **Dynamo Dark** aesthetic.

**Takeaway:** ModelExpress delivers weights peer-to-peer in 11 s (2.6 %, the
single green accent), so the 420 s cold start is now dominated by the cold
JIT-cache warmup cluster — profiling + DeepGEMM compile + CUDA graph capture =
350 s / 83 % — rendered in the coral "cost" family.

## Figure Inventory

| File | Description |
|---|---|
| `images/fig-6-coldstart-anatomy.png` | Stacked single-row cold-start timeline with a 420 s KPI, 8 labeled phases, minute ticks, and a two-column breakdown legend (1600×840 @2x) |

## Type & Pathway

- **Type:** chart — single-row stacked timeline (Gantt-style breakdown bar) with a large KPI number.
- **Pathway:** Python + Plotly via `plotly_dynamo`, rendered to PNG with kaleido at hero width (1600 px).
- **Title treatment:** display / hero (Helvetica Neue Light, title case) + muted Helvetica subtitle.

## Data (source of truth)

Phase durations in seconds; percentages derived (`dur / 420`). Order is time order.

| Phase | Duration | Share | Role → color |
|---|---|---|---|
| Python & vLLM imports & others | 27 s | 6.4 % | setup → grey |
| Engine config & core spawn | 14 s | 3.3 % | setup → grey |
| Worker spawn & distributed init | 17 s | 4.0 % | setup → grey |
| Model load via MX P2P | 11 s | 2.6 % | **MX win → green (single accent)** |
| Memory profiling (JIT wave) | 100 s | 23.8 % | cost → coral (muted) |
| JIT warmup (DeepGEMM compile) | 142 s | 33.8 % | cost peak → coral |
| CUDA graph capture | 108 s | 25.7 % | cost → coral (muted) |
| API server ready | 1 s | 0.2 % | setup → grey |
| **Total** | **420 s (7 m)** | | |

Color encodes role: grey recedes for process/infra setup, green marks the single
ModelExpress win, and the coral family carries the cold-JIT-cache cost cluster
(the peak DeepGEMM phase is the brightest coral).

## Reproduce

```bash
./build.sh                 # bootstraps .venv, renders PNG, lints sources
```

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install plotly==6.9.0 kaleido==1.3.0 PyYAML==6.0.3
.venv/bin/python gen_fig_6_coldstart_anatomy.py
```

## Design System

`design_tokens.yaml` and `plotly_dynamo.py` are copied verbatim from the
canonical blog-figures skill (`docs/skills/blog-figures/examples/`); do not fork
them. Colors, surfaces, borders, and the mono font all come from the tokens.

`.venv/` and `__pycache__/` are build artifacts and are not part of the figure
source; `build.sh` recreates the venv on demand.
