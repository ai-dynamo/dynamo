# LoRA Placement Figures -- Reproduction Guide

Build instructions for all LoRA Placement blog post figures.

## Figure Inventory

All output goes to `../images/`.

| File | Description |
|------|-------------|
| `fig-1-architecture-overview.{svg,png}` | System architecture: control plane, shared state, data plane, workers |
| `fig-2-control-loop.{svg,png}` | Feedback loop: requests to estimator to controller to table to filter |
| `fig-3-mcf-bipartite.{svg,png}` | MCF bipartite flow network: source to LoRA to worker to sink |
| `fig-4-total-churn.{svg,png}` | Total churn comparison: MCF vs HRW vs Random across 4 load patterns |
| `fig-5-churn-free-ratio.{svg,png}` | Churn-free tick ratio: % of zero-churn ticks per algorithm |
| `fig-6-spike-timeline.{svg,png}` | Per-tick churn timeline for the Traffic Spikes scenario |
| `fig-7-churn-efficiency.{svg,png}` | MCF vs HRW churn reduction percentage across all scenarios |

## Prerequisites

```bash
pip3 install plotly kaleido numpy pyyaml
brew install librsvg   # for rsvg-convert (SVG -> PNG)
brew install d2        # only needed to re-render D2 sources
```

## Reproduction

### One-shot build (all figures)

```bash
./build.sh          # figures 1-7 (D2 sources already processed)
./build.sh --d2     # re-render D2 sources first, then all figures
```

### Architecture diagrams (Figures 1-3)

```bash
# From this directory (tools/):

# 1. (Optional) Re-render D2 -> raw SVG (requires d2 CLI)
d2 --layout tala architecture-overview.d2  architecture-overview.svg
d2 --layout tala control-loop.d2           control-loop.svg
d2 --layout tala mcf-bipartite.d2          mcf-bipartite.svg

# 2. Inject legends + padding, write to ../images/
python3 inject_legends.py

# 3. Render SVGs to 2x PNGs
rsvg-convert -z 2 ../images/fig-1-architecture-overview.svg -o ../images/fig-1-architecture-overview.png
rsvg-convert -z 2 ../images/fig-2-control-loop.svg          -o ../images/fig-2-control-loop.png
rsvg-convert -z 2 ../images/fig-3-mcf-bipartite.svg         -o ../images/fig-3-mcf-bipartite.png
```

### Simulation charts (Figures 4-7)

```bash
# Churn bars + efficiency chart (Figures 4, 5, 7):
python3 gen_churn_bars.py

# Spike timeline (Figure 6):
python3 gen_spike_timeline.py
```

Chart data is hardcoded from the LoRA allocation simulation results
(8 workers, K=4 slots, 32 total, 100 LoRAs, 200 ticks, Zipf s=1.0).

## Shared Tools

The following files are symlinked from the Flash Indexer blog tools to
maintain a single source of truth for the Dynamo dark design system:

- `design_tokens.yaml` -- Color and typography tokens
- `plotly_dynamo.py` -- Plotly template builder
- `dynamo.d2` / `theme.d2` -- D2 diagram theme

## Contents

```text
tools/
    README.md                      # This file
    build.sh                       # One-shot build for all figures
    gen_diagrams.py                # Architecture SVG generator (Figures 1-3)
    inject_legends.py              # SVG legend injection (Figures 1-3, for D2 path)
    gen_churn_bars.py              # Churn comparison charts (Figures 4, 5, 7)
    gen_spike_timeline.py          # Spike timeline chart (Figure 6)
    design_tokens.yaml             # Symlink: shared color/typography tokens
    plotly_dynamo.py               # Symlink: Plotly template builder
    dynamo.d2                      # Symlink: D2 theme file
    theme.d2                       # Symlink: shared D2 theme
    architecture-overview.d2       # D2 source for Figure 1
    control-loop.d2                # D2 source for Figure 2
    mcf-bipartite.d2               # D2 source for Figure 3
```
