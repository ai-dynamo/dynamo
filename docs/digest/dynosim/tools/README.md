# DynoSim Figures -- Reproduction Guide

Build instructions for the DynoSim Digest post hero figure.

## Figure Inventory

| File | Description |
|------|-------------|
| `../dynosim-hero.png` | Hero: Pareto frontier of DynoSim-explored configs, with GPU-verified points on the frontier |

## Prerequisites

```bash
pip3 install plotly kaleido numpy pyyaml
```

## Reproduction

```bash
cd tools
./build.sh          # regenerates ../dynosim-hero.png
```

## Design System

`gen_hero.py` renders in the unified **Dynamo Dark** aesthetic, reading the
canonical [`design_tokens.yaml`](design_tokens.yaml) via
[`plotly_dynamo.py`](plotly_dynamo.py) (both copied from the flash-indexer
reference; do not fork). Title is ALL-CAPS Arial weight 700, ground is
`#000000`, accents are token colors, corners are square. The blog-figures
skill (internal, at `docs/skills/blog-figures/`) documents the full design
language.

## Note on the Data

The explored-config point cloud is a **deterministic, representative**
reproduction (fixed RNG seed) of the frontier's shape and axis ranges, not
the original measured DynoSim sweep (which is not committed to this repo).
Every run reproduces the same figure. The cloud is illustrative of "sweep
the space"; no individual point asserts a specific measured config result.
To render from real sweep data, replace the cloud generation in
`gen_hero.py` with a loader for the measured points.
