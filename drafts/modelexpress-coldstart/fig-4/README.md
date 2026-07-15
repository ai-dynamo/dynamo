# fig-4 — Cold Start Time with P2P RDMA and Kernel Artifacts

Horizontal stacked-bar (Gantt-style) comparison of three cold-start paths,
restyled from the ModelExpress cold-start blog reference into the Dynamo Dark
aesthetic.

| File | Description |
|---|---|
| `gen_fig_4_coldstart.py` | Regenerable Plotly generator (data + layout). |
| `design_tokens.yaml` | Canonical Dynamo Dark tokens (copied, do not fork). |
| `plotly_dynamo.py` | Canonical Plotly template (copied, do not fork). |
| `build.sh` | Creates `.venv`, renders `images/fig-4-coldstart-phases.{png,svg}`, lints sources. |
| `images/fig-4-coldstart-phases.png` | 3x raster render. |
| `images/fig-4-coldstart-phases.svg` | Vector render. |

## Data (measured phase durations, seconds)

| Path | Model Loading | Kernel Warmup, Graph Capture, KV Warmup | Others | Total | Speed-up |
|---|---|---|---|---|---|
| Baseline (cold start from VAST, no P2P source) | 1m 10s | 5m 49s | 1m 2s | 8m 1s | — |
| RDMA (P2P RDMA weights only) | 11s | 5m 50s | 59s | 7m | 1.1× |
| RDMA + CACHE (P2P RDMA weights + kernel artifacts) | 9s | 36s | 59s | 1m 44s | 4.6× |

## Reproduce

```bash
./build.sh
```

## Design notes

- **Pathway:** Python + Plotly via `plotly_dynamo` → PNG/SVG (kaleido). Real
  data (durations) → chart pathway.
- **Palette:** Model Loading = CPU blue, warmup = fluorite gold, Others =
  garnet. Dynamo green is reserved for the single winning row (the
  RDMA + kernel-artifact-cache badge and its 4.6× speed-up), so the accent
  points straight at the takeaway.
- **Title:** compact / chart treatment (Arial 18 px, weight 700, uppercase).
- **Legend:** bottom-center per the house multi-figure discipline.

## Lint

```bash
python3 ../../../docs/skills/blog-figures/tools/lint_figures.py \
  gen_fig_4_coldstart.py plotly_dynamo.py --score
```

Scores 100/100 (measured dimensions), zero ERROR/WARN. Lint the source files,
not the `.venv` build artifact.
