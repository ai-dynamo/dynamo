# Flash Indexer Figures -- Reproduction Guide

Build instructions for all Flash Indexer blog post figures.

## Figure Inventory

All output goes to `../images/`.

| File | Description |
|------|-------------|
| `fig-1-kv-event-density.{svg,png}` | KV cache event density heatmap (Mooncake trace) |
| `fig-2-kv-event-flow.{svg,png}` | KV event pipeline: engines -> publishers -> indexer -> router |
| `fig-3-prefix-tree.{svg,png}` | Prefix-aware radix tree with worker tracking |
| `fig-4-concurrency-model.{svg,png}` | Concurrency model: sticky routing + concurrent reads |
| `fig-5-jump-search.{svg,png}` | Positional jump search with lookback |
| `fig-6-indexer-throughput.{svg,png}` | Benchmark: achieved vs. offered throughput (5 backends) |

## Prerequisites

```bash
pip3 install plotly kaleido numpy pyyaml
brew install librsvg   # for rsvg-convert (SVG -> PNG)
brew install d2        # only needed to re-render D2 sources
```

## Reproduction

### One-shot build (all figures)

```bash
./build.sh          # figures 1-6 (D2 sources already processed)
./build.sh --d2     # re-render D2 sources first, then all figures
```

### Architecture diagrams (Figures 2-5)

```bash
# From this directory (tools/):

# 1. (Optional) Re-render D2 -> raw SVG (requires d2 CLI)
d2 --layout tala event-flow.d2      event-flow-raw.svg
d2 --layout elk  radix-tree.d2      radix-tree-raw.svg
d2 --layout tala write-read-path.d2 write-read-path-raw.svg

# 2. Inject legends + padding, write to ../images/
python3 inject_legends.py

# 3. Render SVGs to 2x PNGs
rsvg-convert -z 2 ../images/fig-2-kv-event-flow.svg    -o ../images/fig-2-kv-event-flow.png
rsvg-convert -z 2 ../images/fig-3-prefix-tree.svg       -o ../images/fig-3-prefix-tree.png
rsvg-convert -z 2 ../images/fig-4-concurrency-model.svg -o ../images/fig-4-concurrency-model.png
rsvg-convert -z 2 ../images/fig-5-jump-search.svg       -o ../images/fig-5-jump-search.png
```

### Performance chart (Figure 6)

```bash
python3 gen_throughput.py ../data/sweep_plot.json
```

### KV cache event density heatmap (Figure 1)

```bash
# Synthetic data (no trace file needed):
python3 gen_heatmap.py

# Real Mooncake trace data (98 MB, download separately):
python3 gen_heatmap.py --real-data PATH_TO/kv_events_real.json
```

The real trace data is the first 5% of the
[Mooncake FAST'25 trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl),
replayed across 16 simulated workers with 2,048 blocks/worker (block_size 16).
The file is too large to include in-repo; generate it with
[`mocker`](https://github.com/ai-dynamo/dynamo/tree/main/lib/mocker) or
download the trace and replay it.

The heatmap script reads `design_tokens.yaml` and `plotly_dynamo.py` for the
Dynamo dark theme.

## Contents

```text
tools/
├── README.md                  # This file
├── build.sh                   # One-shot build for all figures
├── inject_legends.py          # SVG legend injection (Figures 2-5)
├── gen_heatmap.py             # KV event heatmap generator (Figure 1)
├── gen_throughput.py           # Throughput chart generator (Figure 6)
├── design_tokens.yaml         # Shared color/typography tokens
├── plotly_dynamo.py           # Plotly template builder
├── dynamo.d2                  # D2 theme file
├── theme.d2                   # Shared D2 theme
├── event-flow.d2              # D2 source for Figure 2
├── radix-tree.d2              # D2 source for Figure 3
└── write-read-path.d2         # D2 source for Figure 4
```
