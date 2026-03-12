# aiperf Result Plotter

Generates latency and throughput plots from `profile_export_aiperf.json` benchmark results.

## Usage

```bash
# Single dataset directory
python3 -m benchmarks.multimodal.plots \
  recipes/qwen3-vl-30b-fp8/embedding_cache/agg/results/gb200-4gpu/1000req_3img_200pool_400word_base64

# Recursive — finds all dataset dirs under a parent
python3 -m benchmarks.multimodal.plots \
  recipes/qwen3-vl-30b-fp8/embedding_cache

# Custom output directory (preserves relative hierarchy)
python3 -m benchmarks.multimodal.plots \
  -o recipes/qwen3-vl-30b-fp8/embedding_cache/plots \
  recipes/qwen3-vl-30b-fp8/embedding_cache
```

When `-o` is not specified, plots are saved to `<dataset_dir>/plots/`.

## Expected directory structure

```
<dataset_dir>/
  <line_name>/          # e.g. cache_on, cache_off
    <x_value>/          # e.g. c4, c8, c16, c32
      profile_export_aiperf.json
```

- **line_name** directories become separate lines on each plot.
- **x_value** directories provide x-axis data points. The alphabetic prefix is split from the numeric value (e.g. `c32` → label `c`, value 32). If all directories share the same prefix, it becomes the x-axis label.

Recursive mode auto-detects dataset directories at any depth and skips `plots/` directories.

## Generated plots

Each dataset directory produces 5 PNG files:

| File | Type | Description |
|---|---|---|
| `request_throughput.png` | single | Request throughput (req/s) |
| `output_token_throughput.png` | single | Output token throughput (tok/s) |
| `request_latency.png` | 2×2 grid | avg, p50, p90, p99 |
| `time_to_first_token.png` | 2×2 grid | avg, p50, p90, p99 |
| `inter_token_latency.png` | 2×2 grid | avg, p50, p90, p99 |
