# aiperf Result Plotter

Generates latency and throughput plots from `profile_export_aiperf.json` benchmark results.

## Usage

```bash
# Recursive — finds all dataset dirs under a parent
python3 -m benchmarks.multimodal.plots \
  recipes/qwen3-vl-30b-fp8/embedding_cache

# Custom output directory (preserves relative hierarchy)
python3 -m benchmarks.multimodal.plots \
  -o recipes/qwen3-vl-30b-fp8/embedding_cache/plots \
  recipes/qwen3-vl-30b-fp8/embedding_cache
```

When `-o` is not specified, plots are saved to `<dataset_dir>/plots/`.

With `-o`, the relative path from the input root is preserved.

## Generated plots

Each dataset directory produces 5 PNG files:

| File | Type | Description |
|---|---|---|
| `request_throughput.png` | single | Request throughput (req/s) |
| `output_token_throughput.png` | single | Output token throughput (tok/s) |
| `request_latency.png` | 2×2 grid | avg, p50, p90, p99 |
| `time_to_first_token.png` | 2×2 grid | avg, p50, p90, p99 |
| `inter_token_latency.png` | 2×2 grid | avg, p50, p90, p99 |

- **line_name** directories (e.g. `cache_on`, `cache_off`) become separate lines on each plot.
- **x_value** directories provide x-axis data points. The alphabetic prefix is split from the numeric value (e.g. `c32` → label `c`, value 32). If all directories share the same prefix, it becomes the x-axis label.

## Example

Given this input tree (recursive mode auto-detects dataset directories at any depth and skips `plots/` directories):

```
recipes/qwen3-vl-30b-fp8/embedding_cache/
  agg/results/
    gb200-4gpu/
      1000req_3img/                       # ← dataset dir
        cache_on/                         # ← line_name
          c4/profile_export_aiperf.json   # ← x_value
          c8/profile_export_aiperf.json
          c16/profile_export_aiperf.json
        cache_off/                        # ← line_name
          c4/profile_export_aiperf.json
          c8/profile_export_aiperf.json
          c16/profile_export_aiperf.json
      500req_1img/                        # ← dataset dir
        cache_on/
          c4/profile_export_aiperf.json
          c8/profile_export_aiperf.json
```

Running:

```bash
python3 -m benchmarks.multimodal.plots \
  -o plots_out \
  recipes/qwen3-vl-30b-fp8/embedding_cache
```

Produces:

```
plots_out/
  agg/results/gb200-4gpu/
    1000req_3img/
      request_throughput.png
      output_token_throughput.png
      request_latency.png
      time_to_first_token.png
      inter_token_latency.png
    500req_1img/
      request_throughput.png
      ...
```
