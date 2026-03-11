# MM Router Benchmark Kit

Multimodal router A/B benchmark: compares MM-aware routing vs round-robin baseline.

## Prerequisites

1. Dynamo repo checked out and built (with vLLM support)
2. etcd + NATS running: `docker compose -f deploy/docker-compose.yml up -d`
3. `aiperf` installed: `pip install aiperf`
4. GPU(s) available

## Quick Start

```bash
cd benchmarks/multimodal/perf_data

# 1. Generate benchmark images (one-time, ~3.7 GB in /tmp/bench_images_512/)
bash setup_bench_images.sh

# 2. Run benchmarks
#    Local (2 workers, Qwen3-VL-2B):
bash run_all_benchmarks.sh

#    Cluster (8 workers, Qwen3-VL-30B-A3B-FP8):
bash run_cluster_benchmarks.sh
```

Paths are auto-detected from the script location. Override with env vars if needed:
```bash
DYNAMO_ROOT=/path/to/dynamo PERF_DIR=/path/to/datasets bash run_all_benchmarks.sh
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_all_benchmarks.sh` | Full A/B sweep: b64 + localhost datasets, 2 workers, Qwen3-VL-2B |
| `run_cluster_benchmarks.sh` | Cluster variant: b64 datasets only, 8 workers, Qwen3-VL-30B-A3B-FP8 |
| `setup_bench_images.sh` | Generates random PNG images to `/tmp/bench_images_512/` |

All run scripts call `examples/backends/vllm/mm_router_worker/run_aiperf_ab.sh` internally.

## JSONL Datasets

Naming convention: `{count}req_{imgs}img_{reuse}pct_reuse_{format}_fixprompt.jsonl`

- **b64_512**: images at `/tmp/bench_images_512/img_XXXX.png` (run `setup_bench_images.sh` first)
- **localhost**: images at `http://127.0.0.1:9999/img_XXXX.png` (requires local image HTTP server)
- **http**: images from COCO dataset URLs (requires internet access)

### Serving images for `localhost` datasets

```bash
cd /tmp/bench_images_512
python3 -m http.server 9999
```

## Results

Results are saved to `results_<timestamp>/` (or `cluster_results_<timestamp>/`) under `PERF_DIR`.
Each dataset produces sub-directories with `baseline/` (round-robin) and `mm/` (MM router) results.
