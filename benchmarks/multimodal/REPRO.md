# Multimodal Benchmark Reproduction Guide

End-to-end steps for reproducing multimodal benchmarks: data generation, serving (vLLM baseline and Dynamo), and profiling with aiperf.

## 1. Generate Benchmark Data

Use the JSONL generator in `benchmarks/multimodal/jsonl/`. Requires PR [#8201](https://github.com/ai-dynamo/dynamo/pull/8201) for sliding-window support.

Each user has a causal session; the image window slides by 1 each turn, giving 4/5 overlap between consecutive requests from the same user:

```
user_0, turn 0: [img0, img1, img2, img3, img4]
user_0, turn 1: [img1, img2, img3, img4, img5]   <- 4/5 overlap
user_0, turn 2: [img2, img3, img4, img5, img6]   <- 4/5 overlap
```

```bash
cd benchmarks/multimodal/jsonl

python main.py sliding-window \
  --num-users 10 \
  --turns-per-user 10 \
  --window-size 5 \
  --user-text-tokens 300 \
  --image-size 512 512 \
  --image-dir /dynamo-tmp/data/bench_images \
  --seed 42
# Output: 10u_10t_5w_300word_base64.jsonl (100 total requests)
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `--num-users` | Concurrent user sessions (default: 10) |
| `--turns-per-user` | Requests per user (default: 20) |
| `--window-size` | Images per request; window slides by 1 each turn (default: 5) |
| `--image-size W H` | PNG dimensions in pixels (default: 512 512) |
| `--image-mode` | `base64` (default, local PNGs) or `http` (COCO URLs) |
| `--image-dir` | Where to store generated PNGs. Use a persistent path, not `/tmp` |

### Important

- **Generate data on the remote cluster**, not locally. Image paths in the JSONL are absolute and must resolve inside the container.
- Use `--image-dir` pointing to a persistent mount (e.g. `/dynamo-tmp/data/bench_images`), not `/tmp` which is ephemeral between srun steps.

## 2. Run vLLM Serve Baseline

Plain `vllm serve` without Dynamo. This is the primary baseline for comparison.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 30 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --port 8000
```

> **Note:** For now we focus on plain `vllm serve` as the baseline. The embedding cache comparison (`--ec-transfer-config`) is not needed yet but the command is here for reference:
>
> ```bash
> vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
>   ... \
>   --ec-transfer-config '{
>     "ec_role": "ec_both",
>     "ec_connector": "DynamoMultimodalEmbeddingCacheConnector",
>     "ec_connector_module_path": "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector",
>     "ec_connector_extra_config": {"multimodal_embedding_cache_capacity_gb": 30}
>   }'
> ```

## 3. Run Dynamo Serve (Aggregated + Embedding Cache)

Dynamo wraps vLLM with its own frontend, tool/reasoning parsers, and embedding cache.

```bash
# Frontend (HTTP endpoint)
python -m dynamo.frontend &

# Backend (vLLM engine)
python -m dynamo.vllm \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --dyn-tool-call-parser qwen3_coder \
  --dyn-reasoning-parser qwen3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-multimodal \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 30 \
  --multimodal-embedding-cache-capacity-gb 30 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384
```

### Key differences from vllm serve

| vllm serve flag | Dynamo equivalent |
|-----------------|-------------------|
| `--enable-auto-tool-choice` | Not needed; use `--dyn-tool-call-parser` |
| `--tool-call-parser X` | `--dyn-tool-call-parser X` |
| `--reasoning-parser X` | `--dyn-reasoning-parser X` |
| `--port 8000` | Handled by `dynamo.frontend` (default 8000) |
| `--ec-transfer-config {...}` | `--multimodal-embedding-cache-capacity-gb 30` |
| _(not needed)_ | `--enable-multimodal` (required for image requests) |

### Environment variables for multimodal

These must be set when running Dynamo with large multimodal payloads:

```bash
export DYN_TCP_MAX_MESSAGE_SIZE=209715200  # 200MB (default 32MB too small for base64 images)
export DYN_HTTP_BODY_LIMIT_MB=200          # 200MB HTTP body limit (default 45MB)
export HF_HUB_OFFLINE=1                    # Skip HF API calls, use local cache
```

## 4. Install and Run aiperf

### Install inside container

```bash
pip install --no-deps -e /aiperf
```

If `/aiperf` is not mounted, install from the repo:

```bash
pip install git+https://github.com/ai-dynamo/aiperf.git
```

### Run benchmark

```bash
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1200 \
AIPERF_DATASET_CONFIGURATION_TIMEOUT=1200 \
aiperf profile \
  -m Qwen/Qwen3.5-397B-A17B-FP8 \
  -u http://localhost:8000 \
  --request-rate 1 \
  --request-count 100 \
  --input-file /dynamo-tmp/data/10u_10t_5w_300word_base64.jsonl \
  --custom-dataset-type single_turn \
  --extra-inputs max_tokens:1024 \
  --extra-inputs min_tokens:1024 \
  --extra-inputs ignore_eos:true \
  --streaming \
  --artifact-dir /dynamo-tmp/logs/04-15/my_benchmark \
  --ui none \
  --no-server-metrics
```

### Key aiperf parameters

| Parameter | Description |
|-----------|-------------|
| `--request-rate` | Requests per second (Poisson arrival). Use 0.2 for low-load, 1+ for stress |
| `--request-count` | Total requests to send |
| `--input-file` | JSONL from step 1 |
| `--custom-dataset-type single_turn` | Required for our JSONL format |
| `--extra-inputs max_tokens:1024` | Output sequence length |
| `--extra-inputs min_tokens:1024` | Force exact OSL (with `ignore_eos:true`) |
| `--artifact-dir` | Where results are saved |
| `--ui none` | Disable TUI (for non-interactive runs) |

### Configure timeouts

The `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` and `AIPERF_DATASET_CONFIGURATION_TIMEOUT` (both in seconds) must be set high enough for large datasets. Base64 encoding of images during configure can take several minutes. Default is too low for datasets with many large images.

### Results

Results are saved to `--artifact-dir`:
- `profile_export_aiperf.json` — full metrics (use `time_to_first_token`, `inter_token_latency`, `total_token_throughput`)
- `profile_export_aiperf.csv` — tabular export
- `inputs.json` — resolved inputs for reproducibility
- `logs/aiperf.log` — detailed execution log

**Do not use** `http_req_time_to_first_byte` — it measures network-level first byte (~9ms), not real TTFT which includes vision preprocessing.
