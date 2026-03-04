# Multimodal Disaggregated Serving Benchmark

## Model
- **Model**: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- **ISL**: ~1111 tokens (text + image)
- **OSL**: ~58 tokens
- **Dataset**: `data_small.jsonl` (single-turn, 1 image + text prompt)

## Benchmark Command

```bash
aiperf profile \
  -m 'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8' \
  --endpoint-type 'chat' \
  -u 'localhost:8000' \
  --streaming \
  --request-count 100 \
  --warmup-request-count 2 \
  --concurrency 16 \
  --osl 500 \
  --input-file '/tmp/data_small.jsonl' \
  --custom-dataset-type 'single_turn' \
  --ui None \
  --no-server-metrics
```

> Adjust `--concurrency` to 16, 32, or 64 as needed.

## Configurations

### 1. Aggregated (Agg) — 2 workers on 2 GPUs

```bash
cd /workspace/examples/backends/trtllm/launch
bash agg_multimodal.sh
```

- Toggle workers in `agg_multimodal.sh` to add/remove agg workers.
- Engine config: `engine_configs/qwen3-vl-30b-a3b-instruct-fp8/agg.yaml`

### 2. Prefill-Decode (PD) — 1 prefill + 1 decode on 2 GPUs

```bash
cd /workspace/examples/backends/trtllm/launch
bash disagg_multimodal.sh
```

- Toggle workers in `disagg_multimodal.sh` to add/remove prefill/decode workers.
- Engine configs: `prefill.yaml` and `decode.yaml`

### 3. Encode-Prefill-Decode (EPD) — encode + prefill + decode on 2 GPUs

```bash
cd /workspace/examples/backends/trtllm/launch
bash epd_perf.sh
```

- Toggle encode workers in `epd_perf.sh` to scale vision encoding independently.
- Engine configs: `encode.yaml`, `prefill.yaml`, and `decode.yaml`

## Setup Notes

1. Copy `data_small.jsonl` to `/tmp/data_small.jsonl` on the target machine.
2. All launch scripts are in `examples/backends/trtllm/launch/`.
3. All engine configs are in `examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct-fp8/`.
4. Verify workers are running with `nvidia-smi` (check GPU memory on both GPUs).
