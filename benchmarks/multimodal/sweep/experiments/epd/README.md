# Aggregated / colocated EPD sweep

This experiment compares Aggregated and colocated EPD topologies for
Dynamo vLLM and SGLang.

Test with `vllm==0.26.0` with the
`vllm/vllm-openai:v0.26.0-ubuntu2404` runtime image and `sglang==0.5.16` with
`lmsysorg/sglang:v0.5.16-cu130-runtime`.

## Download images

```bash
cd benchmarks/multimodal/sweep/experiments/epd
python download_dataset.py \
  --output-dir /to/your/path
```

The downloader downloads 50 images by default. Pass `--count N`, where
`1 <= N <= 50`, to download fewer.

## Run the benchmark

### Run a single workload

```bash
# sglang or vllm
python run_experiment.py \
  --backend sglang \
  --topology aggregate epd \
  --image-count 5 \
  --osl 128 \
  --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --image-dir /to/your/path/images \
  --output-dir /to/your/path/results
```

### Full sweep

```bash
# sglang or vllm
python run_experiment.py \
  --backend sglang \
  --topology aggregate epd \
  --image-count 5,10,30 \
  --image-token-budget 128 256 \
  --osl 128,512,2048 \
  --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --image-dir /to/your/path/images \
  --output-dir /to/your/path/results
```

### Replay fixed request arrivals

Pass `--schedule-file` to run the same timed workload in every selected
backend and topology. Write one JSON object per line with `timestamp` and
`payload` fields. Timestamps are nonnegative milliseconds from the start of
measurement and must be in nondecreasing order. Equal timestamps form a
burst. Each payload is a complete streaming chat request whose `model` matches
`--served-model-name` (by default, the last segment of `--model`).

```jsonl
{"timestamp":0,"payload":{"model":"Qwen3.5-122B-A10B-NVFP4","stream":true,"stream_options":{"include_usage":true},"max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"Describe the image."},{"type":"image_url","image_url":{"url":"https://example.com/first.png"}}]}]}}
{"timestamp":40,"payload":{"model":"Qwen3.5-122B-A10B-NVFP4","stream":true,"stream_options":{"include_usage":true},"max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"Read the text."},{"type":"image_url","image_url":{"url":"https://example.com/second.png"}}]}]}}
```

Replace the example URLs with images reachable by all backends, or use data
URLs. Choose request options supported by each backend in the comparison.

```bash
python run_experiment.py \
  --backend vllm sglang \
  --topology aggregate epd \
  --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --schedule-file /to/your/path/schedule.jsonl \
  --output-dir /to/your/path/results
```

Replay uses the file's payloads and arrivals without a warmup phase.
`--image-dir`, `--image-count`, `--isl`, `--osl`, `--qps`, `--tokenizer`,
and `--image-token-budget` are for generated workloads.

Each `fixed-schedule` cell saves the input as `input.jsonl` and keeps AIPerf's
raw exports. A successful cell records the input SHA-256, achieved requests
per second, and AIPerf's p99 send lag in milliseconds when available. AIPerf
measures this lag relative to the least-late request in the run. A cell with
errors or an incomplete run fails the result gate and leaves its available
exports for diagnosis.
