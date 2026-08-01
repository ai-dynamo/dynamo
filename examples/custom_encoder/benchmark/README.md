<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Paired custom-encoder HTTP benchmark

This benchmark runs the same unique JPEG through two services at once:

- the aggregated custom-encoder plus Qwen2.5-1.5B service;
- a standalone custom encoder with no language model or OpenAI protocol.

For concurrency `x`, the client owns `x` independent pair lanes. Each lane starts
one request against each service and does not take its next image until both
responses finish. There are therefore at most `x` requests at each endpoint and
`2x` HTTP requests in flight overall.

## Encoder-only service

The encoder endpoint accepts one raw JPEG in the multipart field `image`:

```text
POST /encode
Content-Type: multipart/form-data

image: image/jpeg
```

After the vision encoder completes it returns the ten-byte plain-text body
`encoder-ok`. It does not parse prompts, count tokens, generate text, or return an
OpenAI response. The uploaded JPEG bytes are passed directly to the shared Qwen
encoder loader, so this path has no JSON or base64 conversion.

Start it on port 8001:

```bash
python -m examples.custom_encoder.benchmark.encoder_only_server \
  --host 0.0.0.0 \
  --port 8001 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-queue-delay-us 1000
```

The process is ready when `GET /health` returns `{"status":"ready"}`.

## Aggregated service

Start the existing performance-only Qwen2.5 stack on port 8000:

```bash
export DYN_HTTP_PORT=8000
export DYN_SYSTEM_PORT=8081
export DYN_QWEN2_VL_ENCODER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE=1536
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_PATCHES=$((32 * 36 * 36))
export DYN_QWEN2_VL_MAX_BATCH_ITEMS=64
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8,16,32,64
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=300x300,500x500
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0

bash examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh \
  --custom-encoder-max-queue-delay-us 1000 \
  --gpu-memory-utilization 0.4 \
  --no-enable-prefix-caching
```

## Workload

Generate 1,000 unique, randomly mixed images with exact ISL 644 prompts:

```bash
export WORKLOAD_DIR=/dynamo-tmp/logs/paired-custom-encoder/workload

python -m examples.custom_encoder.benchmark.safeguard_proxy_workload generate \
  --output-dir "$WORKLOAD_DIR" \
  --requests 1000 \
  --image-size-count 300x300:500 \
  --image-size-count 500x500:500 \
  --concurrencies 64
```

Generated 300x300 JPEGs are 7 KiB plus or minus 256 bytes. Generated 500x500
JPEGs are 35 KiB plus or minus 256 bytes. No trailing padding is added.

## Run

```bash
python -m examples.custom_encoder.benchmark.run_paired_http_benchmark \
  --input-file "$WORKLOAD_DIR/image_custom_1000_isl644.jsonl" \
  --concurrency 64 \
  --encoder-url http://localhost:8001/encode \
  --aggregated-url http://localhost:8000/v1/chat/completions \
  --output-file /dynamo-tmp/logs/paired-custom-encoder/result.json
```

The client loads and audits the entire workload before timing. Encoder-only
requests use multipart JPEG bytes. Aggregated requests use the same bytes in an
OpenAI image data URI and request exactly seven output tokens without streaming.
Any failed pair invalidates the run; requests are never retried.

The primary result is `paired_images_per_second`, calculated as the number of
completed image pairs divided by total wall time. The report intentionally does
not calculate `2,000 / wall_time` as a combined throughput.
