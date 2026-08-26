<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen2.5 CustomEncoder Benchmark

This benchmark compares two aggregated, single-GPU implementations of the same performance-only Qwen2.5 vision-to-text pipeline:

- `custom-worker-control` uses a Dynamo `LLMEngine` that owns the vision backend and vLLM's offline `LLM`. A dedicated actor collects up to eight requests for 1 ms, then runs image preprocessing, one batched vision pass per image shape, and one blocking `LLM.generate()` call. The next batch cannot begin preprocessing until generation returns.
- `dynamo-vllm-custom` uses the stock aggregated `dynamo.vllm --custom-encoder-class` path. Its encoder and decoder schedule independently through Dynamo and vLLM.

Both arms use `Qwen/Qwen2.5-VL-3B-Instruct` for the vision tower and `Qwen/Qwen2.5-1.5B-Instruct` for text generation. The encoder truncates the native 2048-wide visual output to 1536 columns so it fits the decoder. This is an untrained, performance-only transformation with no model-quality or output-parity claim.

## Workload Contract

The audited workload contains:

- 1000 measured requests and 20 excluded warmups per arm and repetition
- closed-loop concurrency 64
- one shared 644-token raw text prompt and one unique JPEG per measured request
- 500 300x300 images and 500 500x500 images
- decoder input sequence lengths 773 and 976, with an average of 874.5
- exactly seven greedily generated tokens
- canonical measured JSONL SHA-256 `743e859f895ee0e22df2476f74e5d3fa4d48db059273f5fe517634f31d9ef7cc`

Each arm captures the 300x300 and 500x500 vision shapes at batch buckets 1, 2, 4, and 8. The runner requires eight graph captures, 907800 processed image patches across warmup plus measurement, prefix caching, KV-event publication, zero request errors, and matching smoke-test token IDs between arms.

## Run the Comparison

Run inside a Dynamo vLLM development container on one H100 with the audited workload already present:

```bash
export DYN_BENCH_OUTPUT_ROOT=/workspace/logs/qwen25-custom-encoder-comparison
export DYN_BENCH_CONTAINER_IMAGE="${DYN_BENCH_CONTAINER_IMAGE:-unknown}"
export DYN_BENCH_SOURCE_COMMIT="$(git rev-parse HEAD)"

./examples/custom_encoder/benchmark/run_qwen2_5_vl_comparison.sh
```

Override `DYN_BENCH_WORKLOAD_ROOT` if the workload is not at the default audited path. The output directory contains every server and AIPerf log, per-run metrics, source and workload hashes, GPU provenance, `summary.json`, and `report.md`.

Use the launchers directly for a smoke test:

```bash
./examples/custom_encoder/launch/agg_qwen2_5_vl_control.sh \
    --enable-prefix-caching \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

```bash
./examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh \
    --enable-prefix-caching \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```
