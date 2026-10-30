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

## Run a Live AIPerf Demo

For a side-by-side demo, run both implementations either on one two-GPU H100 node or on two statically matched one-GPU H100 nodes. Pin each server to one GPU and, when sharing a node, give the sides distinct HTTP, system-metrics, KV-event, namespace, and cache settings. Start the side-specific AIPerf commands in separate terminals:

```bash
./examples/custom_encoder/benchmark/run_qwen2_5_vl_demo_aiperf.sh control
```

```bash
./examples/custom_encoder/benchmark/run_qwen2_5_vl_demo_aiperf.sh dynamo-vllm
```

The demo runner verifies its selected GPU's model, memory, power limit, maximum SM clock, visible-device count, and workload hash. It executes 20 excluded warmups, waits for the other terminal at a shared-filesystem barrier, then starts both live measurements together. Each side records one-second GPU telemetry and writes to a timestamped artifact directory; both terminals finish with the paired throughput and latency comparison. Set `DYN_DEMO_CONVERSATIONS`, `DYN_DEMO_CONCURRENCY`, `DYN_DEMO_UI`, or `DYN_DEMO_STATS_INTERVAL` to override the default 1000 requests, concurrency 64, lightweight live UI, or five-second metrics refresh.

For the most controlled recording layout, co-locate both warm servers on one H100 with distinct ports and namespaces, set `DYN_VLLM_GPU_MEMORY_UTILIZATION=0.2`, and export `DYN_DEMO_SERIAL_SIDES=1` in both panels. The user still enters `demo-aiperf` once in each terminal. The control panel runs first; the optimized panel visibly waits, then starts automatically after control has released the GPU. This avoids cross-node performance variance while retaining live AIPerf output in both terminals.

For a recording-ready terminal, run `run_qwen2_5_vl_demo_panel.sh` with `control` or `dynamo-vllm`. It starts the server, waits for `/v1/models`, then opens an interactive Bash shell using `demo_shell_rc.sh`. The shell displays a labeled prompt and provides `demo-aiperf` as the only command needed to start that side's run.

Use `run_qwen2_5_vl_demo_server.sh` to start the isolated server processes. The defaults are control on GPU 0 and port 8000, and `dynamo.vllm` on GPU 1 and port 8001. `demo_same_node_layout.py` divides the SLURM CPU allocation symmetrically between the two arms.

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
