<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Baseline Inputs and Existing Harnesses

- Date: 2026-08-28
- Question: Which existing workloads and benchmark behavior should the Planner
  POC baseline reuse without changing deployed infrastructure?

## Findings

- `llm-d-batch-gateway/benchmarks/benchmark.py` already submits Batch jobs,
  polls request counts, drives GuideLLM burst/idle traffic, and queries selected
  Prometheus metrics. Its managed scenarios also install and remove Kubernetes
  resources, so it is not the right execution boundary for this baseline.
- `llm-d-batch-gateway/benchmarks/generate_prompts.py` generates synthetic prompt
  distributions. The POC instead needs the user's fixed GSM8K workload.
- `../run_example.py` implements the required OpenAI-compatible file upload,
  Batch creation, polling, cancellation, and output retrieval flow. It is a
  validation client rather than an evidence-preserving benchmark harness and has
  user-authored uncommitted changes, so this workstream does not modify or import
  it.
- `../llm-d-async-values.yaml` pins llm-d Async `v0.9.0`, Redis sorted-set
  dispatch, a 100 ms poll interval, a batch size of eight, and `gateType:
  constant`. The baseline must capture this effective configuration and leave it
  unchanged.
- `datasets/gsm8k/convert_to_batch_gateway.py`
  creates deterministic OpenAI-compatible JSONL with temperature zero, streaming
  disabled, and a fixed output-token limit. The generated main/test file has
  1,319 records and is suitable as the immutable source.

## Sources

- llm-d Batch Gateway benchmark README and scripts, local checkout, revision
  `v0.5.0-8-g229674c`, accessed 2026-08-28.
- Dynamo llm-d Batch Gateway example README, client, and values files, local
  checkout revision `17583944d48185f9d8de82e2c21ede45bc9039d7`, accessed
  2026-08-28.
- GSM8K dataset card, converter, Parquet shards, and generated Batch JSONL under
  `datasets/gsm8k`, accessed 2026-08-28.

## Relevance to the Experiment

The new harness should reuse the API lifecycle and evidence concepts while
avoiding scenario deployment, teardown, synthetic prompt generation, or changes
to the existing client and manifests.

## Uncertainty

- The live pod names and ConfigMap references may differ from the example. The
  harness must make pod selection configurable and preserve discovery failures.
- Metric endpoints may not be locally forwarded. Explicit metric URLs are
  optional, while Kubernetes pod-proxy metric snapshots are best effort.

## Resulting Decisions

- Normalize a deterministic GSM8K slice into each raw run.
- Drive online traffic with a small standard-library streaming client so the raw
  evidence includes TTFT without installing another benchmark package.
- Make every Kubernetes interaction read-only and capture command exit codes.
