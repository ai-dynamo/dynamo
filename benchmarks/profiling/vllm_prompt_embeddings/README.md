<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM prompt-embedding nsys profiler

This benchmark profiles the synchronous, offline vLLM `LLM.generate` path on
one GPU. It creates one deterministic BF16 CPU tensor with shape
`[515, hidden_size]`, reuses that exact tensor for every request, and requires
75 greedily decoded output tokens. It does not call an HTTP API or tokenize
text; the tensor is synthetic engine input intended for controlled profiling.

The defaults run 20 unprofiled warmup requests followed by 100 sequential,
batch-one measured requests with prefix caching enabled. Full CUDA graphs are
requested for batch size one, the prefix-cache remainder, and the complete
515-token prefill. The post-run audit fails unless every measured prefill and
decode range contains a CUDA graph launch.

## Requirements

- One CUDA GPU supported by vLLM; the reference workload targets an H100.
- vLLM with prompt embeddings and full CUDA graph support.
- NVIDIA Nsight Systems (`nsys`) available on `PATH`.
- The model already available locally when offline mode is enabled.

## Run with nsys

From the Dynamo repository root:

```bash
benchmarks/profiling/vllm_prompt_embeddings/run_nsys.sh \
    qwen25-1.5b-prompt-embeds
```

The run directory defaults to
`logs/nsys/vllm_prompt_embeddings/<run-id>` and must not already exist. Pass a
second positional argument or set `DYN_NSYS_OUTPUT_BASE` to store it elsewhere.

Common overrides are environment variables:

```bash
DYN_PROMPT_EMBEDS_REQUESTS=2 \
DYN_PROMPT_EMBEDS_WARMUP_REQUESTS=0 \
benchmarks/profiling/vllm_prompt_embeddings/run_nsys.sh smoke
```

The remaining overrides are `DYN_PROMPT_EMBEDS_MODEL`,
`DYN_PROMPT_EMBEDS_PROMPT_TOKENS`, `DYN_PROMPT_EMBEDS_OUTPUT_TOKENS`,
`DYN_PROMPT_EMBEDS_BLOCK_SIZE`, `DYN_PROMPT_EMBEDS_MAX_MODEL_LEN`,
`DYN_PROMPT_EMBEDS_GPU_MEMORY_UTILIZATION`, and `DYN_PROMPT_EMBEDS_SEED`.
Use `DYN_NSYS_STAGING_ROOT` when nsys needs a larger temporary filesystem.

## Outputs

Each run retains:

- `requests.jsonl` and `summary.json` with per-request and aggregate latency.
- The finalized `.nsys-rep` and exported SQLite database under `nsys/`.
- `nsys-audit.json`, which records prefill and decode CUDA graph coverage.
- Exact Python and nsys commands, source/dependency/GPU provenance, and hashes.
- A copy and hashes of the exact profiling recipe used for the run.

Open the `.nsys-rep` in Nsight Systems for timeline analysis, or query the
SQLite export for repeatable attribution. A successful run prints
`NSYS_AUDIT=PASS` followed by the observed prefill and decode graph counts.

## Run without nsys

The underlying experiment is also directly callable:

```bash
python3 -m benchmarks.profiling.vllm_prompt_embeddings.main \
    --output-dir logs/prompt-embeddings
```

This still enforces the requested engine configuration and exact output length,
but it does not produce or audit an nsys trace.
