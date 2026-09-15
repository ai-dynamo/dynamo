<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro-0813 Recipes

Recipes for [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813).

> [!NOTE]
> This is a **different checkpoint** from `deepseek-ai/DeepSeek-V4-Pro` used by the sibling
> [`../deepseek-v4-pro`](../deepseek-v4-pro) recipes. Do not share a model cache between them.

## Configurations

Dynamo + vLLM deployment profiles.

|                          | H200 aggregated (agentic + 1M) | H200 disaggregated (agentic + 1M) | GB200 aggregated       | GB200 disaggregated |
| ------------------------ | ----------------------- | -------------------------------- | ---------------------- | ------------------- |
| **GPU** (per worker)     | 8x H200                 | 8x H200 prefill + 8x H200 decode | 8x GB200 (2 nodes)     | 8x GB200 prefill + 8x GB200 decode (2 nodes each) |
| **Mode**                 | Aggregated              | Prefill/decode disaggregated     | Aggregated             | Prefill/decode disaggregated |
| **Framework**            | vLLM                    | vLLM                             | vLLM                   | vLLM |
| **Precision**            | MXFP4 experts + FP8 KV  | MXFP4 experts + FP8 KV           | MXFP4 experts + FP8 KV | MXFP4 experts + FP8 KV |
| **Parallelism**          | TP8/EP8                 | TP8/EP8 prefill / TP8/EP8 decode | TP8/EP8                | TP8/EP8 prefill / TP8/EP8 decode |
| **Routing**              | KV-aware                | KV-aware                         | KV-aware               | KV-aware |
| **Speculative decoding** | None                    | None                             | DSpark k=5             | DSpark k=5 (prefill and decode) |
| **Context length**       | 1,048,576               | 1,048,576                        | 1,048,576              | 1,048,576 |
| **KV cache offloading**  | None                    | None                             | None                   | None |
| **KV transfer**          | N/A                     | NIXL                             | N/A                    | NIXL |

The H200 and GB200 variants use different images because they require different vLLM
versions ¹.

¹ H200 runs vLLM 0.26.0; GB200 requires vLLM 0.28.0. GB200 cannot use 0.26.0 because
DSpark does not run on Blackwell there (DeepGEMM requires `next_n == 1`), and 0.27.x
has an upstream accuracy regression on this checkpoint.

## Supported features

- Modalities: **Text** (this checkpoint is `DeepseekV4ForCausalLM`; it has no vision tower — image input is **not** supported)
- Reasoning
- Tool calling

### Known issue: `message.content` can be `null`

These recipes pass `--dyn-reasoning-parser deepseek_v4`, which splits the model's
thinking trace into `message.reasoning_content`. A generation that never emits a
closing `</think>` — because it was truncated by `max_tokens`, or because the whole
reply is a short constrained value — routes **everything** to `reasoning_content` and
leaves `message.content` as `null`. The answer is still produced; it is in the wrong
field.

This is an upstream vLLM parser bug, not a recipe or serving fault: the grammar and
the generation both work. Tracked as
[vllm-project/vllm#48645](https://github.com/vllm-project/vllm/issues/48645), with a
fix proposed in [vllm-project/vllm#50753](https://github.com/vllm-project/vllm/pull/50753).
Both are open as of 2026-08-28, so the fix is absent from the pinned runtime image.

It shows up most often with `response_format: json_schema` for short values (an enum
lands in `reasoning_content`) and with small `max_tokens`. Practical guidance:

- Read `reasoning_content` as a fallback when `content` is `null`.
- Give reasoning room. This model spends a large budget thinking — `max_tokens` of
  1-2K is frequently consumed before any content token is emitted. Our accuracy runs
  needed a 64K cap to drive the no-answer rate to zero.
- `chat_template_kwargs: {"thinking": false}` populates `content`, but disabling
  thinking costs roughly 21 points of GPQA accuracy. It is a debugging aid, not a
  production setting.

### Known issue: an invalid `json_schema` returns HTTP 500

A `response_format: json_schema` carrying a malformed regex — for example an
unclosed character class such as `"pattern": "[unclosed["` — is rejected with:

```json
{"message":"Failed to generate completions","type":"Internal Server Error","code":500}
```

A client-side schema error should surface as HTTP 400 or 422, not 500. The
validation itself is correct: vLLM raises a `ValueError` when it compiles the
grammar, but the Dynamo HTTP service maps that to a 500 rather than a 4xx. There is
no flag to change this; it needs an error-mapping fix in the frontend.

Validate schemas client-side before submitting them. A 500 from this endpoint does
not necessarily mean the deployment is unhealthy — check the schema first.

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to `deepseek-ai/DeepSeek-V4-Pro-0813`.
3. No CPU KV offload is needed for 1M context on either topology — GPU KV holds ~3.1M tokens,
   about 3x a full 1M request.

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a ReadWriteMany storage
> class available on the target cluster. The checkpoint is ~832 GiB.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

```bash
MODE=agg         # or disagg
SKU=h200         # or gb200 (agentic only)
WORKLOAD=agentic
kubectl apply -f vllm/${MODE}-${SKU}-${WORKLOAD}/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace staging on the
PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

The gate is **joint**: E2E ≥ 50 tok/s/user **and** TTFT p50 < 5 s, where
`E2E = OSL / (TTFT + OSL × ITL)` — the per-user rate *including* time-to-first-token.

Modified Mooncake traces are provided to exercise KV-aware routing and prefix-cache reuse,
see [perf/README.md](perf/README.md) for details.

## Performance results

Joint gate: user output >= 50 tok/s **and** TTFT p50 < 5 s.

### Mooncake trace replay (3,541-row 15% agentic trace, `ignore_eos`)

| Workload | Recipe | SKU | Concurrency | System tok/s/GPU | User output tok/s (P50) | TTFT P50 |
| --- | --- | --- | --- | --- | --- | --- |
| Agentic 64K | `agg-gb200-agentic` | 8x GB200 | 8 | 69.45 | 51.85 | 403 ms |
| Agentic 64K | `disagg-gb200-agentic` | 16x GB200 | 10 | 50.58 | 53.42 | 1,237 ms |
| Agentic 64K | `agg-h200-agentic` | 8x H200 | 4 | 21.4 | 51.8 | 322 ms |
| Agentic 64K | `disagg-h200-agentic` | 16x H200 | 4 | 13.1 | 57.2 | 441 ms |

Each row was measured at its own iso-SLA operating point. All four are complete runs over the
full 3,541-request trace.

On H200, aggregated is the better choice for this workload: it clears the same gate on **half
the GPUs**, at 21.4 versus 13.1 system tok/s/GPU. Disaggregated buys a higher per-user rate
(57.2 vs 51.8) and a slightly lower ITL, so it is an SLA choice rather than a throughput win.
