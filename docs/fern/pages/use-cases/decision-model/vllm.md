---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Decision Models with vLLM
subtitle: Serve the same decision contracts with the experimental vLLM causal executor
---

**Experimental.** Serve OpenAI `POST /v1/decisions` and Jev `POST /v1/systemone` with a vLLM causal worker. Requests and responses use the [same public contracts and SDKs](apis.md); only executor selection, supported limits, and accounting differ from SGLang.

> [!IMPORTANT]
> This guide targets the implementation in [vLLM causal decision support](https://github.com/ai-dynamo/dynamo/pull/18381). Until that implementation merges, use a Dynamo build containing that change; unmodified `main` does not provide this vLLM decision path. This is not a stock-vLLM HTTP endpoint or a vLLM Semantic Router integration.

## Prerequisites

Use released vLLM **0.33.0** with the matching Dynamo Python components and Rust bindings. For a source build, follow [Building from Source](../../developer-guide/advanced-customizations/building-from-source.md). Mounting only the updated Python worker over older bindings does not install the frontend scoring implementation. Keep etcd/NATS available as described in [Local Installation](../../cli/installation/install-dynamo.mdx).

The capability gate also verifies installed cache-counter source files and required scoring API fields. Matching the version string alone is insufficient: unsupported cache-accounting patches must fail qualification rather than fabricate usage. Do not replace the version or bypass these checks to advertise support.

Use `Qwen/Qwen3-0.6B` as the initial text-only example. A model-family name or architecture allowlist entry is not evidence that another checkpoint, native decision head, or modality has been qualified.

## Start the Frontend

In the first terminal:

```bash
DYN_NAMESPACE=decision-model-vllm \
python -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 8000 \
  --enable-systemone-api
```

Use a separate namespace from another deployment. If a frontend is already listening on port 8000, stop that owned example first or choose another port and update the client URLs.

## Start the Worker

In a second terminal with the same Dynamo build and runtime services:

```bash
DYN_NAMESPACE=decision-model-vllm DYN_SYSTEM_PORT=8081 \
python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --served-model-name Qwen/Qwen3-0.6B \
  --systemone-serial \
  --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager --language-model-only \
  --no-enable-prefix-caching \
  --logprobs-mode raw_logprobs \
  --generation-config vllm
```

The recorded single-GPU Qwen validation used a 96 GB GPU with memory fraction `0.1`; smaller GPUs may need a larger fraction so the weights and KV cache fit. Treat a changed configuration as a new validation profile and repeat both API smoke tests. The model and tokenizer revisions are pinned separately above; record template and precision alongside them for reproducibility.

`--systemone-serial` defaults to false. It opts the worker into decision capability publication but does **not** configure scheduling; `--max-num-seqs 1` and raw log-probabilities remain explicit requirements. Ordinary workers that omit the opt-in do not advertise decision capability or change discovery metadata for it. Wait for model readiness, then verify both API calls below; ordinary generation readiness alone does not prove decision capability.

Do not enable data parallelism, LoRA, speculative/diffusion execution, custom schedulers, encoder routing, KV/encoder-cache transfer, or the worker's RL execution mode for this profile. Keep the dedicated worker separate from throughput-oriented chat traffic. Internal scoring transport is enabled without exposing public Generate routes.

## Verify Both APIs

Check the registered alias and readiness, then run the [curl, OpenAI SDK, and Jev/TypeSafe SDK examples](apis.md) unchanged:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/models/Qwen%2FQwen3-0.6B/ready
```

The API examples contain three questions, so successful vLLM responses report `usage.output_tokens: 3`, not SGLang's zero. Input usage aggregates actual rendered prompt tokens across questions. OpenAI responses additionally expose measured cache reads and `total_tokens = input_tokens + output_tokens`; Jev usage contains input and output counts. The example disables prefix caching, so it is not evidence of cross-request cache reuse.

## Execution Boundaries

| Dimension | Causal vLLM profile |
| --- | --- |
| Execution | Dedicated aggregated text-scoring worker; documented example is single-GPU |
| Runtime | vLLM 0.33.0 plus required scoring fields and verified cache-counter sources |
| Question count | At most 128, also bounded by preflight, admission, and context limits |
| Candidates | At most 26 choices; OpenAI score 2–10 levels, Jev score 1–10; predicate uses 2 labels |
| Tokenizer | Distinct single-token candidate labels at the rendered answer position |
| Native work | One actual generated token per question; complete raw candidate score vector required |
| Failure handling | Worker cancellation/error maps to unavailable 503; malformed score vectors or usage remain 500 |
| Isolation | Questions share request-local cache identity and pinned worker placement; sibling execution is serialized |

Local validation of this causal implementation exercised both Dynamo-hosted APIs, official OpenAI 3.26.0 and TypeSafe 0.7.2 clients, and an independent native probability comparison with the same prompt and candidate token IDs. This fixed-case text validation does not establish accuracy, throughput, distributed scaling, or multimodal compatibility.

Native heads such as Clef, vision inputs, diffusion-specific scoring, and native gRPC decision APIs are outside this causal guide. An image-capable model evaluated on text is not a multimodal qualification. Keep those model/executor workstreams separate from the two public HTTP contracts.

Stop the worker and frontend with Ctrl-C in their respective terminals. Do not stop shared runtime services or unrelated deployments. HTTP cancellation observations do not prove physical GPU settlement.
