---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Decision Models with SGLang
sidebar-title: SGLang
subtitle: Run a dedicated text-scoring worker for both Dynamo decision APIs
---

**Experimental.** Run an aggregated SGLang CUDA worker to serve OpenAI `POST /v1/decisions` and Jev `POST /v1/systemone` through NVIDIA Dynamo. Both routes use the same candidate-scoring implementation. They do not expose SGLang's native Decisions wire schema.

## Prerequisites

Use a Dynamo environment built with SGLang and the services described in [Local Installation](../../cli/installation/install-dynamo.mdx). Use a supported GPU and ensure the checkpoint fits available memory. The example uses `Qwen/Qwen3-0.6B` as a small integration-test target, not a claim of decision-task quality or qualification of larger Qwen models.

The worker capability gate accepts SGLang 0.5.19, 0.5.22, and 0.5.23, including their recognized CUDA package suffixes. The current CUDA container pins 0.5.23.

<Warning>
Decision scoring requires `--max-running-requests 1 --disable-overlap-schedule`. The worker processes one inference request at a time, including any chat or completion requests sent to it. Dynamo does not advertise decision scoring if either scheduling requirement is violated. Use a separate worker for throughput-oriented chat traffic to avoid this serialization; separation is a serving recommendation, while the scheduling settings are requirements.
</Warning>

<Info>
Decision capability publication also requires an aggregated token-input worker, one data-parallel rank, and pipeline parallelism of one, with attention data parallelism, speculative decoding, diffusion execution, and priority preemption disabled. These are enforced eligibility checks, not tuning suggestions. CUDA is the qualified device for this guide; other devices and LoRA decision execution are not qualified here.
</Info>

## Start the Frontend

In the first terminal, select a dedicated namespace and enable both public routes:

```bash
DYN_NAMESPACE=decision-model \
python -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 8000 \
  --enable-systemone-api
```

`--enable-systemone-api` defaults to `false` and enables both public decision APIs.

## Start the Worker

In a second terminal using the same Dynamo environment and runtime services:

```bash
DYN_NAMESPACE=decision-model DYN_SYSTEM_PORT=8081 \
python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --tp 1 --dp-size 1 --pp-size 1 \
  --max-running-requests 1 --disable-overlap-schedule \
  --context-length 2048 --max-total-tokens 2048 \
  --disable-piecewise-cuda-graph
```

Wait for the `Published SGLang ... System One capability` message before sending requests. Ordinary generation can work even when the worker does not meet the decision eligibility requirements above.

## Verify Both APIs

List the model and check its serving readiness:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/models/Qwen%2FQwen3-0.6B/ready
```

Run the [curl, OpenAI SDK, and Jev SDK examples](apis.md) against this frontend. Use the exact served model alias. Both endpoints return question answers and usage; this SGLang profile must report `output_tokens: 0` because it requests candidate log-probabilities without generating a token.

## Profile Limits

| Dimension | Current boundary |
| --- | --- |
| Execution | Aggregated CUDA token-input worker; dedicated serialized scheduling |
| Inputs | Text; no image, audio, or video qualification |
| Question types | Choice, predicate/`noul`, and ordered score |
| Question count | Up to 128 per request, also bounded by aggregate preflight and admission limits |
| Candidate count | Choice ceiling 255; score ceiling 10 levels; predicates use 2 labels. Each label must be a distinct single token at the rendered answer position |
| Context | Each rendered question must fit the configured model context; prompts are not silently truncated |
| Worker placement | First question establishes placement; sibling questions stay on that worker and rank |
| Cache isolation | Siblings share a request-local cache identity; independent requests do not share it |

The protocol ceiling is not a promise that this tokenizer can represent 255 distinct labels. Unsupported candidate sets fail before inference. Input-token usage sums the rendered prompts for all questions; cache reads use measured backend counters. See [Decision API Configuration](../../reference/components/frontend-configuration.mdx#decision-apis) for independent byte, token-work, admission, and deadline limits.

The small-model [SGLang numerical qualification fixture](https://github.com/ai-dynamo/dynamo/blob/main/tests/serve/test_systemone_sglang.py) compares Dynamo with an independent native scorer. A passing endpoint smoke test does not replace that numerical check, task-specific accuracy evaluation, or distributed lifecycle testing. Engine-version acceptance alone is not evidence that a model was qualified.

## Stop the Example

Stop the worker and frontend with Ctrl-C in their respective terminals. Keep shared services running if another deployment uses them.
