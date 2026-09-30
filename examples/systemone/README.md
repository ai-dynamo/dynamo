<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# System One Shell Example

**Experimental.** Serve typed decisions with NVIDIA Dynamo and an aggregated SGLang worker. `/v1/systemone` scores candidate tokens without generating an answer string and returns `choice`, `noul`, and `score` results. This example checks the API contract; a general-purpose model does not establish decision quality or probability calibration.

## Prerequisites

Use a Linux environment with Bash 4.3 or later, `setsid`, `curl`, `jq`, an NVIDIA GPU, and a Dynamo build that includes System One support with SGLang installed. Activate the Dynamo virtual environment. Start the discovery and request transport services required by your Dynamo configuration before launching the example; see the [SGLang CLI Template](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/cli-templates/sglang.mdx).

The default model is `Qwen/Qwen3-0.6B`. SGLang downloads missing model assets when launched. To use cached assets, set `MODEL` to a local model directory containing weights, tokenizer assets, and a supported chat template.

## Launch

From the repository root, run:

```bash
bash examples/backends/sglang/launch/agg_systemone.sh
```

The launcher enables `--enable-systemone-api` on the frontend and disables request migration. It launches one aggregated SGLang worker, forwards extra command-line arguments to that worker, and stops its child sessions when a process exits or you press Ctrl+C.

To select another model and ports, run:

```bash
MODEL=Qwen/Qwen3-0.6B DYN_HTTP_PORT=8000 DYN_SYSTEM_PORT=8081 \
  bash examples/backends/sglang/launch/agg_systemone.sh
```

`--model-path` overrides `MODEL`. `CONTEXT_LENGTH` defaults to `4096`. GPU assignment follows `CUDA_VISIBLE_DEVICES`, and the launcher uses the shared SGLang GPU memory helper for test-controlled memory limits. Keep speculative decoding, LoRA adapters, and disaggregated workers disabled for this API.

## Validate

In another terminal, run:

```bash
MODEL=Qwen/Qwen3-0.6B BASE_URL=http://localhost:8000 \
  bash examples/systemone/smoke.sh
```

The smoke test waits up to `READY_TIMEOUT` seconds (default `300`) for `/health` and the selected model in `/v1/models`. It sends one mixed `choice`/`noul`/`score` request, checks typed answer ranges and response headers, verifies zero output tokens, checks `/v1/chat/completions`, and expects HTTP `422` for an empty choice criterion list. Success ends with `System One smoke test passed`. If authentication is enabled, supply `API_KEY` through the environment.

The displayed decision values depend on the model. `noul` is the normalized probability of the affirmative label; `score` is the expected zero-based level index. `x_label_mass` records the unnormalized probability mass of the candidate labels, and the reported values do not imply calibration.

See the [System One API Reference](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/reference/api/systemone.md) for request fields, limits, model compatibility, and errors. The route returns HTTP `404` when the frontend flag is absent.
