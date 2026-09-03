---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speculative Decoding with vLLM
---

Using Speculative Decoding with the vLLM backend.

> **See also**: [Speculative Decoding Overview](overview.md) for cross-backend documentation.

## Prerequisites

- vLLM container with Eagle3 support
- GPU with at least 16GB VRAM
- Hugging Face access token (for gated models)

## Quick Start: Meta-Llama-3.1-8B-Instruct + Eagle3

This guide walks through deploying **Meta-Llama-3.1-8B-Instruct** with **Eagle3** speculative decoding on a single node.

### Step 1: Set Up Your Docker Environment

First, initialize a Docker container using the vLLM backend. See the [vLLM Quickstart Guide](../../knowledge-base/modular-components/backends/vllm/overview.md#quick-start) for details.

```bash
# Launch infrastructure services
docker compose -f dev/docker-compose.yml up -d

# Build the container
./container/build.sh --framework VLLM

# Run the container
./container/run.sh -it --framework VLLM --mount-workspace
```

### Step 2: Get Access to the Llama-3 Model

The **Meta-Llama-3.1-8B-Instruct** model is gated. Request access on Hugging Face:
[Meta-Llama-3.1-8B-Instruct repository](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Approval time varies depending on Hugging Face review traffic.

Once approved, set your access token inside the container:

```bash
export HUGGING_FACE_HUB_TOKEN="insert_your_token_here"
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
```

### Step 3: Run Aggregated Speculative Decoding

```bash
# Requires only one GPU
cd examples/backends/vllm
bash launch/agg_spec_decoding.sh
```

Once the weights finish downloading, the server will be ready for inference requests.

### Step 4: Test the Deployment

```bash
curl http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "messages": [
       {"role": "user", "content": "Write a poem about why Sakura trees are beautiful."}
     ],
     "max_tokens": 250
   }'
```

### Example Output

```json
{
  "id": "cmpl-3e87ea5c-010e-4dd2-bcc4-3298ebd845a8",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "In cherry blossom's gentle breeze ... A delicate balance of life and death, as petals fade, and new life breathes."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 250,
    "total_tokens": 266
  }
}
```

## Configuration

Speculative decoding in vLLM uses Eagle3 as the draft model. The launch script configures:

- Target model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Draft model: Eagle3 variant
- Aggregated serving mode

See `examples/backends/vllm/launch/agg_spec_decoding.sh` for the full configuration.

## Limitations

- Currently only supports Eagle3 as the draft model
- Requires compatible model architectures between target and draft
- Draft checkpoints that share the target's embedding table instead of shipping their
  own `embed_tokens` weights are not loadable on every vLLM build (see below)

### Draft Checkpoints Without Their Own `embed_tokens`

The Quick Start configuration above is not affected: its Eagle3 draft ships its own
embedding table. This section covers draft checkpoints that ship none and expect vLLM to
share the target model's instead. vLLM detects that at load and logs a line such as:

```text
Detected EAGLE model without its own embed_tokens in the checkpoint.
Sharing target model embedding weights with the draft model.
```

> [!WARNING]
> On the `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-kimi-k3-dev.1` image
> (vLLM `0.1.dev19251+g13c59a3da.d20260726`), that sharing path fails at load with
> `AttributeError: 'NoneType' object has no attribute 'weight'`. The traceback lands in
> vLLM's speculative-decoding proposer (`vllm/v1/spec_decode/llm_base_proposer.py`) while
> it compares the target and draft embedding dimensions. Every tensor-parallel worker
> fails identically during `WorkerProc` init, so the engine never becomes ready and no
> request is ever served. Reported with `moonshotai/Kimi-K3` plus the
> `Inferact/Kimi-K3-DSpark` draft on TP16.

The failure is on the vLLM side of the boundary, not in Dynamo: Dynamo passes
`--speculative-config` through to vLLM unchanged and has no setting that changes how the
draft embedding is resolved. Other vLLM builds carry a different version of that code, so
whether a given image is affected has to be checked against the image, not assumed from
this note.

Workarounds: use a draft checkpoint that ships its own `embed_tokens`, run a vLLM build
that loads the shared-embedding path successfully, or drop `--speculative-config` and
serve the target model on its own.

## See Also

| Document | Path |
|----------|------|
| Speculative Decoding Overview | [README.md](overview.md) |
| vLLM Backend Guide | [vLLM README](../../knowledge-base/modular-components/backends/vllm/overview.md) |
| Meta-Llama-3.1-8B-Instruct | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
