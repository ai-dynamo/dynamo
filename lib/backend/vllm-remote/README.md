<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM remote backend

`dynamo-vllm-remote` connects a Dynamo worker to vLLM's native gRPC
`Generate` service. It is a standalone Rust executable.

## Supported

- Aggregated generation
- NIXL prefill/decode generation
- Token and text requests through Dynamo preprocessing
- Sampling, stop conditions, structured output, logprobs, cache options, and priority
- Opaque `kv_transfer_params` handoff

The initial protocol does not support multimodal input, LoRA, KV-aware data
parallel routing, encode workers, beam search, or `n > 1`.

## Run

Start vLLM with its released gRPC listener:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --grpc-port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker explicitly:

```bash
dynamo-vllm-remote \
  --vllm-endpoint 127.0.0.1:50051 \
  --model-path Qwen/Qwen3-0.6B
```

Use `VLLM_GRPC_ENDPOINT` instead of `--vllm-endpoint` when the endpoint is
provided through the environment.

Distribution and container packaging for the executable are intentionally
deferred to a follow-up change.
