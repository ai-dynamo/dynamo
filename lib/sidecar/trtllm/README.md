<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM sidecar

`dynamo-trtllm-sidecar` connects a Dynamo worker to TensorRT-LLM's native
`trtllm.TrtllmService` gRPC `Generate` service. It is a standalone Rust
executable composed with `dynamo_backend_common::run`: TensorRT-LLM runs as its
own process while the sidecar owns Dynamo worker registration, request
conversion, transport, cancellation, and abort.

## Supported

- Aggregated generation
- Token requests through Dynamo preprocessing
- Sampling, stop conditions, structured output (JSON schema / regex / grammar /
  structural tag), and logprobs
- Streaming delta tokens with a terminal usage/finish summary
- `Abort` on cancellation

The initial protocol does **not** support disaggregated (prefill/decode)
serving, multimodal input, LoRA, KV-aware routing, encode workers, beam search,
or `n > 1`. Disaggregation is excluded because the `Generate` response contract
carries no context-phase handoff.

## Run

Start TensorRT-LLM with its native gRPC listener (TRT-LLM `1.3.0rc21`+):

```bash
python -m tensorrt_llm.commands.serve <model> --grpc --host 0.0.0.0 --port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker:

```bash
dynamo-trtllm-sidecar \
  --trtllm-endpoint 127.0.0.1:50051 \
  --model-path <model>
```

Use `TRTLLM_GRPC_ENDPOINT` instead of `--trtllm-endpoint` when the endpoint is
provided through the environment.

Distribution and container packaging for the executable are intentionally
deferred to a follow-up change.
