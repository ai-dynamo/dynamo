<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine mocker server

A CPU-only gRPC server that implements TensorRT-LLM's `openengine.v1` API on top
of the Mocker scheduler. The real `dynamo-trtllm-sidecar` connects to it exactly
as it would to a real TensorRT-LLM engine — no GPU, no model weights.

Token IDs, logprobs, and the disaggregated handoff are synthetic but
deterministic for a given `--seed`. KV-cache accounting, batching, admission,
and prefill/decode timing come from the Mocker scheduler using the configured
engine's scheduling policy.

Request recording is disabled by default. Tests that inspect
`received_requests()` enable `MockerServerConfig::is_request_recording_enabled`;
the recorder retains at most 256 accepted requests.

## Aggregated

```bash
cargo run -p dynamo-trtllm-mocker --bin dynamo-trtllm-mocker-server -- \
  --listen 127.0.0.1:50051 --model Qwen/Qwen3-0.6B --context-length 2048 \
  --extra-engine-args '{"engine":{"speedup_ratio":1000,"block_size":32}}'

cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --grpc-endpoint http://127.0.0.1:50051 --model-path Qwen/Qwen3-0.6B
```

Set the sidecar's `--model-path` to the server's `--model` so the frontend
registers and routes the name you expect. The server does not enforce it: it
serves a request naming any model, because the real engine serves whatever it
was loaded with and rejecting a mismatch would fail requests that work in
production. An empty model is refused.

The mocker needs no weights, but the sidecar still resolves `--model-path` to a
real tokenizer so the frontend can detokenize. Point both at a model that is
present locally (or in the HuggingFace cache) — for example
`--model Qwen/Qwen3-0.6B` and `--model-path Qwen/Qwen3-0.6B`. A name that does
not resolve, such as the default `mocker-model`, fails in the sidecar's model
fetch rather than anywhere in this server.

## Disaggregated

```bash
cargo run -p dynamo-trtllm-mocker --bin dynamo-trtllm-mocker-server -- \
  --listen 127.0.0.1:50051 --model Qwen/Qwen3-0.6B --context-length 2048 \
  --disaggregation-mode prefill \
  --extra-engine-args '{"engine":{"speedup_ratio":1000}}'
cargo run -p dynamo-trtllm-mocker --bin dynamo-trtllm-mocker-server -- \
  --listen 127.0.0.1:50052 --model Qwen/Qwen3-0.6B --context-length 2048 \
  --disaggregation-mode decode \
  --extra-engine-args '{"engine":{"speedup_ratio":1000}}'

cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --grpc-endpoint http://127.0.0.1:50051 --model-path Qwen/Qwen3-0.6B \
  --disaggregation-mode prefill
cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --grpc-endpoint http://127.0.0.1:50052 --model-path Qwen/Qwen3-0.6B \
  --disaggregation-mode decode
```

The two roles are independent processes and need no shared configuration.

The decode role validates the handoff far more strictly than a real engine
would. It requires the opaque `attributes_struct` keys the prefill role wrote,
in their original JSON types, so that a relay which dropped a field, renamed a
key, rounded a fractional number, or flattened a list fails loudly instead of
silently degrading. It also replays the prefill's first generated token, so the
two legs' token accounting matches a real engine's.

## Deliberate limitations

- **KV events are `UNIMPLEMENTED`**, matching the real TensorRT-LLM OpenEngine
  server. `GetKvEventSources` and `SubscribeKvEvents` both refuse, and the
  server publishes no KV events. **Dynamo KV routing therefore cannot be
  exercised against this mocker** — use the worker-level `dynamo.mocker` for
  that. Answering these RPCs here would let a test pass that fails against a
  real engine.
- LoRA lifecycle RPCs are `UNIMPLEMENTED`; `Health` with an inference probe is
  too.
- Text prompts are rejected: the server has no tokenizer and expects
  `token_ids`, which is what the sidecar always sends.
- Multimodal media and per-request LoRA are rejected.
- A validated decode handoff reserves destination KV space and starts decoding
  without recomputing the prompt. The handoff represents a completed transfer;
  no KV bytes move between mocker processes and network transfer time is not
  simulated. A decode request that bypasses prefill still computes its prompt.
- Guided decoding is rejected with `UNIMPLEMENTED`; rejected requests are not recorded.

## `--context-length` interacts with capacity

The sidecar turns an omitted `max_tokens` into `context_length - prompt_len`.
The scheduler caps that budget to the total KV-pool capacity minus the prompt
length, then reserves through completion under `guaranteed_no_evict` without
preemption. A prompt that leaves no room for output is rejected. Set an explicit
`max_tokens` or lower `--context-length` for small-KV experiments.

## Engine arguments

`--extra-engine-args` takes inline JSON or a file path and is merged into
the canonical AISimulate launch configuration. `engine.backend` defaults to `trtllm`;
passing another backend is an error. Put scheduler settings such as `block_size`
and `max_model_len` under `engine`. `--context-length` controls the context
length advertised by the gRPC mock server.
