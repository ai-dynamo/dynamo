<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine mocker server

A CPU-only gRPC server that implements TensorRT-LLM's `openengine.v1` API on top
of the Mocker scheduler. The real `dynamo-trtllm-sidecar` connects to it exactly
as it would to a real TensorRT-LLM engine — no GPU, no model weights.

Token IDs, logprobs, and the disaggregated handoff are synthetic but
deterministic for a given `--seed`. KV-cache accounting, batching, admission,
and prefill/decode timing come from the Mocker scheduler, so capacity and
scheduling behave like a real deployment.

## Aggregated

```bash
cargo run -p dynamo-trtllm-mocker --bin dynamo-trtllm-mocker-server -- \
  --listen 127.0.0.1:50051 --model Qwen/Qwen3-0.6B --context-length 2048 \
  --extra-engine-args '{"speedup_ratio":1000,"block_size":32}'

cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --trtllm-endpoint http://127.0.0.1:50051 --model-path Qwen/Qwen3-0.6B
```

The sidecar's `--model-path` must equal the server's `--model`; the server
answers `NOT_FOUND` otherwise, since it serves exactly one model.

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
  --extra-engine-args '{"speedup_ratio":1000}'
cargo run -p dynamo-trtllm-mocker --bin dynamo-trtllm-mocker-server -- \
  --listen 127.0.0.1:50052 --model Qwen/Qwen3-0.6B --context-length 2048 \
  --disaggregation-mode decode \
  --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --trtllm-endpoint http://127.0.0.1:50051 --model-path Qwen/Qwen3-0.6B \
  --disaggregation-mode prefill
cargo run -p dynamo-trtllm-sidecar --bin dynamo-trtllm-sidecar -- \
  --trtllm-endpoint http://127.0.0.1:50052 --model-path Qwen/Qwen3-0.6B \
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
- Guided decoding is accepted and ignored: the output is unconstrained. The
  request is recorded, so a test can still assert the client mapped the guide
  correctly, but nothing here enforces the grammar.

## `--context-length` interacts with capacity

The sidecar turns an omitted `max_tokens` into `context_length - prompt_len`,
and the TensorRT-LLM scheduling policy (`guaranteed_no_evict`) reserves that
whole budget at admission and never preempts. With the 32768 default and a small
`num_gpu_blocks`, a request that omits `max_tokens` is rejected for capacity
rather than truncated. Lower `--context-length` for small-KV experiments.

## Engine arguments

`--extra-engine-args` takes inline JSON or a file path and is merged into
`MockEngineArgs`. `engine_type` is forced to `trtllm`; passing anything else is
an error. TensorRT-LLM requires `block_size >= 2` (default 32) and rejects
`max_model_len` — use `--context-length` instead.
