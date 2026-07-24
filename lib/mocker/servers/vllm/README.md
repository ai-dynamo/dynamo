<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed vLLM gRPC server

`dynamo-vllm-mocker-server` implements vLLM's native `Generate` and
`GenerateStream` RPCs on CPU, using the Dynamo Mocker scheduler for batching,
KV-capacity, prefix-cache, and timing behavior. Its primary purpose is fast,
repeatable testing of `dynamo-vllm-sidecar` without a model or GPU.

The mock server temporarily imports the generated types exposed by
`dynamo-vllm-sidecar`, whose proto is vendored unchanged from vLLM v0.25.1.
Both consumers will move to vLLM's upstream package once it is published.

## Aggregated serving

Start the mock vLLM endpoint:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000,"block_size":64}'
```

Point the existing Dynamo sidecar at it:

```bash
cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50051 \
  --model-path mocker-model
```

`--extra-engine-args` accepts inline JSON or a JSON file path. The values use
`MockEngineArgs`; `engine_type=vllm`, `dp_size=1`, and
`worker_type=aggregated` are required. Use `--seed` to change the deterministic
synthetic token stream. `--max-concurrent-requests` bounds admitted RPCs
(default `256`) independently of the scheduler's `max_num_seqs`, so accepted
requests can still exercise Mocker queueing.

Synthetic output plans are limited to 32,768 tokens. LiveEngine uses a small,
fixed response buffer for each request and cancels slow consumers rather than
turning declared output length into a second admission-control policy.

## Disaggregated wire-flow

Run separate endpoints for the two emulated vLLM roles:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 --model mocker-model \
  --disaggregation-mode prefill --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50052 --model mocker-model \
  --disaggregation-mode decode --extra-engine-args '{"speedup_ratio":1000}'
```

Then start one sidecar for each endpoint:

```bash
cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50051 --model-path mocker-model \
  --disaggregation-mode prefill

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50052 --model-path mocker-model \
  --disaggregation-mode decode
```

The prefill endpoint returns an opaque vLLM-shaped `kv_transfer_params`
payload, and the decode endpoint validates that the sidecar forwarded it
verbatim — including a non-rendezvous sentinel field, so a dropped opaque field
fails the round trip. No NIXL connection or KV data movement occurs; this mode
tests the sidecar and Dynamo handoff wire-flow only.

## Deliberate limitations

- Token-ID prompts only; the server does not load a tokenizer.
- Deterministic placeholder text, token IDs, and synthetic logprobs rather
  than vLLM sampling.
- One output sequence (`n <= 1`).
- At most 20 logprob candidates per token; larger top-N, explicit token-ID, or
  "all" candidate requests are truncated to 20 rather than returning the full
  set. (vLLM's default `max_logprobs` is also 20, but rejects over-limit
  requests instead of truncating.)
- Length termination only; stop strings, EOS, and structured decoding are
  accepted on the wire but are not simulated.
- Prefix-cache bypass and cache-salt controls are rejected because the Mocker
  server does not emulate their isolation semantics.
- One Mocker data-parallel rank per server process.

The server cancels request-ID scheduler work when a gRPC response stream is
dropped, so cancellation and high-concurrency tests do not leave background
requests consuming simulated capacity.
