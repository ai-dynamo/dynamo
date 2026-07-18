<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed vLLM gRPC server

`dynamo-vllm-mocker-server` implements vLLM's native `Generate` and
`GenerateStream` RPCs on CPU, using the Dynamo Mocker scheduler for batching,
KV-capacity, prefix-cache, and timing behavior. Its primary purpose is fast,
repeatable testing of `dynamo-vllm-sidecar` without a model or GPU.

The mock server and sidecar share the generated types from
`dynamo-vllm-grpc`, whose proto is vendored unchanged from vLLM v0.25.1.

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
synthetic token stream.

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
payload, and the decode endpoint validates that the sidecar forwarded it. No
NIXL connection or KV data movement occurs; this mode tests the sidecar and
Dynamo handoff wire-flow only.

## Deliberate limitations

- Token-ID prompts only; the server does not load a tokenizer.
- Deterministic placeholder text, token IDs, and synthetic logprobs rather
  than vLLM sampling.
- One output sequence (`n <= 1`).
- Length termination only; stop strings, EOS, and structured decoding are
  accepted on the wire but are not simulated.
- One Mocker data-parallel rank per server process.

The server cancels request-ID scheduler work when a gRPC response stream is
dropped, so cancellation and high-concurrency tests do not leave background
requests consuming simulated capacity.
