<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed SGLang gRPC server

`dynamo-sglang-mocker-server` implements the native
`sglang.runtime.v1.SglangService` RPCs used by Dynamo's SGLang sidecar. It runs
on CPU using the Dynamo Mocker scheduler, so sidecar discovery, streaming,
cancellation, capacity, and disaggregated handoff behavior can be tested
without SGLang, a model, or a GPU.

The mock server and sidecar share generated types from `dynamo-sglang-grpc`.
Only tokenized `Generate`, `Abort`, health, and discovery RPCs are implemented;
unrelated SGLang RPCs return `Unimplemented`.

## Aggregated serving

Start the mock SGLang endpoint:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30001 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000,"block_size":4}'
```

Point the existing sidecar at it:

```bash
cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --sglang-endpoint http://127.0.0.1:30001
```

`--extra-engine-args` accepts inline JSON or a JSON file path. The server adds
`engine_type=sglang` when it is omitted; an explicitly different engine type,
`dp_size` other than one, or a non-aggregated Mocker worker type is rejected.
The wire-level role remains controlled by `--disaggregation-mode`.

## Disaggregated wire flow

Run separate endpoints for prefill and decode:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30001 --disaggregation-mode prefill \
  --bootstrap-host 127.0.0.1 --bootstrap-port 8998

cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30002 --disaggregation-mode decode
```

For local loopback testing, give the prefill sidecar an explicit reachable
bootstrap host:

```bash
cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --sglang-endpoint http://127.0.0.1:30001 \
  --bootstrap-host 127.0.0.1

cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --sglang-endpoint http://127.0.0.1:30002
```

The prefill response carries SGLang's bootstrap host, port, and room through
the real sidecar into the decode request. The values are validated, but no
bootstrap socket, NIXL connection, or KV data movement is created.

## Deliberate limitations

- Token-ID prompts only; no tokenizer or model is loaded.
- One output sequence with deterministic synthetic tokens and logprobs.
- Length termination only; sampling, stops, EOS, and structured decoding are
  not simulated.
- One Mocker data-parallel rank per server process.
