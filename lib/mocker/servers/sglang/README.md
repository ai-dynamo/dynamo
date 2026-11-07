<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed SGLang gRPC server

`dynamo-sglang-mocker-server` implements the native
`sglang.api.v1.SglangService` RPCs used by Dynamo's SGLang sidecar, the gRPC
service of SGLang's Rust server. It runs on CPU using the Dynamo Mocker
scheduler, so sidecar discovery, streaming, cancellation, capacity, and
disaggregated handoff behavior can be tested without SGLang, a model, or a GPU.

The mock server temporarily imports the generated contract from
`dynamo_sglang_sidecar::proto`. Only tokenized `Generate`, `Abort`, health, and
discovery RPCs are implemented; unrelated SGLang RPCs, including `Detokenize`
and the HiCache storage RPCs, return `Unimplemented`.

The service follows the Rust server's shapes: typed discovery with
`deployment_info`, `GenerateStreamItem` frames whose terminal carries a typed
`finish_reason`, prompt logprobs on the first frame only, and in-stream error
items with HTTP status codes. A request that does not fit the context ends
with a 400 item, `Abort` ends the stream with a 499 item and reports whether it
matched an active request, and a request beyond the simulated KV capacity ends
with a 503 item.

## Aggregated serving

Start the mock SGLang endpoint:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30001 \
  --model mocker-model \
  --extra-engine-args '{"engine":{"speedup_ratio":1000,"block_size":4}}'
```

Point the existing sidecar at it:

```bash
cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --grpc-endpoint http://127.0.0.1:30001
```

`--extra-engine-args` accepts inline JSON or a JSON file path. The server adds
`engine.backend=sglang` when it is omitted; an explicitly different backend,
`dp_size` other than one, or a non-aggregated `engine.worker_type` is rejected.
The wire-level role remains controlled by `--disaggregation-mode`.
`--max-concurrent-requests` bounds admitted RPCs, including requests waiting
inside the Mocker scheduler.

## KV events

Like regular mock workers, the server publishes KV cache events when prefix
caching is enabled, except in decode mode. It uses the existing Mocker ZMQ
publisher and reports the endpoint through native engine discovery as one
`kv_event_sources` entry at `tcp://127.0.0.1:<port>`, a node-local address like
the Rust server's, so the sidecar must run on the Mocker's host. The sidecar
forwards these events to Dynamo's router.

The event publisher binds a free port by default. For this gRPC server, an
unset `zmq_kv_events_port` selects an automatic ZMQ port. Use a frontend with
`--router-mode kv`.

The event and optional replay sockets bind to all IPv4 interfaces (`0.0.0.0`),
independently of the gRPC `--listen` address. These sockets have no authentication,
and stored events include token IDs from cached prompt and generated-token blocks.

Publishing collects and queues token IDs even when no subscriber is connected.
The background publisher task encodes and sends the events. This work also
occurs when the frontend uses a routing mode that does not consume KV events.

Because discovery reports the publisher at `127.0.0.1`, the sidecar must share
the Mocker's network namespace: the same host, or containers in one Kubernetes
pod. Use a fixed port when other tools need a known event port. For example:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30001 \
  --model mocker-model \
  --extra-engine-args '{"engine":{"speedup_ratio":1000,"block_size":64},"dynamo":{"zmq_kv_events_port":5557}}'

cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --grpc-endpoint 127.0.0.1:30001
```

The sidecar connects to `tcp://127.0.0.1:5557` for events.

An explicit replay client can use the existing optional replay socket by adding
`"zmq_replay_port":5558` under `dynamo` in the engine arguments. SGLang discovery
does not advertise replay, and the sidecar receiver does not use it.
The shared native PUB/SUB path
can lose events before the subscription is ready, and restarting only the
sidecar does not rebuild the index for blocks already in the mock server's
cache. This server uses that existing path without additional recovery.

Set `"enable_prefix_caching":false` under `engine` to disable both prefix caching and KV
events. Decode servers do not publish events. In either case, explicit
`zmq_kv_events_port` and `zmq_replay_port` values are ignored; they do not enable
publishing. As with regular mock workers, publisher setup failures are logged
and serving continues without KV events.

## Disaggregated wire flow

Run separate endpoints for prefill and decode:

Run each command in a separate terminal:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30001 --disaggregation-mode prefill \
  --bootstrap-host 127.0.0.1 --bootstrap-port 8998
```

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 127.0.0.1:30002 --disaggregation-mode decode
```

For local loopback testing, give the prefill sidecar an explicit reachable
bootstrap host. Run each sidecar in a separate terminal:

```bash
cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --grpc-endpoint http://127.0.0.1:30001 \
  --bootstrap-host 127.0.0.1
```

```bash
cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --grpc-endpoint http://127.0.0.1:30002
```

The prefill response carries SGLang's bootstrap host, port, and room through
the real sidecar into the decode request. The values are validated, but no
bootstrap socket, NIXL connection, or KV data movement is created.

## Deliberate limitations

- Token-ID prompts only; no tokenizer or model is loaded.
- One output sequence with deterministic synthetic tokens and logprobs.
- Length and explicit stop-token termination follow SGLang's `min_new_tokens`
  and `ignore_eos` controls. Sampling, string stops, model EOS, and structured
  decoding are not simulated.
- `Abort` releases Mocker scheduler state and ends the stream with a 499 error
  item, matching the Rust server.
- `HealthCheck` reports healthy once the server starts; it does not simulate
  SGLang's one-token probe.
- The HiCache storage RPCs (`GetHiCacheStorage`, `AttachHiCacheStorage`,
  `DetachHiCacheStorage`, `ClearHiCacheStorage`) return `Unimplemented`: the
  Mocker has no storage backend.
- One Mocker data-parallel rank per server process, reported as a single-rank
  deployment.
