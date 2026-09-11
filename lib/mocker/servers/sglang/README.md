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

The mock server temporarily imports the generated contract from
`dynamo_sglang_sidecar::proto`. Only tokenized `Generate`, `Abort`, health, and
discovery RPCs are implemented; unrelated SGLang RPCs return `Unimplemented`.

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
  --grpc-endpoint http://127.0.0.1:30001
```

`--extra-engine-args` accepts inline JSON or a JSON file path. The server adds
`engine_type=sglang` when it is omitted; an explicitly different engine type,
`dp_size` other than one, or a non-aggregated Mocker worker type is rejected.
The wire-level role remains controlled by `--disaggregation-mode`.
`--max-concurrent-requests` bounds admitted RPCs, including requests waiting
inside the Mocker scheduler.

## KV events

Like regular mock workers, the server publishes KV cache events when prefix
caching is enabled, except in decode mode. It uses the existing Mocker ZMQ
publisher and reports the endpoint through native engine discovery. The
sidecar forwards these events to Dynamo's router.

The event publisher binds a free port by default. For this gRPC server, an
unset `zmq_kv_events_port` selects an automatic ZMQ port. The sidecar discovers
the endpoint and replaces its wildcard address with the host from
`--grpc-endpoint`. Use a frontend with `--router-mode kv`.

Automatic ports work for local processes and containers that share a network
namespace, including containers in one Kubernetes pod. Use a fixed port when
port mappings, a Service, or firewall rules need a known event port. For example:

```bash
cargo run -p dynamo-sglang-mocker --bin dynamo-sglang-mocker-server -- \
  --listen 0.0.0.0:30001 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000,"block_size":64,"zmq_kv_events_port":5557}'

cargo run -p dynamo-sglang-sidecar --bin dynamo-sglang-sidecar -- \
  --grpc-endpoint mock-host:30001
```

Replace `mock-host` with a host reachable from the sidecar. Expose both TCP
ports on that host, preserving the event port number. For Docker port mapping,
use `-p 30001:30001 -p 5557:5557` on the mock-server container. The sidecar will
connect to `mock-host:5557` for events.

An explicit replay client can use the existing optional replay socket by adding
`"zmq_replay_port":5558` to the engine arguments. The current SGLang discovery
descriptor does not advertise replay, and the sidecar receiver does not use it.
The shared native PUB/SUB path
can lose events before the subscription is ready, and restarting only the
sidecar does not rebuild the index for blocks already in the mock server's
cache. This server uses that existing path without additional recovery.

Set `"enable_prefix_caching":false` to disable both prefix caching and KV
events. Decode servers do not publish events. As with regular mock workers,
publisher setup failures are logged and serving continues without KV events.

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
- Length termination only; sampling, stops, EOS, and structured decoding are
  not simulated.
- `Abort` releases Mocker scheduler state but does not synthesize SGLang's
  `finish_reason: {"type": "abort"}` terminal; a caller still polling that
  gRPC stream receives `Internal` when its output channel closes.
- One Mocker data-parallel rank per server process.
