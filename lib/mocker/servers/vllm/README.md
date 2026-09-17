<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed vLLM gRPC server

`dynamo-vllm-mocker-server` implements vLLM's native Inference and Control services plus standard gRPC health on CPU. It uses the Dynamo Mocker scheduler for batching, KV capacity, prefix cache, and timing behavior.

The mock server imports the generated types exposed by `dynamo-vllm-sidecar`. The proto files are vendored unchanged from vLLM.

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
  --grpc-endpoint 127.0.0.1:50051
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
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 0.0.0.0:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000,"block_size":64,"zmq_kv_events_port":5557}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --grpc-endpoint mock-host:50051
```

Replace `mock-host` with a host reachable from the sidecar. Expose both TCP
ports on that host, preserving the event port number. For Docker port mapping,
use `-p 50051:50051 -p 5557:5557` on the mock-server container. The sidecar will
connect to `mock-host:5557` for events.

An explicit replay client can use the existing optional replay socket by adding
`"zmq_replay_port":5558` to the engine arguments. The current sidecar receiver
does not consume the advertised replay endpoint. The shared native PUB/SUB path
can lose events before the subscription is ready, and restarting only the
sidecar does not rebuild the index for blocks already in the mock server's
cache. This server uses that existing path without additional recovery.

Set `"enable_prefix_caching":false` to disable both prefix caching and KV
events. Decode servers do not publish events. As with regular mock workers,
publisher setup failures are logged and serving continues without KV events.

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
  --grpc-endpoint 127.0.0.1:50051 \
  --disaggregation-mode prefill

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --grpc-endpoint 127.0.0.1:50052 \
  --disaggregation-mode decode
```

The sidecar discovers model identity through Control. Keep `--disaggregation-mode` for prefill and decode because the current discovery API does not report engine role.

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
