<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker-backed vLLM gRPC server

`dynamo-vllm-mocker-server` implements vLLM's native Inference and Control services plus standard gRPC health on CPU. It uses the Dynamo Mocker scheduler for batching, KV capacity, prefix cache, and timing behavior.

The mock server imports the official `vllm-proto` types exposed by `dynamo-vllm-sidecar`.

## Aggregated serving

Start the mock vLLM endpoint:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"engine":{"speedup_ratio":1000,"block_size":64}}'
```

Point the existing Dynamo sidecar at it:

```bash
cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --grpc-endpoint 127.0.0.1:50051
```

`--extra-engine-args` accepts inline JSON or a JSON file path. The values use
the canonical launch shape: `engine.backend=vllm`, `dp_size=1`, and
`engine.worker_type=aggregated` are required. Use `--seed` to change the deterministic
synthetic token stream. `--max-concurrent-requests` bounds admitted RPCs
(default `256`) independently of the scheduler's `max_num_seqs`, so accepted
requests can still exercise Mocker queueing.

Synthetic output plans are limited to 1,000,000 tokens. LiveEngine uses a small,
fixed response buffer for each request and cancels slow consumers rather than
turning declared output length into a second admission-control policy.

## Image mock deployments

The mock advertises multimodal support and accepts image URLs, data URIs, raw
bytes, and preprocessed feature payloads in aggregated, prefill, and decode
roles. Sources must be non-empty; features must contain non-empty `kwargs`, an
`identifier`, and a positive `length`. Audio and video are rejected. Image
contents remain opaque: the mock does not fetch URLs, decode images, or parse
feature tensors. Its gRPC message limit is 64 MiB, matching the sidecar.

Prompt usage, scheduling, and synthetic output use the supplied token IDs.
The mock does not expand image placeholders or simulate image-aware KV or
encoder caching. Disable prefix caching for these deployments. Prefill returns
the supplied prompt IDs and synthetic handoff metadata, which lets the real
sidecars complete the image P/D request flow without NIXL or KV data transfer.

The launch example runs the real Dynamo HTTP frontend, discovery, routing, and
sidecars. It requires the `ai-dynamo` and `ai-dynamo-runtime` Python packages and
the Rust binaries, but no vLLM installation, model weights, or GPUs:

```bash
cargo build -p dynamo-vllm-mocker -p dynamo-vllm-sidecar
export PATH="$PWD/target/debug:$PATH"

# One aggregated mock worker.
bash lib/mocker/servers/vllm/launch/multimodal.sh aggregated

# Or separate prefill and decode mock workers.
bash lib/mocker/servers/vllm/launch/multimodal.sh disagg
```

The default model is `Qwen/Qwen2.5-VL-3B-Instruct`. The frontend and sidecars
load its configuration, tokenizer, and chat template. Set `MODEL` to a local
metadata directory for offline use. The example uses URL passthrough,
round-robin routing, disabled prefix caching, and file discovery under
`$PWD/.dynamo-mm-mock`; no etcd or NATS service is required. All processes must
share the discovery directory. Ports and namespace can be set through the
environment; run the script with `--help` for details.

After the model appears in `/v1/models`, send a streaming image request:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-VL-3B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="}}]}],"max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}'
```

The response contains synthetic text, terminal usage, and `[DONE]`. It does
not describe the image. Dynamo may inspect image metadata in its own frontend;
the mock does no image processing. The HTTP API uses `image_url` content parts;
raw bytes and preprocessed features are also accepted at the gRPC boundary.

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
  --extra-engine-args '{"engine":{"speedup_ratio":1000,"block_size":64},"dynamo":{"zmq_kv_events_port":5557}}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --grpc-endpoint mock-host:50051
```

Replace `mock-host` with a host reachable from the sidecar. Expose both TCP
ports on that host, preserving the event port number. For Docker port mapping,
use `-p 50051:50051 -p 5557:5557` on the mock-server container. The sidecar will
connect to `mock-host:5557` for events.

An explicit replay client can use the existing optional replay socket by adding
`"zmq_replay_port":5558` under `dynamo` in the engine arguments. The current sidecar receiver
does not consume the advertised replay endpoint. The shared native PUB/SUB path
can lose events before the subscription is ready, and restarting only the
sidecar does not rebuild the index for blocks already in the mock server's
cache. This server uses that existing path without additional recovery.

Set `"enable_prefix_caching":false` under `engine` to disable both prefix caching and KV
events. Decode servers do not publish events. As with regular mock workers,
publisher setup failures are logged and serving continues without KV events.

## Disaggregated wire-flow

Run separate endpoints for the two emulated vLLM roles:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 --model mocker-model \
  --disaggregation-mode prefill --extra-engine-args '{"engine":{"speedup_ratio":1000}}'

cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50052 --model mocker-model \
  --disaggregation-mode decode --extra-engine-args '{"engine":{"speedup_ratio":1000}}'
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
- Length and explicit stop-token termination; stop strings, EOS, and structured
  decoding are accepted on the wire but are not simulated. The synthetic token
  plan is unchanged by minimum-token constraints; stop matching begins after
  that minimum, without simulating logit masking.
- Prefix-cache bypass and cache-salt controls are rejected because the Mocker
  server does not emulate their isolation semantics.
- One Mocker data-parallel rank per server process.

The server cancels request-ID scheduler work when a gRPC response stream is
dropped, so cancellation and high-concurrency tests do not leave background
requests consuming simulated capacity.
