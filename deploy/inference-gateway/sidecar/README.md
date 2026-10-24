<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EPP Decode Sidecar

The EPP decode sidecar is the pod-local HTTP data plane for standalone
disaggregated routing. It accepts the original OpenAI-compatible request from
Gateway and selects one of two paths:

- Without `x-prefiller-host-port`, it proxies the request to the local decode
  engine.
- With exactly one valid `x-prefiller-host-port`, it removes that header and
  invokes the configured backend P/D adapter.

Empty, malformed, repeated, or comma-separated prefill endpoint values return
`502 Bad Gateway` with the OpenAI-style error code `invalid_epp_metadata`.

The binary listens on `0.0.0.0:8000` and proxies to
`http://localhost:8001` by default. Set `DYN_SIDECAR_PORT` and
`DYN_DECODE_ENGINE_PORT` to change the ports. Upstream connections time out
after 10 seconds by default, and stalled response reads time out after 300
seconds without imposing a deadline on the full response stream. Configure
these values in milliseconds with `DYN_SIDECAR_CONNECT_TIMEOUT_MS` and
`DYN_SIDECAR_READ_TIMEOUT_MS`. Active requests drain for 30 seconds during
shutdown before their streams are forced closed. Configure this deadline with
`DYN_SIDECAR_DRAIN_TIMEOUT_MS`.

`GET /health` remains live while requests drain. `GET /ready` returns `200 OK`
while the sidecar accepts requests and `503 Service Unavailable` once draining
starts. Readiness does not indicate that EPP endpoint propagation is complete.

Backend-specific P/D execution is implemented separately. With no adapter
selected, which is the default, requests containing a valid prefill endpoint
return `501 Not Implemented`; decode-only passthrough remains available in every
mode. See the next section for the adapter that handles them.

## Disaggregated P/D: raw-vLLM NIXL adapter

The sidecar ships a P/D adapter for plain vLLM OpenAI workers running the NIXL
pull connector. It is **off by default**; enable it explicitly:

```bash
DYN_SIDECAR_PD_ADAPTER=vllm_nixl
```

`DYN_SIDECAR_PD_ADAPTER` accepts `none` (default) and `vllm_nixl`. Anything else
fails at startup, as does a protocol revision this build does not implement.

With the adapter enabled, a request carrying a valid `x-prefiller-host-port`
runs the P/D exchange:

```text
prefill-only request  ->  selected prefill worker
   <- kv_transfer_params
original request + handoff  ->  the fixed local decode engine
```

KV bytes travel directly between the two workers over NIXL. They never pass
through the sidecar.

The supported protocol is pinned to **vLLM v0.29.0**. The exact handoff shape,
the field-by-field contract, the request variants that are supported or
rejected, the error taxonomy, and the upgrade checklist are in
[`PROTOCOL.md`](PROTOCOL.md).

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DYN_SIDECAR_PD_ADAPTER` | `none` | `none` or `vllm_nixl` |
| `DYN_VLLM_NIXL_PROTOCOL_VERSION` | `vllm-v0.29.0-nixl-pull` | Must match the revision this build implements; a mismatch refuses startup |
| `DYN_MODEL_NAME` | empty | Model both workers serve; recorded at startup |
| `DYN_SIDECAR_MAX_REQUEST_BYTES` | `33554432` (32 MiB) | Maximum request body the P/D path buffers |
| `DYN_SIDECAR_CLIENT_BODY_TIMEOUT_MS` | `30000` (30 s) | Total time allowed to read a client's request body. Bounds a client that sends an under-cap body slowly |
| `DYN_SIDECAR_MAX_PREFILL_RESPONSE_BYTES` | `1048576` (1 MiB) | Maximum prefill response the P/D path buffers |
| `DYN_SIDECAR_PREFILL_DEADLINE_MS` | `60000` (60 s) | Total time allowed for the prefill leg, across every chunk. The read timeout bounds one gap between chunks, not the leg |

Note on the two prefill timeouts: the leg deadline is the ceiling on the whole
prefill exchange, and the read timeout only bounds one gap inside it. With the
defaults the leg ends at 60 s, so a single gap can never reach the 300 s read
timeout, and raising `DYN_SIDECAR_READ_TIMEOUT_MS` alone buys a slow worker no
extra prefill time. Raise `DYN_SIDECAR_PREFILL_DEADLINE_MS` for that.

The decode target is always the locally configured decode engine
(`DYN_DECODE_ENGINE_PORT`). `remote_host` and `remote_port` in the handoff are
connector side-channel information and never change the HTTP destination.

### Minimal two-worker example

Both workers are plain vLLM OpenAI servers with the NIXL connector. Run the
prefill worker with the producer role and the decode worker with the consumer
role, and use a KV load policy that fails rather than recomputes, so a request
that did not actually transfer cannot look like a success.

```bash
# Decode worker (the sidecar's local target).
vllm serve Qwen/Qwen3-0.6B --port 8001 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Prefill worker, reachable at the endpoint EPP selects.
vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Sidecar.
DYN_SIDECAR_PD_ADAPTER=vllm_nixl \
DYN_DECODE_ENGINE_PORT=8001 \
DYN_MODEL_NAME=Qwen/Qwen3-0.6B \
  dynamo-epp-sidecar
```

Then send a chat completion with `x-prefiller-host-port: <prefill-host>:8100`.
Only a trusted local harness should be able to inject that header; the sidecar
must not be exposed to untrusted clients on the P/D path.

### Limitations

- The connector's block lease is not owned by the sidecar. A cancelled request
  in the window between the handoff and the decode worker accepting it relies on
  the producer's own lease expiry. HTTP cancellation here is not proof that GPU
  KV memory was released.
- Neither leg is retried. A retry could duplicate generation or leak the
  producer's block lease, so retries need their own idempotency design.
- `n != 1` is rejected rather than rewritten.
- A request that does not transfer any blocks (a full decode-side prefix-cache
  hit) is a valid, successful exchange. It is not evidence that the NIXL data
  path was exercised; proving that needs an input which genuinely requires
  remote KV.
