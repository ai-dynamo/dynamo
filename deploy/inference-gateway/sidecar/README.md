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

The default `DYN_SIDECAR_PD_BACKEND=unavailable` keeps backend P/D execution
disabled: requests containing a valid prefill endpoint return `501 Not
Implemented`, while decode-only passthrough remains available. Set
`DYN_SIDECAR_PD_BACKEND=sglang` to enable the SGLang adapter. Other values are
rejected at startup.

## SGLang Prefill/Decode Adapter

The adapter supports the native `/v1/chat/completions` protocol in SGLang
release **0.5.19**. For every disaggregated request, it checks both workers'
`GET /server_info` responses for the supported version and the expected
`prefill` or `decode` mode. Other releases and mismatched worker roles are
rejected before inference dispatch. Decode-only passthrough does not use this
adapter or require these probes.
Streaming responses must use SGLang's BOM-free UTF-8 SSE format.

The selected prefill HTTP endpoint comes from EPP's authoritative
`x-prefiller-host-port` header. Gateway must remove any client-supplied value
before EPP sets this header; the sidecar must not be exposed directly to
untrusted callers. The bootstrap host comes from that endpoint,
and the bootstrap port comes from the prefill worker's server information.
The adapter replaces client-supplied bootstrap fields and `rid` values with
sidecar-owned request IDs and one trusted bootstrap tuple shared by both
workers. It preserves the original messages, sampling settings, and streaming
mode. Parallel sampling is limited to `n=1` (including the default when `n` is
omitted).

Prefill and decode requests start concurrently. Only decode response bytes
reach Gateway, and only after the adapter has consumed and validated the full
prefill response, including successful termination of a JSON or SSE response.
An error event or an `abort` finish reason is a failure even with HTTP `200`.
While prefill is pending, the adapter also reads and checks decode's response
body so decode errors can cancel prefill promptly. It retains the exact bytes
read ahead and forwards them before the remaining decode stream.
The sidecar emits no synthetic SSE preamble. Prefill failure, decode failure,
client cancellation, and response-stream teardown cancel the paired work.
The workers transfer KV bytes directly; the sidecar never proxies KV payloads.

The original request and the complete prefill response are each limited to
32 MiB; each server-information response is limited to 1 MiB. The decode
prefix read before prefill completes is also limited to 32 MiB. Exceeding that
limit cancels both legs; after prefill completion, decode streaming has no
total-response-size limit. The configured
connection and idle-read timeouts also apply to adapter upstream requests.
Inference HTTP errors retain their status and `Retry-After` header in an
OpenAI-style error response. Discovery and protocol failures return `502`;
upstream network timeouts return `504`.

Cancellation closes upstream connections and attempts explicit backend aborts
with a five-second deadline. These aborts are best effort: an HTTP `200` from
the abort endpoint does not establish that backend execution or KV transfer
has stopped. The workers must have no separate `admin_api_key`, or the
forwarded inference authorization must also grant access to the abort
endpoint. Shutdown waits for the adapter's outstanding cleanup attempts after
HTTP request draining completes.

EPP owns worker selection and paired reservation accounting. The adapter uses
the shared request, response, and cancellation contract; it does not create or
release EPP reservations itself. A complete GAIE/EPP deployment and real-GPU
KV-transfer validation are outside the adapter's current validation boundary.
