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
- With exactly one valid `x-prefiller-host-port`, it removes EPP-owned headers
  and invokes the configured backend P/D adapter.

Empty, malformed, repeated, or comma-separated prefill endpoint values return
`502 Bad Gateway` with the OpenAI-style error code `invalid_epp_metadata`.

The binary listens on `0.0.0.0:8000` and proxies to
`http://localhost:8001` by default. Set `DYN_SIDECAR_PORT` and
`DYN_DECODE_ENGINE_PORT` to change the ports. Upstream connections time out
after 10 seconds by default, and stalled response reads time out after 300
seconds without imposing a deadline on the full response stream. Configure
these values in milliseconds with `DYN_SIDECAR_CONNECT_TIMEOUT_MS` and
`DYN_SIDECAR_READ_TIMEOUT_MS`.

Backend-specific P/D execution is implemented separately. Until an adapter is
linked, requests containing a valid prefill endpoint return `501 Not
Implemented`; decode-only passthrough remains available.
