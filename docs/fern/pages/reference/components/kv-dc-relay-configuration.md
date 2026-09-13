---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multi-Datacenter KV Relay Configuration
subtitle: CLI arguments, environment variables, resource limits, and diagnostic endpoints
---

**Experimental.** The DC KV Relay collects worker KV-cache events within a data center and
publishes compact cache-locality, serving-readiness, and load information for external consumers.

This reference describes `python -m dynamo.kv_dc_relay`. For deployment, see
[Deploy the DC KV Relay](../../kubernetes/kv-aware-routing/kv-dc-relay.md); for an overview and architecture,
see [DC KV Relay Concepts](../../developer-guide/knowledge-base/modular-components/router/multi-dc-kv-routing.md).

## CLI Arguments

| Argument | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `--dc-id` | `DYN_DC_ID` | Required | Logical data-center identity. Keep stable across restarts; must be nonempty without surrounding whitespace. |
| `--namespaces` | `DYN_RELAY_NAMESPACES` | All visible Dynamo namespaces | Comma-separated Dynamo namespace allowlist. |
| `--namespace-filter` | None | Unset | Legacy single-namespace form of `--namespaces`. |
| `--watch-all` | `DYN_RELAY_WATCH_ALL` | Enabled when no scope is set | Include model cards from every Dynamo namespace visible to the discovery backend. |
| `--endpoint-prefix` | `DYN_RELAY_ENDPOINT_PREFIXES` | No prefix filter | Repeat the CLI option, or use a comma-separated environment value. Match namespace/component/endpoint segments. |
| `--expected-unique-blocks` | `DYN_RELAY_EXPECTED_UNIQUE_BLOCKS` | `1048576` | Positive expected number of unique blocks per pool; sizes the CKF, not a global Relay memory limit. |
| `--bind` | `DYN_RELAY_BIND` | No listener | Plaintext gRPC socket address, such as `127.0.0.1:5561` or `[::1]:5561`. Requires a numeric IP and port. |
| `--help` | None | — | Print CLI usage and exit. |

The protocol and gRPC server are included in the standard build. Omitting `--bind` disables only
the WAN listener; local discovery and pool maintenance still run. Relay has no TLS flags or
certificate configuration. Protect a listener across a trust boundary with an
[optional external sidecar](../../kubernetes/kv-aware-routing/kv-dc-relay.md#optional-mtls-sidecar).

## Precedence and Scope

CLI values override the corresponding environment values. A CLI scope option replaces the whole
environment scope: `--namespaces`, `--namespace-filter`, and `--watch-all` are mutually exclusive.
Likewise, a CLI prefix list replaces, rather than extends, the environment prefix list.

- Lists must contain nonempty, unique entries. Comma-separated lists trim item whitespace.
- Prefixes must match whole endpoint segments, not arbitrary string prefixes. With an explicit
  namespace allowlist, every prefix must belong to one of those namespaces.
- `DYN_RELAY_NAMESPACES` cannot be combined with a true `DYN_RELAY_WATCH_ALL`.
- Boolean environment values accept `1/0`, `true/false`, `yes/no`, or `on/off`, ignoring case
  and surrounding whitespace. Explicit `DYN_RELAY_WATCH_ALL=false` requires a namespace allowlist.
- With neither a CLI nor an environment scope, the Relay watches all visible Dynamo namespaces.

For example, `production`, `production.backend`, and `production.backend.generate` select
successively narrower endpoint scopes. `production.back` does not match `production.backend`.

> [!IMPORTANT]
> Dynamo namespaces are logical discovery scopes, not Kubernetes namespaces. The current
> Kubernetes discovery backend watches only the Relay pod's Kubernetes namespace. Neither
> `--watch-all` nor `--namespaces` expands that Kubernetes watch.

## Runtime Environment

Relay uses the shared `DistributedRuntime`, created by `@dynamo_worker()`.

| Variable | Default | Relay usage |
| --- | --- | --- |
| `DYN_NAMESPACE` | `dynamo` | Namespace of Relay's own runtime endpoints; does not select watched worker namespaces. |
| `DYN_DISCOVERY_BACKEND` | `etcd` | Shared discovery backend; set `kubernetes` for the Kubernetes how-to. |
| `DYN_REQUEST_PLANE` | `tcp` | Shared runtime request transport, independent of the WAN gRPC listener. |
| `DYN_EVENT_PLANE` | Backend-dependent | Match worker event transport: `zmq` for direct TCP events without NATS, or `nats`. Set explicitly in deployment manifests. |
| `NATS_SERVER` | `nats://localhost:4222` | Address of the workers' NATS service when using the NATS event plane. |
| `DYN_SYSTEM_PORT` | Disabled (`-1`) | Enables the runtime HTTP health and metrics server when set to a nonnegative port. |

For connection settings and runtime defaults, see [Runtime Configuration](runtime-configuration.mdx).
For direct event transport and TCP response-stream addressing, see
[TCP-only deployment](../../kubernetes/kv-aware-routing/kv-dc-relay.md#tcp-only-local-planes-no-nats).
For Kubernetes pod identity and RBAC, see [Deploy the DC KV Relay](../../kubernetes/kv-aware-routing/kv-dc-relay.md).
Relay's CLI does not accept the Frontend's runtime CLI flags; configure the runtime through its
environment variables.

## Producer Tuning

These environment-only overrides also work without a WAN listener. All values must be positive
integers. Unknown `DYN_RELAY_*` names are not consumed by the launcher; check spelling.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DYN_RELAY_PUBLICATION_THRESHOLD` | `16` | Primary-residency KV events processed before a publication; not changed buckets or bytes. |
| `DYN_RELAY_PUBLICATION_DELAY_MS` | `1` | Publication coalescing delay in milliseconds. |
| `DYN_RELAY_RECOVERY_ATTEMPT_TIMEOUT_MS` | `30000` | Timeout for a worker recovery attempt. |

The capacity selected by `--expected-unique-blocks` must fit the CBI1 maximum of `16777216`
buckets. The Rust producer rejects a larger derived CKF layout at startup.

## WAN Tuning

Every variable below requires `--bind` or `DYN_RELAY_BIND`, including publication-resource
overrides that are owned by the universal publisher. Setting one without a listener is a startup
error. All values must be positive integers; byte limits use bytes, not MiB.

### Transport and Projection Timing

| Variable | Default | Meaning |
| --- | --- | --- |
| `DYN_RELAY_MAX_MESSAGE_BYTES` | `8388608` | Maximum gRPC encoding and decoding message size. |
| `DYN_RELAY_KEEPALIVE_INTERVAL_MS` | `20000` | HTTP/2 keepalive interval. |
| `DYN_RELAY_KEEPALIVE_TIMEOUT_MS` | `10000` | HTTP/2 keepalive timeout. |
| `DYN_RELAY_POOL_HEARTBEAT_INTERVAL_MS` | `10000` | Pool-stream heartbeat interval after snapshot bootstrap. |
| `DYN_RELAY_READINESS_HEARTBEAT_INTERVAL_MS` | `10000` | Interval for repeating the current readiness snapshot. |
| `DYN_RELAY_SNAPSHOT_PROGRESS_TIMEOUT_MS` | `60000` | Per-frame progress deadline while producing the initial snapshot. |
| `DYN_RELAY_LOAD_WINDOW_MS` | `1000` | Load publication window. |
| `DYN_RELAY_LOAD_FANOUT_CAPACITY` | `16` | Buffered load updates for fanout. |

### Universal Publication Resources

These limits belong to the publisher; the Python launcher exposes their overrides with WAN enabled.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DYN_RELAY_PUBLICATION_QUEUE_CAPACITY` | `16` | Pool-subscriber queue message bound. |
| `DYN_RELAY_PUBLICATION_QUEUE_BYTES` | `16777216` | Pool-subscriber queue byte bound. |
| `DYN_RELAY_PUBLICATION_ENCODING_CONCURRENCY` | `2` | Concurrent snapshot encoders. |
| `DYN_RELAY_MAX_INITIALIZED_POOL_HUBS` | `64` | Resident initialized hubs, including idle hubs eligible for eviction. |

### Stream Admission

| Variable | Default | Meaning |
| --- | --- | --- |
| `DYN_RELAY_MAX_CATALOG_SUBSCRIBERS` | `64` | Concurrent catalog streams. |
| `DYN_RELAY_MAX_POOL_STREAMS_TOTAL` | `64` | Total active pool streams. |
| `DYN_RELAY_MAX_SUBSCRIBERS_PER_POOL` | `64` | Subscribers attached to one pool. |
| `DYN_RELAY_MAX_READINESS_SUBSCRIBERS` | `64` | Concurrent readiness streams. |
| `DYN_RELAY_MAX_LOAD_SUBSCRIBERS` | `64` | Concurrent load streams. |

### Validation Limits

WAN timer values cannot exceed `31536000000` ms. Load fanout capacity cannot exceed `65536`.
Channel and semaphore capacities must also fit their underlying Tokio limits. The message limit
must be at least `4259904` bytes and the queue byte bound at least `4194624` bytes, so each can
hold a maximum CBI1 frame plus its required overhead.

A complete snapshot chunk contains 4 MiB of bucket words plus framing. Configure clients and
sidecars to accept the server's message size; tonic's default 4 MiB receive limit is insufficient
for a maximum chunk. Resource exhaustion and snapshot progress deadlines are described in the
[gRPC contract](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/docs/grpc-contract.md).

## Diagnostic Endpoints

The component registers Dynamo runtime endpoints under
`<DYN_NAMESPACE>.kv_dc_relay_<dc-hash>`, where `dc-hash` is the first 32 hexadecimal characters
of SHA-256 over the UTF-8 DC ID. These are not HTTP paths or WAN RPCs.

| Endpoint | Availability | Request and response |
| --- | --- | --- |
| `health` | Always | Empty request; reports `healthy`, shutdown state, endpoint counts, and WAN state/errors. |
| `stats` | `ckf-diagnostics` build feature | Empty request; returns detailed endpoint/pool statistics. |
| `snapshot` | `ckf-diagnostics` build feature | Requires `serving_endpoint`; returns that endpoint's diagnostic snapshot. |

The WAN listener also exposes gRPC reflection and the standard gRPC health service for
`dynamo.kvrelay.v1.KvEventRelay`. Transport readiness does not prove that pools have been
discovered or that a model can serve requests. Inspect catalog and serving-readiness streams
separately. Terminal host or transport failures stop the component with a nonzero exit status.

## Sources

- [Python argument parsing](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/kv_dc_relay/cli.py)
- [Producer defaults](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/host.rs)
- [gRPC adapter defaults and validation](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/wan/grpc/config.rs)
