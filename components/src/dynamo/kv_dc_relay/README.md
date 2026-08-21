<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DC KV Relay

The DC KV Relay discovers NVIDIA Dynamo inference endpoints, consumes their ordered key-value (KV)
events, and exports one pool stream for each materialized endpoint-local KV domain. Each pool has
one actor-owned Cuckoo-filter (CKF) producer.

## Pool Model

A pool is one atomic Dynamo indexer domain in one data center. The normal derived routing scope
keeps independent serving endpoints in separate pools, even when they advertise the same canonical
model. If two endpoints resolve to the same `PoolId`, the Relay fences that identity instead of
combining their KV state.

The serving endpoint identifies the endpoint that owns the pool; it is not an inference ingress and
is not part of `PoolId`. Each descriptor also publishes the worker roles declared by that
endpoint. Each endpoint pool has one canonical base-model registration and can add Low-Rank
Adaptation (LoRA) registrations backed by that model. Every registration carries its aliases. A
LoRA registration stays in the base model's pool and does not create a separate CKF.

The pool catalog and serving topology are separate projections. The catalog remains endpoint-local.
The topology stream groups endpoints by `(namespace, canonical_model_id)` so Dynamo's
namespace-wide dependencies can be evaluated without merging their CKFs. Topology members link
back to a pool by stable `PoolId`; consumers resolve the current producer generation through the
catalog.

The Relay materializes an endpoint pool only when the endpoint membership has a valid indexer
domain and model registration with no structural conflict, its runtime configuration resolves one
KV-state endpoint, and at least one expected worker/rank advertises an active KV event source. If
any condition stops holding, the Relay withdraws the pool before teardown. The topology member can
remain present while its `pool_id` changes from a value to `None`.

For each pool, the Relay:

- Tracks the exact full hashes owned by every `(worker, dp_rank)` member.
- Refcounts shared hashes so any number of owners contribute exactly one CKF entry.
- Uses full-hash ownership to make unknown removals safe no-ops.
- Maintains the mutable producer CKF and records buckets changed by successful mutations.
- Publishes barrier snapshots and sequenced deltas containing absolute packed-bucket images.

The full hashes and refcounts stay in the Relay because a CKF fingerprint is lossy, can collide,
and has no owner identity.

## Publication and Recovery

The Relay uses separate recovery boundaries for each exported fact:

- Ordered worker KV events update exact rank ownership. A gap or source replacement rebuilds that
  rank before the replacement source becomes active.
- The pool catalog is an authoritative, revisioned snapshot. A withdrawn or fenced generation
  disappears before its actor drains.
- The first `SubscribeKvPool` client initializes that pool's publication hub. The hub captures one
  CKF snapshot and then fans out contiguous deltas through bounded subscriber queues. A lagged
  subscriber reconnects for a fresh snapshot. An initialized hub keeps its mirror until the pool
  generation retires, even when it has no subscribers.
- Serving readiness is a revisioned namespace-wide projection of endpoint availability, typed
  worker dependencies, and LoRA adapter membership. Reconnecting the readiness stream does not
  rebuild CKF state.
- Pool load is emitted as complete, latest-wins windows with independent coverage counts for each
  signal. A lagged load subscriber reconnects without affecting catalog, readiness, or CKF streams.

The Relay publishes pool facts and does not merge or rank independent pools. Consumers choose how
to compare those streams. The readiness projection's `duplicate_role_endpoints` field reports which
typed Prefill/Decode roles are advertised by multiple endpoints for one topology key. It is a
topology fact, not a version-independent statement that the deployment is degraded or that local
rendezvous is disabled.

In prefill/decode (PD) and encode/prefill/decode (EPD) deployments, each endpoint with an active KV
event source has its own pool. Prefill and Decode CKFs are both meaningful and may use different
query semantics. An Encode-only endpoint remains an `ENCODE` member of the serving topology, but
has no pool link and allocates no CKF or load state unless it advertises an active KV event source.
Removing that source withdraws the Encode pool and returns the member to `pool_id: None`.

## Usage

```bash
python -m dynamo.kv_dc_relay \
  --dc-id dc-a \
  --namespaces production-llama
```

Keep `--dc-id` stable for the logical data center across Relay restarts. Use `--namespaces` to
select one or more DynamoGraphDeployment (DGD) namespaces, or use `--watch-all` explicitly. Narrow
the selected scope with repeatable `--endpoint-prefix` values:

```bash
python -m dynamo.kv_dc_relay \
  --dc-id us-west \
  --namespaces llama-fast,llama-slow \
  --endpoint-prefix llama-fast.backend \
  --endpoint-prefix llama-slow.backend
```

`DYN_NAMESPACE` controls only the Relay's runtime namespace and defaults to `dynamo`.
`DYN_RELAY_NAMESPACES` selects watched DGD namespaces when `--namespaces` is not set. Command-line
values take precedence over their corresponding `DYN_RELAY_*` environment variables.

## WAN Server

Build the Python extension with the `kv-dc-relay-wan` feature to enable the gRPC server. Setting
`--bind` requires a server certificate, server key, and client certificate-authority bundle:

```bash
DYN_SYSTEM_PORT=9090 python -m dynamo.kv_dc_relay \
  --dc-id us-west \
  --namespaces llama-fast,llama-slow \
  --bind 0.0.0.0:5560 \
  --tls-server-cert /etc/dynamo/relay/server.crt \
  --tls-server-key /etc/dynamo/relay/server.key \
  --tls-client-ca /etc/dynamo/relay/client-ca.crt
```

The WAN server always uses mutual TLS (mTLS). Omit `--bind` to run the local producer without a WAN
listener. The server exposes these `dynamo.kvrelay.v1.KvEventRelay` methods:

- `GetRelayInfo`
- `WatchKvPoolCatalog`
- `SubscribeKvPool`
- `SubscribeServingReadiness`
- `SubscribeKvPoolLoad`

Set `DYN_SYSTEM_PORT` to expose Relay transport and publication metrics through Dynamo's system
metrics endpoint. The health endpoint reports the WAN listener state and fatal transport errors.

Behavioral bounds (publication cadence, keepalive and heartbeat intervals, queue and
subscriber limits) have self-consistent defaults and are overridden only through
`DYN_RELAY_<KEY>` environment variables — for example `DYN_RELAY_MAX_POOL_STREAMS_TOTAL`,
`DYN_RELAY_MAX_SUBSCRIBERS_PER_POOL`, and `DYN_RELAY_MAX_INITIALIZED_POOL_HUBS` (all default
to 64; the initialized-hub bound is a lifetime cap on per-pool CKF mirrors, not a count of
currently subscribed pools). The full key list is `TUNING_KEYS` in the component CLI.

See [Multi-DC KV Routing and the DC Relay](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/multi-dc-kv-routing.md)
for the pool, identity, consistency, and recovery contracts.
