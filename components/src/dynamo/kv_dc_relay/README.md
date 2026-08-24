<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DC KV Relay

The DC KV Relay discovers Dynamo inference pools, consumes their ordered KV events, and supervises
one actor-owned Cuckoo-filter (CKF) producer for each local pool.

A pool is one atomic Dynamo indexer domain in one data center. Its domain captures cache
compatibility and routing isolation. The Relay does not merge KV state from independent endpoints
or deployments into one actor, even when they serve the same canonical model.

Canonical model names are request-facing bindings. One model can bind to multiple independent
pools, and each pool keeps its own KV stream. LoRA registrations remain attached to the pool of
their backing base model.

For each pool, the Relay:

- Tracks the exact full hashes owned by every `(worker, dp_rank)` member.
- Refcounts shared hashes so any number of owners contribute exactly one CKF entry.
- Uses full-hash ownership to make unknown removals safe no-ops.
- Maintains the mutable producer CKF and records buckets changed by successful mutations.
- Publishes barrier snapshots and sequenced deltas containing absolute packed-bucket images.

The full hashes and refcounts stay in the Relay because a CKF fingerprint is lossy, can collide,
and has no owner identity.

## Recovery boundaries

The Relay shares the normal Dynamo indexer's worker-query recovery path. Ordered KV events handle
live mutations; gaps and source replacement recover exact rank state before the new source epoch
becomes active. A fenced pool is withdrawn before its actor stops.

## Usage

```bash
python -m dynamo.kv_dc_relay --dc-id <stable-dc-id>
```

`--dc-id` must be stable for the logical data center across Relay process restarts. Optional
discovery filters can limit the endpoints supervised by one Relay:

```bash
python -m dynamo.kv_dc_relay \
  --dc-id us-west \
  --namespace-filter dynamo \
  --endpoint-prefix dynamo.backend
```

`DYN_NAMESPACE` controls the namespace used for the Relay's own runtime endpoints and defaults to
`dynamo`.

## Naming invariant

Request-facing model and adapter names must be unique across every namespace one Relay watches.
WAN consumers address published state by model name, so the Relay cannot scope name ownership to a
namespace the way a local frontend can. When one name resolves to conflicting targets anywhere in
the watch scope, the Relay omits that name from every endpoint (fail-closed) instead of picking an
owner; the omission is recorded as a per-endpoint serving conflict. Deployments that reuse one
model name for different targets must split them across separate Relay watch scopes.

## Runtime endpoints

The component always exposes a health endpoint. Builds with the Rust `ckf-diagnostics` feature
also expose Relay statistics and an endpoint-specific producer snapshot. Endpoint component names
include a stable digest of `dc_id`, allowing several DC Relay processes to share a runtime
namespace without colliding.

These diagnostic endpoints are not the WAN publication protocol, and the Relay does not proxy
inference requests. A production global router is expected to transport published state, choose a
DC-local serving pool, and forward requests to that pool.
