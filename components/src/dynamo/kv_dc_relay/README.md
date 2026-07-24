<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DC KV Relay

The DC KV Relay aggregates exact KV-cache ownership inside one data center and publishes a compact
Cuckoo-filter (CKF) projection for multi-DC routing. It discovers workers through the Dynamo
runtime, consumes their ordered KV events, and supervises one actor-owned producer for each local
routing pool.

A pool is one logical indexer domain in one DC. The domain captures cache compatibility and routing
isolation; the DC identity remains stable across Relay restarts and endpoint replacement. Runtime
endpoints are bindings for a pool rather than part of the CKF publication identity.

For each pool, the Relay:

- Tracks the exact full hashes owned by every `(worker, dp_rank)` member.
- Refcounts shared hashes so any number of owners contribute exactly one CKF entry.
- Uses full-hash ownership to make unknown removals safe no-ops.
- Maintains the mutable producer CKF and records buckets changed by successful mutations.
- Publishes barrier snapshots and sequenced deltas containing absolute packed-bucket images.

The full hashes and refcounts stay in the Relay because a CKF fingerprint is lossy, can collide,
and has no owner identity. The global consumer needs only the compact projection required for
cross-DC prefix search.

## Recovery boundaries

Recovery has two stages:

1. **Worker to Relay:** The Relay shares the normal Dynamo indexer's worker-query recovery path.
   Ordered KV events handle live mutations; gaps and source replacement recover exact rank state
   before the new source generation becomes active.
2. **Relay to global consumer:** A new or reconnected lane installs a barrier snapshot, then
   continues with sequenced absolute bucket-image deltas. A missing delta retires that lane and
   requires another snapshot.

The current component uses the producer lifecycle and exposes local diagnostics. An in-process
adapter exercises the complete producer/consumer protocol today. Non-local gRPC transport and
cross-DC request forwarding are separate global-router integration work.

For the complete architecture, pool model, consistency contract, and recovery flow, see
[Multi-DC KV Routing and the DC Relay](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/multi-dc-kv-routing.md).

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

## Runtime endpoints

The component always exposes a health endpoint. Builds with the Rust `ckf-diagnostics` feature
also expose Relay statistics and an endpoint-specific producer snapshot. Endpoint component names
include a stable digest of `dc_id`, allowing several DC Relay processes to share a runtime
namespace without colliding.

These diagnostic endpoints are not the WAN publication protocol, and the Relay does not proxy
inference requests. A production global router is expected to transport published state, choose a
DC-local serving pool, and forward requests to that pool.
