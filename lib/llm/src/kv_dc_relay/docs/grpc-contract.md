<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay: gRPC Contract

Service `dynamo.kvrelay.v1.KvEventRelay`, defined in
[`../protocol/relay.proto`](../protocol/relay.proto). This document is the
contract-level view: RPCs, message semantics, validation, and lifecycle rules.
Byte-level framing of the CKF payload (CBI1) and identity derivation are
specified in [`../protocol/README.md`](../protocol/README.md); the producer
model behind the contract is in [`architecture.md`](architecture.md).

## Envelope and versioning

Every top-level request and response carries `fixed32 contract_marker = 127`
(`0x4B565231`, "KVR1"); every response also carries `protocol_version` and the
typed `RelayIdentity`. Both sides reject a mismatched marker or version with
`FAILED_PRECONDITION`.

`RELAY_PROTOCOL_VERSION` is **1**. Until the contract ships to production,
schema changes are made as clean breaks within v1 — including re-typing
existing field numbers. Version discipline starts when the contract is
released; consumers must be deployed in lockstep with the Relay until then.

All validation is **fail-closed**: unknown enum values (including
`*_UNSPECIFIED`), duplicate set elements, empty required sets, and
inconsistent namespaces are rejected rather than ignored.

## RPCs

| RPC | Kind | Purpose |
| --- | --- | --- |
| `GetRelayInfo` | unary | Protocol version and typed Relay identity |
| `WatchKvPoolCatalog` | server stream | Complete revisioned pool-catalog snapshots |
| `SubscribeKvPool` | server stream | CKF snapshot + deltas for one exact producer generation |
| `SubscribeServingReadiness` | server stream | Complete namespace topology projections |
| `SubscribeKvPoolLoad` | server stream | Complete pool-load windows |

Streaming requests require a non-empty `subscriber_id` (≤ 128 bytes). Each
stream type has an independent subscriber limit; pool publication additionally
bounds total pool streams, subscribers per pool, and initialized publication
hubs. Breaching any bound returns `RESOURCE_EXHAUSTED` and increments
`kv_dc_relay_subscriber_limit_rejected_total{stream,scope}`.

### WatchKvPoolCatalog

Emits `KvPoolCatalogUpdate` — a **complete snapshot** of all live pools per
revision. There are no tombstones anywhere in the contract: a pool absent from
the next snapshot is withdrawn.

`KvPoolDescriptor`:

| Field | Semantics |
| --- | --- |
| `producer: ProducerIdentity` | Pool identity + generation + CKF format. The subscription key for `SubscribeKvPool`. |
| `serving_endpoint` | The Dynamo endpoint owning the pool. Descriptor metadata for consumer-side resolution — **not** an ingress and not a `PoolId` dimension. |
| `registrations[]` | Canonical model id, `ModelTarget` (base, or LoRA `{base_model, adapter}`), normalized aliases. Cross-pool name uniqueness is deliberately not enforced. |
| `query_semantics` | Atomic token→sequence-hash pipeline: `kv_block_size` (> 0 required) + closed `KvQueryHashFormat`. Consumers must reject unknown formats. |
| `pool_roles[]` | Worker roles **declared** by the endpoint's current base cards, independent of liveness (`LEGACY` for cards without a role). Non-empty required. Live roles are published in the topology stream instead. |

### SubscribeKvPool

The request names an exact `expected_producer: ProducerIdentity` taken from
the catalog — not just a pool. A stale generation is rejected with
`FAILED_PRECONDITION` (producer mismatch), an unknown pool with `NOT_FOUND`,
and a generation swap during lazy hub initialization with `UNAVAILABLE`. This
guarantees a subscriber can never silently read a replacement generation's
filter.

The stream delivers `FilterUpdate` frames: `SNAPSHOT_CHUNK`s assembling one
consistent CBI1 image, then contiguous `DELTA`s, with `HEARTBEAT`s in between.
A subscriber that falls behind the bounded queue is terminated with
`RESOURCE_EXHAUSTED` and must resubscribe from scratch.

### SubscribeServingReadiness

Emits complete `ServingReadinessUpdate` snapshots with a monotonic revision
maintained by the topology projection (independent of catalog revisions).
Entry key: `(namespace, canonical_model_id)`.

`TopologyEntry`:

| Field | Semantics |
| --- | --- |
| `state` | `READY` / `UNAVAILABLE` / `UNKNOWN` — serving readiness in core-frontend semantics (namespace-wide role DNF; see the mental model). |
| `present_roles[]` / `missing_roles[]` | Live vs missing typed roles across the namespace. Cleared when `state = UNKNOWN`. |
| `members[]` | One per participating endpoint: `{endpoint, declared roles, optional pool_id}`. `pool_id` is the **stable** pool link, present only while the pool is materialized; the current generation is resolved through the catalog. |
| `duplicate_role_endpoints[]` | Typed roles declared by more than one endpoint under this key. Only `PREFILL`/`DECODE` are valid values. An observable fact; any "disaggregation degraded" interpretation is version-dependent consumer policy. |
| `legacy_fallback_active` | True when any card without a `worker_type` disabled strict gating (readiness = any live worker). |
| `adapters[]` | Per-LoRA readiness: `{canonical_model_id, state, missing_roles}`. Adapters never appear as top-level entries. |

Validation: at least one member; member roles non-empty; entry/member
namespace consistency; duplicate members, adapters, and roles rejected;
`duplicate_role_endpoints` values outside `PREFILL`/`DECODE` rejected.

### SubscribeKvPoolLoad

Emits complete `KvPoolLoadUpdate` windows (`window_sequence`, `observed_ms`,
`window_ms`); a pool absent from the next window is gone. Per-pool
`KvPoolLoadEntry` carries three independent signals, each with
observed/expected rank coverage: KV occupancy (`kv_used_blocks` /
`total_kv_blocks`), decode load (`active_decode_blocks`), and prefill load
(`active_prefill_tokens` / `prefill_token_capacity`). Partial coverage is
reported honestly (`observed < expected`) and must not be read as zero load.

## Consumer lifecycle rules

| Observation | Required consumer reaction |
| --- | --- |
| New `ProducerIdentity` for a known `PoolId` in the catalog | Drop the CKF replica, resubscribe with the new `expected_producer`. |
| Pool absent from a catalog snapshot | Drop its CKF and load state; the topology member loses its `pool_id`. |
| `TopologyEntry` turns `UNAVAILABLE` | Stop routing to this key even though its pools remain published. |
| `TopologyEntry` `UNKNOWN` | Consumer policy; conservatively skip while READY alternatives exist. |
| Stream lag → `RESOURCE_EXHAUSTED` | Resubscribe from scratch (snapshot + deltas). |
| Marker/version mismatch → `FAILED_PRECONDITION` | Deployment skew; do not retry without upgrading. |

Routing is always gated by the topology plane and matched by the pool plane:
pool presence alone never implies routability, and the two planes may disagree
transiently at revision boundaries.
