<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay: Producer Architecture

One Relay runs per data center. It observes the DC-local Dynamo deployment and
publishes two independent, WAN-consumable projections of it. This document
describes the producer model: which facts the Relay consumes, what it derives
from them, and how each Dynamo deployment shape maps onto the published state.
The wire encoding is specified in [`grpc-contract.md`](grpc-contract.md) and
[`../protocol/README.md`](../protocol/README.md).

## Inputs

The Relay consumes only DC-local, positively advertised facts:

| Input | Source | Feeds |
| --- | --- | --- |
| Model deployment cards | discovery plane, namespace-wide query | endpoint membership, declared roles and `needs`, model registrations, query semantics, load capacity |
| Instance availability | per-endpoint instance watches | live worker sets; authoritative once the watch delivers its initial snapshot |
| KV event source advertisements | typed `KvEventSource` discovery entries | pool materialization gate, per-`(worker_id, dp_rank)` ingest and recovery |
| Worker KV-occupancy events | DC event plane (`kv_metrics` subject) | authoritative pool KV-load windows |

The Relay never probes workers, never synthesizes facts it cannot observe, and
never assumes another component's internal state.

## The two projections

```text
Inputs                          Projections (both: complete revisioned snapshots)
──────                          ─────────────────────────────────────────────────
cards ─┬─▶ endpoint membership ─┬─▶ Pool catalog        (KV facts, per endpoint)
       │                        │     KvPoolDescriptor, CKF stream, load stream
availability ──────────────────┤
                                └─▶ Topology projection (serving facts, per
KV sources ────▶ materialization      (namespace, canonical model))
                                        TopologyEntry
```

- The **pool catalog** answers "what KV state exists and how to query it".
- The **topology projection** answers "can this model in this namespace serve a
  request", in the same semantics the Dynamo frontend uses.
- They are deliberately two flat collections, not a tree. A topology member
  links to its pool through the **stable `KvPoolId`**; producer generations are
  resolved through the catalog. Namespace is the topology key, never a pool
  identity dimension: pool lifecycle (fencing, generation swaps, CKF, load) is
  strictly endpoint-local, and namespace names are unique only within one DC.

## Pools

A pool is one serving endpoint's KV publication. `PoolId = (identity_version,
IndexerDomainId, DcId)` where the cache-semantics digest covers the model
source, KV block size, and hash format, and the routing-scope digest covers the
endpoint triple — distinct endpoints always produce distinct pools, and
colliding `PoolId`s are fenced, never merged.

A pool **materializes** only when all three conditions hold for the endpoint:

1. the endpoint's cards are structurally materializable (single indexer domain,
   at least one valid registration, and no `PoolMaterialization` or
   `ServingTopology` conflict);
2. the KV-state endpoint resolves unambiguously;
3. at least one expected worker/rank advertises an **active KV event source**.

Losing a source, domain, pool-only condition, or unambiguous KV-state endpoint
**dematerializes** the pool (catalog withdrawal precedes actor teardown), while
the endpoint remains a topology member with no pool link. A `ServingTopology`
conflict also removes the endpoint from the topology projection. A
`ServingBinding` conflict omits only the registration that cannot be resolved;
other valid endpoint bindings remain eligible. This is why there are no phantom
pools: an endpoint that never publishes KV events — a surface-less encode
worker, for example — never appears in the catalog at all.

Each materialized pool owns exactly one CKF actor. Restart and failure produce
a new producer generation (`ProducerIdentity`); subscriptions are bound to the
exact generation, and per-`(worker_id, dp_rank)` recovery is fenced by source
epochs. Generation swaps churn only the catalog — the topology projection is
untouched because members reference the stable `PoolId`.

Engine-specific block IDs stop at the pool actor's per-source ingest boundary.
The actor derives Dynamo canonical sequence hashes from each event's token hash:
the root sequence hash equals the first token hash, and each child extends its
canonical parent. It retains the engine ID only as source-local lineage for
parent lookup and removals. CKF fingerprints, snapshots, deltas, and WAN
consumers therefore use only canonical hashes. LoRA and cache-namespace salts
are already part of the token hash, so this boundary does not require
backend-specific hashing logic.

## Topology projection

The topology key is `(namespace, canonical model id)`, where the canonical
model id is the served/display name — mirroring the core's topology rendezvous
key. For each key the Relay groups all endpoints in the namespace whose cards
register that model, and evaluates readiness with the **same algorithm as
`Model::evaluate_namespace`** in the Dynamo frontend:

- roles of live typed workers across the whole namespace form `present`;
- a declared typed role with no live worker anywhere becomes `missing`;
- every live unit with a non-empty `needs` disjunctive normal form (DNF) must be
  satisfiable from `present`; dead units' `needs` are ignored;
- ready ⇔ any live worker ∧ nothing missing;
- **legacy fallback**: any card without a `worker_type` (including mixed
  typed/legacy namespaces) disables strict gating entirely — ready ⇔ any live
  worker, reported with `legacy_fallback_active = true`.

State is `UNKNOWN` only while at least one participating endpoint has not yet
delivered an authoritative availability snapshot. An endpoint absent from
membership is a *fact*, not uncertainty: if its role is required, the entry is
`UNAVAILABLE` with that role in `missing_roles`.

`duplicate_role_endpoints` reports an **observable fact**, not a verdict: the
typed PREFILL/DECODE roles declared by more than one endpoint under one key.
In current Dynamo versions this condition disables the DC-local P/D rendezvous
while ordinary serving continues — but that consequence lives in the frontend's
private state, is version-dependent, and is therefore interpretation left to
the consumer. Duplicated AGGREGATED endpoints are legal scale-out and are never
reported.

Availability writes are fenced by slot incarnation: a retired endpoint task can
never overwrite the availability published by its replacement.

Known limitation: readiness projection assumes one base-card/`ModelType` shape
per endpoint. Distinct base cards on the same endpoint that share the same
`(worker_type, needs)` can collapse into one readiness unit.

## Deployment shapes

**Aggregated.** One endpoint → one pool (`pool_roles: [AGGREGATED]`) → one
READY entry with one member. Worker replicas do not split the pool; they are
`(worker_id, dp_rank)` sources of one CKF. Multiple aggregated models in one
namespace are fully independent keys.

**Prefill/decode disaggregation.** Two endpoints → two pools with distinct
`PoolId`s, CKFs, and load entries (`[PREFILL]` and `[DECODE]`) → one entry with
two members. Both CKFs are meaningful — this mirrors the local router, where
both legs are KV-aware: prefill workers publish their own KV events (prefix
reuse on the prefill side drives time to first token (TTFT)), decode events
drive decode-side reuse.
Their query semantics may differ within one topology (prefill hashes are always
the standard format; decode may use a speculative format), so hashes must be
computed per pool. Death of either role turns the entry `UNAVAILABLE` while
both pools stay published.

**Encode/prefill/decode (EPD), multimodal.** Adds an encode endpoint. Encode
participates in readiness (its `needs` DNF requires a P+D pair or an aggregated peer) but does
not publish KV: it stays a member with no pool link unless it ever advertises
an active KV event source, in which case it materializes an ordinary
endpoint-local pool.

**LoRA.** An adapter is not a pool, not a load entry, and not a top-level
topology entry. It is an extra `ModelRegistration` on its base pool's
descriptor and an element of the base entry's `adapters[]`. Base and adapter KV
share one physical CKF, separated by the hash salt (the canonical adapter
name; base requests are unsalted). Adapter readiness requires the base
topology to be ready *and* adapter membership on every applicable role —
DECODE, AGGREGATED, and PREFILL; ENCODE is a required topology dependency of
the base but never adapter-bearing.

**Legacy.** Cards without `worker_type` produce `pool_roles: [LEGACY]` and the
namespace-wide legacy fallback described above.

## What the producer deliberately does not publish

- **Ingress / request targets.** The core records no such fact; the descriptor's
  `serving_endpoint` is the pool-owning worker endpoint, not an address for
  client requests.
- **Derived indexes** (model → pool, alias → model). Consumers build them from
  catalog snapshots; the Relay does not enforce cross-pool name uniqueness.
  Same-target repeats are allowed, but a consumer omits a lookup
  name that maps to more than one distinct `BindingIdentity`.
- **Cross-endpoint KV merges.** One CKF never aggregates another endpoint's KV.
- **Router scheduler load.** Scheduler events are replica-local and lack publisher identity, so
  the Relay cannot aggregate them authoritatively across router replicas.
- **KV-state endpoints and per-adapter load.**
