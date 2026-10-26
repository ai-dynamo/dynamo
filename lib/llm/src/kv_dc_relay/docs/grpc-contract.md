<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay: gRPC Contract

Service `dynamo.kvrelay.v1.KvEventRelay`, defined in
[`../wan/grpc/protocol/relay.proto`](../wan/grpc/protocol/relay.proto). This document is the
contract-level view: RPCs, message semantics, validation, and lifecycle rules.
The transport-neutral CKF payload (CBI1) is specified in the
[protocol README](../wan/grpc/protocol/README.md#cbi1-payload); the producer
model behind the contract is in [`architecture.md`](architecture.md).

## Envelope and versioning

Every top-level request and response carries a `fixed32 contract_marker` at
field number 127. Its required value is `0x4B565231` (`KVR1`). Every response
also carries `protocol_version` and the typed `RelayIdentity`. The server rejects
a mismatched request marker with `FAILED_PRECONDITION` / `CONTRACT_MISMATCH`.
Consumers reject a mismatched response marker or version before applying any state.

`RELAY_PROTOCOL_VERSION` is **1**. Relay and consumers must use the same compatibility major,
not necessarily the same Dynamo release.

`RelayIdentity` separates the backing runtime instance (`drt_instance_id`) from the Relay's
lifecycle (`relay_incarnation`). Each `KvDcRelay::start` call generates a new incarnation,
including restarts that reuse the same runtime instance.

### v1 Evolution

- Keep package `dynamo.kvrelay.v1`, protocol version `1`, and marker `KVR1` for additive changes.
- Preserve existing field numbers, types, meanings, RPC names, and error reason names/numbers.
  Reserve both numbers and names when removing fields or enum values; never reuse them.
- Add fields only when absence preserves previous behavior. Readers ignore unknown protobuf
  fields; writers cannot assume old readers retain them when reconstructing a request.
- New enum values and `oneof` alternatives are not automatically semantically compatible.
  Advertise new per-pool formats only with an explicit discriminator; old readers quarantine
  the affected entry. Do not introduce new mandatory filter frame kinds into a v1 stream.
- Identity composition, existing hash/CKF interpretations, and required behavior cannot change
  under v1. Use a new package/service major for incompatible changes and keep v1 during migration.

### Unknown Types and Validation

Validate the envelope first, then each complete snapshot entry independently. The wire helpers
return errors without mutating state; `WireIdentityError::is_unsupported()` distinguishes an
unsupported entry from malformed known data. `*_UNSPECIFIED`, missing required fields, duplicate
sets, and namespace inconsistencies remain invalid. `ServingReadinessState.UNKNOWN` is a valid
state, not an unknown enum value.

| Observation | Required scope and action |
| --- | --- |
| Unknown pool identity source/version, CKF/hash format, pool role, or model target | Quarantine the containing pool, including its previous CKF/load state. Keep unrelated supported pools from the complete snapshot. |
| Unknown topology state or role, including member/adapter dependencies | Quarantine the entire topology entry; never remove the unknown dependency and infer `READY`. Keep unrelated entries. |
| Unknown filter frame kind or unsupported CBI1 payload | Invalidate only that pool replica and close its stream. Do not skip a frame and continue applying deltas. |
| Missing or unknown `ModelTarget.target` alternative | Treat the pool as unsupported. Protobuf readers cannot distinguish an unknown alternative from an unset `oneof`; never infer a base target. |
| Malformed entry with a trustworthy key | Quarantine that entry; never retain its old usable state as a fallback. |
| Invalid envelope, duplicate top-level keys, or entries that cannot be identified safely | Invalidate the affected state plane; do not partially apply an ambiguous snapshot. |

An unsupported entry is not permission to use default semantics. A newly received complete
snapshot replaces the previous supported view, even if some entries are quarantined. These are
consumer requirements; this package does not implement a consumer state machine.

### Producer Identity Key

`ProducerIdentity` is the frozen v1 subscription key. Equality includes:

- `pool_id.identity_version`, both domain digests **and their sources**, and `dc_id`;
- `producer_incarnation` and `layout_generation`;
- every existing `ckf_format` field: version, seed, bucket count, fingerprint bits, and slots per bucket.

Rust callers use `ProducerKey::try_from(&identity)` to validate and compare these explicit fields.
Metadata belongs in `KvPoolDescriptor`, not in the identity or its nested messages. Endpoint,
registrations, roles, and future descriptive fields do not participate in generation equality.
Their updates still require catalog/query/readiness revalidation, but do not alone restart a CKF
subscription. Never change the meaning of an existing key field or silently drop it from equality.

### Machine-Readable Errors

Relay application errors include the ASCII trailing metadata key `kv-relay-error-reason`.
Its value is the exact protobuf `RelayErrorReason` name, for example
`RELAY_ERROR_REASON_PRODUCER_CHANGED`. This is a custom trailer, not `google.rpc.Status`
inside `grpc-status-details-bin`; no error payload is inserted into the data stream.
Rust callers can use `relay_error_reason(&tonic::Status)`. Diagnostic message text is not stable.

The table omits the common `RELAY_ERROR_REASON_` prefix:

| Reason | gRPC code | Client action |
| --- | --- | --- |
| `CONTRACT_MISMATCH` | `FAILED_PRECONDITION` | Stop retries with this contract; use a compatible client/server major. |
| `INVALID_REQUEST` | `INVALID_ARGUMENT` | Fix the request; do not retry unchanged. |
| `UNSUPPORTED_FEATURE` | `FAILED_PRECONDITION` | Quarantine the requested pool; wait for supported semantics or upgrade. |
| `POOL_NOT_FOUND` | `NOT_FOUND` | Refresh catalog, drop withdrawn state; subscribe only to an advertised producer. |
| `PRODUCER_CHANGED` | `FAILED_PRECONDITION` | Drop the old replica, refresh catalog, and subscribe to the current key. |
| `RESOURCE_LIMIT` | `RESOURCE_EXHAUSTED` | Retry admission with bounded exponential backoff and jitter; reduce subscriptions if persistent. |
| `SUBSCRIBER_LAGGED` | `RESOURCE_EXHAUSTED` | Invalidate this stream's state and reopen from a full snapshot/window with backoff. Other planes remain independent. |
| `SNAPSHOT_PROGRESS_TIMEOUT` | `RESOURCE_EXHAUSTED` | Discard incomplete assembly, fix draining/backpressure, and reopen with backoff. |
| `PUBLICATION_UNAVAILABLE` | `UNAVAILABLE` | Invalidate the affected stream state, refresh catalog for pool streams, and reconnect with backoff. |
| `INVALID_PUBLICATION` | `FAILED_PRECONDITION` | Invalidate the replica and refresh catalog; do not keep retrying the same fenced generation. |
| `INTERNAL` | `INTERNAL` | Invalidate the affected stream state, refresh catalog, and retry with backoff; surface persistent failures. |

Older servers, proxies, and gRPC itself can return statuses without this trailer. Unknown or
`UNSPECIFIED` reasons are equivalent to absent reasons: use the status code, never message parsing.
For `FAILED_PRECONDITION`, refresh identity/catalog and verify the envelope; do not blindly retry
an unchanged request. For `RESOURCE_EXHAUSTED` or `UNAVAILABLE`, reconnect with backoff; a pool
stream always starts with a new snapshot. Request errors need correction. Authentication and
authorization errors belong to the external proxy. An unexpected clean EOF also invalidates the
affected stream state and requires reconnecting; it is not proof that cached state remains fresh.

## RPCs

| RPC | Kind | Purpose |
| --- | --- | --- |
| `GetRelayInfo` | unary | Protocol version and typed Relay identity |
| `WatchKvPoolCatalog` | server stream | Complete revisioned pool-catalog snapshots |
| `SubscribeKvPool` | server stream | CKF snapshot + deltas for one exact producer generation |
| `SubscribeServingReadiness` | server stream | Complete namespace topology projections |
| `SubscribeKvPoolLoad` | server stream | Complete pool-load windows |
| `SubscribeServingLoad` | server stream | Complete scheduler-load and frontend-load windows |

Streaming requests require a non-empty `subscriber_id` (≤ 128 UTF-8 bytes,
no control characters). It is a diagnostic label, not an authentication identity or resume
token. Reusing it does not replace another stream or bypass admission limits. Each
stream type has an independent subscriber limit; pool publication additionally
bounds total pool streams, subscribers per pool, and initialized publication
hubs. Breaching any bound returns `RESOURCE_EXHAUSTED`.

The server's default maximum message size is 8 MiB. Configure consumers and sidecars to accept
that size as well: a full snapshot chunk carries 4 MiB of bucket words plus CBI1 and protobuf
headers, exceeding tonic's default 4 MiB client receive limit. Rust clients can use
`KvEventRelayClient::max_decoding_message_size(8 * 1024 * 1024)`.

## Transport security boundary

Relay serves plaintext HTTP/2 gRPC without authentication. mTLS is an optional external
sidecar, not a Relay feature. Across a trust boundary, expose only the authenticated proxy
and keep Relay's listener private. Certificates, authorization, and rotation belong to
the proxy; see the [deployment guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/kv-aware-routing/kv-dc-relay.md#optional-mtls-sidecar).

## Stream contracts

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

Same-target name repeats across pools are allowed. If one lookup name resolves to more than one
distinct `BindingIdentity`, a consumer name index omits it fail-closed.

Catalog revisions may skip numbers when watch notifications coalesce. Each update is complete;
replace the local catalog with the first snapshot after reconnecting.

#### Query Semantics and Model Targets

`KvQuerySemantics` defines one atomic token-to-sequence-hash format:

- `DYNAMO_STANDARD_V1` uses `kv_block_size` token windows.
- `DYNAMO_EAGLE_V1` uses `kv_block_size + 1` token windows with a `kv_block_size` stride.

Each format also fixes token and multimodal encoding, request-wide LoRA/cache-namespace salting,
local block hashing, and rolling sequence hashing. Compute hashes separately for each pool using
its declared semantics; Prefill and Decode formats can differ. Unknown or unspecified formats
must not fall back to a known format.

The shared [hashing implementation and test vectors](../../../../kv-router/src/protocols.rs)
define `compute_block_hash_for_seq` and `compute_seq_hash_for_block`. CBI1 transports bucket
images; it does not replace this token-to-sequence-hash pipeline.

The producer emits one canonical base target and any LoRA targets backed by that model. For a LoRA
request, the hash salt is the canonical `ModelTarget.lora.adapter`, not an alias or the
registration's lookup ID. Base-model requests use no LoRA salt. Base and adapter KV entries share
one physical pool while retaining distinct cache keys.

### SubscribeKvPool

The request names an exact `expected_producer: ProducerIdentity` taken from
the catalog — not just a pool. A stale generation is rejected with
`FAILED_PRECONDITION` (producer mismatch), an unknown pool with `NOT_FOUND`,
and retirement while waiting for admission or lazy hub initialization can return
`UNAVAILABLE` through cancellation. A mismatch detected by the identity check after
initialization still returns `FAILED_PRECONDITION`. Refresh the catalog before retrying
these responses. A subscriber can never silently read a replacement generation's filter.

The stream delivers `FilterUpdate` frames: `SNAPSHOT_CHUNK`s assembling one
consistent CBI1 image, then contiguous `DELTA`s, with `HEARTBEAT`s in between.
Heartbeats start only after the complete initial snapshot and do not advance its sequence.
These payloads come directly from the universal publisher; the gRPC adapter does not re-encode
the CKF or maintain an independent publication hub.
A subscriber that falls behind the bounded queue is terminated with
`RESOURCE_EXHAUSTED` and must resubscribe from scratch.
The initial snapshot producer has an independent per-frame progress timeout. If a client stops
draining the bounded bootstrap queue, the Relay stops that producer and releases encoder admission
at the deadline. The stream yields `RESOURCE_EXHAUSTED` after gRPC resumes polling it; this
client-local timeout does not fence the producer generation. A client that never resumes can retain
one of the bounded total pool-stream slots until its transport disconnects or the Relay shuts down.

### SubscribeServingReadiness

Emits complete `ServingReadinessUpdate` snapshots with a monotonic revision
maintained by the topology projection (independent of catalog revisions).
Entry key: `(namespace, canonical_model_id)`.

`TopologyEntry`:

| Field | Semantics |
| --- | --- |
| `state` | `READY` / `UNAVAILABLE` / `UNKNOWN` — serving readiness in core-frontend semantics (namespace-wide role disjunctive normal form (DNF); see the mental model). |
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
`window_ms`); a pool absent from the next window is gone. Nothing is sent before
the first window, one `window_ms` after the Relay starts serving. Per-pool
`KvPoolLoadEntry` carries worker-authoritative KV occupancy
(`kv_used_blocks` / `total_kv_blocks`) and observed/expected rank coverage.
An aggregate is authoritative only when every declared rank has reported and
every rank has a nonzero known capacity. Incomplete usage is encoded with
`observed < expected`; unknown capacity is encoded as zero capacity. Aggregate
sums use the same saturating `u64` arithmetic as the local load aggregator, so
the Relay forwards a saturated value as `u64::MAX`. Neither incomplete coverage
nor zero capacity may be read as zero load.

Full rank coverage does not prove current rank liveness. The Relay retains the latest value per
rank without a rank timestamp, and a change-deduplicating publisher need not heartbeat or retry.
A new WAN window therefore cannot establish per-rank freshness; that requires publisher
heartbeats or rank-scoped freshness metadata.

An empty `pools` list is a valid idle window. Use local receive time to track WAN stream freshness;
`observed_ms` is producer wall-clock metadata, not a cross-data-center freshness guarantee.
Router scheduler load is published separately by `SubscribeServingLoad`.

### SubscribeServingLoad

Emits complete `ServingLoadUpdate` windows (`window_sequence`, `observed_ms`, `window_ms`) at
the same cadence and with the same fanout bound as `SubscribeKvPoolLoad`; a lagging subscriber
is terminated with `SUBSCRIBER_LAGGED`. Its subscriber limit is independent. A pool or model
absent from the next window is gone. Like `SubscribeKvPoolLoad`, the stream sends nothing until
the Relay has built its first window, so a subscriber never sees a placeholder empty window.

A Relay that predates this RPC answers `UNIMPLEMENTED`: serving load is unavailable from it, which
is not an outage.

Producers publish at least once per second, and the Relay treats a report as fresh for three
of those intervals; a producer must not publish less often. Producers publish only with
`DYN_SERVING_LOAD_PUBLISH` set to a truthy value. Set it on every frontend (including the routers
embedded in frontends) and on standalone routers. Without it the Relay still serves the stream,
but every view is `UNAVAILABLE`, and it logs a warning after 30 s without any publisher.

| Entry | Key | Contents |
| --- | --- | --- |
| `PoolServingLoad` | `ProducerIdentity` | Every catalog pool, including pools without data. Router scheduler load (`active_prefill_tokens`, `active_decode_blocks`) and deployment facts. `requests` is unset. |
| `ModelServingLoad` | `(namespace, canonical_model_id)` | Every model a catalog pool registers in the namespace or a recently published frontend serves there. Frontend request lifecycle and token load, frontend coverage, and stable `serving_pools` links. |

Each `LoadView.status` states coverage of its expected sources:

| Status | Pools (router schedulers) | Models (frontends in the namespace) |
| --- | --- | --- |
| `COMPLETE` | Every expected router has a recent report of every declared rank. | The roster is known and every expected frontend published recently. |
| `DEGRADED` | Some reports are missing, or the router roster is unknown. | Some expected frontends are silent, or the frontend roster is unknown. |
| `UNAVAILABLE` | No router reported recently. | No frontend published recently. |

These states describe report coverage only. `UNAVAILABLE` here means no report was heard; it
says nothing about whether the pool or model can serve, which is what
`SubscribeServingReadiness` reports.

`COMPLETE` means every source publishing this stream has reported; it does not mean every
process that should publish is alive, or publishes at all. A frontend or router that does not
publish (the gate is off, or it predates this RPC) is invisible to the Relay, so a fleet where only
some of them publish reads `COMPLETE` with a partial sum. Enable the gate on every frontend and
router serving the namespace. A source is expected from when the Relay first sees its
discovery record until one freshness window (three publish intervals) later, and after that
only while it keeps reporting. So a source that is starting holds its views `DEGRADED` until it
reports, and a source that stops reporting, whether it exited or was partitioned, drops out of
the expected set when its last report expires; `expected_frontends` and `observed_frontends`
fall with it. The roster is unknown until the Relay's discovery watch delivers a complete set,
and again while that watch is re-established. ZMQ-broker publishers do not register, so in
broker mode the expected sources are exactly the ones heard recently.

Gauges (`active_*`, `requests_awaiting_first_token`, `requests_generating`, and the in-flight
token counts) are set exactly when the view is `COMPLETE`: a partial sum would read as lower load.
Model `*_total` counters are always set. They count what one `relay_incarnation` observed: a
frontend first heard from its first frame, or after it restarts, contributes everything it
reports, and one first heard later contributes only its growth from then on. They absorb frontend
restarts and restart from zero with each `relay_incarnation`; compute rates only within one
incarnation.
Fields that belong to the other view kind are never set. `PoolDeploymentStatus` fields are unset
while the pool's runtime configuration is invalid or, for `max_concurrency`, while a worker
does not declare its limit.

Router `active_prefill_tokens` is the scheduler's live estimate of prefill work it has assigned
and not yet seen complete, the same number the router schedules with. With a prefill model
configured it decays as prefill progresses; without one it stays at the full effective prompt
tokens until prefill completes.

Within one scheduler group, router reports overlap, so the Relay takes each rank's maximum.
Distinct groups are disjoint, so it sums them. A standalone router that reconnects counts only
its latest report. Frontends are heard in every namespace that holds a catalog pool or a served model, and in
the Relay's own namespace, where frontends without an exact namespace publish. Those frontends
serve every namespace's models, so their load of a model that catalog pools register in exactly
one other namespace counts under that namespace, and their coverage counts toward it. A model
registered in several namespaces stays under the Relay's namespace.

Totals stay monotonic within a frontend incarnation. Three cases still miscount; they are bounded
and never make a counter go backwards:

- A frame in which a frontend's total for a model falls (an alias was unregistered) counts no
  growth, including any real growth in that frame.
- An alias moved to another model carries its history into the new model's totals.
- An alias removed and then registered again counts its history twice.

Other known limits:

- A restarted standalone router reports under a new group, so for up to three publish intervals
  its previous process's reports are still fresh and both are summed.
- A router reports a dimension it does not track as 0: prefill pools always report
  `active_decode_blocks = 0`, as does any router with that tracking disabled.
- Within a replica group the maximum undercounts what replicas do not share: output blocks, which
  each replica tracks locally, and two routers for one endpoint in the same process, which ignore
  each other's events.
- A KServe gRPC frontend stops publishing when it starts draining, so it drops out of the expected
  set while its last requests finish.
- With a Python chat processor (`--dyn-chat-processor vllm` or `sglang`), prompt tokens reach the
  frontend with the first output chunk, so `awaiting_first_token_input_tokens` stays near zero
  through queueing and prefill.

Validate with `validate_serving_load_update` before applying a window: an invalid envelope,
a missing relay identity, a pool without an identifiable producer, or a duplicate pool or model
key invalidates the whole serving-load plane. Then validate each entry with
`validate_pool_serving_load` or `validate_model_serving_load`. An unknown `DataStatus` or
unsupported pool identity (`is_unsupported()`) quarantines that entry only; a malformed entry
with a trustworthy key, including a presence-rule violation, is also quarantined. Never read an
unset gauge as zero.

## Consumer lifecycle rules

| Observation | Required consumer reaction |
| --- | --- |
| New `ProducerIdentity` for a known `PoolId` in the catalog | Drop the CKF replica, resubscribe with the new `expected_producer`. |
| Pool absent from a catalog snapshot | Drop its CKF and load state. Any topology reference to it is unresolved until the independent views converge. |
| `TopologyEntry` turns `UNAVAILABLE` | Record authoritative unavailability; published pools do not override it. |
| `TopologyEntry` `UNKNOWN` | Record readiness as unknown, not ready or authoritatively unavailable. |
| Stream lag → `RESOURCE_EXHAUSTED` | Resubscribe from scratch (snapshot + deltas). |
| Marker/version mismatch → `FAILED_PRECONDITION` | Deployment skew; do not retry without upgrading. |

Pool presence and serving readiness are separate facts. Their snapshots can temporarily
disagree; neither stream updates the other atomically. Consumers decide how to use these facts.

Keep catalog, pool filters, readiness, pool load, and serving load as independent state machines. Install a filter
only after validating its complete snapshot; apply deltas only when Relay identity, producer
identity, format, and base sequence match the installed state. A gap, malformed CBI1 frame, or
identity drift invalidates only the affected pool replica. Replace readiness from the first
complete snapshot after reconnecting. Replace load from the next complete window; load gaps
require reopening only that stream and do not invalidate CKF state.
