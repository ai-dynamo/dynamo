# Dynamo Decode Migration Roadmap

## Objective

Dynamo coordinates live decode-to-decode migration while every compatible
SGLang worker remains a normally registered model worker. Each worker exposes
the same `generate`, `migration_prepare`, `migration_sync`, and
`migration_finalize` endpoints. Dynamo selects source and destination instances
using role metadata, live load/KV capacity, cache affinity, and a transfer
compatibility identifier.

No worker is hidden by suppressing its model deployment card. Fast and slow are
routing classes, not different worker implementations.

The matching engine-level contract and state invariants live in the SGLang
worktree's root `ROADMAP.md`.

## What Is Retained From Earlier Decode Disaggregation

The historical `warnold/sglang-dd` branch had several useful properties:

- workers exposed normal generation plus a migration control endpoint;
- Dynamo retained the exact instance ID selected for generation;
- source migration calls used direct instance routing;
- destination selection could use KV-aware routing;
- one frontend stream survived the worker handoff;
- buffered output was reconciled with an explicit token watermark;
- the old source stream remained alive until destination execution was proven.

We retain those behaviors. We do not retain component-per-tier naming,
`this_seqlen` as a hardware role, MDC suppression for intermediate tiers,
rank-encoded bootstrap rooms, or a destructive one-call migration operation.

## Worker Metadata

Migration-enabled workers publish normal MDCs with runtime metadata similar to:

```json
{
  "decode_migration": {
    "protocol_version": 1,
    "decode_class": "fast",
    "transport": "nixl",
    "compatibility_id": "..."
  }
}
```

Every worker in the migration pool starts both the normal decode receiver and
the source-side bootstrap service. The model card exposes the stable NIXL
bootstrap host/port for the worker. DP rank is selected and carried explicitly
in routing/control messages. The
opaque bootstrap room must not encode it.

TP size is published as topology metadata, but is not part of strict
compatibility equality. Current SGLang NIXL supports heterogeneous TP by slicing
GQA/MHA KV heads (optionally through the staging path) and directly transferring
replicated MLA KV. The router must validate that a particular source/destination
TP transformation is supported rather than requiring equal TP sizes.

The migration coordinator is selected by request metadata, for example:

```json
{
  "nvext": {
    "decode_migration": {
      "enabled": true,
      "source_class": "fast",
      "destination_class": "slow"
    }
  }
}
```

Ordinary requests continue through ordinary model routing. Migration requests
enter the coordinator/operator, which performs constrained source and
destination selection. This avoids changing model visibility to enforce the
feature path.

## Control Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant D as Dynamo migration operator
    participant F as Fast source
    participant S as Slow destination

    C->>D: generate with decode-migration nvext
    D->>F: generate (constrained route)
    F-->>D: tokens
    D->>S: migration_prepare(reserve, source endpoint/rank)
    S-->>D: migration ID, reservation, opaque room
    D->>F: migration_sync(quiesce, destination room)
    F-->>D: exact frontiers; pending sender installed
    D->>S: migration_prepare(arm exact frontier)
    S-->>D: ready transfer ticket
    F->>S: NIXL KV write after receiver arming
    D->>S: migration_finalize(activate)
    S-->>D: destination stream ready
    D->>F: migration_finalize(release)
    S-->>D: continued tokens
    D-->>C: one continuous response stream
```

The two destination prepare calls are one idempotent state transition:
`reserved -> ready`. They are separate because a destination can reserve
capacity before the source's exact committed frontier is known.

## Routing Requirements

### Source

- Choose from workers matching model and `source_class`.
- Record the routed instance ID and DP rank from the generation route.
- Direct all source control calls to that same instance/rank.

### Destination

- Choose a different compatible instance matching `destination_class`.
- Consider free KV blocks, active reservations, cache affinity, expected output
  length, transport compatibility, and reservation TTL.
- Bind the returned reservation to destination instance/rank. Never round-robin
  a later control call for that migration.

### Normal MDC publication

Delete the prototype `internal_decode_migration_worker` path. Worker registration
always happens. Until the generic router can dispatch an `nvext` request through
the migration operator, the local harness may use explicit worker components,
but that is test configuration rather than a production visibility mechanism.

## Coordinator State

Maintain one typed record per migration:

```text
migration_id
request_id
source instance + rank + stream
destination instance + rank + reservation
bootstrap room/generation
logical, committed, transferred, emitted watermarks
phase
deadline
terminal outcome
```

Phases:

```text
SOURCE_STREAMING
DESTINATION_RESERVED
SOURCE_QUIESCED
DESTINATION_READY
TRANSFERRING
DESTINATION_RECEIVED
DESTINATION_ACTIVE
SOURCE_RELEASED
ABORTED
```

Control retries must be idempotent. A repeated response cannot advance or
reverse stream ownership twice.

## Stream Ownership

The coordinator is the only component that emits to the client across handoff.
It tracks output positions, not chunk counts. `--stream-interval` may coalesce
several tokens and cannot be used as a migration frontier.

Before destination activation, the source remains authoritative and can resume.
After activation, destination output becomes authoritative. The coordinator
forwards any source-committed but not-yet-emitted tail exactly once and trims any
destination replay prefix by token position.

Client cancellation is propagated to whichever side owns or retains state:

- before destination activation: abort destination reservation and cancel or
  resume/release source as appropriate;
- after destination activation: cancel destination and best-effort release any
  retained source state.

## Migration Policy

The coordinator owns *when* to hand off; SGLang owns *what KV range is stable*.
Keeping these concerns separate lets a request use a token-count threshold, a
semantic boundary, an SLA signal, or a router decision without changing the
prepare/sync/finalize protocol.

The prototype accepts repeated `--migrate-on-token-id` arguments. When any are
present, they take precedence over `--migrate-after-tokens`. For Qwen3, token
ID `151668` (`</think>`) moves the visible-answer phase to the slow worker only
after hidden reasoning has completed. The boundary token is forwarded before
quiescing, so stream reconciliation remains position-based and works when a
stream chunk contains several tokens. If the request finishes in that chunk,
there is no useful continuation to migrate and the coordinator returns normally.

Incremental migration should not add a second policy-specific protocol. The
coordinator should keep the same destination reservation alive, ask the source
for monotonically increasing stable ranges, and use a final quiescent sync at
the semantic or SLA boundary.

## Implementation Sequence

### 1. Refresh the existing local prototype

- Add all four endpoints to every SGLang worker.
- Replace source `migration_prepare` with source `migration_sync`.
- Add destination reservation and receiver-ready control states.
- Pass DP rank explicitly and generate opaque rooms.
- Preserve the existing deterministic stream reconciliation and rollback tests.

### 2. Remove registration suppression

- Delete `internal_decode_migration_worker`.
- Always register worker MDCs.
- Attach decode class and compatibility metadata through runtime config.
- Keep the standalone coordinator's own model alias only for the local harness.

### 3. Integrate constrained routing

- Replace hard-coded `fast` and `slow` singleton clients with model-manager
  discovery and constrained routers.
- Route source generation and retain the selected instance/rank.
- Reserve destination at migration time and direct all subsequent calls.
- Reject self-migration unless explicitly allowed for testing.

### 4. Add incremental sync

- Keep destination reservation alive while source continues decoding.
- Arm and transfer successive stable ranges.
- Use one control migration ID and per-range room/generation IDs.
- Final sync quiesces source; activation and release remain unchanged.

## Test Matrix

Pure coordinator tests:

- reservation, quiescent sync, arm, transfer, activate, release ordering;
- duplicate and out-of-order control responses;
- stream interval reconciliation;
- source finish before reserve, while reserved, and while quiescing;
- cancellation in every phase;
- destination reserve, arm, receive, and activation failures;
- source resume after every pre-activation failure;
- explicit source/destination rank propagation;
- opaque room values independent of rank.

Live two-worker tests:

- deterministic migrated output equals source-only baseline;
- token-count and semantic-token migration triggers;
- a thinking request that finishes before its semantic boundary does not migrate;
- stream intervals 1 and 4;
- finish just below and just above migration threshold;
- injected destination failure with source rollback;
- disconnect during handoff and successful subsequent request;
- both workers retain normal MDC registration;
- logs prove the selected source/destination instance and rank, destination
  reservation/arming, NIXL transfer completion, activation, and source release.

Verified mixed-TP cases on B200:

- Qwen3-8B source TP4 to destination TP1 with NIXL heterogeneous-TP staging,
  stream intervals 1 and 4.
- DeepSeek-V2-Lite source TP4 to destination TP1 through replicated MLA KV,
  Triton attention, deterministic inference, and stream intervals 1 and 4.

The tested image's FlashInfer MLA package was one patch below the SGLang minimum,
and FlashMLA dense decode is restricted to SM90a, so Triton was used for the
DeepSeek-V2-Lite B200 run.

For these tiny 28-29 token test transfers, cold prepare-to-transfer-complete
times were approximately 16 seconds for Qwen heterogeneous TP and 1.3-1.8
seconds for MLA. Subsequent transfers in the same worker lifetime were about
0.51-0.53 seconds for both. These include control-plane polling and kernel/path
warmup and are not bandwidth measurements; a dedicated long-context benchmark
is required before drawing SLA conclusions.

## Verified Prototype Status

The one-shot prototype gate is met. The local harness has exercised normal MDC
registration, destination-prepared opaque rooms, explicit ranks, exact stream
frontiers, rollback, cancellation, and mixed-TP handoff. Coordinator unit tests
cover coalesced chunks, unforwarded committed tails, destination failure and
source resume, completion before count or semantic triggers, disabled migration,
and cancellation before activation.

The Qwen3-8B TP4-to-TP1 paired thinking check migrated 20/20 requests at
`</think>` with 95% baseline and 95% migrated GSM8K accuracy. All hidden
reasoning and extracted answers matched pairwise. The production coordinator
module, rather than a benchmark-local shim, was also exercised in a separate
2/2 end-to-end smoke run.

This does not make the feature production complete. The remaining PR path is to
replace singleton `fast`/`slow` endpoints with constrained model-manager
routing, replace the scheduler-wide pause with per-request parking, enforce
leases and compatibility in the router, and add incremental stable-range sync
and long-context transfer benchmarks.
