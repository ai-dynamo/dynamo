# Dynamo Decode Migration Roadmap

## Objective

Implement opt-in decode-to-decode request migration inside Dynamo's normal Rust
frontend pipeline. Keep ordinary model registration, discovery, preprocessing,
prefill routing, admission, metrics, and HTTP serving. Do not introduce a
coordinator model or hide worker model deployment cards.

Fast and slow are scheduling policies expressed with existing worker taints.
Every migration-enabled SGLang worker exposes the same generation and control
endpoints and may act as source or destination.

## Implemented Architecture

The frontend is enabled with:

```text
--router-mode kv --enable-decode-migration
```

The Rust `DecodeMigration` operator is inserted after token preprocessing and
before `PrefillRouter`:

```text
HTTP/OpenAI frontend
  -> preprocessing
  -> existing failure-retry Migration operator
  -> token backend
  -> DecodeMigration
  -> PrefillRouter
  -> normal decode backend
```

Requests without `nvext.decode_migration` pass through unchanged. Requests with
a migration policy require the frontend flag and KV routing.

The old Python `decode_migration_frontend.py` and synthetic coordinator models
are removed. Python remains only in the normal SGLang worker adapter and in test
clients.

## Request Contract

A request provides independent source and destination constraints and exactly one
tagged trigger:

```json
{
  "nvext": {
    "decode_migration": {
      "source": {"required_taints": ["decode/fast"]},
      "destination": {"required_taints": ["decode/slow"]},
      "trigger": {"type": "token_id", "token_id": 151668}
    }
  }
}
```

Supported triggers:

- `token_id`: fire when that token appears anywhere in a source stream chunk;
- `sequence_length`: fire when prompt plus forwarded output reaches the target.

There is no `any_of` schema. Trigger composition is intentionally deferred.
Qwen3 token `151668` is `</think>` for the tested tokenizer.

The prototype supports one non-beam output sequence and rejects guided decoding.

## Taints and Worker Registration

Workers keep the normal `dynamo.backend.generate` endpoint and model card. Pools
are distinguished with repeatable worker taints, for example:

```text
--worker-taint decode/fast
--worker-taint decode/slow
```

The request's source and destination constraints use Dynamo's existing
`RoutingConstraints`. Source constraints are applied only during source
selection; they are not copied into generic routing fields consumed by prefill
selection.

Workers publish diagnostic migration metadata:

```json
{
  "decode_migration": {
    "protocol_version": 1,
    "transport": "nixl",
    "tp_size": 4,
    "pp_size": 1,
    "compatibility_id": "model:page-size:pp:nixl"
  }
}
```

Current limitation: the Rust selector does not yet consume this metadata for
hard compatibility filtering. The test deployment uses taints to identify known
compatible pools. Destination prepare fails closed if migration is disabled, but
that is later than selection and quiescence planning should be. Compatibility
filtering is a required upstream step.

Do not add `can_source`, `can_receive`, decode-specific worker classes, or
`topology_domains` for this prototype. Capability and scheduling policy are
separate concerns.

## Routing Sequence

1. Select a source with the KV router and source taint constraints.
2. Pin `backend_instance_id`, `decode_worker_id`, and DP rank on the request.
3. Invoke the downstream pipeline.
4. Let `PrefillRouter` perform its normal prompt-prefill and prompt-KV handoff to
   the pinned source. In aggregated mode it remains a passthrough.
5. Forward source output and evaluate the single trigger over token positions.
6. Select a different destination with independent destination constraints.
7. Call source `migration_sync(phase=describe)` for its bootstrap address.
8. Call destination `migration_prepare` to create a logical session and opaque
   room.
9. Call source `migration_sync(phase=quiesce)` to retain the source and capture
   exact KV and stream frontiers.
10. Call destination `migration_prepare` again with the exact continuation to arm
    its NIXL receiver.
11. Dispatch the destination continuation directly to the selected instance. It
    does not re-enter `PrefillRouter`.
12. Reconcile destination replay by token position.
13. On the first valid destination output, activate the destination and poll
    source commit until NIXL reports transfer success.
14. Continue the destination stream as the same client response.

The source remains authoritative through step 13.

## Stream Ownership

Dynamo tracks numeric output positions, not chunks. This is required because
`--stream-interval` can coalesce the trigger token with later tokens.

The operator:

- forwards the source trigger chunk before handoff;
- obtains exact committed and pending frontiers from SGLang;
- emits committed source tokens that were not yet visible;
- trims tokens replayed by the destination;
- reduces destination `max_tokens` and `min_tokens` by committed output count;
- never migrates a source item that already contains a finish reason.

A destination continuation is explicitly marked `is_destination=true`. The
worker must find the stream created by prepare; otherwise it fails instead of
falling back to ordinary generation.

## Failure and Cancellation Semantics

Before source commit:

- destination selection, describe, or reserve failure leaves source generation
  in place;
- source finishing during quiesce aborts the logical destination reservation;
- destination arm, dispatch, transfer, or first-output failure aborts the
  destination and resumes the retained source;
- cancellation aborts destination state and cancels the retained source.

After source commit, the destination owns the stream. A later destination error
is returned to the client; the source is no longer resumable.

`MigrationCleanup` provides drop-time best-effort cleanup when the client stream
is abandoned during the transaction. The destination context is linked to the
parent context so cancellation propagates after handoff.

## What Is Complete

- Normal Rust frontend integration behind one flag.
- Operator placement before `PrefillRouter`.
- Existing KV-router selection and exact worker pinning.
- Independent source and destination taints.
- One typed trigger per request.
- Normal worker model cards and generate endpoints.
- Source describe/quiesce and destination reserve/arm/finalize RPCs.
- Opaque rooms and explicit rank fields in request state.
- Stream-interval reconciliation, finish races, cancellation, rollback, and
  fail-closed destination dispatch.
- Qwen3-0.6B TP1-to-TP1 live tests at stream intervals 1 and 4.
- Qwen3-8B TP4-to-TP1 live migration using the SGLang staging path.

## Prototype Limitations

- Destination `reserve_tokens` is advisory handler state, not a hard KV lease.
- Compatibility metadata is not yet enforced by selection.
- Source quiescence pauses the whole SGLang scheduler and permits one source
  migration at a time.
- The live topology is DP=1; rank fields are explicit, but multi-DP control
  routing is not yet proven.
- Transfer is one-shot.
- There is no automatic destination retry after a selected destination fails.
- Phase-specific metrics and deadlines are incomplete.
- Advanced generation modes and state not represented in the continuation are
  rejected or out of scope.

## Upstream Plan

### 1. Split reviewable SGLang and Dynamo changes

SGLang PR:

- source frontier and retained-request transaction;
- destination receive support on ordinary workers;
- exact destination admission ownership;
- I/O structures, cleanup, and engine tests.

Dynamo PR:

- frontend flag and typed `nvext` schema;
- Rust operator and pipeline insertion;
- taint parsing/publication and migration metadata;
- normal SGLang adapter control endpoints;
- black-box deployment and tests.

Keep generic NIXL changes independent where practical.

### 2. Enforce compatibility before quiescence

Use worker runtime metadata to filter protocol version, transport, model revision,
page size, KV dtype/layout, PP layout, and supported TP transformations. Reject a
request before source generation when no compatible source/destination pool can
exist, and retry destination selection when a candidate disappears before
reserve.

### 3. Add hard destination leases

Have SGLang return a lease containing reserved capacity, expiry, destination
rank, room generation, and capability fingerprint. Expose free and reserved KV
capacity to routing. Cleanup must be idempotent across timeout and cancellation.

### 4. Replace scheduler-wide pause

Implement per-request source parking in SGLang, then allow unrelated generation
and multiple concurrent migrations. Dynamo transaction ownership does not need
to change.

### 5. Add incremental sync

Reuse the same destination lease and migration ID. Before the final trigger,
issue non-quiescent syncs for monotonic stable ranges. At the trigger, quiesce,
copy the final delta, activate, and commit through the existing path. No new
worker role, prefill path, or trigger language is required.

### 6. Add production routing and observability

Add reservation-aware destination scoring, phase deadlines, terminal outcome
metrics, transfer bytes/pages, first destination token latency, rollback count,
and SLO measurements from `measurement_plan.md`.

## Test Requirements

Rust tests:

- pass-through and global enable validation;
- source/destination taint selection and source exclusion;
- source pinning before prefill routing;
- token and sequence triggers across coalesced chunks;
- malformed trigger and unsupported generation validation;
- reserve, quiesce, arm, first-output, commit, and rollback ordering;
- transport/control failures and dropped client streams;
- source finish before trigger and during quiesce.

Python worker tests:

- concurrent prepare serialization;
- abort racing with arm;
- missing prepared destination stream fails closed;
- repeated abort and consumed-stream cleanup.

Live tests:

- deterministic baseline parity;
- stream intervals 1 and 4;
- finish before trigger and immediately after handoff;
- cancellation and successful subsequent requests;
- concurrent trigger attempts;
- equal and heterogeneous TP;
- Qwen3 thinking-boundary paired accuracy;
- no scheduler exception or KV-pool leak signatures.

## Validation Record

As of June 12, 2026, the current worktrees pass 18 focused Rust migration
tests, 9 Python worker-handler tests, 13 focused SGLang destination/frontier/cache
tests, Python binding and `dynamo-llm` compile checks, and clean rebuilt-image
black-box suites at stream intervals 1 and 4. The rebuilt image also passes the
Qwen3-8B TP4-to-TP1 heterogeneous staging suite, including finish races,
cancellation and recovery, and concurrent trigger attempts.

The paired Qwen3-8B GSM8K run collected 20 committed `</think>` migrations after
skipping one completion with no boundary. Fast-only scored 19/20; migrated
TP4-to-TP1 scored 18/20; extracted answers agreed on 18/20. There were no
scheduler exceptions or KV-pool leak signatures. This passes the configured
one-regression smoke gate, but the observed 5-point difference on 20 samples is
not sufficient to claim accuracy neutrality. The next measurement must include a
larger sample, a same-TP migration control, and token-prefix diagnostics to
separate heterogeneous-TP numerical divergence from transfer defects.
