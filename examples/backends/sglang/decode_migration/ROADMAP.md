# Dynamo Decode Migration Roadmap

## Current Design

Decode migration is an opt-in operator in Dynamo's normal Rust frontend:

```text
HTTP -> preprocessing -> DecodeMigration -> PrefillRouter -> backend
```

It is enabled by `--router-mode kv --enable-decode-migration`. Requests without
`nvext.decode_migration` are unchanged. Migration requests provide independent
source and destination `RoutingConstraints` and one `token_id` or
`sequence_length` trigger.

Workers keep the normal model card and `dynamo.backend.generate` endpoint.
Repeatable taints such as `decode/fast` and `decode/slow` identify scheduling
pools; migration support is worker capability metadata, not a separate class or
model.

## Transaction

1. Select and pin the source before `PrefillRouter`.
2. Run the normal prefill path and forward source output.
3. At the trigger, select a different destination.
4. Describe the source, reserve a destination session and opaque room, then
   quiesce the source.
5. Arm and dispatch the exact destination continuation.
6. Reconcile source tail and destination replay by token position.
7. After valid destination output, activate it and commit the source.

The source stays authoritative until commit. Destination selection, reserve,
arm, transfer, or first-output failures resume the retained source. Client
cancellation aborts both sides as needed. Same-rank handoffs are serialized so
SGLang's current coarse pause returns no `busy` fail-open requests.

The control endpoints are `migration_prepare`, `migration_sync`, and
`migration_finalize`. Rooms are opaque; rank is explicit request state.

## Guarantees

- Trigger detection scans coalesced stream chunks.
- A source item with a finish reason is never migrated.
- Committed but unstreamed source tokens are emitted once.
- Destination replay is trimmed and remaining token limits are adjusted.
- A destination continuation must consume the stream created by prepare; it
  cannot silently fall back to fresh generation.
- Before commit, dropped client streams receive best-effort source and
  destination cleanup.

## Remaining Work

### Compatibility-aware routing

Filter candidates by protocol version, model revision, page size, KV
dtype/layout, PP layout, transport, and supported TP transformation before
source quiescence. Retry destination selection when a candidate disappears.

### Hard destination leases

Move advisory reservation state into SGLang and expose granted KV capacity,
expiry, destination rank, and capability fingerprint. Make reserve, grow, arm,
activate, abort, and expiry idempotent.

### Per-request source parking

Replace SGLang's scheduler-wide pause with a parked request so unrelated decode
and independent migrations continue. Dynamo's transaction ownership should not
need to change.

### Incremental KV sync

Reuse one lease and migration ID to copy monotonically increasing stable ranges
before the trigger. At the trigger, quiesce, copy the final delta, activate, and
commit through the existing path.

### Production hardening

Add multi-DP routing tests, destination retry, phase deadlines, reservation-aware
scoring, transfer and cleanup metrics, and fault injection for worker loss,
timeouts, lost responses, and lease expiry.

## Validation

The branch passes 19 focused Rust migration tests and 9 Python SGLang-handler
tests, including stream coalescing, finish races, cancellation, rollback, and
dropped streams. Live coverage includes Qwen3-0.6B TP1-to-TP1 and Qwen3-8B
TP4-to-TP1 with NIXL staging.

In a 200-sample Qwen3-8B GSM8K run at temperature 1.0 and 32K context, all 200
requests reached `</think>` and committed migration. There were no busy
rejections, rollbacks, invalid responses, or truncations. Accuracy was 96.0%
without migration and 96.5% with migration (`p=1.0`, paired McNemar exact).
