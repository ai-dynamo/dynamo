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
2. While source decode runs, select and reserve a destination, bootstrap its
   receiver, and prepare the source sender.
3. Inspect the source stream in Dynamo and apply the configured trigger policy.
4. Ask SGLang to quiesce generically at Dynamo's acknowledged output frontier.
5. Bind and dispatch the exact destination continuation.
6. Reconcile source tail and destination replay by token position.
7. After valid destination output, activate it and commit the source.

Destination discovery and reservation fail open while the source is still
decoding. Once quiescence starts, the transaction owns the detached request:
post-quiesce failures abort the destination, cancel the source, and return an
error. Client cancellation uses the same cleanup path. Concurrent requests may
migrate independently from the same source worker.

The control endpoints are `migration_prepare`, `migration_sync`, and
`migration_finalize`. Source `prepare` and `quiesce` are distinct phases:
prepare establishes transport state but does not park generation; quiesce has no
trigger fields and stops the request at a safe scheduler boundary. Rooms are
opaque and rank is explicit request state.

## Guarantees

- Trigger detection scans coalesced stream chunks.
- Trigger interpretation is owned by Dynamo; SGLang's migration API is agnostic
  to sequence lengths, token IDs, and reasoning formats.
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

### Incremental KV sync

Reuse one lease and migration ID to copy monotonically increasing stable ranges
before the trigger. At the trigger, quiesce, copy the final delta, activate, and
commit through the existing path.

### Production hardening

Add multi-DP routing tests, destination retry, phase deadlines, reservation-aware
scoring, transfer and cleanup metrics, and fault injection for worker loss,
timeouts, lost responses, and lease expiry.

## Validation

The generic prepare/quiesce branch passes 24 focused Rust controller tests, 12
Python handler tests, and 120 focused SGLang tests plus 9 subtests. Coverage
includes pre-admission prepare, stream coalescing, overlap overshoot, finish
races, cancellation, one-way failure cleanup, and dropped streams.

Local Qwen3-0.6B TP1-to-TP1 validation used two B200s, NIXL, overlap scheduling,
and `--stream-interval 1`. A warm low-load OSL-128 run measured 34.9 ms p50 /
36.1 ms p95 client handoff. A simultaneous 32-request burst completed 32/32
migrations with 525.4 ms p50 / 855.1 ms p95 handoff. Dynamo setup was 32.5 ms
p50, trigger-to-source-park was 13.0 ms, and source-park-to-NIXL-issue was 3.4
ms. Most loaded delay was concurrent NIXL receiver completion and destination
admission, not Dynamo trigger detection. A 64-request burst migrated 64/64.

Live coverage also includes Qwen3-8B TP4-to-TP1 with NIXL staging.

In a 200-sample Qwen3-8B GSM8K run at temperature 1.0 and 32K context, all 200
requests reached `</think>` and committed migration. There were no busy
rejections, rollbacks, invalid responses, or truncations. Accuracy was 96.0%
without migration and 96.5% with migration (`p=1.0`, paired McNemar exact).
