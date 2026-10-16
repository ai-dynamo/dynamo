# KV Event Publisher Invariants

## Shared processing

Legacy and state-agent publishers should share decoding, normalization,
coalescing, deduplication, ordinary local-index admission, event-plane
partitioning, and transport mechanics when their semantics match.

Treat the legacy publisher's actual delivery and failure guarantees as the
baseline. Do not add blocking, acknowledgements, retries, quarantine, or
fail-closed behavior to a parallel publisher path without a concrete contract
or lifecycle requirement. If the legacy behavior is unsafe, fix and share the
common stage instead of hardening only one path.

## Relaxed data-plane guarantees

KV routing events are advisory:

- ZMQ ingestion must not wait for downstream processing or publication.
- Ordinary Store/Remove events use queue-admission semantics, not completion
  acknowledgements.
- Raw sequence gaps and ordinary local/publication failures warn and continue.
- Outbound IDs remain monotonic and are never reused.
- Do not add per-event or per-chunk acknowledgements or publication retries.

`Cleared` is different: it is an ordering barrier and must complete across every
affected physical tier before publication.

## Intentional state-agent differences

Keep state-agent-specific identity, protocol validation, attachment lifecycle,
ordered Worker resets, status/discovery metadata, and recovery fencing separate.
Fail closed only when an event cannot be interpreted safely, identity is
inconsistent, or a reset barrier fails.

The in-process foundation does not establish durable KVCR continuity. Do not
strengthen the ordinary event pipeline to imply that guarantee.
