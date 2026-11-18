# lib/router/protocol

`dynamo-router-protocol` holds the identities and worker-facing wire types every router crate names. It is data only (serde derives and conversions) with no async, logging, or Dynamo dependencies; hosts and router crates import it directly.

## Entry rule

A type enters protocol only if it is serialized across a process boundary to an engine, a worker, or a sidecar, or is an identity every crate names.

Everything else belongs to the crate that produces it. When a second crate needs such a type, it depends on the producer; it does not lift the type here.

## `AttemptId` and `AttemptIdMinter`

`AttemptId` has no public constructor. `AttemptIdMinter::mint` is the only source of a value and yields fresh ids only; the slot tracker holds the one instance and mints at booking. Every other crate receives an `AttemptId` on a replica event, a booking descriptor, or an indexer lease and carries it unchanged. There is no `AttemptId::new`, no `From<u64>`, and no `#[doc(hidden)]` escape. The chosen-value constructors `AttemptId::for_testing` and `AttemptId::for_benchmark` exist only behind the `testing` and `bench` features; no normal dependency may enable either.
