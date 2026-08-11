# N-2 Worker / Frontend Compatibility

Assume N-2 mixed-version operation between workers and frontends during rolling
updates. Frontends and workers from the current release and the two immediately
previous releases may coexist in any combination and must interoperate across
the worker/frontend boundary. This includes a single frontend concurrently
discovering worker cards from multiple supported releases for the same logical
deployment or worker set. Differences caused only by supported wire evolution
must not split otherwise compatible workers, fail discovery closed, or remove
healthy serving state. N-3 and older combinations are unsupported unless a
narrower temporary exception is explicitly documented.

Treat model deployment cards, discovery metadata, and worker/frontend wire
formats owned by this crate as cross-process compatibility surfaces. Consider
both age directions: an older worker with a newer frontend and a newer worker
with an older frontend, anywhere within the supported window.

- Normal, default deployment paths should remain operable across the N-2
  compatibility window. Unconditional parsing or discovery failures on those
  paths are generally compatibility bugs.
- Prefer tolerant readers and conservative writers. Accept known legacy fields
  when they are safe to interpret, and continue emitting required legacy fields
  for the supported compatibility window.
- Silent degradation may be acceptable when it is safe and bounded. It must not
  erase an essential capability, violate an invariant, or produce misleading
  state that makes an otherwise valid deployment unusable.
- Conditional failures may be acceptable for explicitly enabled features whose
  semantics cannot be represented safely by the other version. Prefer a
  targeted unsupported-feature error over a generic deserialization failure.

This expectation applies to cross-process worker/frontend wiring, including the
frontend's aggregation of mixed-version worker metadata. It does not establish
a compatibility window for direct worker-to-worker protocols or a general
stability promise for in-process Rust APIs; those are internal unless explicitly
documented otherwise.

## Compatibility Shims

Keep version-specific compatibility code narrow, close to the wire boundary,
and temporary. Translate legacy input into canonical state immediately, and
derive legacy output from canonical state. Do not plumb legacy fields through
internal APIs, make core logic branch on them, or rely on their presence after
deserialization. When both representations are present, the current
representation is authoritative unless a specific compatibility rule says
otherwise.

Every shim must name its compatibility window and removal condition, for
example:

```rust
// Compatibility with v1.2 workers and frontends during v1.4 rolling upgrades.
// TODO(v1.5): Remove when v1.2 falls outside the N-2 compatibility window.
```

Remove the shim when the corresponding version leaves the supported window;
do not carry compatibility branches forward indefinitely.
