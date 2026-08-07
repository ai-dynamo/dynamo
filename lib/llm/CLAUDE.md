# N-1 Worker / Frontend Compatibility

Assume N-1 mixed-version operation between workers and frontends during rolling
updates: current-version components must interoperate with components from the
immediately previous release. N-2 and older combinations are unsupported unless
a narrower temporary exception is explicitly documented. Treat model deployment
cards, discovery metadata, and worker/frontend wire formats owned by this crate
as cross-process compatibility surfaces. Consider both directions: a
previous-version worker with a current frontend and a current worker with a
previous-version frontend.

- Normal, default deployment paths should remain operable across the N-1
  boundary. Unconditional parsing or discovery failures on those paths are
  generally compatibility bugs.
- Prefer tolerant readers and conservative writers. Accept known legacy fields
  when they are safe to interpret, and continue emitting required legacy fields
  for the supported compatibility window.
- Silent degradation may be acceptable when it is safe and bounded. It must not
  erase an essential capability, violate an invariant, or produce misleading
  state that makes an otherwise valid deployment unusable.
- Conditional failures may be acceptable for explicitly enabled features whose
  semantics cannot be represented safely by the other version. Prefer a
  targeted unsupported-feature error over a generic deserialization failure.

This expectation applies to cross-process worker/frontend wiring. It is not a
general stability promise for in-process Rust APIs, which are internal unless
explicitly documented otherwise.

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
// Compatibility with v1.3 workers during v1.4 rolling upgrades.
// TODO(v1.5): Remove when v1.3 falls outside the N-1 compatibility window.
```

Remove the shim when the corresponding version leaves the supported window;
do not carry compatibility branches forward indefinitely.
