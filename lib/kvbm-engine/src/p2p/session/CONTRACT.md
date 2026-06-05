# P2P Session Contract

This document is the source of truth for the `kvbm-engine` P2P
session API (`lib/kvbm-engine/src/p2p/session/`). It pins the
invariants every downstream consumer relies on — conditional-disagg
(`kvbm-connector`), remote-search, bidirectional prefix sharing,
late-failure propagation. Each claim cites code and test files;
add an executable test before changing an invariant here.

Code path: `lib/kvbm-engine/src/p2p/session/`
- `mod.rs` — `Session` trait + `Frame` enum + stream types + the
  `PeerCommitted` / `PeerAvailable` snapshot enums.
- `velo.rs` — production `VeloSession` / `VeloSessionFactory`.
- `testing.rs` — `MockSession` / `MockSessionFactory` (feature
  `testing`); observable parity with `VeloSession` for trait-only
  callers.
- `manager.rs` — `SessionManager` lifecycle holder with a
  configurable watchdog (default 30s).

---

## 1. Scope

A `Session` is a one-to-one bidirectional channel between two
KVBM instances, correlated by a `SessionId`. It carries:

- **Holder-side intent** (the hashes we will provide).
- **Holder-side capability** (the blocks pinned in G2, ready to
  pull).
- **Wire-level pull authorization** (`Frame::Pull` ↔
  `Frame::PullComplete` ↔ `Frame::PullAck`).
- **Cooperative shutdown** (`finalize`) and **abort** (`close`).

It is **not**:

- **Multi-puller** — one peer per session.
- **Request-scoped** — the protocol allows session reuse, though
  current callers (CD) tear down per-request.
- **Auto-cleaning** — the caller must drop its `Arc<dyn Session>`
  on `LifecycleEvent::Detached` to release the inner state.
  `SessionManager` provides a watchdog as a safety net only.
- **Order-symmetric across all surfaces** — see §3 invariants.
  Streams preserve order; sorted snapshots do not.

---

## 2. Trait surface

All methods are on `pub trait Session: Send + Sync`
(`mod.rs:220`). Section numbering follows the trait order.

### 2.1 `session_id() -> SessionId` (`mod.rs:222`)

Stable correlation id, constant across the session's lifetime.
Useful for joining audit log lines.

### 2.2 `endpoint() -> Option<SessionEndpoint>` (`mod.rs:226`)

The endpoint the peer uses to `attach` to us. `Some` after `open`
or `attach`; transitions to `None` only on a teardown path
(currently not exercised). Both holder and puller have a `Some`
endpoint post-attach — there is no role distinguisher at this
surface.

### 2.3 `commit(hashes: Vec<SequenceHash>) -> Result<()>` (`mod.rs:235`)

Holder-side. Declares hashes we will provide. Monotonic-add:
calling twice with overlapping sets is allowed; the union is the
committed state. Sends `Frame::Commit { hashes }` on the wire
(`velo.rs:857`). Peer's monitor pushes `CommitDelta::Added` into
the commits stream (`velo.rs:540`).

- **Synchronous enqueue** — frames go out in the order their
  public methods are called, with no spawn race
  (`velo.rs:855-859`).
- **Order on the wire** equals call order on a single session.
- **Errors after `finish_commits`** — calling `commit` after
  `finish_commits` returns `Err`; the contract guarantees the
  committed set is final once sealed. Same guard in
  `MockSession::commit`.
- **Pinned by tests**:
  `session_stream_ordering.rs::mock_paired_commits_preserves_order_across_calls`,
  `velo_loopback_commits_preserves_order_across_calls`,
  `session_seal_immutability.rs::mock_commit_after_finish_commits_errors`,
  `velo_commit_after_finish_commits_errors`.

### 2.4 `finish_commits() -> Result<()>` (`mod.rs:239`)

Holder-side. Marks the commit set complete. Idempotent
(`velo.rs:869-878` early-returns on the second call). Sends
`Frame::CommitsClosed`; peer pushes `CommitDelta::Closed` AND
sets the seal flag (`velo.rs:542-548`), so subsequent
`peer_committed()` calls return `PeerCommitted::Sealed`.

### 2.5 `make_available(blocks: Vec<ImmutableBlock<G2>>) -> Result<()>` (`mod.rs:246`)

Holder-side. Marks committed hashes as actually pullable.

- **Precondition validated synchronously, pre-mutation**: every
  block's hash must already be in the local committed set; error
  otherwise. The same precondition is upheld on `MockSession`
  (`testing.rs::make_available_errors_if_not_committed`).
- **Pin lifecycle**: holder pins each block (strong G2 ref); pin
  is dropped automatically when the puller's PullAck for that
  hash arrives (`velo.rs:606-619`).
- **Order on the wire** equals call order; sends
  `Frame::Available { blocks }` (peer pushes
  `AvailabilityDelta::Available`).
- **Errors after `finish_availability`** — calling
  `make_available` after `finish_availability` returns `Err`; same
  immutability guarantee as commit/finish_commits.
- **Pinned by tests**:
  `session_stream_ordering.rs::*_availability_preserves_order_across_calls`,
  `session_seal_immutability.rs::mock_make_available_after_finish_availability_errors`,
  `velo_make_available_after_finish_availability_errors`.

### 2.6 `finish_availability() -> Result<()>` (`mod.rs:251`)

Holder-side. Marks availability set complete. Idempotent. Sends
`Frame::Drained`; peer pushes `AvailabilityDelta::Drained` AND
sets the seal flag (`velo.rs:565-571`), so subsequent
`peer_available()` calls return `PeerAvailable::Sealed`.

### 2.7 `commits() -> CommitStream` (`mod.rs:260`)

Puller-side. Stream of `CommitDelta` from the peer.

- **Subscribe-once**: second call panics
  (`velo.rs::ReplayStream::subscribe`, mirrored in
  `testing.rs::take_commits_stream`,
  `velo.rs:1496+::replay_stream_subscribe_twice_panics`).
- **Replay-on-subscribe**: items buffered before the first
  subscribe are coalesced — multiple `Added` deltas merge into a
  single `Added` carrying the concatenated hash list; `Closed`,
  if seen pre-subscribe, is preserved as the last item
  (`velo.rs::build_commit_stream`,
  `testing.rs::drain_commit_buffer`).
- **Live-after-subscribe**: items arriving after first subscribe
  are NOT coalesced; each `Frame::Commit` produces a separate
  `CommitDelta::Added`.
- **Order**: concatenating all `Added` payloads (replay + live)
  reproduces the holder's call order across all `commit()` calls.
- **Pinned by tests**:
  `session_stream_ordering.rs::*_commits_preserves_order_across_calls`,
  existing `velo_session_loopback.rs::replay_on_late_subscribe_coalesces`.

### 2.8 `availability() -> AvailabilityStream` (`mod.rs:265`)

Same shape as `commits()` for `AvailabilityDelta`. `Drained`
terminator instead of `Closed`. Same ordering / replay /
subscribe-once invariants.

**Consumer gotcha: deltas are arrival-ordered, NOT
position-ordered.** A holder may publish availability in any
number of `Available` deltas — `make_available([a, b])` followed
by `make_available([c, d])` is equivalent to
`make_available([a, b, c, d])` on the WIRE (a single drained
set), but the puller's `availability()` stream surfaces them as
two distinct deltas in call order. Any consumer that drains
incrementally and **appends results to a flat log** (e.g., for
later positional pairing with vLLM-allocated destinations) MUST
re-sort by absolute position before treating the log as
positionally indexed; appending in arrival order silently
mis-pairs when the holder publishes in two-plus deltas.

Holders that publish in a single `make_available` call mask this
gotcha for the consumer, but no contract guarantee binds them to
that shape — and any future per-tier promotion / chunked-output
flow legitimately splits availability. Treat single-delta arrival
as a coincidence, not an invariant.

Pinned by `lib/kvbm-connector/tests/cd_prefill_e2e.rs::cd_prefill_kick_onboard_robust_to_split_delta_availability`
(prefill `kick_onboard` consumer; broken pre-fix because it took
`registered_g2.iter().skip(suffix_start)` in arrival order
instead of sorting by `expected_hashes` position).

### 2.9 `peer_committed() -> PeerCommitted` (`mod.rs:273`)

Puller-side snapshot of peer's committed set with seal status as
type-level discriminant.

- `PeerCommitted::Open(v)` — peer has not yet signaled
  `CommitsClosed`; `v` MAY grow. **Do not use to size a pull.**
- `PeerCommitted::Sealed(v)` — peer has signaled `CommitsClosed`;
  `v` is final and safe to size a pull.

**Immutability of Sealed is enforced two ways**:
1. Holder-side: `commit` errors after `finish_commits` (§2.3) — a
   well-behaved holder cannot enqueue a post-`CommitsClosed`
   `Frame::Commit`.
2. Puller-side dispatch (defense-in-depth): a `Frame::Commit`
   arriving after `Frame::CommitsClosed` is dropped with
   `tracing::error!`; `peer_committed` is NOT mutated
   (`velo.rs::dispatch_frame` Frame::Commit handler). `MockSession`
   has equivalent protection via `inject_peer_commit`'s
   `stream_state.is_terminated()` short-circuit.

Pinned by
`session_seal_immutability.rs::velo_sealed_committed_does_not_grow_under_post_closed_commit_frame`
and `velo_loopback_sealed_snapshot_is_stable_across_calls`.

Hash order inside the inner `Vec` is **sorted** (`BTreeSet`
iteration), NOT holder's call order. Callers that need call-order
data must drain the `commits()` stream.

Helpers on the enum: `as_slice()`, `is_sealed()`, `len()`,
`is_empty()`. Derived: `Debug`, `Clone`, `PartialEq`, `Eq`.

Upgrade path documented: if profiling shows the `Sealed` copy
cost is hot, swap `Sealed(Vec<T>)` → `Sealed(Arc<[T]>)`;
consumers using `as_slice()` / `is_sealed()` remain
source-compatible (`mod.rs:108-141`).

### 2.10 `peer_available() -> PeerAvailable` (`mod.rs:279`)

Same shape for `PeerAvailable::{Open, Sealed}(Vec<CommittedBlock>)`.
Sealed immutability enforced identically to §2.9 — holder errors
on `make_available` after `finish_availability`; puller drops any
`Frame::Available` arriving after `Frame::Drained`. Pinned by
`session_seal_immutability.rs::velo_sealed_available_does_not_grow_under_post_drained_avail_frame`.

### 2.11 `pull(hashes, dst) -> Result<Vec<MutableBlock<G2>>>` (`mod.rs:286`)

Puller-side. Pulls each `hashes[i]` from peer into `dst[i]`.

**Preconditions (validated synchronously, pre-wire):**
- `hashes.len() == dst.len()` else error
  (`velo.rs:953-960`, `testing.rs::pull_errors_on_length_mismatch`).
- Every `h ∈ hashes` must be in the local peer-available map
  (which is populated from inbound `Frame::Available` frames at
  `velo.rs:556-560`) — else synchronous error, no `Frame::Pull`
  sent. The validation does NOT consult the `PeerAvailable::Open`
  vs `Sealed` discriminant; callers that need finality MUST drain
  `availability()` to `Drained` first.
  - Pinned by `velo_session_loopback.rs::pull_validation_synchronous_error`,
    `testing.rs::pull_errors_if_not_peer_available`.

**Pairing (the core contract):**
- `dst[i]` receives data the peer published for `hashes[i]`,
  regardless of `hashes` order relative to the holder's
  `make_available` call order.
- Implementation: sequential lookup
  `peer_block_ids[i] = peer_available[hashes[i]]`
  (`velo.rs:991-1002`), zipped on same `i` with
  `dst_block_ids[i]` to produce `Vec<PullRef>`
  (`velo.rs:1066-1073`).
- Pinned by `session_pull_pairing.rs::mock_paired_pull_preserves_caller_order`
  (trait-level) and `velo_loopback_frame_pull_carries_caller_order_hashes`
  (wire-level).

**Wire correlation:**
- Allocates `pull_id` (atomic), installs a oneshot in
  `pending_pulls`, sends `Frame::Pull { pull_id, hashes }`, awaits
  `Frame::PullComplete { pull_id }` from peer
  (`velo.rs:1029-1056`).
- Then drives `InstanceLeader::rdma_pull_with_opts` with the
  pair-zipped refs.
- Then enqueues `Frame::PullAck { pull_id }` so holder releases
  pins (`velo.rs:1093-1101`).

**Pin lifecycle (holder side):**
- Holder records inbound `Frame::Pull { pull_id, hashes }` in
  `inbound_pulls[pull_id]` (`velo.rs:572-591`).
- Holder drops pins for those hashes when `Frame::PullAck`
  arrives (`velo.rs:606-619`).
- Pinned by `velo_session_loopback.rs::pull_ack_drops_holder_pins`.

**Out of scope (deferred — needs worker-equipped infra):**
- Full RDMA happy path with data landing at `dst`. The
  `peer_block_ids ↔ dst_block_ids` zip is correct by
  construction; current tests verify the shell. End-to-end RDMA
  pairing is a follow-up.

**Commit-then-late-availability — supported pattern, no helper:**
- Pulling a hash that has been COMMITTED but not yet
  MADE-AVAILABLE is the central pattern exercised by the
  conditional-disagg prefix-promotion path (kvbm-connector
  Stage 1: G1→G2 promoted prefix; Stage 2: G3→G2 staged
  prefix). The holder commits the planned prefix hashes
  up-front at GNMT, calls `finish_commits` (sealing the planned
  set), and later calls `make_available` from a per-request
  promotion task as the G2 blocks land. The session API
  permits this — `commit` accepts hashes whose backing blocks
  do not exist yet — and the puller-side contract is
  **explicit**: the caller must observe each hash in
  `peer_available` (via `availability()` / `peer_available()`)
  BEFORE calling `pull`. Any wrapper that internally drives
  "wait for availability, then pull" can layer on top without
  a wire change; see §8 "drain-then-pull abstraction" for the
  proposed shape.

### 2.12 `lifecycle() -> LifecycleStream` (`mod.rs:299`)

Either-side stream of `LifecycleEvent::{Attached, Detached, Failed}`.
Subscribe-once, replay-on-subscribe in order. Detached emitted
from cooperative finalize, from `close` abort, from peer-side
velo `Finalized` sentinel, and from velo heartbeat
loss. `Failed` emitted from inbound `Frame::Error`.

### 2.13 `finalize(reason: Option<String>)` (`mod.rs:325`)

Either-side cooperative shutdown. Idempotent.

- Sends `CommitsClosed` + `Drained` terminators if not already
  sent (`velo.rs:1118-1136`).
- Sends `Frame::Finished` exactly once (`velo.rs:1141-1149`).
- When BOTH sides have called `finalize`, each side independently
  triggers velo's `StreamSender::finalize`; the peer's monitor
  sees the velo `Finalized` sentinel and emits
  `LifecycleEvent::Detached`.
- Pinned by `velo_session_loopback.rs::finalize_rendezvous_triggers_both_side_velo_finalize`.

### 2.14 `close(reason: Option<String>)` (`mod.rs:337`)

Either-side abort. Implies `finish_commits` + `finish_availability`
(`velo.rs:1168+`), then calls velo's `StreamSender::finalize`
directly. Does NOT wait for peer to rendezvous. Use for
fatal-error / aborted-request scenarios.

- Pinned by `velo_session_loopback.rs::close_from_holder_terminates_streams`
  (peer sees Detached + Closed + Drained).

---

## 3. Cross-method invariants

1. **`available ⊆ committed`** (precondition on `make_available`,
   enforced both sides — `mod.rs:8`, `testing.rs:1130-1135`).
2. **Monotonic-add sets** — committed and available only grow
   within a session lifetime; never remove.
3. **Pin lifecycle** — `make_available` pins (strong G2 ref);
   PullAck drops per-hash; `close` drains remaining pins.
4. **Subscribe-once per stream** — second `commits()` /
   `availability()` / `lifecycle()` call panics.
5. **Stream replay-on-subscribe** — coalesces consecutive Added /
   Available items into one per stream; preserves the closed
   terminator.
6. **Bidirectional symmetry** — both sides may concurrently
   commit, make_available, pull. No implicit role asymmetry in
   the state machine.
   - Pinned by `session_bidirectional.rs::velo_loopback_bidirectional_publish_drain_pull`.
7. **Seal flag tracks peer terminator frame** —
   `peer_commits_closed` flips when inbound `Frame::CommitsClosed`
   arrives; `peer_avail_drained` flips when inbound `Frame::Drained`
   arrives. Both feed the `PeerCommitted` / `PeerAvailable` enum
   discriminant.
8. **Sealed snapshot is final** — once `peer_committed()` returns
   `Sealed(v)`, subsequent calls return an equal set. Enforced
   bilaterally: holder errors on `commit`/`make_available` after
   `finish_*`; puller drops post-terminator `Frame::Commit` /
   `Frame::Available`. See §2.3, §2.5, §2.9, §2.10. Pinned by
   `session_seal_immutability.rs`.
9. **Holder-side wire-order is race-free under concurrent
   commit/finish** — `VeloSession::commit` and `finish_commits`
   serialise on the same `commits_closed` mutex, held from
   flag-check through `enqueue_frame`. A concurrent `finish_commits`
   either runs entirely before the commit's check (commit returns
   Err) or entirely after the commit's enqueue (CommitsClosed lands
   after the Commit on the wire). Same lock discipline for
   `make_available` ↔ `finish_availability` via `avail_drained`.
   Pinned by
   `session_seal_immutability.rs::velo_concurrent_commit_and_finish_commits_no_silent_drops`.
   `MockSession` does NOT make this race-free guarantee — partner
   injection happens outside the inner-state lock to avoid
   cross-session lock nesting; mocks are for sequential
   test-fixture use only.

---

## 4. Production callsite audit

| File:line | Method used | Order dependency | Backed by |
|---|---|---|---|
| `lib/kvbm-connector/src/connector/leader/disagg/coordinator/driver.rs:521-560` | `commits()`, `availability()` streams (drain to terminator) | Wire receive order; relies on the closed-terminator signal | §2.7, §2.8 (stream order pinned by `session_stream_ordering.rs`) |

No other production code reads `peer_committed()` / `peer_available()`
snapshots today. The snapshots are exposed for diagnostics +
future abstractions (e.g., "drain-then-pull" wrappers) that can
layer on without wire changes.

`session.inject_peer_*` calls in 25+ kvbm-connector tests are
inherent helpers on `MockSession`, not trait methods. They
populate test fixtures and are unaffected by trait changes.

---

## 5. Bidirectional symmetry

Both sides — the one that called `factory.open(...)` and the one
that called `factory.attach(...)` — materialize identical
`VeloSessionInner` state post-attach. There is no holder/puller
distinguisher at the trait surface or the inner-state layout.

Concurrent operations supported:
- Both sides commit + make_available + finish concurrently.
- Both sides drain peer's streams concurrently.
- Both sides observe `Sealed` snapshots once peer finishes.
- Both sides call `pull(...)` on peer's hashes concurrently.

End-to-end pinned by `session_bidirectional.rs`; the test
asserts each side sees the other's call order in streams, both
snapshots are `Sealed` after rendezvous, and both pull paths
reach the wire (Frame::Pull → PullComplete → rdma step).

---

## 6. Lifecycle and termination

| Path | Method | Effect on peer |
|---|---|---|
| Cooperative shutdown | `finalize` on both sides | Both observe `LifecycleEvent::Detached` after rendezvous |
| Abort | `close` on one side | Peer observes `LifecycleEvent::Detached` + `CommitDelta::Closed` + `AvailabilityDelta::Drained` |
| Peer crash | velo heartbeat loss | Surviving side observes `LifecycleEvent::Detached` |
| Protocol error | inbound `Frame::Error` | Observer pushes `LifecycleEvent::Failed` |
| Watchdog timeout | `SessionManager` (default 30s) | Evicts un-terminated sessions; the held `Arc` drops |

Existing tests:
- `close_from_holder_terminates_streams` — abort propagation.
- `finalize_rendezvous_triggers_both_side_velo_finalize` —
  cooperative rendezvous.
- `active_session_count_returns_to_zero` — clean teardown via
  rendezvous + drop; no Arc leaks.
- `pull_ack_drops_holder_pins` — pin release on PullAck.
- `lifecycle_stream_replay_preserves_order` (`velo.rs`
  unit-internal) — replay ordering.

Out of scope (deferred — see §8):
- Race resolution when `Frame::Error` arrives during finalize
  rendezvous.
- Exactly-once terminal guarantee under repeated `finalize()`.
- `SessionManager` watchdog timeout coverage.

---

## 7. Mock-vs-Velo parity

`MockSession` matches `VeloSession`'s observable behaviour for
all trait-surface callers:

- Subscribe-once panic ✓
- Replay-on-subscribe coalescing ✓
- `close` implies `finish_commits` + `finish_availability` ✓
- Synchronous pre-mutation precondition validation ✓
- Seal flag tracking on the snapshot enum ✓
  (`testing.rs::inject_peer_finish_commits` and
  `inject_peer_drained` set the flag before pushing the
  terminator)

Tests cross both transports where feasible:
- `session_pull_pairing.rs` — mock + velo.
- `session_stream_ordering.rs` — mock + velo.
- `session_bidirectional.rs` — velo only (symmetric paired
  velo is the higher-confidence proof; mock paired mode is
  trivially symmetric by construction).

Mock-only (manual-resolution `resolve_pull` API) is reserved for
in-process test composition that doesn't need a wire.

---

## 8. Out of scope / deferred

Each item below is a known gap with a follow-up issue or task:

- **RDMA happy-path pull** — verifying `dst[i]` actually receives
  the peer's block bytes end-to-end. Requires worker-equipped
  test infra (NIXL/CUDA). The wire shell + pair zipping is
  pinned; data-landing is by inspection.
- **Lifecycle race tests** — `Frame::Error` during finalize,
  exactly-once terminal under repeated finalize, watchdog
  eviction. Big invariants are covered (§6); race semantics
  deferred.
- **Snapshot `Arc<[T]>` migration** — current
  `PeerCommitted::Sealed(Vec<T>)` clones on each call. Source-
  compatible upgrade path via `.as_slice()` / `.is_sealed()`.
  Defer until profiling shows it matters.
- **"Drain-then-pull" abstraction** — a wrapper that internally
  drains `availability()` to `Sealed`, then calls `pull(...)`
  with the full set. Can layer on this contract without wire
  change.

Add new tests under `lib/kvbm-engine/tests/` and cite them here
when filling these gaps.
