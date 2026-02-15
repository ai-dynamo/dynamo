# slot_v2 State Machine Specification

## Overview

`slot_v2` replaces the original `slot` module's two-lock design (entry-level
`ParkingMutex<EventState>` + slot-level `ParkingMutex<SlotStateInner>` with
atomic guards) with a single entry-level `ParkingMutex<EventState>` that holds
all per-entry state including waker registration.

This eliminates:
- **Race 1** (stale completion leakage): old `begin_generation` conditionally
  cleared completion based on `waiter_count`, leaking stale results to
  new-generation waiters.
- **Race 2 fragility** (drop-then-signal): old `finalize_completion` dropped
  the entry lock before signaling the slot, creating a window where a new
  generation could start before waiters were notified.

## State Variables

All fields are protected by a single `parking_lot::Mutex`:

```rust
struct EventState {
    last_triggered: Generation,              // highest completed generation
    active_generation: Option<Generation>,   // currently pending generation
    wakers: Vec<Waker>,                      // registered waiter wakers
    poisoned: BTreeMap<Generation, PoisonArc>, // poison history per generation
    retired: bool,                           // permanently unusable
}
```

## Lifecycle Phases

```
    begin_generation()
Idle ──────────────────> Active
  ^                        |
  |   finalize_completion()|
  |                        v
  +──────────────────── Completing
```

All transitions happen under a single lock acquisition.

### Idle

- `active_generation = None`
- `wakers` is empty (drained by prior `finalize_completion`)
- Entry is available for reuse via the free list

### Active

- `active_generation = Some(gen)`
- Waiters may register wakers via `poll_waiter`
- Only one generation can be active at a time

### Completing

- `finalize_completion` sets `last_triggered`, clears `active_generation`,
  stores poison (if applicable), drains wakers, then wakes them
- Transitions back to Idle

## Operations

### begin_generation

1. Acquire lock
2. Validate: no active generation, not retired, not overflowed
3. Compute `next = last_triggered + 1`
4. Drain stale wakers (`std::mem::take`)
5. Set `active_generation = Some(next)`
6. Release lock
7. Wake stale wakers (outside lock)

### finalize_completion(generation, completion)

1. Acquire lock
2. Validate: `active_generation == Some(generation)`
3. Set `last_triggered = generation`
4. Clear `active_generation`
5. Insert/remove from poison map
6. Drain wakers
7. Release lock
8. Wake all drained wakers (outside lock)

### register_local_waiter(generation)

1. Acquire lock
2. If `generation <= last_triggered`: return Ready or Poisoned
3. If `active_generation == Some(generation)`: return Pending
4. Otherwise: return InvalidGeneration error

### poll_waiter(observed_generation, cx)

1. Acquire lock
2. If `observed_generation <= last_triggered`: return completion result
3. If `active_generation.is_none()`: return "generation expired" error
4. Register waker with deduplication (`will_wake` check)
5. Return Pending

### retire

1. Acquire lock
2. Set `retired = true`, clear `active_generation`

## Invariants

- **I1: Generation monotonicity** — `last_triggered` only increases. Each
  `begin_generation` computes `last_triggered + 1`.

- **I2: Single completion per generation** — `active_generation` guard ensures
  only one generation is active. `finalize_completion` validates
  `active_generation == Some(generation)`.

- **I3: Completion visibility** — Waiter resolution is determined by
  `observed_generation <= last_triggered` (success) plus the `poisoned`
  BTreeMap (error). Both are set under the same lock that the waiter reads.

- **I4: No stale completion leakage** — There is no stored completion value
  that could leak. Waiters resolve via generation comparison + poison map.
  `begin_generation` unconditionally drains stale wakers.

- **I5: No lost wakeups** — `finalize_completion` sets `last_triggered` and
  drains wakers in the same lock scope. Any waiter that registered before
  the drain will be woken. Any waiter that polls after the drain will see
  `observed_generation <= last_triggered` and resolve immediately.

- **I6: Stale waiter resolution** — Waiters from generation N check
  `observed_generation <= last_triggered` on every poll. After generation N
  completes, this check succeeds regardless of what generation is currently
  active. `begin_generation` also flushes stale wakers defensively.

## Concurrency Rules

A single `parking_lot::Mutex` per entry serializes all state mutations. This
eliminates the need for:

- Atomic `waiter_count` (no conditional clearing)
- Atomic `completed` flag (redundant with lock-protected state)
- Separate slot-level `generation` counter (entry-level suffices)
- Manual waker deduplication across lock boundaries

The only concurrency pattern is: acquire lock, read/write state, release lock,
then wake drained wakers outside the lock (waker invocation only enqueues
tasks on the runtime, it does not poll them synchronously).
