# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Build
cargo build -p velo-event

# Run all tests
cargo test -p velo-event

# Run a single test
cargo test -p velo-event <test_name>

# Check (no codegen)
cargo check -p velo-event
```

## Architecture

`velo-event` is a generational event system for coordinating async awaiters with minimal overhead. Events can be triggered (success) or poisoned (error), and entries are recycled across generations.

### Trait layer (`event.rs`, `manager.rs`, `guard.rs`)

- **`Event`** — single event that can be triggered or poisoned exactly once. Cloning produces a second handle to the same event.
- **`EventManager`** — creates/manages a collection of events: `new_event`, `awaiter`, `poll`, `trigger`, `poison`, `merge_events`, `force_shutdown`.
- **`EventGuard<E: Event>`** — RAII wrapper that automatically poisons the event on drop unless explicitly triggered. Prevents events from being silently lost.

### Local implementation (`local/`)

- **`LocalEventSystem`** — the concrete `EventManager` implementation. Uses `DashMap` for concurrent event storage with a free-list for entry recycling. `EventManager` is implemented on `Arc<LocalEventSystem>`, not `LocalEventSystem` directly.
- **`LocalEvent`** — concrete `Event` backed by an `Arc<LocalEventInner>` holding a reference to the system, entry, and handle.

### Handle encoding (`handle.rs`)

`EventHandle` packs identity into a single `u128`: `[system_id: 64][local_index: 32][generation: 32]`. Bit 31 of `local_index` distinguishes local (bit set) from distributed (bit clear) handles. Both local and distributed systems have unique non-zero `system_id` values. `LocalEventSystem` validates that handles belong to the system that created them.

### Slot machinery

- **`slot/`** — Original implementation. Has a known stale-completion race (Race 1).
  Frozen pending removal.
- **`slot_v2/`** — Single-lock replacement. See [docs/slot-state-machine.md](docs/slot-state-machine.md)
  for invariants. Any change to `slot_v2/` must preserve all invariants (I1-I6)
  and update the document.

Active module is `slot_v2`. Key types:
- **`EventEntry`** — per-index state machine with a single `ParkingMutex<EventState>` protecting generation tracking, waker registration, and poison history.
- **`EventAwaiter`** — `Future` impl that resolves to `Result<()>`. Supports both immediate (already-complete) and pending modes. Delegates poll to `EventEntry::poll_waiter`.
- **`CompletionKind`** — `Triggered` | `Poisoned(Arc<EventPoison>)`.

### Factory (`factory.rs`)

`DistributedEventFactory` creates a `LocalEventSystem` pre-configured with a `system_id` for distributed (Nova-managed) deployments.

## Key Design Decisions

- `EventManager` is implemented for `Arc<LocalEventSystem>`, requiring callers to wrap the system in an `Arc`. `LocalEventSystem::new()` returns `Arc<Self>` directly.
- Slot entries track a `BTreeMap<Generation, PoisonArc>` for poison history, allowing past-generation poison queries.
- Generation overflow causes entry retirement and a new entry allocation (transparent retry loop in `new_event_inner`).
- `force_shutdown` poisons all pending events and rejects future allocations via an `AtomicBool` flag.
