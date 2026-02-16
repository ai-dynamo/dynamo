# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Build
cargo build -p velo-events

# Run all tests
cargo test -p velo-events

# Run a single test
cargo test -p velo-events <test_name>

# Check (no codegen)
cargo check -p velo-events
```

## Architecture

`velo-events` is a generational event system for coordinating async awaiters with minimal overhead. Events can be triggered (success) or poisoned (error), and entries are recycled across generations.

### Trait layer (`event.rs`, `manager.rs`)

- **`Event`** — RAII guard for a single event. Dropping without calling `trigger(self)` or `poison(self, ...)` auto-poisons the event. `into_handle(self)` disarms the guard and returns the bare handle. `trigger` and `poison` consume `self`, preventing double-completion at compile time.
- **`EventManager`** — creates/manages a collection of events: `new_event`, `awaiter`, `poll`, `trigger`, `poison`, `merge_events`, `force_shutdown`.

### Local implementation (`local/`)

- **`LocalEventSystem`** — the concrete `EventManager` implementation. Uses `DashMap` for concurrent event storage with a free-list for entry recycling. `EventManager` is implemented on `Arc<LocalEventSystem>`, not `LocalEventSystem` directly.
- **`LocalEvent`** — concrete `Event` using `Option<LocalEventInner>` take-pattern. The `Drop` impl poisons with a static `LazyLock<Arc<str>>` reason when the inner is still `Some`.

### Handle encoding (`handle.rs`)

`EventHandle` packs identity into a single `u128`: `[system_id: 64][local_index: 32][generation: 32]`. Bit 31 of `local_index` distinguishes local (bit set) from distributed (bit clear) handles. Both local and distributed systems have unique non-zero `system_id` values. `LocalEventSystem` validates that handles belong to the system that created them.

### Slot machinery (`slot/`)

Single-lock synchronization primitives. See [docs/slot-state-machine.md](docs/slot-state-machine.md)
for invariants. Any change to `slot/` must preserve all invariants (I1-I6)
and update the document.

Key types:
- **`EventEntry`** — per-index state machine with a single `ParkingMutex<EventState>` protecting generation tracking, waker registration, and poison history.
- **`EventAwaiter`** — `Future` impl that resolves to `Result<()>`. Supports both immediate (already-complete) and pending modes. Delegates poll to `EventEntry::poll_waiter`.
- **`CompletionKind`** — `Triggered` | `Poisoned(Arc<EventPoison>)`.

### Factory (`factory.rs`)

`DistributedEventFactory` creates a `LocalEventSystem` pre-configured with a `system_id` for distributed (Nova-managed) deployments.

## Key Design Decisions

- `Event` is an RAII guard by default — dropping without triggering auto-poisons. `into_handle()` is the explicit opt-out for manager-level operations. `Clone` is intentionally not implemented; each event is a unique ownership token.
- `EventManager` is implemented for `Arc<LocalEventSystem>`, requiring callers to wrap the system in an `Arc`. `LocalEventSystem::new()` returns `Arc<Self>` directly.
- Slot entries track a `BTreeMap<Generation, PoisonArc>` for poison history, allowing past-generation poison queries.
- Generation overflow causes entry retirement and a new entry allocation (transparent retry loop in `new_event_inner`).
- `force_shutdown` poisons all pending events and rejects future allocations via an `AtomicBool` flag.
