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

`EventHandle` packs identity into a single `u128`: `[worker_id: 64][local_index: 32][generation: 32]`. Local-only handles have `worker_id == 0`. Distributed handles embed a non-zero worker id for global uniqueness.

### Slot machinery (`slot/` — frozen, do not modify)

Internal synchronization primitives for the generational slot system:
- **`EventEntry`** — per-index state machine tracking active generation, completion history, and poison records. Entries are reused across generations until they overflow or retire.
- **`ActiveSlot`** / **`ActiveSlotState`** — lock-based completion + waker storage. Wakers are deduplicated for `select!` compatibility.
- **`EventAwaiter`** — `Future` impl that resolves to `Result<()>`. Supports both immediate (already-complete) and pending modes. Tracks waiter count to prevent premature completion cleanup.
- **`CompletionKind`** — `Triggered` | `Poisoned(Arc<EventPoison>)`.

### Factory (`factory.rs`)

`DistributedEventFactory` creates a `LocalEventSystem` pre-configured with a `worker_id` for distributed (Nova-managed) deployments.

## Key Design Decisions

- `EventManager` is implemented for `Arc<LocalEventSystem>`, requiring callers to wrap the system in an `Arc`. `LocalEventSystem::new()` returns `Arc<Self>` directly.
- Slot entries track a `BTreeMap<Generation, PoisonArc>` for poison history, allowing past-generation poison queries.
- Generation overflow causes entry retirement and a new entry allocation (transparent retry loop in `new_event_inner`).
- `force_shutdown` poisons all pending events and rejects future allocations via an `AtomicBool` flag.
