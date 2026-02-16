# velo-events

A generational event system for coordinating async tasks with [minimal overhead](https://drive.google.com/file/d/1s9M1I-dUbhqWLrMFB5ehPSM-qDQBGPZG).

Events can be created, awaited, merged into precondition graphs, and poisoned
on failure. The local implementation lives in this crate; a distributed event
system can be built on top via active messaging.

## Core concepts

| Operation | What it does |
|-----------|-------------|
| **Create** | `system.new_event()` allocates a pending event and returns an `Event` — an RAII guard you can trigger or await. |
| **Await** | `system.awaiter(handle)?.await` suspends the current task until the event completes (or is poisoned). |
| **Merge** | `system.merge_events(vec![a, b, c])` creates a new event that completes only after **all** inputs complete — this is how you build precondition graphs. |
| **Poison** | Events can fail with a reason string. Dropping an `Event` without triggering it auto-poisons so events are never silently lost. |

## Usage

### Create, trigger, await

```rust,no_run
use std::sync::Arc;
use velo_events::{EventManager, LocalEventSystem, Event};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = LocalEventSystem::new();

    let event = system.new_event()?;
    let handle = event.handle();

    // Spawn a task that waits for the event
    let sys: std::sync::Arc<LocalEventSystem> = Arc::clone(&system);
    let waiter = tokio::spawn(async move {
        sys.awaiter(handle)?.await
    });

    // Complete the event — consumes self, disarms the drop guard
    event.trigger()?;
    waiter.await??;
    Ok(())
}
```

### RAII drop safety

`Event` is an RAII guard: dropping it without calling `trigger()` or `poison()`
automatically poisons the event so waiters are never silently abandoned. Both
`trigger` and `poison` consume `self`, preventing double-completion at compile
time.

To opt out of auto-poisoning (e.g. when handing ownership to a manager-level
operation), call `into_handle()`:

```rust,no_run
use velo_events::{EventManager, LocalEventSystem, Event};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = LocalEventSystem::new();
    let event = system.new_event()?;
    let handle = event.handle();

    // If this function returns early or panics, the event
    // drops and is automatically poisoned.
    do_work()?;

    event.trigger()?; // success — consumes the event
    Ok(())
}

fn do_work() -> anyhow::Result<()> { Ok(()) }
```

### Merging events (precondition graphs)

`merge_events` lets you express "wait for all of these before proceeding":

```rust,no_run
use velo_events::{EventManager, LocalEventSystem, Event};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = LocalEventSystem::new();

    let load_weights = system.new_event()?;
    let load_tokenizer = system.new_event()?;

    // merged event completes only after both inputs complete
    let ready = system.merge_events(vec![
        load_weights.handle(),
        load_tokenizer.handle(),
    ])?;

    load_weights.trigger()?;
    load_tokenizer.trigger()?;

    system.awaiter(ready)?.await?;
    Ok(())
}
```

Because merged events are themselves events, you can merge merges to build
arbitrary DAGs of preconditions.

### Poison propagation

When an event is poisoned, all awaiters receive an error containing the
reason. Merged events accumulate poison reasons from their inputs:

```rust,no_run
use velo_events::{Event, EventManager, LocalEventSystem, EventPoison};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = LocalEventSystem::new();

    let a = system.new_event()?;
    let b = system.new_event()?;
    let merged = system.merge_events(vec![a.handle(), b.handle()])?;

    system.poison(a.handle(), "a failed")?;
    system.poison(b.handle(), "b failed")?;

    let err = system.awaiter(merged)?.await.unwrap_err();
    let poison = err.downcast::<EventPoison>()?;
    assert!(poison.reason().contains("a failed"));
    assert!(poison.reason().contains("b failed"));
    Ok(())
}
```

### Application responsibility

In distributed systems, concurrent trigger/poison calls cannot be coordinated
through the type system alone. Application logic must carefully manage how
events are completed.

**Pattern: don't use trigger/poison as if/else on one event.** Poison reasons
are kept in a `BTreeMap` history per entry, so poison strings persist in memory.
Instead, create a separate event per outcome arm and use `tokio::select!` to
race them:

```rust,no_run
use velo_events::{EventManager, LocalEventSystem, Event};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let system = LocalEventSystem::new();

    let success_event = system.new_event()?;
    let failure_event = system.new_event()?;

    let success_handle = success_event.handle();
    let failure_handle = failure_event.handle();

    // Producer decides which arm:
    // success_event.trigger()? OR failure_event.trigger()?

    // Consumer races:
    let success_awaiter = system.awaiter(success_handle)?;
    let failure_awaiter = system.awaiter(failure_handle)?;
    tokio::select! {
        ok = success_awaiter => { ok?; /* success path */ }
        err = failure_awaiter => { err?; /* failure path */ }
    }
    Ok(())
}
```

## Distributed events

For distributed deployments, `DistributedEventFactory` creates an event system
whose handles embed a non-zero `worker_id` for global uniqueness. The local
event machinery stays the same — coordination across workers is handled by an
active-messaging layer built on top.

```rust,no_run
use velo_events::DistributedEventFactory;

let factory = DistributedEventFactory::new(0x42.try_into().unwrap());
let system = factory.event_manager();
// handles produced by this system carry system_id = 0x42
```

## Resources

- [Design document (PDF)](https://drive.google.com/file/d/1s9M1I-dUbhqWLrMFB5ehPSM-qDQBGPZG)
- [NotebookLM podcast overview](https://notebooklm.google.com/notebook/e99c3e2a-a04e-4200-a21e-3e69d8f2ba73/audio)
