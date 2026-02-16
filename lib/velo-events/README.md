# velo-events

A generational event system for coordinating async tasks with [minimal overhead](https://drive.google.com/file/d/1s9M1I-dUbhqWLrMFB5ehPSM-qDQBGPZG).

Events can be created, awaited, merged into precondition graphs, and poisoned
on failure. The local implementation lives in this crate; a distributed event
system can be built on top via active messaging.

## Core concepts

| Operation | What it does |
|-----------|-------------|
| **Create** | `manager.new_event()` allocates a pending event and returns an `Event` — an RAII guard you can trigger or await. |
| **Await** | `manager.awaiter(handle)?.await` suspends the current task until the event completes (or is poisoned). |
| **Merge** | `manager.merge_events(vec![a, b, c])` creates a new event that completes only after **all** inputs complete — this is how you build precondition graphs. |
| **Poison** | Events can fail with a reason string. Dropping an `Event` without triggering it auto-poisons so events are never silently lost. |

## Usage

### Create, trigger, await

```rust,no_run
use velo_events::EventManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manager = EventManager::local();

    let event = manager.new_event()?;
    let handle = event.handle();

    // Spawn a task that waits for the event
    let mgr = manager.clone();
    let waiter = tokio::spawn(async move {
        mgr.awaiter(handle)?.await
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
use velo_events::EventManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manager = EventManager::local();
    let event = manager.new_event()?;
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
use velo_events::EventManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manager = EventManager::local();

    let load_weights = manager.new_event()?;
    let load_tokenizer = manager.new_event()?;

    // merged event completes only after both inputs complete
    let ready = manager.merge_events(vec![
        load_weights.handle(),
        load_tokenizer.handle(),
    ])?;

    load_weights.trigger()?;
    load_tokenizer.trigger()?;

    manager.awaiter(ready)?.await?;
    Ok(())
}
```

Because merged events are themselves events, you can merge merges to build
arbitrary DAGs of preconditions.

### Poison propagation

When an event is poisoned, all awaiters receive an error containing the
reason. Merged events accumulate poison reasons from their inputs:

```rust,no_run
use velo_events::{EventManager, EventPoison};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manager = EventManager::local();

    let a = manager.new_event()?;
    let b = manager.new_event()?;
    let merged = manager.merge_events(vec![a.handle(), b.handle()])?;

    manager.poison(a.handle(), "a failed")?;
    manager.poison(b.handle(), "b failed")?;

    let err = manager.awaiter(merged)?.await.unwrap_err();
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
use velo_events::EventManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manager = EventManager::local();

    let success_event = manager.new_event()?;
    let failure_event = manager.new_event()?;

    let success_handle = success_event.handle();
    let failure_handle = failure_event.handle();

    // Producer decides which arm:
    // success_event.trigger()? OR failure_event.trigger()?

    // Consumer races:
    let success_awaiter = manager.awaiter(success_handle)?;
    let failure_awaiter = manager.awaiter(failure_handle)?;
    tokio::select! {
        ok = success_awaiter => { ok?; /* success path */ }
        err = failure_awaiter => { err?; /* failure path */ }
    }
    Ok(())
}
```

## Distributed events

For distributed deployments, `EventBackend` and `EventSystemBase` are public
so you can implement custom routing. Create a base with an explicit system_id,
implement `EventBackend` to route local vs remote handles, and pass both to
`EventManager::new`:

```rust,no_run
use velo_events::{EventSystemBase, EventBackend, EventManager, EventHandle, EventAwaiter};
use anyhow::Result;
use std::sync::Arc;

struct MyDistributedBackend {
    local: Arc<EventSystemBase>,
    // router: MyRouter,
}

impl EventBackend for MyDistributedBackend {
    fn trigger(&self, handle: EventHandle) -> Result<()> {
        if handle.system_id() == self.local.system_id() {
            self.local.trigger_inner(handle)   // fast local path
        } else {
            todo!("route over network")
        }
    }

    fn poison(&self, handle: EventHandle, reason: Arc<str>) -> Result<()> {
        if handle.system_id() == self.local.system_id() {
            self.local.poison_inner(handle, reason)
        } else {
            todo!("route over network")
        }
    }

    fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter> {
        if handle.system_id() == self.local.system_id() {
            self.local.awaiter_inner(handle)
        } else {
            todo!("route over network")
        }
    }
}

let base = EventSystemBase::distributed(0x42);
let backend = Arc::new(MyDistributedBackend { local: base.clone() });
let manager = EventManager::new(base, backend);
// handles produced by this manager carry system_id = 0x42
```

For simpler cases where you just need handles stamped with a system_id (without
custom routing), `DistributedEventFactory` is a convenience wrapper:

```rust,no_run
use velo_events::DistributedEventFactory;

let factory = DistributedEventFactory::new(0x42.try_into().unwrap());
let manager = factory.event_manager();
// handles produced by this manager carry system_id = 0x42
```

## Resources

- [Design document (PDF)](https://drive.google.com/file/d/1s9M1I-dUbhqWLrMFB5ehPSM-qDQBGPZG)
- [NotebookLM podcast overview](https://notebooklm.google.com/notebook/e99c3e2a-a04e-4200-a21e-3e69d8f2ba73/audio)
