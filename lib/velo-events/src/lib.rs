// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # velo-event
//!
//! A generational event system for coordinating async tasks with minimal overhead.
//!
//! Events can be created, awaited, merged into precondition graphs, and poisoned
//! on failure. The local implementation lives in this crate; a distributed event
//! system can be built on top via active messaging.
//!
//! ## Core concepts
//!
//! | Operation | What it does |
//! |-----------|-------------|
//! | **Create** | `system.new_event()` allocates a pending event and returns an [`Event`] handle you can trigger or await. |
//! | **Await** | `system.awaiter(handle)?.await` suspends the current task until the event completes (or is poisoned). |
//! | **Merge** | `system.merge_events(vec![a, b, c])` creates a new event that completes only after **all** inputs complete — this is how you build precondition graphs. |
//! | **Poison** | Events can fail with a reason string. [`EventGuard`] auto-poisons on drop so events are never silently lost. |
//!
//! ## Usage
//!
//! ### Create, trigger, await
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use velo_event::{EventManager, LocalEventSystem, Event};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = LocalEventSystem::new();
//!
//!     let event = system.new_event()?;
//!     let handle = event.handle();
//!
//!     // Spawn a task that waits for the event
//!     let sys = Arc::clone(&system);
//!     let waiter = tokio::spawn(async move {
//!         sys.awaiter(handle)?.await
//!     });
//!
//!     // Complete the event
//!     event.trigger()?;
//!     waiter.await??;
//!     Ok(())
//! }
//! ```
//!
//! ### Merging events (precondition graphs)
//!
//! [`EventManager::merge_events`] lets you express "wait for all of these before proceeding":
//!
//! ```rust,no_run
//! use velo_event::{EventManager, LocalEventSystem, Event};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = LocalEventSystem::new();
//!
//!     let load_weights = system.new_event()?;
//!     let load_tokenizer = system.new_event()?;
//!
//!     // merged event completes only after both inputs complete
//!     let ready = system.merge_events(vec![
//!         load_weights.handle(),
//!         load_tokenizer.handle(),
//!     ])?;
//!
//!     load_weights.trigger()?;
//!     load_tokenizer.trigger()?;
//!
//!     system.awaiter(ready)?.await?;
//!     Ok(())
//! }
//! ```
//!
//! Because merged events are themselves events, you can merge merges to build
//! arbitrary DAGs of preconditions.
//!
//! ### EventGuard — automatic cleanup
//!
//! [`EventGuard`] is an RAII wrapper that poisons the event on drop unless you
//! explicitly trigger it. This prevents events from being silently forgotten
//! when a task panics or returns early:
//!
//! ```rust,no_run
//! use velo_event::{EventManager, LocalEventSystem, Event};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = LocalEventSystem::new();
//!     let event = system.new_event()?;
//!
//!     let guard = event.into_guard();
//!
//!     // If this function returns early or panics, the guard
//!     // drops and poisons the event automatically.
//!     do_work()?;
//!
//!     guard.trigger()?; // success — disarms the guard
//!     Ok(())
//! }
//!
//! fn do_work() -> anyhow::Result<()> { Ok(()) }
//! ```
//!
//! ### Poison propagation
//!
//! When an event is poisoned, all awaiters receive an error containing the
//! reason. Merged events accumulate poison reasons from their inputs:
//!
//! ```rust,no_run
//! use velo_event::{Event, EventManager, LocalEventSystem, EventPoison};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let system = LocalEventSystem::new();
//!
//!     let a = system.new_event()?;
//!     let b = system.new_event()?;
//!     let merged = system.merge_events(vec![a.handle(), b.handle()])?;
//!
//!     system.poison(a.handle(), "a failed")?;
//!     system.poison(b.handle(), "b failed")?;
//!
//!     let err = system.awaiter(merged)?.await.unwrap_err();
//!     let poison = err.downcast::<EventPoison>()?;
//!     assert!(poison.reason().contains("a failed"));
//!     assert!(poison.reason().contains("b failed"));
//!     Ok(())
//! }
//! ```
//!
//! ## Distributed events
//!
//! For distributed deployments, [`DistributedEventFactory`] creates an event system
//! whose handles embed a non-zero `system_id` for global uniqueness. The local
//! event machinery stays the same — coordination across systems is handled by an
//! active-messaging layer built on top.
//!
//! [`LocalEventSystem`] enforces that handles belong to the system that created
//! them. Passing a handle from one system to another will return an error.
//! A distributed event system must implement [`EventManager`] with routing
//! logic to forward operations on remote handles to the correct owning system.
//!
//! ```rust,no_run
//! use velo_event::DistributedEventFactory;
//!
//! let factory = DistributedEventFactory::new(0x42.try_into().unwrap());
//! let system = factory.event_manager();
//! // handles produced by this system carry system_id = 0x42
//! ```

// Public trait API
mod event;
mod guard;
mod manager;

// Public types
pub mod factory;
mod handle;
mod status;

// Local implementation
pub mod local;

// Internal synchronization (frozen — do not modify)
pub(crate) mod slot;

// ── Re-exports ───────────────────────────────────────────────────────

pub use event::Event;
pub use factory::DistributedEventFactory;
pub use guard::EventGuard;
pub use handle::EventHandle;
pub use local::{LocalEvent, LocalEventSystem};
pub use manager::EventManager;
pub use slot::EventAwaiter;
pub use status::{EventPoison, EventStatus, Generation};

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tokio::task::yield_now;

    fn create_system() -> Arc<LocalEventSystem> {
        LocalEventSystem::new()
    }

    use std::sync::Arc;

    #[tokio::test]
    async fn wait_resolves_after_trigger() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let waiter = {
            let system = Arc::clone(&system);
            tokio::spawn(async move { system.awaiter(handle)?.await })
        };

        yield_now().await;
        event.trigger()?;
        waiter.await??;
        Ok(())
    }

    #[tokio::test]
    async fn wait_ready_if_triggered_first() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        event.trigger()?;
        system.awaiter(handle)?.await?;
        Ok(())
    }

    #[tokio::test]
    async fn poison_is_visible() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        system.poison(handle, "boom")?;
        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "boom");
        Ok(())
    }

    #[tokio::test]
    async fn entry_reused_after_completion() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();
        let index = handle.local_index();
        let generation = handle.generation();

        event.trigger()?;
        system.awaiter(handle)?.await?;

        let next = system.new_event()?;
        let next_handle = next.handle();
        assert_eq!(next_handle.local_index(), index);
        assert_eq!(next_handle.generation(), generation + 1);
        Ok(())
    }

    #[tokio::test]
    async fn multiple_waiters_wake() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let mut waiters = Vec::new();
        for _ in 0..8 {
            let system_clone = Arc::clone(&system);
            waiters.push(tokio::spawn(
                async move { system_clone.awaiter(handle)?.await },
            ));
        }

        yield_now().await;
        event.trigger()?;
        for waiter in waiters {
            waiter.await??;
        }
        Ok(())
    }

    #[tokio::test]
    async fn merge_triggers_after_dependencies() -> Result<()> {
        let system = create_system();
        let first = system.new_event()?;
        let second = system.new_event()?;

        let merged = system.merge_events(vec![first.handle(), second.handle()])?;

        first.trigger()?;
        second.trigger()?;

        system.awaiter(merged)?.await?;
        Ok(())
    }

    #[tokio::test]
    async fn merge_poison_accumulates_reasons() -> Result<()> {
        let system = create_system();
        let first = system.new_event()?;
        let second = system.new_event()?;

        let merged = system.merge_events(vec![first.handle(), second.handle()])?;

        system.poison(first.handle(), "first failed")?;
        system.poison(second.handle(), "second failed")?;

        let err = system.awaiter(merged)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert!(poison.reason().contains("first failed"));
        assert!(poison.reason().contains("second failed"));
        Ok(())
    }

    #[tokio::test]
    async fn force_shutdown_poison_pending() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let waiter = {
            let system = system.clone();
            tokio::spawn(async move { system.awaiter(handle)?.await })
        };

        yield_now().await;
        system.force_shutdown("shutdown");

        let err = waiter.await.unwrap().unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "shutdown");
        Ok(())
    }

    #[tokio::test]
    async fn new_event_fails_after_force_shutdown() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        system.force_shutdown("shutdown");

        let err = match system.new_event() {
            Ok(_) => panic!("expected shutdown to block new events"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("shutdown"));

        let err = system.awaiter(event.handle())?.await.unwrap_err();
        assert!(err.downcast::<EventPoison>().is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn force_shutdown_is_idempotent() -> Result<()> {
        let system = create_system();
        let _ = system.new_event()?;
        system.force_shutdown("shutdown");
        system.force_shutdown("shutdown");
        assert!(system.new_event().is_err());
        Ok(())
    }

    // ── Trait-based tests ─────────────────────────────────────────────

    fn exercise_manager(mgr: &impl EventManager) -> Result<()> {
        let event = mgr.new_event()?;
        let handle = event.handle();

        assert_eq!(mgr.poll(handle)?, EventStatus::Pending);
        mgr.trigger(handle)?;
        assert_eq!(mgr.poll(handle)?, EventStatus::Ready);
        Ok(())
    }

    #[tokio::test]
    async fn trait_exercise_manager() -> Result<()> {
        let system = create_system();
        exercise_manager(&system)
    }

    #[tokio::test]
    async fn trait_exercise_guard_poison_on_drop() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        {
            let _guard = event.into_guard();
            // guard drops here without trigger → poisons the event
        }

        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert!(
            poison
                .reason()
                .contains("event dropped without being triggered")
        );
        Ok(())
    }

    #[tokio::test]
    async fn trait_exercise_guard_trigger() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let guard = event.into_guard();
        guard.trigger()?;

        system.awaiter(handle)?.await?;
        Ok(())
    }

    // ── DistributedEventFactory (factory.rs) ─────────────────────────

    #[tokio::test]
    async fn distributed_factory_stamps_system_id() -> Result<()> {
        use crate::factory::DistributedEventFactory;

        let factory = DistributedEventFactory::new(0x42.try_into().unwrap());
        assert_eq!(factory.system_id(), 0x42);

        let mgr = factory.event_manager();
        let event = mgr.new_event()?;
        let handle = event.handle();
        assert_eq!(handle.system_id(), 0x42);
        assert!(handle.is_distributed());

        // system() returns the same underlying system
        assert!(Arc::ptr_eq(factory.system(), &mgr));

        event.trigger()?;
        mgr.awaiter(handle)?.await?;
        Ok(())
    }

    // ── EventHandle accessors (handle.rs) ────────────────────────────

    #[test]
    fn handle_round_trip_raw() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();

        let raw = handle.raw();
        let reconstructed = EventHandle::from_raw(raw);
        assert_eq!(handle, reconstructed);
    }

    #[test]
    fn handle_system_id_local() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();
        assert_ne!(handle.system_id(), 0);
        assert!(handle.is_local());
    }

    #[test]
    fn handle_with_generation() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();

        let new_handle = handle.with_generation(99);
        assert_eq!(new_handle.generation(), 99);
        assert_eq!(new_handle.local_index(), handle.local_index());
        assert_eq!(new_handle.system_id(), handle.system_id());
    }

    #[test]
    fn handle_display() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();
        let display = format!("{}", handle);
        assert!(display.contains("EventHandle"));
        assert!(display.contains("system="));
        assert!(display.contains("index="));
        assert!(display.contains("generation="));
        assert!(display.contains("local"));
    }

    // ── EventGuard poison / awaiter (guard.rs) ───────────────────────

    #[tokio::test]
    async fn guard_explicit_poison() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let guard = event.into_guard();
        guard.poison("explicit")?;

        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "explicit");
        Ok(())
    }

    #[tokio::test]
    async fn guard_awaiter() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let guard = event.into_guard();
        let awaiter = guard.awaiter()?;

        // Also verify guard.handle()
        assert_eq!(guard.handle(), handle);

        guard.trigger()?;
        awaiter.await?;
        Ok(())
    }

    #[tokio::test]
    async fn guard_with_custom_reason() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        {
            let _guard = event.into_guard_with_reason("custom drop reason");
            // guard drops here without trigger
        }

        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert!(poison.reason().contains("custom drop reason"));
        Ok(())
    }

    // ── LocalEvent::poison (local/event.rs) ──────────────────────────

    #[tokio::test]
    async fn event_poison_directly() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        event.poison("direct reason")?;

        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "direct reason");
        Ok(())
    }

    // ── EventPoison Display and accessors (status.rs) ────────────────

    #[tokio::test]
    async fn poison_display_and_reason_arc() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        system.poison(handle, "test reason")?;
        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();

        // Display impl
        let display = format!("{}", poison);
        assert!(display.contains("poisoned"));
        assert!(display.contains("test reason"));

        // reason_arc accessor
        let arc = poison.reason_arc();
        assert_eq!(&**arc, "test reason");

        // handle accessor
        assert_eq!(poison.handle(), handle);

        // std::error::Error impl — no source
        assert!(std::error::Error::source(&poison).is_none());
        Ok(())
    }

    // ── System-level edge cases (local/system.rs) ────────────────────

    #[tokio::test]
    async fn poison_reason_helper() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        system.poison(handle, "oops")?;

        let reason = system.poison_reason(handle);
        assert!(reason.is_some());
        assert_eq!(&*reason.unwrap(), "oops");
        Ok(())
    }

    // ── Local vs distributed flag ────────────────────────────────────

    #[test]
    fn is_local_vs_distributed() {
        // Local system produces local handles
        let local = create_system();
        let event = local.new_event().unwrap();
        let handle = event.handle();
        assert!(handle.is_local());
        assert!(!handle.is_distributed());
        assert_ne!(handle.system_id(), 0);

        // Distributed factory produces distributed handles
        let factory = DistributedEventFactory::new(0x99.try_into().unwrap());
        let mgr = factory.event_manager();
        let event = mgr.new_event().unwrap();
        let handle = event.handle();
        assert!(handle.is_distributed());
        assert!(!handle.is_local());
        assert_eq!(handle.system_id(), 0x99);
    }

    // ── Cross-system validation tests ────────────────────────────────

    #[tokio::test]
    async fn cross_system_awaiter_rejected() -> Result<()> {
        let system_a = create_system();
        let system_b = create_system();

        let event = system_a.new_event()?;
        let handle = event.handle();

        match system_b.awaiter(handle) {
            Ok(_) => panic!("expected error for cross-system awaiter"),
            Err(err) => assert!(err.to_string().contains("belongs to system")),
        }
        Ok(())
    }

    #[tokio::test]
    async fn cross_system_trigger_rejected() -> Result<()> {
        let system_a = create_system();
        let system_b = create_system();

        let event = system_a.new_event()?;
        let handle = event.handle();

        let err = system_b.trigger(handle).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_system_poison_rejected() -> Result<()> {
        let system_a = create_system();
        let system_b = create_system();

        let event = system_a.new_event()?;
        let handle = event.handle();

        let err = system_b.poison(handle, "bad").unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_system_poll_rejected() -> Result<()> {
        let system_a = create_system();
        let system_b = create_system();

        let event = system_a.new_event()?;
        let handle = event.handle();

        let err = system_b.poll(handle).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_system_merge_rejected() -> Result<()> {
        let system_a = create_system();
        let system_b = create_system();

        let event = system_a.new_event()?;
        let handle = event.handle();

        let err = system_b.merge_events(vec![handle]).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_type_local_on_distributed_rejected() -> Result<()> {
        let local = create_system();
        let factory = DistributedEventFactory::new(0x10.try_into().unwrap());
        let distributed = factory.event_manager();

        let event = local.new_event()?;
        let handle = event.handle();

        let err = distributed.trigger(handle).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_type_distributed_on_local_rejected() -> Result<()> {
        let local = create_system();
        let factory = DistributedEventFactory::new(0x20.try_into().unwrap());
        let distributed = factory.event_manager();

        let event = distributed.new_event()?;
        let handle = event.handle();

        let err = local.trigger(handle).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_distributed_systems_rejected() -> Result<()> {
        let factory_a = DistributedEventFactory::new(0x30.try_into().unwrap());
        let factory_b = DistributedEventFactory::new(0x40.try_into().unwrap());
        let mgr_a = factory_a.event_manager();
        let mgr_b = factory_b.event_manager();

        let event = mgr_a.new_event()?;
        let handle = event.handle();

        let err = mgr_b.trigger(handle).unwrap_err();
        assert!(err.to_string().contains("belongs to system"));
        Ok(())
    }
}
