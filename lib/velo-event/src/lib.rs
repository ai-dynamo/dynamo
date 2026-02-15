// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generational event system for coordinating local awaiters with minimal overhead.
//!
//! # Overview
//!
//! This crate provides trait-based abstractions for an event system:
//!
//! - [`Event`] — a single event that can be triggered or poisoned
//! - [`EventManager`] — creates and manages a collection of events
//! - [`EventGuard`] — RAII wrapper that poisons on drop
//! - [`EventHandle`] — unique identifier for an event
//!
//! The [`local`] module provides the concrete local implementation.

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
    async fn distributed_factory_stamps_worker_id() -> Result<()> {
        use crate::factory::DistributedEventFactory;

        let factory = DistributedEventFactory::new(0x42);
        assert_eq!(factory.worker_id(), 0x42);

        let mgr = factory.event_manager();
        let event = mgr.new_event()?;
        let handle = event.handle();
        assert_eq!(handle.worker_id(), 0x42);
        assert!(!handle.is_local_only());

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
    fn handle_worker_id_local() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();
        assert_eq!(handle.worker_id(), 0);
        assert!(handle.is_local_only());
    }

    #[test]
    fn handle_with_generation() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();

        let new_handle = handle.with_generation(99);
        assert_eq!(new_handle.generation(), 99);
        assert_eq!(new_handle.local_index(), handle.local_index());
        assert_eq!(new_handle.worker_id(), handle.worker_id());
    }

    #[test]
    fn handle_display() {
        let system = create_system();
        let event = system.new_event().unwrap();
        let handle = event.handle();
        let display = format!("{}", handle);
        assert!(display.contains("EventHandle"));
        assert!(display.contains("index="));
        assert!(display.contains("generation="));
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
}
