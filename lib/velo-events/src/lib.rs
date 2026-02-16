// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

// Public trait API
mod event;
mod manager;

// Public types
pub mod factory;
mod handle;
mod status;

// Local implementation
pub mod local;

// Internal synchronization (see docs/slot-state-machine.md)
pub(crate) mod slot;

// ── Re-exports ───────────────────────────────────────────────────────

pub use event::Event;
pub use factory::DistributedEventFactory;
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
        let handle = event.into_handle();

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
    async fn trait_exercise_drop_poison() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        {
            let _event = event;
            // event drops here without trigger → poisons the event
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
    async fn trait_exercise_trigger() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        event.trigger()?;

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

    // ── Event poison / awaiter ──────────────────────────────────────

    #[tokio::test]
    async fn event_explicit_poison() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        event.poison("explicit")?;

        let err = system.awaiter(handle)?.await.unwrap_err();
        let poison = err.downcast::<EventPoison>().unwrap();
        assert_eq!(poison.reason(), "explicit");
        Ok(())
    }

    #[tokio::test]
    async fn event_awaiter() -> Result<()> {
        let system = create_system();
        let event = system.new_event()?;
        let handle = event.handle();

        let awaiter = event.awaiter()?;

        // Verify event.handle() still works before consuming
        assert_eq!(event.handle(), handle);

        event.trigger()?;
        awaiter.await?;
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

    // ── slot regression tests ────────────────────────────────────────

    #[tokio::test]
    async fn race1_no_stale_completion_leakage() -> Result<()> {
        // Regression test for Race 1: stale completion visible to new-generation waiters.
        //
        // Scenario: waiter from gen N is still alive when gen N+1 starts.
        // In the old slot module, begin_generation would skip clearing completion
        // when waiter_count > 0, causing gen N+1 waiters to see gen N's result.
        // The slot design eliminates this structurally.
        let system = create_system();

        // Gen 1: create event and a waiter (keeps waiter alive across generation boundary)
        let event1 = system.new_event()?;
        let handle1 = event1.handle();
        let _waiter1 = system.awaiter(handle1)?;

        // Complete gen 1
        event1.trigger()?;

        // Gen 2: same entry reused from free list
        let event2 = system.new_event()?;
        let handle2 = event2.handle();
        assert_eq!(handle2.local_index(), handle1.local_index());
        assert_eq!(handle2.generation(), handle1.generation() + 1);

        // Create waiter for gen 2 — must be Pending, not stale Ready from gen 1
        let waiter2 = system.awaiter(handle2)?;

        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        let mut waiter2 = waiter2;
        let poll = std::pin::Pin::new(&mut waiter2).poll(&mut cx);
        assert!(
            poll.is_pending(),
            "Gen N+1 waiter should be Pending, not resolved with stale completion"
        );

        // Complete gen 2 and verify it resolves
        event2.trigger()?;
        waiter2.await?;
        Ok(())
    }

    #[tokio::test]
    async fn stale_waiter_resolves_after_generation_transition() -> Result<()> {
        // Test that a waiter from gen N resolves correctly even after gen N+1 starts.
        let system = create_system();

        let event1 = system.new_event()?;
        let handle1 = event1.handle();

        // Create a waiter for gen 1
        let waiter1 = system.awaiter(handle1)?;

        // Complete gen 1
        event1.trigger()?;

        // Start gen 2 (same entry reused)
        let event2 = system.new_event()?;
        assert_eq!(event2.handle().local_index(), handle1.local_index());

        // Waiter from gen 1 should still resolve correctly
        waiter1.await?;
        Ok(())
    }

    #[tokio::test]
    async fn stale_waiter_with_poison_resolves_after_generation_transition() -> Result<()> {
        // Poisoned gen N waiter resolves correctly after gen N+1 begins.
        let system = create_system();

        let event1 = system.new_event()?;
        let handle1 = event1.handle();
        let waiter1 = system.awaiter(handle1)?;

        // Poison gen 1
        system.poison(handle1, "gen1 failed")?;

        // Start gen 2
        let _event2 = system.new_event()?;

        // Waiter from gen 1 should see the poison
        let err = waiter1.await.unwrap_err();
        let poison = err.downcast::<EventPoison>()?;
        assert_eq!(poison.reason(), "gen1 failed");
        Ok(())
    }
}
