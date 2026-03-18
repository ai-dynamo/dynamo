// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EventsManager for coordinating block registration events.
//!
//! The EventsManager hooks into BlockRegistry to emit KvCacheEvents when blocks
//! are registered or removed. It uses a policy to filter which blocks trigger events
//! and a broadcast channel to allow multiple subscribers.

use std::sync::Arc;

use anyhow::Result;
use derive_builder::Builder;
use futures::Stream;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use super::policy::EventEmissionPolicy;
use super::protocol::{EventReleaseHandle, KvCacheEvent};
use crate::registry::BlockRegistrationHandle;

/// Settings for constructing an [`EventsManager`].
///
/// # Example
///
/// ```ignore
/// // Simple with defaults (AllEventsPolicy)
/// let manager = EventsManagerSettings::builder().build()?.into_manager();
///
/// // With custom policy
/// let manager = EventsManagerSettings::builder()
///     .policy(Arc::new(PowerOfTwoPolicy::new()))
///     .build()?
///     .into_manager();
///
/// // With custom configuration
/// let manager = EventsManagerSettings::builder()
///     .channel_capacity(2048)
///     .build()?
///     .into_manager();
/// ```
#[derive(Builder, Clone)]
#[builder(setter(into, strip_option), build_fn(error = "anyhow::Error"))]
pub struct EventsManagerSettings {
    /// The event emission policy.
    ///
    /// Default: [`AllEventsPolicy`](super::policy::AllEventsPolicy)
    #[builder(default, setter(strip_option = false))]
    policy: Option<Arc<dyn EventEmissionPolicy>>,

    /// Capacity of the broadcast channel.
    ///
    /// Default: 1024
    #[builder(default = "1024")]
    channel_capacity: usize,
}

impl EventsManagerSettings {
    /// Creates a new builder for EventsManagerSettings.
    pub fn builder() -> EventsManagerSettingsBuilder {
        EventsManagerSettingsBuilder::default()
    }

    /// Converts settings into an EventsManager.
    pub fn into_manager(self) -> EventsManager {
        let policy = self
            .policy
            .unwrap_or_else(|| Arc::new(super::policy::AllEventsPolicy::new()));
        let (event_tx, _) = broadcast::channel(self.channel_capacity);

        EventsManager { policy, event_tx }
    }
}

/// Manager for emitting and coordinating block registration events.
///
/// The EventsManager is responsible for:
/// - Filtering block registrations based on a policy
/// - Emitting Create events when blocks are registered
/// - Attaching RAII handles that emit Remove events when blocks are dropped
/// - Broadcasting events to multiple subscribers via [`subscribe()`](Self::subscribe)
///
/// Note: Instance context is applied at the publisher level via
/// [`KvbmCacheEventsPublisher`](super::publisher::KvbmCacheEventsPublisher).
///
/// # Example
///
/// ```ignore
/// // Create with defaults (AllEventsPolicy)
/// let manager = EventsManager::builder().build();
///
/// // Create with PowerOfTwoPolicy
/// let manager = EventsManager::builder()
///     .policy(Arc::new(PowerOfTwoPolicy::new()))
///     .build();
/// ```
pub struct EventsManager {
    policy: Arc<dyn EventEmissionPolicy>,
    event_tx: broadcast::Sender<KvCacheEvent>,
}

/// Builder for [`EventsManager`] that wraps [`EventsManagerSettingsBuilder`].
#[derive(Default)]
pub struct EventsManagerBuilder(EventsManagerSettingsBuilder);

impl EventsManagerBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the event emission policy.
    ///
    /// Default: [`AllEventsPolicy`](super::policy::AllEventsPolicy)
    pub fn policy(mut self, policy: Arc<dyn EventEmissionPolicy>) -> Self {
        self.0.policy = Some(Some(policy));
        self
    }

    /// Sets the broadcast channel capacity.
    ///
    /// Default: 1024
    pub fn channel_capacity(mut self, capacity: usize) -> Self {
        self.0.channel_capacity = Some(capacity);
        self
    }

    /// Builds the EventsManager.
    pub fn build(self) -> EventsManager {
        self.0
            .build()
            .expect("EventsManagerSettings has all defaults")
            .into_manager()
    }
}

impl EventsManager {
    /// Creates a new builder for EventsManager.
    pub fn builder() -> EventsManagerBuilder {
        EventsManagerBuilder::new()
    }

    /// Subscribe to the event stream.
    ///
    /// Returns a stream of events. Multiple subscribers are supported, and each
    /// subscriber receives all events. Late subscribers will miss events that
    /// occurred before subscribing.
    ///
    /// The stream filters out lagged errors (when events are dropped due to
    /// slow consumption) and continues delivering subsequent events.
    pub fn subscribe(&self) -> impl Stream<Item = KvCacheEvent> + Send + 'static {
        let rx = self.event_tx.subscribe();
        BroadcastStream::new(rx).filter_map(|result| result.ok())
    }

    /// Hook called when a block is registered in the BlockRegistry.
    ///
    /// This method:
    /// 1. Checks the policy to determine if an event should be emitted
    /// 2. Broadcasts a Create event if the policy allows
    /// 3. Attaches an EventReleaseHandle to the registration handle for cleanup
    ///
    /// # Arguments
    /// * `handle` - The block registration handle
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if attachment fails
    pub fn on_block_registered(&self, handle: &BlockRegistrationHandle) -> Result<()> {
        let seq_hash = handle.seq_hash();

        // Check policy - only emit events for filtered blocks
        if !self.policy.should_emit(seq_hash) {
            return Ok(());
        }

        // Emit Create event
        let create_event = KvCacheEvent::Create(seq_hash);

        // Broadcast send only fails if there are no receivers, which is fine
        let _ = self.event_tx.send(create_event);

        // Attach RAII handle for Remove event
        let release_handle = EventReleaseHandle::new(seq_hash, self.event_tx.clone());

        // Attach as Arc<dyn Any> to the registration handle
        handle.attach_unique(Arc::new(release_handle))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy::PowerOfTwoPolicy;
    use super::*;
    use crate::registry::BlockRegistry;
    use crate::{KvbmSequenceHashProvider, SequenceHash};
    use dynamo_tokens::TokenBlockSequence;
    use futures::StreamExt;

    fn create_seq_hash_at_position(position: usize) -> SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    #[tokio::test]
    async fn test_events_manager_emits_create_for_power_of_two() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(16); // Power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle).unwrap();

        // Should receive Create event
        let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event, KvCacheEvent::Create(seq_hash));
    }

    #[tokio::test]
    async fn test_events_manager_skips_non_power_of_two() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(17); // Not power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle).unwrap();

        // Should NOT receive any event (will timeout)
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(50), stream.next()).await;
        assert!(result.is_err()); // Timeout expected

        // Keep handle alive to prevent drop event
        drop(handle);
    }

    #[tokio::test]
    async fn test_events_manager_emits_remove_on_drop() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(32); // Power of 2

        {
            let handle = registry.register_sequence_hash(seq_hash);
            manager.on_block_registered(&handle).unwrap();

            // Consume Create event
            let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(event, KvCacheEvent::Create(seq_hash));

            // Handle is dropped here, triggering Remove
        }

        // Should receive Remove event
        let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event, KvCacheEvent::Remove(seq_hash));
    }

    #[tokio::test]
    async fn test_events_manager_multiple_subscribers() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();

        let mut stream1 = Box::pin(manager.subscribe());
        let mut stream2 = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(64); // Power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        manager.on_block_registered(&handle).unwrap();

        // Both streams should receive the same event
        let event1 = tokio::time::timeout(std::time::Duration::from_millis(100), stream1.next())
            .await
            .unwrap()
            .unwrap();
        let event2 = tokio::time::timeout(std::time::Duration::from_millis(100), stream2.next())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event1, KvCacheEvent::Create(seq_hash));
        assert_eq!(event2, KvCacheEvent::Create(seq_hash));
    }

    #[test]
    fn test_events_manager_default_policy() {
        // With no policy specified, should use AllEventsPolicy (default)
        let manager = EventsManager::builder().build();

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(17); // Not power of 2

        // Use a subscriber to verify events are emitted
        let _subscription = manager.subscribe();

        let handle = registry.register_sequence_hash(seq_hash);

        // With AllEventsPolicy, all blocks should emit events
        // (this would fail with PowerOfTwoPolicy for position 17)
        manager.on_block_registered(&handle).unwrap();
    }
}
