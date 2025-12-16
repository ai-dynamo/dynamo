// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EventsManager for coordinating block registration events.
//!
//! The EventsManager hooks into BlockRegistry to emit KvCacheEvents when blocks
//! are registered or removed. It uses a policy to filter which blocks trigger events
//! and maintains a channel for publishing events to the hub.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use super::policy::EventEmissionPolicy;
use super::protocol::{EventReleaseHandle, InstanceId, KvCacheEvent};
use crate::v2::logical::blocks::BlockRegistrationHandle;

/// Manager for emitting and coordinating block registration events.
///
/// The EventsManager is responsible for:
/// - Filtering block registrations based on a policy
/// - Emitting Create events when blocks are registered
/// - Attaching RAII handles that emit Remove events when blocks are dropped
/// - Publishing events through a channel for consumption by the hub
pub struct EventsManager {
    policy: Arc<dyn EventEmissionPolicy>,
    instance_id: InstanceId,
    cluster_id: String,
    event_tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl EventsManager {
    /// Creates a new EventsManager.
    ///
    /// # Arguments
    /// * `policy` - Policy for determining which blocks should emit events
    /// * `instance_id` - Unique identifier for this worker instance
    /// * `cluster_id` - Cluster/deployment identifier for event routing
    ///
    /// # Returns
    /// A tuple of (EventsManager, receiver for consuming events)
    pub fn new(
        policy: Arc<dyn EventEmissionPolicy>,
        instance_id: InstanceId,
        cluster_id: String,
    ) -> (Self, mpsc::UnboundedReceiver<KvCacheEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let manager = Self {
            policy,
            instance_id,
            cluster_id,
            event_tx,
        };

        (manager, event_rx)
    }

    /// Hook called when a block is registered in the BlockRegistry.
    ///
    /// This method:
    /// 1. Checks the policy to determine if an event should be emitted
    /// 2. Sends a Create event if the policy allows
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
        let create_event = KvCacheEvent::Create {
            seq_hash,
            instance_id: self.instance_id,
            cluster_id: self.cluster_id.clone(),
        };

        if self.event_tx.send(create_event).is_err() {
            tracing::warn!("Failed to send Create event for seq_hash {:?}", seq_hash);
        }

        // Attach RAII handle for Remove event
        let release_handle =
            EventReleaseHandle::new(seq_hash, self.instance_id, self.event_tx.clone());

        // Attach as Arc<dyn Any> to the registration handle
        handle.attach_unique(Arc::new(release_handle))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy::PowerOfTwoPolicy;
    use super::*;
    use crate::v2::logical::blocks::BlockRegistry;
    use crate::{KvbmSequenceHashProvider, SequenceHash};
    use dynamo_tokens::TokenBlockSequence;

    fn create_seq_hash_at_position(position: usize) -> SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    #[tokio::test]
    async fn test_events_manager_emits_create_for_power_of_two() {
        let policy = Arc::new(PowerOfTwoPolicy::new());
        let (manager, mut event_rx) = EventsManager::new(policy, 12345, "test-cluster".to_string());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(16); // Power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle).unwrap();

        // Should receive Create event
        let event = event_rx.try_recv().unwrap();
        match event {
            KvCacheEvent::Create {
                seq_hash: received_hash,
                instance_id,
                cluster_id,
            } => {
                assert_eq!(received_hash, seq_hash);
                assert_eq!(instance_id, 12345);
                assert_eq!(cluster_id, "test-cluster");
            }
            _ => panic!("Expected Create event"),
        }
    }

    #[tokio::test]
    async fn test_events_manager_skips_non_power_of_two() {
        let policy = Arc::new(PowerOfTwoPolicy::new());
        let (manager, mut event_rx) = EventsManager::new(policy, 12345, "test-cluster".to_string());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(17); // Not power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle).unwrap();

        // Should NOT receive any event
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_events_manager_emits_remove_on_drop() {
        let policy = Arc::new(PowerOfTwoPolicy::new());
        let (manager, mut event_rx) = EventsManager::new(policy, 12345, "test-cluster".to_string());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(32); // Power of 2

        {
            let handle = registry.register_sequence_hash(seq_hash);
            manager.on_block_registered(&handle).unwrap();

            // Consume Create event
            let _create = event_rx.try_recv().unwrap();

            // Handle is dropped here, triggering Remove
        }

        // Should receive Remove event
        let event = event_rx.try_recv().unwrap();
        match event {
            KvCacheEvent::Remove {
                seq_hash: received_hash,
                instance_id,
            } => {
                assert_eq!(received_hash, seq_hash);
                assert_eq!(instance_id, 12345);
            }
            _ => panic!("Expected Remove event"),
        }
    }
}
