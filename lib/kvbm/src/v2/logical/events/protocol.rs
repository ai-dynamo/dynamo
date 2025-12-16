// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event types for KV cache coordination across workers.
//!
//! This module defines the event protocol used to track block registrations
//! and removals across distributed workers. Events are emitted when blocks
//! at power-of-2 positions are registered or released.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::SequenceHash;

/// Instance identifier for a worker node (u128).
pub type InstanceId = u128;

/// Events emitted when blocks are registered or removed from the cache.
///
/// These events are used by the KvbmHub to maintain a distributed view
/// of which workers have which blocks cached.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCacheEvent {
    /// A block has been registered in a worker's cache.
    Create {
        /// The positional sequence hash identifying the block.
        seq_hash: SequenceHash,
        /// The worker instance that registered the block.
        instance_id: InstanceId,
        /// The cluster/deployment identifier for event routing.
        cluster_id: String,
    },
    /// A block has been removed from a worker's cache.
    Remove {
        /// The positional sequence hash identifying the block.
        seq_hash: SequenceHash,
        /// The worker instance that removed the block.
        instance_id: InstanceId,
    },
}

/// RAII handle that triggers a Remove event when dropped.
///
/// This handle is attached to a BlockRegistrationHandle as an Arc<dyn Any>.
/// When all references to the block are dropped, this handle's Drop implementation
/// sends a Remove event to clean up the hub's tracking state.
pub struct EventReleaseHandle {
    seq_hash: SequenceHash,
    instance_id: InstanceId,
    event_tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl EventReleaseHandle {
    /// Creates a new release handle.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash of the block
    /// * `instance_id` - The worker instance identifier
    /// * `event_tx` - Channel sender for emitting the Remove event
    pub fn new(
        seq_hash: SequenceHash,
        instance_id: InstanceId,
        event_tx: mpsc::UnboundedSender<KvCacheEvent>,
    ) -> Self {
        Self {
            seq_hash,
            instance_id,
            event_tx,
        }
    }
}

impl Drop for EventReleaseHandle {
    fn drop(&mut self) {
        let event = KvCacheEvent::Remove {
            seq_hash: self.seq_hash,
            instance_id: self.instance_id,
        };
        if self.event_tx.send(event).is_err() {
            tracing::warn!(
                "Failed to send Remove event for seq_hash {:?}",
                self.seq_hash
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvbmSequenceHashProvider;
    use dynamo_tokens::TokenBlockSequence;

    #[test]
    fn test_event_serialization() {
        let tokens = vec![1u32, 2, 3, 4];
        let seq = TokenBlockSequence::from_slice(&tokens, tokens.len() as u32, Some(1337));
        let seq_hash = seq.blocks()[0].kvbm_sequence_hash();

        let create_event = KvCacheEvent::Create {
            seq_hash,
            instance_id: 12345,
            cluster_id: "test-cluster".to_string(),
        };

        let serialized = serde_json::to_string(&create_event).unwrap();
        let deserialized: KvCacheEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(create_event, deserialized);

        let remove_event = KvCacheEvent::Remove {
            seq_hash,
            instance_id: 12345,
        };

        let serialized = serde_json::to_string(&remove_event).unwrap();
        let deserialized: KvCacheEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(remove_event, deserialized);
    }

    #[tokio::test]
    async fn test_release_handle_drop() {
        let tokens = vec![1u32, 2, 3, 4];
        let seq = TokenBlockSequence::from_slice(&tokens, tokens.len() as u32, Some(1337));
        let seq_hash = seq.blocks()[0].kvbm_sequence_hash();

        let (tx, mut rx) = mpsc::unbounded_channel();

        {
            let _handle = EventReleaseHandle::new(seq_hash, 12345, tx);
            // Handle is dropped here
        }

        // Should receive Remove event
        let event = rx.recv().await.unwrap();
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
