// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event types for KV cache coordination across workers.
//!
//! This module defines the event protocol used to track block registrations
//! and removals across distributed workers. Events are emitted when blocks
//! at power-of-2 positions are registered or released.
//!
//! The event types are organized in three layers:
//! - [`KvCacheEvent`]: Individual events for internal streaming
//! - [`KvCacheEvents`]: Batched events with multiple sequence hashes
//! - [`KvbmCacheEvents`]: Wire format with instance/cluster context

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::SequenceHash;

/// Instance identifier for a worker node (u128).
pub type InstanceId = u128;

/// Individual event emitted when a block is registered or removed.
///
/// This is the simplified internal event type. Instance and cluster context
/// are provided at the batch level via [`KvbmCacheEvents`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCacheEvent {
    /// A block has been registered in a worker's cache.
    Create(SequenceHash),
    /// A block has been removed from a worker's cache.
    Remove(SequenceHash),
}

/// Batched events with multiple sequence hashes.
///
/// Events are batched by type - either all creates or all removes.
/// Create events are sorted by position ascending (low to high) for efficient
/// radix tree insertion. Remove events are sorted by position descending
/// (high to low) for efficient radix tree removal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCacheEvents {
    /// Multiple blocks have been registered.
    Create(Vec<SequenceHash>),
    /// Multiple blocks have been removed.
    Remove(Vec<SequenceHash>),
    /// Publisher is shutting down.
    Shutdown,
}

/// Wire format for publishing batched events.
///
/// This is the complete message format sent over the wire, including
/// instance context that applies to all events in the batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvbmCacheEvents {
    /// The batched events.
    pub events: KvCacheEvents,
    /// The worker instance that generated these events.
    pub instance_id: InstanceId,
}

/// RAII handle that triggers a Remove event when dropped.
///
/// This handle is attached to a [`crate::registry::BlockRegistrationHandle`] as an [`std::sync::Arc<dyn std::any::Any>`].
/// When all references to the block are dropped, this handle's Drop implementation
/// sends a Remove event to clean up the hub's tracking state.
pub struct EventReleaseHandle {
    seq_hash: SequenceHash,
    event_tx: broadcast::Sender<KvCacheEvent>,
}

impl EventReleaseHandle {
    /// Creates a new release handle.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash of the block
    /// * `event_tx` - Broadcast channel sender for emitting the Remove event
    pub fn new(seq_hash: SequenceHash, event_tx: broadcast::Sender<KvCacheEvent>) -> Self {
        Self { seq_hash, event_tx }
    }
}

impl Drop for EventReleaseHandle {
    fn drop(&mut self) {
        let event = KvCacheEvent::Remove(self.seq_hash);
        // Broadcast send only fails if there are no receivers, which is fine
        let _ = self.event_tx.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvbmSequenceHashProvider;
    use dynamo_tokens::TokenBlockSequence;
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    #[test]
    fn test_event_serialization() {
        let tokens = vec![1u32, 2, 3, 4];
        let seq = TokenBlockSequence::from_slice(&tokens, tokens.len() as u32, Some(1337));
        let seq_hash = seq.blocks()[0].kvbm_sequence_hash();

        let create_event = KvCacheEvent::Create(seq_hash);
        let serialized = serde_json::to_string(&create_event).unwrap();
        let deserialized: KvCacheEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(create_event, deserialized);

        let remove_event = KvCacheEvent::Remove(seq_hash);
        let serialized = serde_json::to_string(&remove_event).unwrap();
        let deserialized: KvCacheEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(remove_event, deserialized);
    }

    #[test]
    fn test_batch_events_serialization() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let seq = TokenBlockSequence::from_slice(&tokens, 4, Some(1337));
        let seq_hashes: Vec<_> = seq
            .blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect();

        let batch = KvbmCacheEvents {
            events: KvCacheEvents::Create(seq_hashes.clone()),
            instance_id: 12345,
        };

        let serialized = serde_json::to_string(&batch).unwrap();
        let deserialized: KvbmCacheEvents = serde_json::from_str(&serialized).unwrap();
        assert_eq!(batch, deserialized);
    }

    #[tokio::test]
    async fn test_release_handle_drop() {
        let tokens = vec![1u32, 2, 3, 4];
        let seq = TokenBlockSequence::from_slice(&tokens, tokens.len() as u32, Some(1337));
        let seq_hash = seq.blocks()[0].kvbm_sequence_hash();

        let (tx, rx) = broadcast::channel(16);
        let mut stream = BroadcastStream::new(rx);

        {
            let _handle = EventReleaseHandle::new(seq_hash, tx);
            // Handle is dropped here
        }

        // Should receive Remove event
        let event = stream.next().await.unwrap().unwrap();
        assert_eq!(event, KvCacheEvent::Remove(seq_hash));
    }
}
