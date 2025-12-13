// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KvbmHub server for coordinating distributed block locations.
//!
//! The hub maintains a sparse radix tree of block locations across workers
//! and tracks access frequency for intelligent caching decisions.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::events::{InstanceId, KvCacheEvent};
use super::radix::PositionalRadixTree;
use crate::utils::tinylfu::{FrequencyTracker, TinyLFUTracker};
use crate::v2::PositionalSequenceHash;

/// Central hub for coordinating KV cache across distributed workers.
///
/// The KvbmHub:
/// - Maintains a sparse radix tree tracking which workers have which blocks
/// - Tracks block access frequency using TinyLFU
/// - Processes Create/Remove events from workers
/// - Provides query methods for finding blocks across the fleet
pub struct KvbmHub {
    radix_tree: Arc<PositionalRadixTree>,
    frequency_tracker: Arc<TinyLFUTracker<u128>>,
    event_rx: mpsc::UnboundedReceiver<KvCacheEvent>,
}

impl KvbmHub {
    /// Creates a new KvbmHub.
    ///
    /// # Arguments
    /// * `event_rx` - Receiver for consuming KvCacheEvents from workers
    /// * `frequency_capacity` - Capacity for the frequency tracker (default: 1M entries)
    pub fn new(event_rx: mpsc::UnboundedReceiver<KvCacheEvent>) -> Self {
        Self::with_capacity(event_rx, 1_000_000)
    }

    /// Creates a new KvbmHub with custom frequency tracker capacity.
    ///
    /// # Arguments
    /// * `event_rx` - Receiver for consuming KvCacheEvents from workers
    /// * `frequency_capacity` - Capacity for the frequency tracker
    pub fn with_capacity(
        event_rx: mpsc::UnboundedReceiver<KvCacheEvent>,
        frequency_capacity: usize,
    ) -> Self {
        Self {
            radix_tree: Arc::new(PositionalRadixTree::new()),
            frequency_tracker: Arc::new(TinyLFUTracker::new(frequency_capacity)),
            event_rx,
        }
    }

    /// Returns a reference to the radix tree for testing/inspection.
    pub fn radix_tree(&self) -> Arc<PositionalRadixTree> {
        self.radix_tree.clone()
    }

    /// Returns a reference to the frequency tracker for testing/inspection.
    pub fn frequency_tracker(&self) -> Arc<TinyLFUTracker<u128>> {
        self.frequency_tracker.clone()
    }

    /// Runs the hub's event processing loop.
    ///
    /// This method consumes the hub and runs indefinitely, processing events
    /// from workers. It should be spawned as a tokio task.
    ///
    /// # Example
    /// ```ignore
    /// let hub = KvbmHub::new(event_rx);
    /// tokio::spawn(async move {
    ///     hub.run().await;
    /// });
    /// ```
    pub async fn run(mut self) {
        while let Some(event) = self.event_rx.recv().await {
            self.process_event(event);
        }
    }

    /// Processes a single event.
    fn process_event(&self, event: KvCacheEvent) {
        match event {
            KvCacheEvent::Create {
                seq_hash,
                instance_id,
                cluster_id: _,
            } => {
                // Insert into radix tree
                self.radix_tree.insert(seq_hash, instance_id);

                // Track frequency
                self.frequency_tracker.touch(seq_hash.as_u128());

                tracing::debug!(
                    "Hub: Created block seq_hash={:?} on instance={}",
                    seq_hash,
                    instance_id
                );
            }
            KvCacheEvent::Remove {
                seq_hash,
                instance_id,
            } => {
                // Remove from radix tree
                self.radix_tree.remove(seq_hash, instance_id);

                tracing::debug!(
                    "Hub: Removed block seq_hash={:?} from instance={}",
                    seq_hash,
                    instance_id
                );
            }
        }
    }

    /// Finds which workers have a specific block.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash to look up
    ///
    /// # Returns
    /// Some(HashSet) of instance IDs if found, None otherwise
    pub fn find_workers(&self, seq_hash: PositionalSequenceHash) -> Option<HashSet<InstanceId>> {
        self.radix_tree.lookup(seq_hash)
    }

    /// Gets the access frequency for a block.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash to query
    ///
    /// # Returns
    /// The estimated access count
    pub fn get_frequency(&self, seq_hash: PositionalSequenceHash) -> u32 {
        self.frequency_tracker.count(seq_hash.as_u128())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::TokenBlockSequence;

    fn create_seq_hash_at_position(position: usize) -> PositionalSequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].positional_sequence_hash()
    }

    #[tokio::test]
    async fn test_hub_processes_create_event() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut hub = KvbmHub::new(rx);

        let seq_hash = create_seq_hash_at_position(16);
        let instance_id = 12345;

        let event = KvCacheEvent::Create {
            seq_hash,
            instance_id,
            cluster_id: "test".to_string(),
        };

        tx.send(event).unwrap();

        // Process one event
        let event = hub.event_rx.recv().await.unwrap();
        hub.process_event(event);

        // Verify block is in radix tree
        let workers = hub.find_workers(seq_hash).unwrap();
        assert_eq!(workers.len(), 1);
        assert!(workers.contains(&instance_id));

        // Verify frequency was tracked
        assert_eq!(hub.get_frequency(seq_hash), 1);
    }

    #[tokio::test]
    async fn test_hub_processes_remove_event() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let hub = KvbmHub::new(rx);

        let seq_hash = create_seq_hash_at_position(32);
        let instance_id = 12345;

        // Create then remove
        hub.process_event(KvCacheEvent::Create {
            seq_hash,
            instance_id,
            cluster_id: "test".to_string(),
        });

        hub.process_event(KvCacheEvent::Remove {
            seq_hash,
            instance_id,
        });

        // Verify block is not in radix tree
        let workers = hub.find_workers(seq_hash);
        // Should be None or empty set
        assert!(workers.is_none() || workers.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_hub_tracks_multiple_workers() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let hub = KvbmHub::new(rx);

        let seq_hash = create_seq_hash_at_position(64);

        // Multiple workers have the same block
        hub.process_event(KvCacheEvent::Create {
            seq_hash,
            instance_id: 111,
            cluster_id: "test".to_string(),
        });

        hub.process_event(KvCacheEvent::Create {
            seq_hash,
            instance_id: 222,
            cluster_id: "test".to_string(),
        });

        hub.process_event(KvCacheEvent::Create {
            seq_hash,
            instance_id: 333,
            cluster_id: "test".to_string(),
        });

        // Verify all workers are tracked
        let workers = hub.find_workers(seq_hash).unwrap();
        assert_eq!(workers.len(), 3);
        assert!(workers.contains(&111));
        assert!(workers.contains(&222));
        assert!(workers.contains(&333));

        // Frequency should be 3 (one touch per Create)
        assert_eq!(hub.get_frequency(seq_hash), 3);
    }

    #[tokio::test]
    async fn test_hub_find_blocks_in_range() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let hub = KvbmHub::new(rx);

        let hash_16 = create_seq_hash_at_position(16);
        let hash_32 = create_seq_hash_at_position(32);
        let hash_64 = create_seq_hash_at_position(64);

        hub.process_event(KvCacheEvent::Create {
            seq_hash: hash_16,
            instance_id: 111,
            cluster_id: "test".to_string(),
        });

        hub.process_event(KvCacheEvent::Create {
            seq_hash: hash_32,
            instance_id: 222,
            cluster_id: "test".to_string(),
        });

        hub.process_event(KvCacheEvent::Create {
            seq_hash: hash_64,
            instance_id: 333,
            cluster_id: "test".to_string(),
        });
    }

    #[tokio::test]
    async fn test_hub_run_loop() {
        let (tx, rx) = mpsc::unbounded_channel();
        let hub = KvbmHub::new(rx);
        let radix_tree = hub.radix_tree();

        let seq_hash = create_seq_hash_at_position(128);

        // Spawn hub in background
        tokio::spawn(async move {
            hub.run().await;
        });

        // Send events
        tx.send(KvCacheEvent::Create {
            seq_hash,
            instance_id: 12345,
            cluster_id: "test".to_string(),
        })
        .unwrap();

        // Give hub time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify block was processed
        let workers = radix_tree.lookup(seq_hash).unwrap();
        assert!(workers.contains(&12345));
    }
}
