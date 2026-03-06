// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event batching for efficient publishing.
//!
//! The [`EventBatcher`] transforms a stream of individual [`KvCacheEvent`]s into
//! batched [`KvbmCacheEvents`] for efficient wire transmission.

use std::num::NonZeroUsize;
use std::time::Duration;

use async_stream::stream;
use futures::Stream;
use futures::StreamExt;
use tokio::pin;

use super::protocol::{InstanceId, KvCacheEvent, KvCacheEvents, KvbmCacheEvents};
use crate::SequenceHash;

/// Configuration for event batching.
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Maximum time to wait before flushing a batch. Default: 10ms.
    pub window_duration: Duration,
    /// Maximum number of events in a batch before flushing. Default: 1024.
    pub max_batch_size: NonZeroUsize,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_millis(10),
            max_batch_size: NonZeroUsize::new(1024).unwrap(),
        }
    }
}

impl BatchingConfig {
    /// Sets the window duration.
    pub fn with_window(mut self, duration: Duration) -> Self {
        self.window_duration = duration;
        self
    }

    /// Sets the maximum batch size.
    pub fn with_max_size(mut self, size: NonZeroUsize) -> Self {
        self.max_batch_size = size;
        self
    }
}

/// Tracks which type of events we're currently batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchType {
    Create,
    Remove,
}

impl BatchType {
    fn from_event(event: &KvCacheEvent) -> Self {
        match event {
            KvCacheEvent::Create(_) => BatchType::Create,
            KvCacheEvent::Remove(_) => BatchType::Remove,
        }
    }
}

/// Event batcher that transforms individual events into batched wire format.
///
/// The batcher collects contiguous same-type events and flushes them as batches when:
/// - The event type switches (Create -> Remove or vice versa)
/// - The maximum batch size is reached
/// - The window duration expires
///
/// Batches are sorted for optimal radix tree operations:
/// - Create events: sorted by position ascending (low to high)
/// - Remove events: sorted by position descending (high to low)
pub struct EventBatcher {
    config: BatchingConfig,
    instance_id: InstanceId,
}

impl EventBatcher {
    /// Creates a new EventBatcher.
    ///
    /// # Arguments
    /// * `config` - Batching configuration
    /// * `instance_id` - Worker instance ID to include in batched events
    pub fn new(config: BatchingConfig, instance_id: InstanceId) -> Self {
        Self {
            config,
            instance_id,
        }
    }

    /// Transform an input stream of events into a batched output stream.
    ///
    /// # Arguments
    /// * `input` - Stream of individual KvCacheEvents
    ///
    /// # Returns
    /// A stream of batched KvbmCacheEvents ready for wire transmission.
    pub fn batch<S>(self, input: S) -> impl Stream<Item = KvbmCacheEvents> + Send
    where
        S: Stream<Item = KvCacheEvent> + Send + 'static,
    {
        let config = self.config;
        let instance_id = self.instance_id;

        stream! {
            pin!(input);

            let mut current_batch: Vec<SequenceHash> = Vec::with_capacity(config.max_batch_size.get());
            let mut current_type: Option<BatchType> = None;
            let mut deadline = tokio::time::Instant::now() + config.window_duration;

            loop {
                let timeout = tokio::time::sleep_until(deadline);

                tokio::select! {
                    biased;

                    maybe_event = input.next() => {
                        match maybe_event {
                            Some(event) => {
                                let event_type = BatchType::from_event(&event);
                                let seq_hash = match &event {
                                    KvCacheEvent::Create(h) | KvCacheEvent::Remove(h) => *h,
                                };

                                // Check if we need to flush due to type switch
                                if let Some(current) = current_type
                                    && current != event_type && !current_batch.is_empty() {
                                        // Flush current batch before switching types
                                        let batch = Self::make_batch(
                                            &mut current_batch,
                                            current,
                                            instance_id,
                                        );
                                        yield batch;
                                        deadline = tokio::time::Instant::now() + config.window_duration;
                                    }

                                current_type = Some(event_type);
                                current_batch.push(seq_hash);

                                // Check if we need to flush due to size
                                if current_batch.len() >= config.max_batch_size.get() {
                                    let batch = Self::make_batch(
                                        &mut current_batch,
                                        event_type,
                                        instance_id,
                                    );
                                    yield batch;
                                    current_type = None;
                                    deadline = tokio::time::Instant::now() + config.window_duration;
                                }
                            }
                            None => {
                                // Input stream ended, flush remaining
                                if let Some(batch_type) = current_type
                                    && !current_batch.is_empty() {
                                        let batch = Self::make_batch(
                                            &mut current_batch,
                                            batch_type,
                                            instance_id,
                                        );
                                        yield batch;
                                    }
                                break;
                            }
                        }
                    }

                    _ = timeout => {
                        // Timer expired, flush if we have anything
                        if let Some(batch_type) = current_type
                            && !current_batch.is_empty() {
                                let batch = Self::make_batch(
                                    &mut current_batch,
                                    batch_type,
                                    instance_id,
                                );
                                yield batch;
                                current_type = None;
                            }
                        deadline = tokio::time::Instant::now() + config.window_duration;
                    }
                }
            }
        }
    }

    fn make_batch(
        hashes: &mut Vec<SequenceHash>,
        batch_type: BatchType,
        instance_id: InstanceId,
    ) -> KvbmCacheEvents {
        // Sort based on batch type
        match batch_type {
            BatchType::Create => {
                // Sort ascending by position for efficient radix tree insertion
                hashes.sort_by_key(|h| h.position());
            }
            BatchType::Remove => {
                // Sort descending by position for efficient radix tree removal
                hashes.sort_by_key(|h| std::cmp::Reverse(h.position()));
            }
        }

        let sorted_hashes = std::mem::take(hashes);

        let events = match batch_type {
            BatchType::Create => KvCacheEvents::Create(sorted_hashes),
            BatchType::Remove => KvCacheEvents::Remove(sorted_hashes),
        };

        KvbmCacheEvents {
            events,
            instance_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvbmSequenceHashProvider;
    use dynamo_tokens::TokenBlockSequence;
    use futures::stream;

    fn create_seq_hash_at_position(position: usize) -> SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    #[tokio::test]
    async fn test_batcher_batches_creates() {
        let config = BatchingConfig::default().with_window(Duration::from_millis(50));
        let batcher = EventBatcher::new(config, 12345);

        let events = vec![
            KvCacheEvent::Create(create_seq_hash_at_position(10)),
            KvCacheEvent::Create(create_seq_hash_at_position(5)),
            KvCacheEvent::Create(create_seq_hash_at_position(15)),
        ];

        let input = stream::iter(events);
        let mut output = Box::pin(batcher.batch(input));

        let batch = output.next().await.unwrap();
        assert!(matches!(batch.events, KvCacheEvents::Create(_)));
        assert_eq!(batch.instance_id, 12345);

        // Verify sorted ascending by position
        if let KvCacheEvents::Create(hashes) = &batch.events {
            assert_eq!(hashes.len(), 3);
            for i in 1..hashes.len() {
                assert!(hashes[i - 1].position() <= hashes[i].position());
            }
        }
    }

    #[tokio::test]
    async fn test_batcher_batches_removes() {
        let config = BatchingConfig::default().with_window(Duration::from_millis(50));
        let batcher = EventBatcher::new(config, 12345);

        let events = vec![
            KvCacheEvent::Remove(create_seq_hash_at_position(10)),
            KvCacheEvent::Remove(create_seq_hash_at_position(5)),
            KvCacheEvent::Remove(create_seq_hash_at_position(15)),
        ];

        let input = stream::iter(events);
        let mut output = Box::pin(batcher.batch(input));

        let batch = output.next().await.unwrap();
        assert!(matches!(batch.events, KvCacheEvents::Remove(_)));

        // Verify sorted descending by position
        if let KvCacheEvents::Remove(hashes) = &batch.events {
            assert_eq!(hashes.len(), 3);
            for i in 1..hashes.len() {
                assert!(hashes[i - 1].position() >= hashes[i].position());
            }
        }
    }

    #[tokio::test]
    async fn test_batcher_flushes_on_type_switch() {
        let config = BatchingConfig::default().with_window(Duration::from_secs(60)); // Long window
        let batcher = EventBatcher::new(config, 12345);

        let events = vec![
            KvCacheEvent::Create(create_seq_hash_at_position(10)),
            KvCacheEvent::Create(create_seq_hash_at_position(11)),
            KvCacheEvent::Remove(create_seq_hash_at_position(5)), // Type switch!
            KvCacheEvent::Create(create_seq_hash_at_position(12)),
        ];

        let input = stream::iter(events);
        let mut output = Box::pin(batcher.batch(input));

        // First batch should be creates (flushed due to type switch)
        let batch1 = output.next().await.unwrap();
        assert!(matches!(batch1.events, KvCacheEvents::Create(ref h) if h.len() == 2));

        // Second batch should be removes (flushed due to type switch)
        let batch2 = output.next().await.unwrap();
        assert!(matches!(batch2.events, KvCacheEvents::Remove(ref h) if h.len() == 1));

        // Third batch should be the remaining create (flushed on stream end)
        let batch3 = output.next().await.unwrap();
        assert!(matches!(batch3.events, KvCacheEvents::Create(ref h) if h.len() == 1));
    }

    #[tokio::test]
    async fn test_batcher_flushes_on_max_size() {
        let config = BatchingConfig::default()
            .with_window(Duration::from_secs(60)) // Long window
            .with_max_size(NonZeroUsize::new(3).unwrap());
        let batcher = EventBatcher::new(config, 12345);

        let events = vec![
            KvCacheEvent::Create(create_seq_hash_at_position(1)),
            KvCacheEvent::Create(create_seq_hash_at_position(2)),
            KvCacheEvent::Create(create_seq_hash_at_position(3)),
            KvCacheEvent::Create(create_seq_hash_at_position(4)),
            KvCacheEvent::Create(create_seq_hash_at_position(5)),
        ];

        let input = stream::iter(events);
        let mut output = Box::pin(batcher.batch(input));

        // First batch should have 3 (max size)
        let batch1 = output.next().await.unwrap();
        assert!(matches!(batch1.events, KvCacheEvents::Create(ref h) if h.len() == 3));

        // Second batch should have remaining 2 (flushed on stream end)
        let batch2 = output.next().await.unwrap();
        assert!(matches!(batch2.events, KvCacheEvents::Create(ref h) if h.len() == 2));
    }

    #[tokio::test]
    async fn test_batcher_flushes_on_timeout() {
        let config = BatchingConfig::default().with_window(Duration::from_millis(50));
        let batcher = EventBatcher::new(config, 12345);

        // Create a channel-based stream so we can control timing
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let input = tokio_stream::wrappers::ReceiverStream::new(rx);
        let mut output = Box::pin(batcher.batch(input));

        // Send one event
        tx.send(KvCacheEvent::Create(create_seq_hash_at_position(10)))
            .await
            .unwrap();

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should receive batch due to timeout
        let batch = tokio::time::timeout(Duration::from_millis(100), output.next())
            .await
            .unwrap()
            .unwrap();

        assert!(matches!(batch.events, KvCacheEvents::Create(ref h) if h.len() == 1));

        // Clean up
        drop(tx);
    }
}
