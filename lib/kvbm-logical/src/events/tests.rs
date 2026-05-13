// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the events pipeline.
//!
//! These tests verify the end-to-end flow from BlockRegistry through
//! EventsManager, EventBatcher, and KvbmCacheEventsPublisher.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use tokio::sync::mpsc;

use super::batcher::BatchingConfig;
use super::manager::EventsManager;
use super::protocol::{KvCacheEvent, KvCacheEvents, KvbmCacheEvents};
use super::publisher::KvbmCacheEventsPublisher;
use crate::pubsub::Publisher;
use crate::registry::BlockRegistry;
use crate::{KvbmSequenceHashProvider, SequenceHash};
use dynamo_tokens::TokenBlockSequence;

fn create_seq_hash_at_position(position: usize) -> SequenceHash {
    let tokens_per_block = 4;
    let total_tokens = (position + 1) * tokens_per_block;
    let tokens: Vec<u32> = (0..total_tokens as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
    seq.blocks()[position].kvbm_sequence_hash()
}

/// Mock publisher that captures published events via channel.
struct MockPublisher {
    captured_tx: mpsc::UnboundedSender<KvbmCacheEvents>,
}

impl MockPublisher {
    fn new(captured_tx: mpsc::UnboundedSender<KvbmCacheEvents>) -> Self {
        Self { captured_tx }
    }
}

impl Publisher for MockPublisher {
    fn publish(&self, _subject: &str, payload: Bytes) -> Result<()> {
        let events: KvbmCacheEvents = rmp_serde::from_slice(&payload)?;
        self.captured_tx.send(events).ok();
        Ok(())
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// Full pipeline test: BlockRegistry -> EventsManager -> Batcher -> Publisher
#[tokio::test]
async fn test_full_event_pipeline() {
    // 1. Setup - AllEventsPolicy is the default
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    // 2. Create mock publisher that captures events
    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    // 3. Build pipeline
    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
        .build()
        .unwrap();

    // 4. Register blocks (triggers Create events)
    let seq_hashes: Vec<_> = (0..5).map(create_seq_hash_at_position).collect();
    let handles: Vec<_> = seq_hashes
        .iter()
        .map(|&hash| {
            let handle = registry.register_sequence_hash(hash);
            manager.on_block_registered(&handle).unwrap();
            handle
        })
        .collect();

    // 5. Wait for batch window
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 6. Verify Create batch received
    let batch = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(batch.events, KvCacheEvents::Create(_)));
    assert_eq!(batch.instance_id, 12345);

    // Verify sorted by position ascending
    if let KvCacheEvents::Create(hashes) = &batch.events {
        assert_eq!(hashes.len(), 5);
        for i in 1..hashes.len() {
            assert!(
                hashes[i - 1].position() <= hashes[i].position(),
                "Create events should be sorted ascending by position"
            );
        }
    }

    // 7. Drop handles (triggers Remove events)
    drop(handles);
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 8. Verify Remove batch received
    let batch = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(batch.events, KvCacheEvents::Remove(_)));

    // Verify sorted by position descending
    if let KvCacheEvents::Remove(hashes) = &batch.events {
        assert_eq!(hashes.len(), 5);
        for i in 1..hashes.len() {
            assert!(
                hashes[i - 1].position() >= hashes[i].position(),
                "Remove events should be sorted descending by position"
            );
        }
    }
}

/// Test that type switches cause immediate flush
#[tokio::test]
async fn test_type_switch_flushes_batch() {
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    // Use long window so we know flushes are due to type switch, not timeout
    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(BatchingConfig::default().with_window(Duration::from_secs(60)))
        .build()
        .unwrap();

    // Register block (Create event)
    let hash1 = create_seq_hash_at_position(10);
    let handle1 = registry.register_sequence_hash(hash1);
    manager.on_block_registered(&handle1).unwrap();

    // Drop block (Remove event) - should flush pending Create first
    drop(handle1);

    // Register another block (Create event) - should flush pending Remove
    let hash2 = create_seq_hash_at_position(20);
    let handle2 = registry.register_sequence_hash(hash2);
    manager.on_block_registered(&handle2).unwrap();

    // Give time for events to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Should receive: Create batch (flushed on type switch to Remove)
    let batch1 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(batch1.events, KvCacheEvents::Create(_)),
        "First batch should be Create"
    );

    // Should receive: Remove batch (flushed on type switch to Create)
    let batch2 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(batch2.events, KvCacheEvents::Remove(_)),
        "Second batch should be Remove"
    );

    drop(handle2);
}

/// Test max batch size triggers flush
#[tokio::test]
async fn test_max_batch_size_flush() {
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(
            BatchingConfig::default()
                .with_window(Duration::from_secs(60)) // Long window
                .with_max_size(NonZeroUsize::new(3).unwrap()),
        )
        .build()
        .unwrap();

    // Register 5 blocks
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let hash = create_seq_hash_at_position(i);
            let handle = registry.register_sequence_hash(hash);
            manager.on_block_registered(&handle).unwrap();
            handle
        })
        .collect();

    // Give time for events to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Should receive first batch with 3 events (max size reached)
    let batch1 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    if let KvCacheEvents::Create(hashes) = &batch1.events {
        assert_eq!(
            hashes.len(),
            3,
            "First batch should have max_size (3) events"
        );
    } else {
        panic!("Expected Create batch");
    }

    // Drop handles to allow remove events to proceed
    drop(handles);
}

/// Test multiple subscribers receive same events
#[tokio::test]
async fn test_multiple_subscribers() {
    let manager = Arc::new(EventsManager::builder().build());

    let mut stream1 = Box::pin(manager.subscribe());
    let mut stream2 = Box::pin(manager.subscribe());

    let registry = BlockRegistry::new();
    let hash = create_seq_hash_at_position(42);
    let handle = registry.register_sequence_hash(hash);
    manager.on_block_registered(&handle).unwrap();

    // Both streams should receive the Create event
    let event1 = tokio::time::timeout(Duration::from_millis(100), stream1.next())
        .await
        .unwrap()
        .unwrap();
    let event2 = tokio::time::timeout(Duration::from_millis(100), stream2.next())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event1, KvCacheEvent::Create(hash));
    assert_eq!(event2, KvCacheEvent::Create(hash));

    // Drop handle to trigger Remove
    drop(handle);

    // Both should receive Remove
    let event1 = tokio::time::timeout(Duration::from_millis(100), stream1.next())
        .await
        .unwrap()
        .unwrap();
    let event2 = tokio::time::timeout(Duration::from_millis(100), stream2.next())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event1, KvCacheEvent::Remove(hash));
    assert_eq!(event2, KvCacheEvent::Remove(hash));
}

/// Test that events are properly serialized with msgpack
#[tokio::test]
async fn test_msgpack_serialization() {
    let hash = create_seq_hash_at_position(10);
    let batch = KvbmCacheEvents {
        events: KvCacheEvents::Create(vec![hash]),
        instance_id: 12345,
    };

    // Serialize with msgpack
    let bytes = rmp_serde::to_vec(&batch).unwrap();

    // Deserialize
    let decoded: KvbmCacheEvents = rmp_serde::from_slice(&bytes).unwrap();

    assert_eq!(decoded.instance_id, 12345);
    assert!(matches!(decoded.events, KvCacheEvents::Create(ref h) if h.len() == 1));
}
