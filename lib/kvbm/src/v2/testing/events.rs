// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end testing utilities for the events pipeline.
//!
//! This module provides test infrastructure for verifying the complete event flow
//! from BlockManager registration through EventsManager, batching, and publishing.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use derive_builder::Builder;
use futures::StreamExt;

use crate::logical::blocks::BlockMetadata;
use crate::logical::manager::BlockManager;
use crate::v2::InstanceId;
use crate::v2::distributed::pubsub::{StubBus, Subscriber, Subscription};
use crate::v2::logical::events::{
    BatchingConfig, EventsManager, KvbmCacheEvents, KvbmCacheEventsPublisher,
};
use crate::v2::testing::managers::TestManagerBuilder;

// =============================================================================
// Events Pipeline Fixture
// =============================================================================

/// Configuration for creating an EventsPipelineFixture.
///
/// Uses `derive_builder` with a custom async build function since fixture
/// construction requires async operations (subscription setup).
///
/// # Example
///
/// ```ignore
/// // BEFORE: 15 lines of setup
/// let events_manager = Arc::new(EventsManager::builder().build());
/// let bus = StubBus::default();
/// let publisher = Arc::new(bus.publisher());
/// let subscriber = bus.subscriber();
/// let mut subscription = subscriber.subscribe("kvbm.events").await?;
/// let _publisher = KvbmCacheEventsPublisher::builder()
///     .instance_id(12345)
///     .event_stream(events_manager.subscribe())
///     .publisher(publisher)
///     .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
///     .subject("kvbm.events")
///     .build()?;
///
/// // AFTER: 3 lines
/// let mut fixture = EventsPipelineFixture::builder()
///     .batching_window(Duration::from_millis(50))
///     .build_async().await?;
/// let manager = fixture.create_manager::<G1>(100, 4);
/// ```
#[derive(Builder)]
#[builder(setter(into, strip_option), build_fn(skip), pattern = "owned")]
#[allow(dead_code)] // Fields are read via builder pattern
pub struct EventsPipelineConfig {
    /// Instance ID for events (default: random v4 UUID).
    #[builder(default)]
    instance_id: Option<InstanceId>,

    /// Batching window duration (default: 50ms).
    #[builder(default = "Duration::from_millis(50)")]
    batching_window: Duration,

    /// Subject for publishing events (default: "kvbm.events").
    #[builder(default = "\"kvbm.events\".to_string()")]
    subject: String,
}

impl EventsPipelineConfigBuilder {
    /// Builds the fixture asynchronously.
    ///
    /// This is a custom build function because fixture construction requires
    /// async operations (setting up the subscription).
    pub async fn build_async(self) -> Result<EventsPipelineFixture> {
        let instance_id = self
            .instance_id
            .flatten()
            .unwrap_or_else(InstanceId::new_v4);
        let batching_window = self.batching_window.unwrap_or(Duration::from_millis(50));
        let subject = self.subject.unwrap_or_else(|| "kvbm.events".to_string());

        // Create EventsManager
        let events_manager = Arc::new(EventsManager::builder().build());

        // Create stub pubsub
        let bus = StubBus::default();
        let publisher_arc = Arc::new(bus.publisher());
        let subscriber = bus.subscriber();

        // Subscribe BEFORE publishing (stub doesn't buffer)
        let subscription = subscriber.subscribe(&subject).await?;

        // Build the publishing pipeline - convert InstanceId to u128
        let publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(instance_id.as_u128())
            .event_stream(events_manager.subscribe())
            .publisher(publisher_arc)
            .batching_config(BatchingConfig::default().with_window(batching_window))
            .subject(&subject)
            .build()?;

        Ok(EventsPipelineFixture {
            events_manager,
            subscription,
            publisher,
            bus,
            instance_id,
            subject,
        })
    }
}

/// Test fixture that encapsulates the full events pipeline setup.
///
/// This reduces the ~15 lines of boilerplate for setting up:
/// - EventsManager
/// - StubBus (publisher + subscriber)
/// - Subscription
/// - KvbmCacheEventsPublisher with batching
pub struct EventsPipelineFixture {
    /// The EventsManager instance.
    pub events_manager: Arc<EventsManager>,
    /// Subscription for receiving published events.
    pub subscription: Subscription,
    /// The publisher (held to keep it alive).
    #[allow(dead_code)]
    publisher: KvbmCacheEventsPublisher,
    /// The stub bus (held for reference).
    #[allow(dead_code)]
    bus: StubBus,
    /// Instance ID used for events.
    pub instance_id: InstanceId,
    /// Subject for events.
    pub subject: String,
}

impl EventsPipelineFixture {
    /// Creates a new builder for the fixture.
    ///
    /// Use `.build_async().await?` to construct the fixture.
    pub fn builder() -> EventsPipelineConfigBuilder {
        EventsPipelineConfigBuilder::default()
    }

    /// Creates a BlockManager with events integration.
    ///
    /// # Arguments
    /// * `block_count` - Number of blocks in the manager
    /// * `block_size` - Tokens per block
    pub fn create_manager<M: BlockMetadata>(
        &self,
        block_count: usize,
        block_size: usize,
    ) -> BlockManager<M> {
        TestManagerBuilder::<M>::new()
            .block_count(block_count)
            .block_size(block_size)
            .events_manager(self.events_manager.clone())
            .build()
    }

    /// Receive a batch of events with timeout.
    ///
    /// Returns `None` if timeout expires before receiving events.
    pub async fn receive_batch(&mut self, timeout: Duration) -> Option<KvbmCacheEvents> {
        match tokio::time::timeout(timeout, self.subscription.next()).await {
            Ok(Some(msg)) => rmp_serde::from_slice(&msg.payload).ok(),
            _ => None,
        }
    }

    /// Receive a batch with default timeout (500ms).
    pub async fn receive_batch_default(&mut self) -> Option<KvbmCacheEvents> {
        self.receive_batch(Duration::from_millis(500)).await
    }

    /// Wait for the batching window to flush, then receive events.
    ///
    /// This sleeps for a bit longer than the batching window to ensure
    /// events are flushed, then attempts to receive.
    pub async fn flush_and_receive(
        &mut self,
        batching_window: Duration,
    ) -> Option<KvbmCacheEvents> {
        tokio::time::sleep(batching_window + Duration::from_millis(50)).await;
        self.receive_batch(Duration::from_millis(500)).await
    }

    /// Returns a reference to the EventsManager.
    pub fn events_manager(&self) -> &Arc<EventsManager> {
        &self.events_manager
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;

    use crate::v2::G1;
    use crate::v2::distributed::pubsub::{StubBus, Subscriber};
    use crate::v2::logical::events::{
        BatchingConfig, EventsManager, KvCacheEvents, KvbmCacheEvents, KvbmCacheEventsPublisher,
        PowerOfTwoPolicy,
    };
    use crate::v2::testing::managers::TestManagerBuilder;
    use crate::v2::testing::token_blocks;

    /// Full end-to-end test: G1 BlockManager -> EventsManager -> Batcher -> Publisher -> Subscriber
    ///
    /// This test verifies the complete event pipeline:
    /// 1. Token sequences are created and registered with the BlockManager
    /// 2. EventsManager emits Create events via BlockRegistry integration
    /// 3. EventBatcher batches and sorts events
    /// 4. KvbmCacheEventsPublisher serializes and publishes via stub
    /// 5. StubSubscriber receives the batched events
    #[tokio::test]
    async fn test_full_events_pipeline_with_block_manager() {
        // 1. Create EventsManager (default AllEventsPolicy)
        let events_manager = Arc::new(EventsManager::builder().build());

        // 2. Create G1 BlockManager with EventsManager integrated via builder
        let block_count = 100;
        let block_size = 4;
        let manager = TestManagerBuilder::<G1>::new()
            .block_count(block_count)
            .block_size(block_size)
            .events_manager(events_manager.clone())
            .build();

        // 4. Create stub pubsub for testing
        let bus = StubBus::default();
        let publisher = Arc::new(bus.publisher());
        let subscriber = bus.subscriber();

        // 5. Subscribe BEFORE publishing (stub doesn't buffer)
        let mut subscription = subscriber
            .subscribe("kvbm.events")
            .await
            .expect("Should subscribe");

        // 6. Build the publishing pipeline
        let _events_publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(12345)
            .event_stream(events_manager.subscribe())
            .publisher(publisher)
            .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
            .subject("kvbm.events")
            .build()
            .expect("Should build publisher");

        // 7. Create token sequence and register blocks via BlockManager
        // This automatically triggers events through the registry -> events_manager chain
        let num_blocks = 5;
        let token_sequence = token_blocks::create_token_sequence(num_blocks, block_size, 0);

        // Allocate, complete, and register blocks
        let allocated_blocks = manager
            .allocate_blocks(num_blocks)
            .expect("Should allocate blocks");

        // Complete blocks with token data
        let complete_blocks: Vec<_> = allocated_blocks
            .into_iter()
            .zip(token_sequence.blocks())
            .map(|(block, token_block)| {
                block
                    .complete(token_block.clone())
                    .expect("Should complete")
            })
            .collect();

        // Register blocks - this triggers Create events via the integrated EventsManager
        let _immutable_blocks = manager.register_blocks(complete_blocks);

        // 8. Wait for batch window to flush
        tokio::time::sleep(Duration::from_millis(100)).await;

        // 9. Receive and verify the batched events
        let msg = tokio::time::timeout(Duration::from_millis(500), subscription.next())
            .await
            .expect("Should receive within timeout")
            .expect("Should have message");

        // Deserialize the msgpack payload
        let batch: KvbmCacheEvents =
            rmp_serde::from_slice(&msg.payload).expect("Should deserialize");

        assert_eq!(batch.instance_id, 12345);

        match &batch.events {
            KvCacheEvents::Create(hashes) => {
                assert_eq!(hashes.len(), num_blocks);

                // Verify sorted ascending by position
                for i in 1..hashes.len() {
                    assert!(
                        hashes[i - 1].position() <= hashes[i].position(),
                        "Create events should be sorted ascending"
                    );
                }
            }
            KvCacheEvents::Remove(_) => panic!("Expected Create events, got Remove"),
            KvCacheEvents::Shutdown => panic!("Expected Create events, got Shutdown"),
        }
    }

    /// Test events with PowerOfTwoPolicy filtering.
    ///
    /// Only blocks at power-of-2 positions should emit events.
    #[tokio::test]
    async fn test_events_with_power_of_two_policy() {
        // Use PowerOfTwoPolicy instead of default AllEventsPolicy
        let events_manager = Arc::new(
            EventsManager::builder()
                .policy(Arc::new(PowerOfTwoPolicy::new()))
                .build(),
        );

        let block_size = 4;
        let manager = TestManagerBuilder::<G1>::new()
            .block_count(100)
            .block_size(block_size)
            .events_manager(events_manager.clone())
            .build();

        let bus = StubBus::default();
        let publisher = Arc::new(bus.publisher());
        let subscriber = bus.subscriber();

        let mut subscription = subscriber
            .subscribe("kvbm.events")
            .await
            .expect("Should subscribe");

        let _events_publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(12345)
            .event_stream(events_manager.subscribe())
            .publisher(publisher)
            .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
            .subject("kvbm.events")
            .build()
            .expect("Should build publisher");

        // Create sequence with 32 blocks (positions 0-31)
        // Power of 2 positions: 1, 2, 4, 8, 16
        let num_blocks = 32;
        let token_sequence = token_blocks::create_token_sequence(num_blocks, block_size, 0);

        let allocated_blocks = manager
            .allocate_blocks(num_blocks)
            .expect("Should allocate blocks");

        let complete_blocks: Vec<_> = allocated_blocks
            .into_iter()
            .zip(token_sequence.blocks())
            .map(|(block, token_block)| {
                block
                    .complete(token_block.clone())
                    .expect("Should complete")
            })
            .collect();

        let _immutable_blocks = manager.register_blocks(complete_blocks);

        // Wait for batch window
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should receive only power-of-2 position events
        let msg = tokio::time::timeout(Duration::from_millis(500), subscription.next())
            .await
            .expect("Should receive within timeout")
            .expect("Should have message");

        let batch: KvbmCacheEvents =
            rmp_serde::from_slice(&msg.payload).expect("Should deserialize");

        match &batch.events {
            KvCacheEvents::Create(hashes) => {
                // Verify all received are at power-of-2 positions
                for hash in hashes {
                    let pos = hash.position();
                    assert!(
                        pos.is_power_of_two(),
                        "Position {} should be power of 2",
                        pos
                    );
                }
            }
            KvCacheEvents::Remove(_) => panic!("Expected Create events"),
            KvCacheEvents::Shutdown => panic!("Expected Create events"),
        }
    }

    /// Test that Remove events are emitted when blocks are evicted from the pool.
    ///
    /// Note: Dropping ImmutableBlocks returns them to the pool (not dropped).
    /// Remove events only fire when blocks are actually evicted from the pool
    /// due to capacity limits (LRU eviction).
    #[tokio::test]
    async fn test_remove_events_on_pool_eviction() {
        let events_manager = Arc::new(EventsManager::builder().build());

        // Small pool capacity to force eviction
        let block_count = 10;
        let block_size = 4;
        let manager = TestManagerBuilder::<G1>::new()
            .block_count(block_count)
            .block_size(block_size)
            .events_manager(events_manager.clone())
            .build();

        let bus = StubBus::default();
        let publisher = Arc::new(bus.publisher());
        let subscriber = bus.subscriber();

        let mut subscription = subscriber
            .subscribe("kvbm.events")
            .await
            .expect("Should subscribe");

        let _events_publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(12345)
            .event_stream(events_manager.subscribe())
            .publisher(publisher)
            .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
            .subject("kvbm.events")
            .build()
            .expect("Should build publisher");

        // Fill the pool completely with first batch of blocks
        let first_batch_size = block_count;
        let token_sequence1 = token_blocks::create_token_sequence(first_batch_size, block_size, 0);

        let allocated_blocks = manager
            .allocate_blocks(first_batch_size)
            .expect("Should allocate blocks");

        let complete_blocks: Vec<_> = allocated_blocks
            .into_iter()
            .zip(token_sequence1.blocks())
            .map(|(block, token_block)| {
                block
                    .complete(token_block.clone())
                    .expect("Should complete")
            })
            .collect();

        // Register and immediately drop handles to return blocks to pool
        let _first_batch = manager.register_blocks(complete_blocks);

        // Wait for Create batch
        tokio::time::sleep(Duration::from_millis(100)).await;

        let msg = tokio::time::timeout(Duration::from_millis(500), subscription.next())
            .await
            .expect("Should receive Create batch")
            .expect("Should have message");
        let batch: KvbmCacheEvents = rmp_serde::from_slice(&msg.payload).unwrap();
        assert!(
            matches!(batch.events, KvCacheEvents::Create(ref h) if h.len() == first_batch_size)
        );

        // Drop first batch to return blocks to pool
        drop(_first_batch);

        // Now allocate more blocks - this should trigger eviction of old blocks
        // since we're reusing the same pool slots with new sequence hashes
        let second_batch_size = block_count;
        let token_sequence2 =
            token_blocks::create_token_sequence(second_batch_size, block_size, 1000);

        let allocated_blocks = manager
            .allocate_blocks(second_batch_size)
            .expect("Should allocate blocks for second batch");

        let complete_blocks: Vec<_> = allocated_blocks
            .into_iter()
            .zip(token_sequence2.blocks())
            .map(|(block, token_block)| {
                block
                    .complete(token_block.clone())
                    .expect("Should complete")
            })
            .collect();

        let _second_batch = manager.register_blocks(complete_blocks);

        // Wait for events
        tokio::time::sleep(Duration::from_millis(100)).await;

        // We should receive Remove events for evicted blocks and Create events for new blocks.
        // The order depends on the event batching (type switches cause flush).
        // Collect all events received
        let mut received_creates = 0;
        let mut received_removes = 0;

        // Try to receive messages with timeout
        while let Ok(Some(msg)) =
            tokio::time::timeout(Duration::from_millis(200), subscription.next()).await
        {
            let batch: KvbmCacheEvents = rmp_serde::from_slice(&msg.payload).unwrap();
            match batch.events {
                KvCacheEvents::Create(hashes) => received_creates += hashes.len(),
                KvCacheEvents::Remove(hashes) => received_removes += hashes.len(),
                KvCacheEvents::Shutdown => {} // Ignore shutdown events in counting
            }
        }

        // Should have created second batch
        assert_eq!(
            received_creates, second_batch_size,
            "Should receive Create events for second batch"
        );

        // Should have removed first batch (evicted due to pool reuse)
        assert_eq!(
            received_removes, first_batch_size,
            "Should receive Remove events for evicted blocks"
        );
    }
}
