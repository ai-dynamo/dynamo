// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event publisher for KV cache coordination.
//!
//! The [`KvbmCacheEventsPublisher`] consumes batched events from an [`EventBatcher`]
//! and publishes them to a messaging system via the [`Publisher`] trait.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use tokio::task::JoinHandle;

use super::batcher::{BatchingConfig, EventBatcher};
use super::protocol::{InstanceId, KvCacheEvent, KvbmCacheEvents};
use crate::pubsub::Publisher;

/// Builder for constructing a [`KvbmCacheEventsPublisher`].
///
/// # Example
///
/// ```ignore
/// let publisher = KvbmCacheEventsPublisher::builder()
///     .instance_id(manager.instance_id())
///     .event_stream(manager.subscribe())
///     .publisher(nats_publisher)
///     .subject("kvbm.events")
///     .build()?;
/// ```
pub struct KvbmCacheEventsPublisherBuilder<S, P> {
    instance_id: Option<InstanceId>,
    event_stream: Option<S>,
    publisher: Option<Arc<P>>,
    batching_config: BatchingConfig,
    subject: String,
}

impl<S, P> Default for KvbmCacheEventsPublisherBuilder<S, P> {
    fn default() -> Self {
        Self {
            instance_id: None,
            event_stream: None,
            publisher: None,
            batching_config: BatchingConfig::default(),
            subject: "kvbm.events".to_string(),
        }
    }
}

impl<S, P> KvbmCacheEventsPublisherBuilder<S, P>
where
    S: Stream<Item = KvCacheEvent> + Send + 'static,
    P: Publisher + 'static,
{
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the instance ID for batched events.
    pub fn instance_id(mut self, id: InstanceId) -> Self {
        self.instance_id = Some(id);
        self
    }

    /// Sets the event stream to consume.
    pub fn event_stream(mut self, stream: S) -> Self {
        self.event_stream = Some(stream);
        self
    }

    /// Sets the publisher for sending batched events.
    pub fn publisher(mut self, publisher: Arc<P>) -> Self {
        self.publisher = Some(publisher);
        self
    }

    /// Sets the batching configuration.
    pub fn batching_config(mut self, config: BatchingConfig) -> Self {
        self.batching_config = config;
        self
    }

    /// Sets the subject/topic to publish events to.
    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = subject.into();
        self
    }

    /// Builds and starts the publisher.
    ///
    /// This spawns a background task that consumes events from the stream,
    /// batches them according to the configuration, and publishes them.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> Result<KvbmCacheEventsPublisher> {
        let instance_id = self
            .instance_id
            .ok_or_else(|| anyhow!("instance_id is required"))?;
        let event_stream = self
            .event_stream
            .ok_or_else(|| anyhow!("event_stream is required"))?;
        let publisher = self
            .publisher
            .ok_or_else(|| anyhow!("publisher is required"))?;

        let batcher = EventBatcher::new(self.batching_config, instance_id);
        let batched_stream = batcher.batch(event_stream);
        let subject = self.subject;

        let handle = tokio::spawn(async move {
            Self::run_publish_loop(batched_stream, publisher, subject, instance_id).await;
        });

        Ok(KvbmCacheEventsPublisher {
            handle: Some(handle),
        })
    }

    async fn run_publish_loop(
        batched_stream: impl Stream<Item = KvbmCacheEvents>,
        publisher: Arc<P>,
        subject: String,
        instance_id: InstanceId,
    ) {
        futures::pin_mut!(batched_stream);

        while let Some(batch) = batched_stream.next().await {
            // Serialize using msgpack
            let payload = match rmp_serde::to_vec(&batch) {
                Ok(bytes) => Bytes::from(bytes),
                Err(e) => {
                    tracing::error!("Failed to serialize batch: {}", e);
                    continue;
                }
            };

            if let Err(e) = publisher.publish(&subject, payload) {
                tracing::error!("Failed to publish batch: {}", e);
            }
        }

        // Emit shutdown event
        let shutdown_event = KvbmCacheEvents {
            events: super::protocol::KvCacheEvents::Shutdown,
            instance_id,
        };
        if let Ok(bytes) = rmp_serde::to_vec(&shutdown_event)
            && let Err(e) = publisher.publish(&subject, Bytes::from(bytes))
        {
            tracing::error!("Failed to publish shutdown event: {}", e);
        }

        // Flush on stream end
        if let Err(e) = publisher.flush().await {
            tracing::error!("Failed to flush publisher: {}", e);
        }
    }
}

/// Publisher that consumes batched events and publishes them to a messaging system.
///
/// The publisher runs as a background task and can be stopped by calling
/// [`abort()`](Self::abort) or by dropping it.
pub struct KvbmCacheEventsPublisher {
    handle: Option<JoinHandle<()>>,
}

impl KvbmCacheEventsPublisher {
    /// Creates a new builder for constructing a publisher.
    pub fn builder<S, P>() -> KvbmCacheEventsPublisherBuilder<S, P>
    where
        S: Stream<Item = KvCacheEvent> + Send + 'static,
        P: Publisher + 'static,
    {
        KvbmCacheEventsPublisherBuilder::new()
    }

    /// Waits for the publisher task to complete.
    ///
    /// The task completes when the input event stream ends.
    pub async fn join(mut self) -> Result<(), tokio::task::JoinError> {
        if let Some(handle) = self.handle.take() {
            handle.await
        } else {
            Ok(())
        }
    }

    /// Aborts the publisher task.
    pub fn abort(&self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }

    /// Checks if the publisher task is finished.
    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }
}

impl Drop for KvbmCacheEventsPublisher {
    fn drop(&mut self) {
        // Abort the task on drop to clean up
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvbmSequenceHashProvider;
    use crate::pubsub::Publisher;
    use dynamo_tokens::TokenBlockSequence;
    use futures::future::BoxFuture;
    use std::sync::Mutex;
    use std::time::Duration;

    fn create_seq_hash_at_position(position: usize) -> crate::SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    /// Mock publisher that captures published events.
    struct MockPublisher {
        captured: Mutex<Vec<KvbmCacheEvents>>,
    }

    impl MockPublisher {
        fn new() -> Self {
            Self {
                captured: Mutex::new(Vec::new()),
            }
        }

        fn take_captured(&self) -> Vec<KvbmCacheEvents> {
            std::mem::take(&mut *self.captured.lock().unwrap())
        }
    }

    impl Publisher for MockPublisher {
        fn publish(&self, _subject: &str, payload: Bytes) -> Result<()> {
            let events: KvbmCacheEvents = rmp_serde::from_slice(&payload)?;
            self.captured.lock().unwrap().push(events);
            Ok(())
        }

        fn flush(&self) -> BoxFuture<'static, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn test_publisher_publishes_batched_events() {
        let mock_publisher = Arc::new(MockPublisher::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let event_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let config = BatchingConfig::default().with_window(Duration::from_millis(50));

        let publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(12345)
            .event_stream(event_stream)
            .publisher(mock_publisher.clone())
            .batching_config(config)
            .subject("test.events")
            .build()
            .unwrap();

        // Send some events
        tx.send(KvCacheEvent::Create(create_seq_hash_at_position(10)))
            .await
            .unwrap();
        tx.send(KvCacheEvent::Create(create_seq_hash_at_position(20)))
            .await
            .unwrap();

        // Close the channel to signal end
        drop(tx);

        // Wait for publisher to finish
        publisher.join().await.unwrap();

        let captured = mock_publisher.take_captured();
        // Should have 1 batch + 1 shutdown event
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].instance_id, 12345);

        if let super::super::protocol::KvCacheEvents::Create(hashes) = &captured[0].events {
            assert_eq!(hashes.len(), 2);
        } else {
            panic!("Expected Create events");
        }

        // Verify shutdown event
        assert_eq!(captured[1].instance_id, 12345);
        assert!(
            matches!(
                captured[1].events,
                super::super::protocol::KvCacheEvents::Shutdown
            ),
            "Expected Shutdown event, got {:?}",
            captured[1].events
        );
    }

    #[tokio::test]
    async fn test_publisher_builder_validation() {
        let (_, rx) = tokio::sync::mpsc::channel::<KvCacheEvent>(10);
        let event_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let mock_publisher = Arc::new(MockPublisher::new());

        // Missing instance_id
        let result = KvbmCacheEventsPublisherBuilder::new()
            .event_stream(event_stream)
            .publisher(mock_publisher)
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_publisher_abort() {
        let mock_publisher = Arc::new(MockPublisher::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let event_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let publisher = KvbmCacheEventsPublisher::builder()
            .instance_id(12345)
            .event_stream(event_stream)
            .publisher(mock_publisher)
            .build()
            .unwrap();

        // Abort before sending anything
        publisher.abort();

        // Should complete quickly
        let result = tokio::time::timeout(Duration::from_millis(100), publisher.join()).await;
        assert!(result.is_ok());

        // Clean up
        drop(tx);
    }
}
