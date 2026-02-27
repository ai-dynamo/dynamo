// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory stub implementation of the PubSub traits for testing.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use parking_lot::RwLock;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use super::{Message, Publisher, Subscriber, Subscription};

/// Shared state for stub publisher/subscriber pairs.
#[derive(Clone)]
pub struct StubBus {
    inner: Arc<StubBusInner>,
}

struct StubBusInner {
    /// Map of subject patterns to broadcast channels.
    channels: RwLock<HashMap<String, broadcast::Sender<Message>>>,
    /// Channel capacity for new subscriptions.
    capacity: usize,
}

impl Default for StubBus {
    fn default() -> Self {
        Self::new(256)
    }
}

impl StubBus {
    /// Create a new stub bus with the specified channel capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(StubBusInner {
                channels: RwLock::new(HashMap::new()),
                capacity,
            }),
        }
    }

    /// Create a publisher for this bus.
    pub fn publisher(&self) -> StubPublisher {
        StubPublisher { bus: self.clone() }
    }

    /// Create a subscriber for this bus.
    pub fn subscriber(&self) -> StubSubscriber {
        StubSubscriber { bus: self.clone() }
    }

    fn get_or_create_channel(&self, subject: &str) -> broadcast::Sender<Message> {
        let channels = self.inner.channels.read();
        if let Some(tx) = channels.get(subject) {
            return tx.clone();
        }
        drop(channels);

        let mut channels = self.inner.channels.write();
        // Double-check after acquiring write lock
        if let Some(tx) = channels.get(subject) {
            return tx.clone();
        }

        let (tx, _) = broadcast::channel(self.inner.capacity);
        channels.insert(subject.to_string(), tx.clone());
        tx
    }
}

/// Stub implementation of the [`Publisher`] trait for testing.
pub struct StubPublisher {
    bus: StubBus,
}

impl StubPublisher {
    /// Create a new stub publisher with a dedicated bus.
    pub fn new() -> (Self, StubSubscriber) {
        let bus = StubBus::default();
        (bus.publisher(), bus.subscriber())
    }
}

impl Publisher for StubPublisher {
    fn publish(&self, subject: &str, payload: Bytes) -> Result<()> {
        let tx = self.bus.get_or_create_channel(subject);
        let msg = Message {
            subject: subject.to_string(),
            payload,
        };
        // Ignore send errors (no receivers is ok)
        let _ = tx.send(msg);
        Ok(())
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        // In-memory delivery is synchronous, nothing to flush
        async { Ok(()) }.boxed()
    }
}

/// Stub implementation of the [`Subscriber`] trait for testing.
pub struct StubSubscriber {
    bus: StubBus,
}

impl Subscriber for StubSubscriber {
    fn subscribe(&self, subject: &str) -> BoxFuture<'static, Result<Subscription>> {
        let tx = self.bus.get_or_create_channel(subject);
        let rx = tx.subscribe();

        let stream: BoxStream<'static, Message> = BroadcastStream::new(rx)
            .filter_map(|result| async move { result.ok() })
            .boxed();

        async move { Ok(stream) }.boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_stub_pubsub() {
        let bus = StubBus::default();
        let publisher = bus.publisher();
        let subscriber = bus.subscriber();

        // Subscribe first
        let mut sub = subscriber.subscribe("test.subject").await.unwrap();

        // Publish a message
        publisher
            .publish("test.subject", Bytes::from("hello"))
            .unwrap();

        // Receive the message
        let msg = sub.next().await.unwrap();
        assert_eq!(msg.subject, "test.subject");
        assert_eq!(msg.payload.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn test_stub_multiple_subscribers() {
        let bus = StubBus::default();
        let publisher = bus.publisher();

        let mut sub1 = bus.subscriber().subscribe("multi").await.unwrap();
        let mut sub2 = bus.subscriber().subscribe("multi").await.unwrap();

        publisher
            .publish("multi", Bytes::from("broadcast"))
            .unwrap();

        let msg1 = sub1.next().await.unwrap();
        let msg2 = sub2.next().await.unwrap();

        assert_eq!(msg1.payload.as_ref(), b"broadcast");
        assert_eq!(msg2.payload.as_ref(), b"broadcast");
    }
}
