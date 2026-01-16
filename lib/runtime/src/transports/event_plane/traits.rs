// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic event plane traits for transport-agnostic pub/sub.
//!
//! These traits abstract over different event transports (NATS, ZMQ) to provide
//! a unified interface for publishing and subscribing to events.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::pin::Pin;

/// Event envelope containing metadata for sequencing, timing, and gap detection.
///
/// Every event published through the event plane is wrapped in an envelope that
/// includes the publisher's identity, sequence number, and timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    /// The topic/subject the event was published to
    pub topic: String,
    /// Unique identifier of the publisher (typically discovery instance_id)
    pub publisher_id: u64,
    /// Monotonically increasing sequence number per publisher
    pub sequence: u64,
    /// Unix timestamp in milliseconds when the event was published
    pub published_at: u64,
    /// The serialized event payload
    #[serde(with = "bytes_serde")]
    pub payload: Bytes,
}

/// Serde helper for Bytes serialization
mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        Ok(Bytes::from(bytes))
    }
}

/// A stream of event envelopes from a subscription.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<EventEnvelope>> + Send>>;

/// A stream of typed events with their envelopes.
pub type TypedEventStream<T> = Pin<Box<dyn Stream<Item = Result<(EventEnvelope, T)>> + Send>>;

/// Transport-agnostic event publisher trait.
///
/// Implementations of this trait can publish events to various backends
/// (NATS, ZMQ, etc.) while maintaining consistent envelope metadata.
///
/// # Example
///
/// ```ignore
/// let event_plane = EventPlane::for_component(&component);
/// event_plane.publish("kv-events", &my_event).await?;
/// ```
#[async_trait]
pub trait GenericEventPublisher: Send + Sync {
    /// Publish a serializable event to the specified topic.
    ///
    /// The event will be wrapped in an [`EventEnvelope`] with metadata
    /// (publisher_id, sequence, published_at) before being sent.
    ///
    /// # Arguments
    /// * `topic` - The topic/subject to publish to (e.g., "kv-events")
    /// * `event` - The event to publish (must be serializable)
    async fn publish<T: Serialize + Send + Sync>(&self, topic: &str, event: &T) -> Result<()>;

    /// Publish raw bytes to the specified topic.
    ///
    /// The bytes will be wrapped in an [`EventEnvelope`] before being sent.
    /// Use this when you have pre-serialized data.
    async fn publish_bytes(&self, topic: &str, bytes: Vec<u8>) -> Result<()>;

    /// Get the unique identifier of this publisher.
    ///
    /// This ID is included in every [`EventEnvelope`] and is used for
    /// gap detection on the subscriber side.
    fn publisher_id(&self) -> u64;
}

/// Transport-agnostic event subscriber trait.
///
/// Implementations of this trait can subscribe to events from various backends
/// (NATS, ZMQ, etc.) and receive them as a stream of [`EventEnvelope`]s.
///
/// # Example
///
/// ```ignore
/// let event_plane = EventPlane::for_component(&component);
/// let mut stream = event_plane.subscribe("kv-events").await?;
/// while let Some(envelope) = stream.next().await {
///     let envelope = envelope?;
///     println!("Received event from publisher {}", envelope.publisher_id);
/// }
/// ```
#[async_trait]
pub trait GenericEventSubscriber: Send + Sync {
    /// Subscribe to events on the specified topic.
    ///
    /// Returns a stream of [`EventEnvelope`]s. The envelope contains
    /// metadata useful for gap detection and debugging.
    ///
    /// # Arguments
    /// * `topic` - The topic/subject to subscribe to (e.g., "kv-events")
    async fn subscribe(&self, topic: &str) -> Result<EventStream>;

    /// Subscribe to events with automatic deserialization.
    ///
    /// Returns a stream of tuples containing both the envelope and
    /// the deserialized event payload.
    ///
    /// # Arguments
    /// * `topic` - The topic/subject to subscribe to
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize events into
    async fn subscribe_typed<T: DeserializeOwned + Send + 'static>(
        &self,
        topic: &str,
    ) -> Result<TypedEventStream<T>>;
}

/// Helper trait for objects that implement both publishing and subscribing.
pub trait GenericEventPlane: GenericEventPublisher + GenericEventSubscriber {}

impl<T: GenericEventPublisher + GenericEventSubscriber> GenericEventPlane for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_envelope_serialization() {
        let envelope = EventEnvelope {
            topic: "test-topic".to_string(),
            publisher_id: 12345,
            sequence: 1,
            published_at: 1700000000000,
            payload: Bytes::from("test payload"),
        };

        let json = serde_json::to_string(&envelope).unwrap();
        let deserialized: EventEnvelope = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.topic, "test-topic");
        assert_eq!(deserialized.publisher_id, 12345);
        assert_eq!(deserialized.sequence, 1);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.payload, Bytes::from("test payload"));
    }
}
