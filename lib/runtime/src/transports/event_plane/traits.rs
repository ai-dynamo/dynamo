// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event plane types for transport-agnostic pub/sub.

use anyhow::Result;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventEnvelope {
    /// Unique identifier of the publisher (typically discovery instance_id)
    pub publisher_id: u64,
    /// Monotonically increasing sequence number per publisher
    pub sequence: u64,
    /// Unix timestamp in milliseconds when the event was published
    pub published_at: u64,
    /// The topic this event was published to
    pub topic: String,
    /// The serialized event payload
    #[serde(with = "bytes_serde")]
    pub payload: Bytes,
}

/// Serde helper for Bytes serialization with MessagePack
mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_envelope_msgpack_serialization() {
        let envelope = EventEnvelope {
            publisher_id: 12345,
            sequence: 1,
            published_at: 1700000000000,
            topic: "test-topic".to_string(),
            payload: Bytes::from("test payload"),
        };

        let msgpack = rmp_serde::to_vec(&envelope).unwrap();
        let deserialized: EventEnvelope = rmp_serde::from_slice(&msgpack).unwrap();

        assert_eq!(deserialized.publisher_id, 12345);
        assert_eq!(deserialized.sequence, 1);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.topic, "test-topic");
        assert_eq!(deserialized.payload, Bytes::from("test payload"));
    }
}
