// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS implementation for the generic event plane.
//!
//! This module provides NATS Core-based pub/sub for the event plane,
//! wrapping messages in EventEnvelope for sequencing and gap detection.

use anyhow::Result;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

use crate::DistributedRuntime;

use super::{EventEnvelope, EventStream, TypedEventStream};

/// Internal envelope format for NATS transport.
///
/// This is serialized as JSON and wraps the actual event payload
/// with metadata for sequencing and publisher identification.
///
/// Note: The payload is serialized as a JSON array of bytes by default.
/// This is inefficient for large payloads but acceptable for Phase 1.
/// Future optimization: use base64 or msgpack encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct NatsEnvelope {
    pub publisher_id: u64,
    pub sequence: u64,
    pub published_at: u64,
    pub payload: Vec<u8>,
}

/// Subscribe to a NATS subject and return a stream of EventEnvelopes.
pub(super) async fn subscribe_to_nats(
    drt: &Arc<DistributedRuntime>,
    subject: String,
) -> Result<EventStream> {
    let nats_subscriber = drt.kv_router_nats_subscribe(subject.clone()).await?;

    let stream = nats_subscriber.map(move |msg| {
        // Parse the envelope from the NATS message payload
        let envelope: NatsEnvelope = serde_json::from_slice(&msg.payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse event envelope: {}", e))?;

        Ok(EventEnvelope {
            topic: msg.subject.to_string(),
            publisher_id: envelope.publisher_id,
            sequence: envelope.sequence,
            published_at: envelope.published_at,
            payload: Bytes::from(envelope.payload),
        })
    });

    Ok(Box::pin(stream))
}

/// Subscribe to a NATS subject and return a typed stream with auto-deserialization.
pub(super) async fn subscribe_typed_to_nats<T: serde::de::DeserializeOwned + Send + 'static>(
    drt: &Arc<DistributedRuntime>,
    subject: String,
) -> Result<TypedEventStream<T>> {
    let nats_subscriber = drt.kv_router_nats_subscribe(subject.clone()).await?;

    let stream = nats_subscriber.map(move |msg| {
        // Parse the envelope from the NATS message payload
        let envelope: NatsEnvelope = serde_json::from_slice(&msg.payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse event envelope: {}", e))?;

        // Deserialize the inner payload to the requested type
        let typed_event: T = serde_json::from_slice(&envelope.payload)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize event payload: {}", e))?;

        let event_envelope = EventEnvelope {
            topic: msg.subject.to_string(),
            publisher_id: envelope.publisher_id,
            sequence: envelope.sequence,
            published_at: envelope.published_at,
            payload: Bytes::from(envelope.payload.clone()),
        };

        Ok((event_envelope, typed_event))
    });

    Ok(Box::pin(stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nats_envelope_serialization() {
        let envelope = NatsEnvelope {
            publisher_id: 12345,
            sequence: 1,
            published_at: 1700000000000,
            payload: b"test payload".to_vec(),
        };

        let json = serde_json::to_string(&envelope).unwrap();
        let deserialized: NatsEnvelope = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.publisher_id, 12345);
        assert_eq!(deserialized.sequence, 1);
        assert_eq!(deserialized.published_at, 1700000000000);
        assert_eq!(deserialized.payload, b"test payload".to_vec());
    }

    #[test]
    fn test_nats_envelope_binary_payload() {
        let envelope = NatsEnvelope {
            publisher_id: 1,
            sequence: 0,
            published_at: 0,
            payload: vec![0, 1, 2, 255],
        };

        let json = serde_json::to_string(&envelope).unwrap();
        // Verify payload is serialized as JSON array
        assert!(json.contains("[0,1,2,255]") || json.contains("[0, 1, 2, 255]"));

        let deserialized: NatsEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.payload, vec![0, 1, 2, 255]);
    }
}
