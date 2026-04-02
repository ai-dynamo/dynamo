// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event plane codec for serializing/deserializing envelopes and payloads.

use anyhow::Result;
use bytes::Bytes;
use serde::{Serialize, de::DeserializeOwned};

use super::EventEnvelope;

/// Codec for serializing and deserializing event envelopes and payloads.
///
/// Currently only supports MessagePack for all transports.
#[derive(Debug, Clone, Copy)]
pub enum Codec {
    Msgpack(MsgpackCodec),
}

impl Default for Codec {
    fn default() -> Self {
        Codec::Msgpack(MsgpackCodec)
    }
}

impl Codec {
    /// Encode an EventEnvelope to wire bytes
    pub fn encode_envelope(&self, envelope: &EventEnvelope) -> Result<Bytes> {
        match self {
            Codec::Msgpack(c) => c.encode_envelope(envelope),
        }
    }

    /// Decode wire bytes to an EventEnvelope
    pub fn decode_envelope(&self, bytes: &Bytes) -> Result<EventEnvelope> {
        match self {
            Codec::Msgpack(c) => c.decode_envelope(bytes),
        }
    }

    /// Encode a typed payload to bytes (for embedding in envelope)
    pub fn encode_payload<T: Serialize>(&self, payload: &T) -> Result<Bytes> {
        match self {
            Codec::Msgpack(c) => c.encode_payload(payload),
        }
    }

    /// Decode payload bytes to a typed value
    pub fn decode_payload<T: DeserializeOwned>(&self, bytes: &Bytes) -> Result<T> {
        match self {
            Codec::Msgpack(c) => c.decode_payload(bytes),
        }
    }

    /// Codec name for debugging
    pub fn name(&self) -> &'static str {
        match self {
            Codec::Msgpack(c) => c.name(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MsgpackCodec;

impl MsgpackCodec {
    pub fn encode_envelope(&self, envelope: &EventEnvelope) -> Result<Bytes> {
        Ok(Bytes::from(rmp_serde::to_vec_named(envelope)?))
    }

    pub fn decode_envelope(&self, bytes: &Bytes) -> Result<EventEnvelope> {
        Ok(rmp_serde::from_slice(bytes)?)
    }

    pub fn encode_payload<T: Serialize>(&self, payload: &T) -> Result<Bytes> {
        Ok(Bytes::from(rmp_serde::to_vec_named(payload)?))
    }

    pub fn decode_payload<T: DeserializeOwned>(&self, bytes: &Bytes) -> Result<T> {
        Ok(rmp_serde::from_slice(bytes)?)
    }

    pub fn name(&self) -> &'static str {
        "msgpack"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
    struct TestEvent {
        worker_id: u64,
        message: String,
    }

    #[test]
    fn test_msgpack_codec_envelope_roundtrip() {
        let codec = MsgpackCodec;

        let envelope = EventEnvelope {
            publisher_id: 12345,
            sequence: 42,
            published_at: 1700000000000,
            topic: "test-topic".to_string(),
            payload: Bytes::from("test payload"),
        };

        let encoded = codec.encode_envelope(&envelope).unwrap();
        let decoded = codec.decode_envelope(&encoded).unwrap();

        assert_eq!(decoded.publisher_id, envelope.publisher_id);
        assert_eq!(decoded.sequence, envelope.sequence);
        assert_eq!(decoded.published_at, envelope.published_at);
        assert_eq!(decoded.topic, envelope.topic);
        assert_eq!(decoded.payload, envelope.payload);
    }

    #[test]
    fn test_msgpack_codec_payload_roundtrip() {
        let codec = MsgpackCodec;

        let event = TestEvent {
            worker_id: 123,
            message: "hello world".to_string(),
        };

        let encoded = codec.encode_payload(&event).unwrap();
        let decoded: TestEvent = codec.decode_payload(&encoded).unwrap();

        assert_eq!(decoded, event);
    }
}
