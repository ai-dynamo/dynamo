// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire protocol definitions for stream anchors.

use bytes::Bytes;
use dynamo_identity::{InstanceId, WorkerId};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// Bit layout for AnchorId (u128):
// - Bits 0-63: owner_worker (u64)
// - Bits 64-79: local_index (u16)
// - Bits 80-127: generation (u48)
const WORKER_BITS: u32 = 64;
const INDEX_BITS: u32 = 16;
const GENERATION_BITS: u32 = 48;

const INDEX_SHIFT: u32 = WORKER_BITS;
const GENERATION_SHIFT: u32 = WORKER_BITS + INDEX_BITS;

const WORKER_MASK: u128 = (1u128 << WORKER_BITS) - 1;
const INDEX_MASK: u128 = ((1u128 << INDEX_BITS) - 1) << INDEX_SHIFT;
const GENERATION_MASK: u128 = ((1u128 << GENERATION_BITS) - 1) << GENERATION_SHIFT;

/// Maximum generation value before slot retirement.
pub const MAX_GENERATION: u64 = (1u64 << GENERATION_BITS) - 1;

/// Maximum number of anchor slots per worker.
pub const MAX_ANCHOR_SLOTS: usize = u16::MAX as usize;

/// Unique identifier for a stream anchor.
///
/// Encodes the owner worker, local slot index, and generation in a single u128
/// for efficient wire transfer. The generation counter prevents ABA problems
/// when slots are reused.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnchorId(u128);

impl AnchorId {
    /// Create a new anchor ID from components.
    pub fn new(owner_worker: WorkerId, local_index: u16, generation: u64) -> Self {
        debug_assert!(
            generation <= MAX_GENERATION,
            "generation exceeds maximum value"
        );

        let raw = (owner_worker.as_u64() as u128)
            | ((local_index as u128) << INDEX_SHIFT)
            | ((generation as u128) << GENERATION_SHIFT);

        Self(raw)
    }

    /// Reconstruct from raw u128 value.
    #[inline]
    pub fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Get the raw u128 value.
    #[inline]
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Get the owner worker ID.
    #[inline]
    pub fn owner_worker(&self) -> WorkerId {
        WorkerId::from_u64((self.0 & WORKER_MASK) as u64)
    }

    /// Get the local slot index.
    #[inline]
    pub fn local_index(&self) -> u16 {
        ((self.0 & INDEX_MASK) >> INDEX_SHIFT) as u16
    }

    /// Get the generation counter.
    #[inline]
    pub fn generation(&self) -> u64 {
        ((self.0 & GENERATION_MASK) >> GENERATION_SHIFT) as u64
    }

    /// Create a new ID with an updated generation.
    pub fn with_generation(&self, generation: u64) -> Self {
        Self::new(self.owner_worker(), self.local_index(), generation)
    }
}

impl fmt::Debug for AnchorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnchorId")
            .field("worker", &self.owner_worker())
            .field("index", &self.local_index())
            .field("generation", &self.generation())
            .finish()
    }
}

impl fmt::Display for AnchorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Anchor({}/{}/{})",
            self.owner_worker(),
            self.local_index(),
            self.generation()
        )
    }
}

// =============================================================================
// Nova RPC Request/Response Types
// =============================================================================

/// Request to attach to a stream anchor (sent by sender to receiver).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachRequest {
    /// The anchor ID to attach to.
    pub anchor_id: AnchorId,
    /// Unique session ID for this attachment.
    pub session_id: Uuid,
    /// Instance ID of the sender.
    pub sender_instance: InstanceId,
}

/// Response to an attach request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachResponse {
    /// Endpoint where data frames should be sent.
    pub stream_endpoint: String,
    /// Suggested heartbeat interval for the sender.
    /// Sender should send heartbeats at this interval to keep stream alive.
    pub heartbeat_interval_ms: Option<u64>,
    /// Receiver's message timeout in milliseconds.
    pub message_timeout_ms: Option<u64>,
}

/// Request to cancel a stream (sent by receiver to sender).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// The anchor ID being cancelled.
    pub anchor_id: AnchorId,
    /// Session ID of the attachment being cancelled.
    pub session_id: Uuid,
    /// Reason for cancellation.
    pub reason: String,
}

/// Acknowledgment of cancel request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    /// Whether the cancel was acknowledged.
    pub received: bool,
}

/// Raw wire frame for deserialization when we don't know T yet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawWireFrame {
    /// The target anchor ID.
    pub anchor_id: AnchorId,
    /// The frame payload as a JSON value.
    pub frame: serde_json::Value,
}

/// Frame types for the data stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamFrame<T> {
    /// Data payload (will be serialized by transport).
    Data(T),

    /// Raw pre-serialized data payload.
    ///
    /// Use this when you have data that's already serialized in your preferred
    /// format (protobuf, flatbuffers, msgpack, etc.) and you don't want the
    /// transport to re-serialize it.
    ///
    /// The bytes are sent as-is without any additional serialization overhead.
    #[serde(with = "serde_bytes")]
    RawData(Vec<u8>),

    /// Heartbeat to keep connection alive.
    Heartbeat,

    /// Sender attached to the anchor.
    Attached {
        /// Instance ID of the attached sender.
        sender_instance: InstanceId,
        /// Session ID for this attachment.
        session_id: Uuid,
    },

    /// Sender detaching (may re-attach later).
    Detached,

    /// Stream finalized (no more data).
    Finalized,

    /// Attach timeout - no sender attached within deadline.
    AttachTimeout,

    /// Message timeout - no message received within deadline while attached.
    MessageTimeout,

    /// Stream cancelled by receiver.
    Cancelled(String),

    /// Error frame.
    Error(String),
}

impl<T> StreamFrame<T> {
    /// Create a data frame.
    pub fn data(item: T) -> Self {
        Self::Data(item)
    }

    /// Create a raw data frame from pre-serialized bytes.
    ///
    /// Use this when you have data that's already serialized and you want to
    /// avoid double-serialization overhead.
    pub fn raw_data(bytes: impl Into<Vec<u8>>) -> Self {
        Self::RawData(bytes.into())
    }

    /// Create a raw data frame from `Bytes`.
    pub fn raw_bytes(bytes: Bytes) -> Self {
        Self::RawData(bytes.to_vec())
    }

    /// Create an error frame.
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    /// Returns true if this is a data frame (either Data or RawData).
    pub fn is_data(&self) -> bool {
        matches!(self, Self::Data(_) | Self::RawData(_))
    }

    /// Returns true if this is a control frame (not data).
    pub fn is_control(&self) -> bool {
        !self.is_data()
    }

    /// If this is a RawData frame, return the bytes.
    pub fn into_raw_data(self) -> Option<Vec<u8>> {
        match self {
            Self::RawData(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// If this is a RawData frame, return a reference to the bytes.
    pub fn as_raw_data(&self) -> Option<&[u8]> {
        match self {
            Self::RawData(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Create a cancelled frame.
    pub fn cancelled(reason: impl Into<String>) -> Self {
        Self::Cancelled(reason.into())
    }

    /// Returns true if this frame indicates the stream has ended.
    ///
    /// Terminal frames are: Finalized, AttachTimeout, MessageTimeout, Cancelled.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Finalized
                | Self::AttachTimeout
                | Self::MessageTimeout
                | Self::Cancelled(_)
        )
    }

    /// Returns true if this is a timeout frame.
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::AttachTimeout | Self::MessageTimeout)
    }

    /// If this is a Cancelled frame, return the reason.
    pub fn as_cancelled(&self) -> Option<&str> {
        match self {
            Self::Cancelled(reason) => Some(reason),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_id_round_trip() {
        let worker = WorkerId::from_u64(0x1234_5678_9ABC_DEF0);
        let index = 0xABCD;
        let generation = 0x1234_5678_9ABC;

        let id = AnchorId::new(worker, index, generation);

        assert_eq!(id.owner_worker(), worker);
        assert_eq!(id.local_index(), index);
        assert_eq!(id.generation(), generation);
    }

    #[test]
    fn anchor_id_max_values() {
        let worker = WorkerId::from_u64(u64::MAX);
        let index = u16::MAX;
        let generation = MAX_GENERATION;

        let id = AnchorId::new(worker, index, generation);

        assert_eq!(id.owner_worker(), worker);
        assert_eq!(id.local_index(), index);
        assert_eq!(id.generation(), generation);
    }

    #[test]
    fn anchor_id_with_generation() {
        let worker = WorkerId::from_u64(42);
        let index = 100;
        let generation = 1;

        let id = AnchorId::new(worker, index, generation);
        let next_id = id.with_generation(2);

        assert_eq!(next_id.owner_worker(), worker);
        assert_eq!(next_id.local_index(), index);
        assert_eq!(next_id.generation(), 2);
    }

    #[test]
    fn anchor_id_serialization() {
        let id = AnchorId::new(WorkerId::from_u64(42), 100, 1);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: AnchorId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
