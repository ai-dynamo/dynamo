// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StreamFrame<T>`]: the seven-variant wire frame enum for the streaming protocol.

use serde::{Deserialize, Serialize};

/// A single frame traveling over the streaming wire.
///
/// `T` is the user-defined item payload type. It must implement `Serialize` and
/// `Deserialize` at call sites. `String` is used as the error carrier in
/// [`SenderError`](StreamFrame::SenderError) and
/// [`TransportError`](StreamFrame::TransportError) to avoid requiring
/// `T: std::error::Error` and to ensure msgpack compatibility.
///
/// Consumers MUST NOT be exposed to [`StreamFrame::Heartbeat`] — the API layer
/// filters heartbeat frames before surfacing items to user code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamFrame<T> {
    /// A data item carrying the user payload `T`.
    Item(T),
    /// A soft error reported by the sender. Does not terminate the stream;
    /// the sender may continue sending `Item` frames afterwards.
    SenderError(String),
    /// The sender was dropped without an explicit detach or finalize.
    /// This is the last frame the anchor receives from a sender that exited abruptly.
    Dropped,
    /// The sender explicitly detached; a new sender may attach to the same anchor.
    Detached,
    /// The stream is permanently closed; no further items will arrive.
    Finalized,
    /// Internal liveness ping. Never exposed to the user API layer.
    Heartbeat,
    /// The transport layer encountered an unrecoverable error.
    TransportError(String),
}

// ---------------------------------------------------------------------------
// Error types for the consumer (AnchorStream) and producer (StreamSender) APIs
// ---------------------------------------------------------------------------

/// Errors surfaced to [`AnchorStream`](crate::anchor::AnchorStream) consumers.
///
/// These represent problems that occurred on the sender side or during
/// transport/deserialization. The stream may or may not continue producing
/// items after a `StreamError`, depending on the variant.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// The sender reported a soft error via [`StreamFrame::SenderError`].
    #[error("sender error: {0}")]
    SenderError(String),
    /// The sender was dropped without an explicit detach or finalize.
    #[error("sender dropped without explicit detach/finalize")]
    SenderDropped,
    /// The transport layer encountered an unrecoverable error.
    #[error("transport error: {0}")]
    TransportError(String),
    /// A received frame could not be deserialized into `StreamFrame<T>`.
    #[error("deserialization error: {0}")]
    DeserializationError(String),
}

/// Errors surfaced to [`StreamSender`] callers when a send operation fails.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// The underlying channel is closed (receiver dropped or anchor removed).
    #[error("channel closed")]
    ChannelClosed,
    /// The item could not be serialized into the wire format.
    #[error("serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_frame_variants() {
        // Verify all seven variants compile and serialize/deserialize via rmp-serde
        let frames: Vec<StreamFrame<u32>> = vec![
            StreamFrame::Item(42u32),
            StreamFrame::SenderError("soft error".to_string()),
            StreamFrame::Dropped,
            StreamFrame::Detached,
            StreamFrame::Finalized,
            StreamFrame::Heartbeat,
            StreamFrame::TransportError("connection reset".to_string()),
        ];

        for frame in &frames {
            let encoded = rmp_serde::to_vec(frame).expect("serialize StreamFrame");
            let decoded: StreamFrame<u32> =
                rmp_serde::from_slice(&encoded).expect("deserialize StreamFrame");
            // Verify the variant survived the round-trip
            match (frame, &decoded) {
                (StreamFrame::Item(a), StreamFrame::Item(b)) => assert_eq!(a, b),
                (StreamFrame::SenderError(a), StreamFrame::SenderError(b)) => assert_eq!(a, b),
                (StreamFrame::Dropped, StreamFrame::Dropped) => {}
                (StreamFrame::Detached, StreamFrame::Detached) => {}
                (StreamFrame::Finalized, StreamFrame::Finalized) => {}
                (StreamFrame::Heartbeat, StreamFrame::Heartbeat) => {}
                (StreamFrame::TransportError(a), StreamFrame::TransportError(b)) => {
                    assert_eq!(a, b)
                }
                _ => panic!("variant mismatch after round-trip"),
            }
        }
    }

    #[test]
    fn test_stream_frame_item_direct_payload() {
        // Item(T) contains T directly, not Result<T, String>
        let frame = StreamFrame::Item(42u32);
        let encoded = rmp_serde::to_vec(&frame).expect("serialize");
        let decoded: StreamFrame<u32> = rmp_serde::from_slice(&encoded).expect("deserialize");
        match decoded {
            StreamFrame::Item(val) => assert_eq!(val, 42u32),
            other => panic!("expected Item(42), got {:?}", other),
        }
    }

    #[test]
    fn test_stream_frame_sender_error_round_trip() {
        let frame = StreamFrame::<u32>::SenderError("msg".to_string());
        let encoded = rmp_serde::to_vec(&frame).expect("serialize");
        let decoded: StreamFrame<u32> = rmp_serde::from_slice(&encoded).expect("deserialize");
        match decoded {
            StreamFrame::SenderError(msg) => assert_eq!(msg, "msg"),
            other => panic!("expected SenderError, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_error_display() {
        assert_eq!(
            StreamError::SenderError("msg".to_string()).to_string(),
            "sender error: msg"
        );
        assert_eq!(
            StreamError::SenderDropped.to_string(),
            "sender dropped without explicit detach/finalize"
        );
        assert_eq!(
            StreamError::TransportError("msg".to_string()).to_string(),
            "transport error: msg"
        );
        assert_eq!(
            StreamError::DeserializationError("msg".to_string()).to_string(),
            "deserialization error: msg"
        );
    }

    #[test]
    fn test_send_error_display() {
        assert_eq!(SendError::ChannelClosed.to_string(), "channel closed");
        assert_eq!(
            SendError::SerializationError("msg".to_string()).to_string(),
            "serialization error: msg"
        );
    }
}
