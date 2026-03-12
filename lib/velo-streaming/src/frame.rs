// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StreamFrame<T>`]: the six-variant wire frame enum for the streaming protocol.

use serde::{Deserialize, Serialize};

/// A single frame traveling over the streaming wire.
///
/// `T` is the user-defined item payload type. It must implement `Serialize` and
/// `Deserialize` at call sites. `String` is used as the error carrier to avoid
/// requiring `T: std::error::Error` and to ensure msgpack compatibility.
///
/// Consumers MUST NOT be exposed to [`StreamFrame::Heartbeat`] — the API layer
/// filters heartbeat frames before surfacing items to user code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamFrame<T> {
    /// A data item — `Ok(T)` for a successful item, `Err(String)` for a soft error.
    Item(Result<T, String>),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_frame_variants() {
        // Verify all six variants compile and serialize/deserialize via rmp-serde
        let frames: Vec<StreamFrame<u32>> = vec![
            StreamFrame::Item(Ok(42u32)),
            StreamFrame::Item(Err("soft error".to_string())),
            StreamFrame::Dropped,
            StreamFrame::Detached,
            StreamFrame::Finalized,
            StreamFrame::Heartbeat,
            StreamFrame::TransportError("connection reset".to_string()),
        ];

        for frame in &frames {
            let encoded = rmp_serde::to_vec(frame).expect("serialize StreamFrame");
            let _decoded: StreamFrame<u32> =
                rmp_serde::from_slice(&encoded).expect("deserialize StreamFrame");
        }
    }
}
