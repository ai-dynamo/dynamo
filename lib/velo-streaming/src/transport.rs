// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport abstraction for frame-level ordered delivery.
//!
//! This module defines the [`FrameTransport`] / [`FrameReader`] / [`FrameWriter`]
//! trait boundary consumed by [`AnchorManager`] (Phase 7) and implemented by
//! [`VeloFrameTransport`] (Phase 9).

use anyhow::Result;

/// Transport abstraction for frame-level ordered delivery.
///
/// # Ordered-Delivery Contract
///
/// All frames — including data frames (`Item`, `Heartbeat`) **and** sentinel
/// frames (`Dropped`, `Detached`, `Finalized`, `TransportError`) — MUST travel
/// the **same physical channel** established by [`FrameTransport::bind`] /
/// [`FrameTransport::connect`]. Sentinels MUST NOT be injected via a side
/// channel; the FIFO ordering guarantee of the underlying channel is
/// load-bearing for the correctness of the streaming protocol.
///
/// Implementations MUST preserve send order: a frame sent before another MUST
/// be received before that other frame on the corresponding [`FrameReader`].
///
/// # Usage
///
/// The transport operates at the raw byte level. Callers are responsible for
/// serializing [`crate::frame::StreamFrame`] values to `Vec<u8>` before
/// calling [`FrameWriter::send`], and for deserializing bytes received from
/// [`FrameReader::recv`].
pub trait FrameTransport: Send + Sync {
    /// Bind a new receive endpoint for the given anchor.
    ///
    /// Returns `(endpoint_address, reader)` where `endpoint_address` is an
    /// opaque string the sender passes to [`connect`][FrameTransport::connect]
    /// and `reader` is the exclusive receive half of the channel.
    ///
    /// The channel established by `bind` / `connect` MUST provide ordered,
    /// loss-free delivery of all frames including sentinels.
    fn bind(&self, anchor_id: u64) -> Result<(String, Box<dyn FrameReader>)>;

    /// Connect a write endpoint to the given anchor.
    ///
    /// - `endpoint`: the address string returned by a prior [`bind`][FrameTransport::bind] call.
    /// - `anchor_id`: identifies which anchor this writer is attached to.
    /// - `session_id`: unique session identifier for this attachment; used by
    ///   the transport to route frames to the correct reader.
    ///
    /// Returns a [`FrameWriter`] for sending frames to the bound reader.
    fn connect(
        &self,
        endpoint: &str,
        anchor_id: u64,
        session_id: u64,
    ) -> Result<Box<dyn FrameWriter>>;
}

/// Receive half of a frame transport channel.
///
/// Frames are yielded in the order they were sent. `None` indicates the
/// channel has been closed cleanly by the writer (via [`FrameWriter::close`]).
///
/// The `recv` method takes `&mut self` — a [`FrameReader`] is exclusively
/// owned by a single reader task and is not shared across threads.
pub trait FrameReader: Send + Sync {
    /// Receive the next frame.
    ///
    /// Returns:
    /// - `Some(Ok(bytes))` — a frame was received successfully.
    /// - `Some(Err(e))` — a transport-level error occurred; the channel
    ///   should be considered unusable after this.
    /// - `None` — the write half has been cleanly closed; no more frames
    ///   will be delivered.
    ///
    /// Frames are guaranteed to be delivered in send order.
    fn recv(&mut self) -> Option<Result<Vec<u8>>>;
}

/// Send half of a frame transport channel.
///
/// A [`FrameWriter`] is exclusively owned by a single [`StreamSender`] and is
/// not shared across threads. The `send` and `close` methods take `&mut self`
/// and `Box<Self>` respectively to enforce this exclusivity.
///
/// [`StreamSender`]: crate::frame::StreamFrame
pub trait FrameWriter: Send {
    /// Send a serialized frame to the bound reader.
    ///
    /// The transport MUST deliver this frame before any frame sent in a
    /// subsequent `send` call — ordered delivery is a hard invariant.
    fn send(&mut self, data: &[u8]) -> Result<()>;

    /// Flush and close the write channel, consuming the writer.
    ///
    /// After `close`, the corresponding [`FrameReader::recv`] MUST return
    /// `None` once all previously sent frames have been delivered.
    fn close(self: Box<Self>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof that all three traits are object-safe.
    ///
    /// If any trait violates `dyn` object safety, this function will fail to
    /// compile. The function itself is never called — its existence is the test.
    fn _assert_object_safe(
        _transport: &dyn FrameTransport,
        _reader: Box<dyn FrameReader>,
        _writer: Box<dyn FrameWriter>,
    ) {
    }
}
