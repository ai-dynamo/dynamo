// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport abstraction for frame-level ordered delivery.
//!
//! This module defines the [`FrameTransport`] trait boundary consumed by
//! [`AnchorManager`] (Phase 7) and implemented by [`VeloFrameTransport`]
//! (Phase 9).
//!
//! Transport endpoints are concrete [`flume::Receiver<Vec<u8>>`] and
//! [`flume::Sender<Vec<u8>>`] channel halves rather than trait objects.
//! This enables mixed sync/async usage: synchronous `send()` in Drop impls,
//! `send_async().await` on the normal data path, and `recv_async().await` in
//! the reader pump.

use anyhow::Result;
use futures::future::BoxFuture;

/// Transport abstraction for frame-level ordered delivery.
///
/// # Ordered-Delivery Contract
///
/// All frames -- including data frames (`Item`, `Heartbeat`) **and** sentinel
/// frames (`Dropped`, `Detached`, `Finalized`, `TransportError`) -- MUST travel
/// the **same physical channel** established by [`FrameTransport::bind`] /
/// [`FrameTransport::connect`]. Sentinels MUST NOT be injected via a side
/// channel; the FIFO ordering guarantee of the underlying channel is
/// load-bearing for the correctness of the streaming protocol.
///
/// Implementations MUST preserve send order: a frame sent before another MUST
/// be received before that other frame on the corresponding
/// [`flume::Receiver`].
///
/// # Usage
///
/// The transport operates at the raw byte level. Callers are responsible for
/// serializing [`crate::frame::StreamFrame`] values to `Vec<u8>` before
/// sending via [`flume::Sender::send_async`], and for deserializing bytes
/// received from [`flume::Receiver::recv_async`].
///
/// # Async Design
///
/// Both `bind` and `connect` return [`BoxFuture`] to support async
/// implementations (e.g., network setup). The heap allocation is acceptable
/// because these are setup-path calls, not per-frame hot-path operations.
pub trait FrameTransport: Send + Sync {
    /// Bind a new receive endpoint for the given anchor.
    ///
    /// Returns `(endpoint_address, receiver)` where `endpoint_address` is an
    /// opaque string the sender passes to [`connect`][FrameTransport::connect]
    /// and `receiver` is the receive half of the frame channel.
    ///
    /// The channel established by `bind` / `connect` MUST provide ordered,
    /// loss-free delivery of all frames including sentinels.
    fn bind(&self, anchor_id: u64) -> BoxFuture<'_, Result<(String, flume::Receiver<Vec<u8>>)>>;

    /// Connect a write endpoint to the given anchor.
    ///
    /// - `endpoint`: the address string returned by a prior [`bind`][FrameTransport::bind] call.
    ///   **Implementations must convert this `&str` to an owned `String` before
    ///   entering the async block** (the future's lifetime is tied to `&self`,
    ///   not to `endpoint`).
    /// - `anchor_id`: identifies which anchor this writer is attached to.
    /// - `session_id`: unique session identifier for this attachment; used by
    ///   the transport to route frames to the correct reader.
    ///
    /// Returns a [`flume::Sender<Vec<u8>>`] for sending frames to the bound receiver.
    fn connect(
        &self,
        endpoint: &str,
        anchor_id: u64,
        session_id: u64,
    ) -> BoxFuture<'_, Result<flume::Sender<Vec<u8>>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof that [`FrameTransport`] is object-safe.
    ///
    /// If the trait violates `dyn` object safety, this function will fail to
    /// compile. The function itself is never called -- its existence is the test.
    fn _assert_object_safe(_transport: &dyn FrameTransport) {}
}
