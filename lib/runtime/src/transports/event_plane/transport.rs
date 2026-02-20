// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport trait abstractions for the event plane.
//!
//! These traits define the low-level interface for different transport backends
//! (NATS, ZMQ) to implement. The event plane uses these traits to provide a
//! unified pub/sub API.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;

use crate::discovery::EventTransportKind;

/// Stream of raw wire bytes from a subscription.
pub type WireStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Transport-agnostic interface for publishing events.
#[async_trait]
pub trait EventTransportTx: Send + Sync {
    /// Publish raw envelope bytes to a subject/topic.
    async fn publish(&self, subject: &str, envelope_bytes: Bytes) -> Result<()>;

    fn kind(&self) -> EventTransportKind;
}

/// Transport-agnostic interface for subscribing to events.
#[async_trait]
pub trait EventTransportRx: Send + Sync {
    /// Subscribe to a subject/topic and return a stream of raw envelope bytes.
    async fn subscribe(&self, subject: &str) -> Result<WireStream>;

    fn kind(&self) -> EventTransportKind;
}
