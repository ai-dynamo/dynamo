// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PubSub abstraction for distributed messaging.
//!
//! This module provides traits for publish/subscribe messaging patterns,
//! with implementations for NATS and an in-memory stub for testing.

use anyhow::Result;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::BoxStream;

#[cfg(feature = "nats")]
mod nats;
mod stub;

#[cfg(feature = "nats")]
pub use self::nats::{NatsConfig, NatsPublisher, NatsSubscriber};
pub use stub::{StubBus, StubPublisher, StubSubscriber};

/// Message received from a subscription.
#[derive(Debug, Clone)]
pub struct Message {
    /// The subject the message was published to.
    pub subject: String,
    /// The message payload.
    pub payload: Bytes,
}

/// A subscription stream that yields messages.
pub type Subscription = BoxStream<'static, Message>;

pub use dynamo_kvbm_logical::pubsub::Publisher;

/// Subscriber trait for receiving messages from subjects.
///
/// Subscribers receive messages published to matching subjects.
/// Subject patterns support wildcards:
/// - `*` matches a single token (e.g., `foo.*.bar`)
/// - `>` matches one or more tokens at the tail (e.g., `foo.>`)
pub trait Subscriber: Send + Sync {
    /// Subscribe to a subject pattern, returning a message stream.
    ///
    /// The returned stream yields messages as they arrive. The subscription
    /// remains active until the stream is dropped.
    fn subscribe(&self, subject: &str) -> BoxFuture<'static, Result<Subscription>>;
}
