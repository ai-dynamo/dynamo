// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publisher trait for distributed messaging.
//!
//! This module provides the `Publisher` trait used by the event pipeline.
//! Concrete implementations (NATS, Stub) remain in `dynamo-kvbm`.

use anyhow::Result;
use bytes::Bytes;
use futures::future::BoxFuture;

/// Publisher trait for sending messages to subjects.
///
/// Publishers are responsible for sending messages to named subjects.
/// Messages are delivered to all subscribers matching the subject pattern.
pub trait Publisher: Send + Sync {
    /// Publish a message to a subject.
    ///
    /// This queues the message for delivery and returns immediately.
    /// Use [`flush`](Publisher::flush) to ensure delivery.
    fn publish(&self, subject: &str, payload: Bytes) -> Result<()>;

    /// Flush pending messages to ensure delivery.
    ///
    /// Returns when all previously published messages have been acknowledged
    /// by the messaging system.
    fn flush(&self) -> BoxFuture<'static, Result<()>>;
}
