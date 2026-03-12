// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data source abstractions for the Dynamo TUI.
//!
//! Each source (ETCD, NATS, Metrics) implements the [`Source`] trait,
//! which allows mocking in tests. Sources send [`AppEvent`]s through
//! an mpsc channel to the main app loop.

pub mod etcd;
pub mod metrics;
pub mod nats;

use crate::model::{Namespace, NatsStats, PrometheusMetrics};

/// Events sent from data sources to the app event loop.
#[derive(Debug)]
pub enum AppEvent {
    /// ETCD discovery tree was updated.
    DiscoveryUpdate(Vec<Namespace>),

    /// NATS statistics were refreshed.
    NatsUpdate(NatsStats),

    /// Prometheus metrics were scraped.
    MetricsUpdate(PrometheusMetrics),

    /// A source encountered a connection error.
    SourceError { source: String, message: String },

    /// Keyboard/terminal input event.
    Input(crossterm::event::KeyEvent),

    /// Periodic tick for UI refresh.
    Tick,
}

/// Trait for data sources. All sources are async and send events via a channel.
///
/// This trait enables mocking in tests — each source can be replaced
/// with a mock that sends pre-determined events.
#[async_trait::async_trait]
pub trait Source: Send + 'static {
    /// Run the source, sending events to `tx` until `cancel` is triggered.
    async fn run(
        self: Box<Self>,
        tx: tokio::sync::mpsc::Sender<AppEvent>,
        cancel: tokio_util::sync::CancellationToken,
    );
}
