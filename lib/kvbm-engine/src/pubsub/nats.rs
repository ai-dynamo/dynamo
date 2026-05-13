// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS implementation of the PubSub traits.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_nats::Client;
use bytes::Bytes;
use flume::{Receiver, Sender};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use tokio::sync::oneshot;
use tracing::error;

use super::{Message, Publisher, Subscriber, Subscription};

/// Configuration for NATS publisher/subscriber.
#[derive(Debug, Clone)]
pub struct NatsConfig {
    /// NATS server URL (e.g., "nats://localhost:4222").
    pub server_url: String,
    /// Optional subject prefix prepended to all subjects.
    pub subject_prefix: Option<String>,
}

impl NatsConfig {
    /// Create a new NATS configuration.
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            subject_prefix: None,
        }
    }

    /// Set an optional subject prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.subject_prefix = Some(prefix.into());
        self
    }

    /// Connect to the NATS server and return a client.
    pub async fn connect(&self) -> Result<Client> {
        async_nats::connect(&self.server_url)
            .await
            .context("failed to connect to NATS server")
    }

    /// Format a subject with the optional prefix.
    fn format_subject(&self, subject: &str) -> String {
        match &self.subject_prefix {
            Some(prefix) => format!("{}.{}", prefix, subject),
            None => subject.to_string(),
        }
    }
}

/// Command sent to the publisher background task.
enum PublishCommand {
    /// Publish a message to a subject.
    Publish { subject: String, payload: Bytes },
    /// Flush pending messages and notify when complete.
    Flush { done: oneshot::Sender<Result<()>> },
}

/// NATS implementation of the [`Publisher`] trait.
///
/// Uses a background task with a flume channel to handle async publishes.
pub struct NatsPublisher {
    tx: Sender<PublishCommand>,
    config: Arc<NatsConfig>,
}

impl NatsPublisher {
    /// Create a new NATS publisher from a client and configuration.
    ///
    /// Spawns a background task to handle async publish operations.
    pub fn new(client: Client, config: NatsConfig) -> Self {
        let (tx, rx) = flume::unbounded();
        let config = Arc::new(config);

        tokio::spawn(Self::run_publish_loop(client, rx));

        Self { tx, config }
    }

    /// Create a new NATS publisher by connecting to the server.
    pub async fn connect(config: NatsConfig) -> Result<Self> {
        let client = config.connect().await?;
        Ok(Self::new(client, config))
    }

    /// Background task that processes publish commands.
    async fn run_publish_loop(client: Client, rx: Receiver<PublishCommand>) {
        while let Ok(cmd) = rx.recv_async().await {
            match cmd {
                PublishCommand::Publish { subject, payload } => {
                    if let Err(e) = client.publish(subject, payload).await {
                        error!("failed to publish message: {e}");
                    }
                }
                PublishCommand::Flush { done } => {
                    let result = client.flush().await.context("failed to flush");
                    // Ignore send error (receiver may have dropped)
                    let _ = done.send(result);
                }
            }
        }
    }
}

impl Publisher for NatsPublisher {
    fn publish(&self, subject: &str, payload: Bytes) -> Result<()> {
        let subject = self.config.format_subject(subject);
        self.tx
            .send(PublishCommand::Publish { subject, payload })
            .map_err(|_| anyhow::anyhow!("publisher task has terminated"))
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        let (done_tx, done_rx) = oneshot::channel();
        let tx = self.tx.clone();

        async move {
            tx.send(PublishCommand::Flush { done: done_tx })
                .map_err(|_| anyhow::anyhow!("publisher task has terminated"))?;
            done_rx
                .await
                .map_err(|_| anyhow::anyhow!("publisher task has terminated"))?
        }
        .boxed()
    }
}

/// NATS implementation of the [`Subscriber`] trait.
pub struct NatsSubscriber {
    client: Client,
    config: NatsConfig,
}

impl NatsSubscriber {
    /// Create a new NATS subscriber from a client and configuration.
    pub fn new(client: Client, config: NatsConfig) -> Self {
        Self { client, config }
    }

    /// Create a new NATS subscriber by connecting to the server.
    pub async fn connect(config: NatsConfig) -> Result<Self> {
        let client = config.connect().await?;
        Ok(Self::new(client, config))
    }
}

impl Subscriber for NatsSubscriber {
    fn subscribe(&self, subject: &str) -> BoxFuture<'static, Result<Subscription>> {
        let subject = self.config.format_subject(subject);
        let client = self.client.clone();
        async move {
            let subscriber = client
                .subscribe(subject)
                .await
                .context("failed to subscribe")?;

            let stream: BoxStream<'static, Message> = subscriber
                .map(|msg| Message {
                    subject: msg.subject.to_string(),
                    payload: msg.payload,
                })
                .boxed();

            Ok(stream)
        }
        .boxed()
    }
}
