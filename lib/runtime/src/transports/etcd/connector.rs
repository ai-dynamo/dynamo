// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ErrorContext, Result, error};
use etcd_client::ConnectOptions;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Manages ETCD client connections with reconnection support
pub struct Connector {
    /// The actual ETCD client, protected by RwLock for safe updates during reconnection
    /// WARNING: Do not recursively acquire a read lock when the current thread already holds one
    client: RwLock<etcd_client::Client>,
    /// Configuration for connecting to ETCD
    etcd_urls: Vec<String>,
    connect_options: Option<ConnectOptions>,
    /// Initial backoff duration for reconnection attempts
    pub initial_backoff: Duration,
    /// Maximum backoff duration for reconnection attempts
    pub max_backoff: Duration,
}

impl Connector {
    /// Create a new connector with an established connection
    pub async fn new(
        etcd_urls: Vec<String>,
        connect_options: Option<ConnectOptions>,
    ) -> Result<Arc<Self>> {
        // Connect to ETCD
        let client = Self::connect(&etcd_urls, &connect_options).await?;

        Ok(Arc::new(Self {
            client: RwLock::new(client),
            etcd_urls,
            connect_options,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(5),
        }))
    }

    /// Connect to ETCD cluster
    async fn connect(
        etcd_urls: &[String],
        connect_options: &Option<ConnectOptions>,
    ) -> Result<etcd_client::Client> {
        etcd_client::Client::connect(etcd_urls.to_vec(), connect_options.clone())
            .await
            .with_context(|| {
                format!(
                    "Unable to connect to etcd server at {}. Check etcd server status",
                    etcd_urls.join(", ")
                )
            })
    }

    /// Get a clone of the current ETCD client
    pub fn get_client(&self) -> etcd_client::Client {
        self.client.read().clone()
    }

    /// Reconnect to ETCD cluster with retry logic
    /// Respects the deadline and returns error if exceeded
    pub async fn reconnect(&self, deadline: std::time::Instant) -> Result<()> {
        tracing::warn!("Reconnecting to ETCD cluster at: {:?}", self.etcd_urls);

        let mut backoff = self.initial_backoff;

        loop {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(error!(
                    "Unable to reconnect to ETCD cluster: deadline exceeded"
                ));
            }
            let remaining = deadline.saturating_duration_since(now);
            backoff = std::cmp::min(std::cmp::min(backoff, remaining / 2), self.max_backoff);
            sleep(backoff).await;

            match Self::connect(&self.etcd_urls, &self.connect_options).await {
                Ok(new_client) => {
                    tracing::info!("Successfully reconnected to ETCD cluster");
                    // Update the client behind the lock
                    let mut client_guard = self.client.write();
                    *client_guard = new_client;
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(
                        "Reconnection failed (remaining time: {:?}): {}",
                        remaining,
                        e
                    );
                    backoff *= 2;
                }
            }
        }
    }

    /// Get the ETCD URLs
    pub fn etcd_urls(&self) -> &[String] {
        &self.etcd_urls
    }

    /// Get the connection options
    pub fn connect_options(&self) -> &Option<ConnectOptions> {
        &self.connect_options
    }
}
