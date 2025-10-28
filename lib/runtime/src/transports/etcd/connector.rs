// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ErrorContext, Result, error};
use etcd_client::ConnectOptions;
use parking_lot::RwLock;
use std::{sync::Arc, time::Duration};
use tokio::{sync::Mutex, time::sleep};

/// Manages ETCD client connections with reconnection support
pub struct Connector {
    /// The actual ETCD client, protected by RwLock for safe updates during reconnection
    /// WARNING: Do not recursively acquire a read lock when the current thread already holds one
    client: RwLock<etcd_client::Client>,
    /// Configuration for connecting to ETCD
    etcd_urls: Vec<String>,
    connect_options: Option<ConnectOptions>,
    /// Tracks the current backoff duration and last successful connect time
    /// The Mutex ensures only one reconnect operation runs at a time
    backoff_state: Mutex<BackoffState>,
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
            backoff_state: Mutex::new(BackoffState::default()),
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
    ///
    /// Backoff behavior:
    /// - Starts at 0 (immediate reconnect) if this is the first reconnect or enough time has passed
    ///   since the last reconnect
    /// - Increments exponentially for continuous failures
    /// - Resets to 0 only when: this is a new call AND current_time > last_connect_time + residual_backoff
    ///
    /// The mutex ensures only one reconnect operation runs at a time globally
    pub async fn reconnect(&self, deadline: std::time::Instant) -> Result<()> {
        let mut backoff_state = self.backoff_state.lock().await;

        tracing::warn!("Reconnecting to ETCD cluster at: {:?}", self.etcd_urls);
        backoff_state.attempt_reset();

        loop {
            backoff_state.apply_backoff(deadline).await;
            if std::time::Instant::now() >= deadline {
                return Err(error!(
                    "Unable to reconnect to ETCD cluster: deadline exceeded"
                ));
            }

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
                        deadline.saturating_duration_since(std::time::Instant::now()),
                        e
                    );
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

#[derive(Debug)]
struct BackoffState {
    /// Initial backoff duration for reconnection attempts
    pub initial_backoff: Duration,
    /// Minimum backoff duration for reconnection attempts
    pub min_backoff: Duration,
    /// Maximum backoff duration for reconnection attempts
    pub max_backoff: Duration,
    /// Current backoff duration (starts at 0 for immediate reconnect)
    current_backoff: Duration,
    /// Last time a connection establishment was attempted
    last_connect_attempt: std::time::Instant,
}

impl Default for BackoffState {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(500),
            min_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_secs(5),
            current_backoff: Duration::ZERO,
            last_connect_attempt: std::time::Instant::now(),
        }
    }
}

impl BackoffState {
    /// Reset backoff to 0 if enough time has passed since the last connection
    pub fn attempt_reset(&mut self) {
        if std::time::Instant::now() > self.last_connect_attempt + self.current_backoff {
            tracing::debug!("Resetting backoff to 0 (first reconnect or enough time has passed)");
            self.current_backoff = Duration::ZERO;
        }
    }

    /// Apply backoff and update backoff state for possible next connection attempt
    pub async fn apply_backoff(&mut self, deadline: std::time::Instant) {
        if self.current_backoff > Duration::ZERO {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let backoff = std::cmp::min(self.current_backoff, remaining / 2);
            let backoff = std::cmp::min(backoff, self.max_backoff);
            let backoff = std::cmp::max(backoff, self.min_backoff);
            self.current_backoff = backoff * 2;

            tracing::debug!(
                "Applying backoff of {:?} (remaining time: {:?})",
                backoff,
                remaining
            );
            sleep(backoff).await;
        } else {
            self.current_backoff = self.initial_backoff;
        }
        self.last_connect_attempt = std::time::Instant::now();
    }
}
