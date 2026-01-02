// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow as error};
use dashmap::DashMap;
use etcd_client::ConnectOptions;
use futures::future::{BoxFuture, FutureExt, Shared};
use parking_lot::RwLock;
use std::{sync::Arc, time::Duration};
use tokio::{sync::Mutex, time::sleep};

/// Type alias for the shared reconnection future
type ReconnectFuture = Shared<BoxFuture<'static, Result<(), Arc<anyhow::Error>>>>;

/// Manages ETCD client connections with reconnection support
#[derive(Clone)]
pub struct Client {
    /// The actual ETCD client, protected by RwLock for safe updates during reconnection
    /// WARNING: Do not recursively acquire a read lock when the current thread already holds one
    client: Arc<RwLock<etcd_client::Client>>,
    /// Configuration for connecting to ETCD
    etcd_urls: Arc<Vec<String>>,
    connect_options: Arc<Option<ConnectOptions>>,
    /// Tracks the current backoff duration and last successful connect time
    /// The Mutex ensures only one reconnect operation runs at a time
    backoff_state: Arc<Mutex<BackoffState>>,
    /// Shared reconnection futures for deduplication
    /// Only one reconnection happens at a time; concurrent callers share the future
    reconnect_pending: Arc<DashMap<(), ReconnectFuture>>,
}

impl Client {
    /// Create a new connector with an established connection
    pub async fn new(
        etcd_urls: Vec<String>,
        connect_options: Option<ConnectOptions>,
        initial_backoff: Duration,
        min_backoff: Duration,
        max_backoff: Duration,
    ) -> Result<Self> {
        // Connect to ETCD
        let client = Self::connect(&etcd_urls, &connect_options).await?;

        Ok(Self {
            client: Arc::new(RwLock::new(client)),
            etcd_urls: Arc::new(etcd_urls),
            connect_options: Arc::new(connect_options),
            backoff_state: Arc::new(Mutex::new(BackoffState::new(
                initial_backoff,
                min_backoff,
                max_backoff,
            ))),
            reconnect_pending: Arc::new(DashMap::new()),
        })
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

    /// Ensure the client is connected, triggering reconnection if needed.
    ///
    /// This method deduplicates concurrent reconnection attempts - only one
    /// reconnection happens at a time, with all callers sharing the same future.
    ///
    /// # Arguments
    /// * `deadline` - Deadline for reconnection attempts
    /// * `force` - If true, start reconnection even if not already in progress
    ///
    /// Returns Ok(()) if connected, Err if reconnection failed.
    pub async fn ensure_connected(&self, deadline: std::time::Instant, force: bool) -> Result<()> {
        // Check if reconnection already in progress
        if let Some(shared_future_ref) = self.reconnect_pending.get(&()) {
            let shared = shared_future_ref.clone();
            drop(shared_future_ref); // Release DashMap lock before await
            let result = shared.await.map_err(|e| anyhow::anyhow!("{}", e));
            if result.is_err() {
                // Clean up failed future so subsequent calls can retry
                self.reconnect_pending.remove(&());
            }
            return result;
        }

        // If not forced, assume we're connected (lightweight path)
        if !force {
            return Ok(());
        }

        // Start new reconnection (deduplicated)
        use dashmap::mapref::entry::Entry;
        let shared_future = match self.reconnect_pending.entry(()) {
            Entry::Occupied(entry) => {
                // Another thread started reconnection, use their future
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                // We're first, create the shared future
                let client = self.clone();
                let shared = async move { client.reconnect_impl(deadline).await.map_err(Arc::new) }
                    .boxed()
                    .shared();

                entry.insert(shared.clone());
                shared
            }
        };

        let result = shared_future.await.map_err(|e| anyhow::anyhow!("{}", e));
        if result.is_err() {
            // Clean up failed future so subsequent calls can retry
            self.reconnect_pending.remove(&());
        }
        result
    }

    /// Internal implementation of reconnection with retry logic.
    /// Respects the deadline and returns error if exceeded.
    ///
    /// Backoff behavior:
    /// - Starts at 0 (immediate reconnect) if this is the first reconnect or enough time has passed
    ///   since the last reconnect
    /// - Increments exponentially for continuous failures
    /// - Resets to 0 only when: this is a new call AND current_time > last_connect_time + residual_backoff
    ///
    /// The mutex ensures only one reconnect operation runs at a time globally
    async fn reconnect_impl(&self, deadline: std::time::Instant) -> Result<()> {
        let mut backoff_state = self.backoff_state.lock().await;

        tracing::warn!("Reconnecting to ETCD cluster at: {:?}", self.etcd_urls);
        backoff_state.attempt_reset();

        loop {
            backoff_state.apply_backoff(deadline).await;
            if std::time::Instant::now() >= deadline {
                // Clear the pending reconnection before returning error
                self.reconnect_pending.remove(&());
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

                    // Clear the pending reconnection
                    self.reconnect_pending.remove(&());

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
    #[allow(dead_code)]
    pub fn etcd_urls(&self) -> &[String] {
        &self.etcd_urls
    }

    /// Get the connection options
    #[allow(dead_code)]
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
    /// Create a new BackoffState with custom parameters.
    pub fn new(initial_backoff: Duration, min_backoff: Duration, max_backoff: Duration) -> Self {
        Self {
            initial_backoff,
            min_backoff,
            max_backoff,
            current_backoff: Duration::ZERO,
            last_connect_attempt: std::time::Instant::now(),
        }
    }

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
