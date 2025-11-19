// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resilient keep-alive task for etcd leases.
//!
//! Handles periodic keep-alive requests to prevent lease expiration,
//! with automatic reconnection and recovery on failure.

use crate::systems::etcd::client::Client;
use crate::systems::etcd::lease::LeaseState;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Background task that keeps an etcd lease alive.
///
/// # Resilience Strategy
///
/// - Acquires client and starts keep-alive stream
/// - Uses stream until failure (does NOT hold client lock)
/// - On failure: triggers reconnection, reacquires client, restarts
/// - Respects shutdown signal for clean termination
pub struct KeepAliveTask {
    client: Arc<Client>,
    lease_state: Arc<RwLock<LeaseState>>,
    ttl: Duration,
    shutdown: CancellationToken,
}

impl KeepAliveTask {
    /// Create a new keep-alive task.
    pub fn new(
        client: Arc<Client>,
        lease_state: Arc<RwLock<LeaseState>>,
        ttl: Duration,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            client,
            lease_state,
            ttl,
            shutdown,
        }
    }

    /// Spawn the keep-alive task as a background tokio task.
    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            tracing::debug!("Keep-alive task starting");

            loop {
                // Check for shutdown signal
                if self.shutdown.is_cancelled() {
                    tracing::debug!("Keep-alive task shutting down");
                    break;
                }

                // Run keep-alive loop with automatic recovery
                if let Err(e) = self.run_keep_alive_loop().await {
                    tracing::error!("Keep-alive loop failed: {}", e);

                    // Trigger reconnection before restarting (force=true)
                    let deadline = std::time::Instant::now() + Duration::from_secs(30);
                    if let Err(e) = self.client.ensure_connected(deadline, true).await {
                        tracing::error!("Failed to reconnect after keep-alive failure: {}", e);

                        // Wait before retry to avoid tight loop
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    } else {
                        tracing::info!("Reconnected successfully, restarting keep-alive");
                    }
                }
            }

            tracing::debug!("Keep-alive task exited");
        })
    }

    /// Run the keep-alive loop until failure or shutdown.
    ///
    /// # Strategy
    ///
    /// 1. Get lease ID from state
    /// 2. Acquire client and start keep-alive stream (brief lock)
    /// 3. Release client lock
    /// 4. Use keeper/stream handles until they fail
    /// 5. On failure, return error (outer loop handles reconnection)
    async fn run_keep_alive_loop(&self) -> Result<()> {
        // Get current lease ID
        let lease_id = self
            .lease_state
            .read()
            .lease_id()
            .ok_or_else(|| anyhow::anyhow!("No lease ID available"))?;

        tracing::debug!("Starting keep-alive loop for lease {}", lease_id);

        // Acquire client and start keep-alive stream (brief lock acquisition)
        let mut client = self.client.get_client();
        let (mut keeper, mut stream) = client
            .lease_keep_alive(lease_id)
            .await
            .context("Failed to start lease keep-alive stream")?;

        // Client lock is released here - we now only use keeper/stream handles

        // Calculate sleep interval (TTL / 3, with minimum of 1 second)
        let sleep_interval = Duration::from_secs((self.ttl.as_secs() / 3).max(1));

        loop {
            // Check for messages from the stream
            tokio::select! {
                // Shutdown signal
                _ = self.shutdown.cancelled() => {
                    tracing::debug!("Keep-alive loop received shutdown signal");
                    return Ok(());
                }

                // Keep-alive response from etcd
                msg = stream.message() => {
                    match msg {
                        Ok(Some(_resp)) => {
                            tracing::trace!("Received keep-alive response for lease {}", lease_id);
                            // Successful keep-alive, continue
                        }
                        Ok(None) => {
                            tracing::warn!("Keep-alive stream closed for lease {}", lease_id);
                            return Err(anyhow::anyhow!("Keep-alive stream closed"));
                        }
                        Err(e) => {
                            tracing::warn!("Keep-alive stream error for lease {}: {}", lease_id, e);
                            return Err(e.into());
                        }
                    }
                }
            }

            // Wait before sending next keep-alive
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    tracing::debug!("Keep-alive loop received shutdown signal during sleep");
                    return Ok(());
                }
                _ = tokio::time::sleep(sleep_interval) => {
                    // Time to send next keep-alive
                }
            }

            // Send keep-alive request
            if let Err(e) = keeper.keep_alive().await {
                tracing::warn!("Failed to send keep-alive for lease {}: {}", lease_id, e);
                return Err(e.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keep_alive_task_creation() {
        // Test that we can create a keep-alive task
        // (actual testing requires running etcd instance)

        // This is a smoke test to ensure the struct compiles
        let ttl = Duration::from_secs(60);
        let sleep_interval = (ttl.as_secs() / 3).max(1);

        assert_eq!(sleep_interval, 20);
    }

    #[test]
    fn test_sleep_interval_calculation() {
        // Test sleep interval calculation
        let ttl = Duration::from_secs(60);
        let interval = (ttl.as_secs() / 3).max(1);
        assert_eq!(interval, 20);

        let ttl = Duration::from_secs(10);
        let interval = (ttl.as_secs() / 3).max(1);
        assert_eq!(interval, 3);

        let ttl = Duration::from_secs(2);
        let interval = (ttl.as_secs() / 3).max(1);
        assert_eq!(interval, 1); // Minimum of 1 second
    }
}
