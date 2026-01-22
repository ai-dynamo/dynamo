// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation execution with automatic retry and reconnection.
//!
//! Wraps etcd operations to handle transient connection failures transparently.

use crate::peer::{DiscoveryError, DiscoveryQueryError};
use crate::systems::etcd::client::Client;
use crate::systems::etcd::error::{EtcdErrorClass, classify_error};
use anyhow::Result;
use futures::future::BoxFuture;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Executes etcd operations with automatic reconnection on transient errors.
#[derive(Clone)]
pub struct OperationExecutor {
    client: Arc<Client>,
    default_timeout: Duration,
    max_retries: u32,
}

impl OperationExecutor {
    /// Create a new operation executor.
    pub fn new(client: Arc<Client>, default_timeout: Duration, max_retries: u32) -> Self {
        Self {
            client,
            default_timeout,
            max_retries,
        }
    }

    /// Execute a query operation with automatic retry on reconnectable errors.
    ///
    /// # Arguments
    ///
    /// * `op` - Function that performs the etcd operation given a client
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - Operation succeeded
    /// * `Err(DiscoveryQueryError::NotFound)` - Key not found (expected)
    /// * `Err(DiscoveryQueryError::Backend)` - Fatal error or timeout
    ///
    /// # Behavior
    ///
    /// 1. Acquire client (brief RwLock read)
    /// 2. Execute operation
    /// 3. On reconnectable error:
    ///    - Trigger reconnection via `ensure_connected()`
    ///    - Retry operation
    /// 4. On NotFound: return DiscoveryQueryError::NotFound
    /// 5. On Fatal error: return DiscoveryQueryError::Backend
    pub async fn execute_query<F, T>(&self, op: F) -> Result<T, DiscoveryQueryError>
    where
        F: Fn(etcd_client::Client) -> BoxFuture<'static, Result<T, etcd_client::Error>>,
    {
        let deadline = Instant::now() + self.default_timeout;
        let mut retry_count = 0;

        loop {
            // Check deadline
            if Instant::now() >= deadline {
                return Err(DiscoveryQueryError::Backend(Arc::new(anyhow::anyhow!(
                    "Operation timed out after {:?}",
                    self.default_timeout
                ))));
            }

            // Await any in-progress reconnection (lightweight check)
            if let Err(e) = self.client.ensure_connected(deadline, false).await {
                return Err(DiscoveryQueryError::Backend(Arc::new(e)));
            }

            // Acquire client (brief lock)
            let client = self.client.get_client();

            // Execute operation
            match op(client).await {
                Ok(result) => {
                    return Ok(result);
                }
                Err(err) => {
                    // Classify the error to determine action
                    match classify_error(err) {
                        EtcdErrorClass::Reconnectable(kind) => {
                            retry_count += 1;
                            if retry_count >= self.max_retries {
                                tracing::error!(
                                    "Max retries ({}) exceeded for reconnectable error: {:?}",
                                    self.max_retries,
                                    kind
                                );
                                return Err(DiscoveryQueryError::Backend(Arc::new(
                                    anyhow::anyhow!("Max retries exceeded: {}", kind),
                                )));
                            }

                            tracing::debug!(
                                "Reconnectable error (attempt {}/{}): {:?}, retrying...",
                                retry_count,
                                self.max_retries,
                                kind
                            );

                            // Trigger reconnection (force=true)
                            if let Err(e) = self.client.ensure_connected(deadline, true).await {
                                tracing::error!("Failed to reconnect: {}", e);
                                return Err(DiscoveryQueryError::Backend(Arc::new(e)));
                            }

                            // Loop will retry operation
                            continue;
                        }
                        EtcdErrorClass::NotFound => {
                            return Err(DiscoveryQueryError::NotFound);
                        }
                        EtcdErrorClass::Fatal(e) => {
                            return Err(DiscoveryQueryError::Backend(Arc::new(e)));
                        }
                    }
                }
            }
        }
    }

    /// Execute a write operation (register/unregister) with automatic retry.
    ///
    /// Similar to `execute_query` but returns `DiscoveryError` instead.
    pub async fn execute_write<F>(&self, op: F) -> Result<(), DiscoveryError>
    where
        F: Fn(etcd_client::Client) -> BoxFuture<'static, Result<(), etcd_client::Error>>,
    {
        let deadline = Instant::now() + self.default_timeout;
        let mut retry_count = 0;

        loop {
            // Check deadline
            if Instant::now() >= deadline {
                return Err(DiscoveryError::Backend(anyhow::anyhow!(
                    "Operation timed out after {:?}",
                    self.default_timeout
                )));
            }

            // Await any in-progress reconnection (lightweight check)
            if let Err(e) = self.client.ensure_connected(deadline, false).await {
                return Err(DiscoveryError::Backend(e));
            }

            // Acquire client (brief lock)
            let client = self.client.get_client();

            // Execute operation
            match op(client).await {
                Ok(()) => {
                    return Ok(());
                }
                Err(err) => {
                    // Classify the error to determine action
                    match classify_error(err) {
                        EtcdErrorClass::Reconnectable(kind) => {
                            retry_count += 1;
                            if retry_count >= self.max_retries {
                                tracing::error!(
                                    "Max retries ({}) exceeded for reconnectable error: {:?}",
                                    self.max_retries,
                                    kind
                                );
                                return Err(DiscoveryError::Backend(anyhow::anyhow!(
                                    "Max retries exceeded: {}",
                                    kind
                                )));
                            }

                            tracing::debug!(
                                "Reconnectable error (attempt {}/{}): {:?}, retrying...",
                                retry_count,
                                self.max_retries,
                                kind
                            );

                            // Trigger reconnection (force=true)
                            if let Err(e) = self.client.ensure_connected(deadline, true).await {
                                tracing::error!("Failed to reconnect: {}", e);
                                return Err(DiscoveryError::Backend(e));
                            }

                            // Loop will retry operation
                            continue;
                        }
                        EtcdErrorClass::NotFound => {
                            // For writes, NotFound might be valid (e.g., deleting non-existent key)
                            // Treat as success
                            tracing::debug!("Write operation: key not found (treating as success)");
                            return Ok(());
                        }
                        EtcdErrorClass::Fatal(e) => {
                            return Err(DiscoveryError::Backend(e));
                        }
                    }
                }
            }
        }
    }

    /// Get the underlying client reference.
    #[allow(dead_code)]
    pub fn client(&self) -> &Arc<Client> {
        &self.client
    }
}
