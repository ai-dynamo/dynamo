// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lease management for etcd peer discovery.
//!
//! Handles lease creation, validation, and renewal. Attempts to reuse
//! existing leases when reconnecting to avoid unnecessary re-registration.
//!
//! # Lease Revocation Limitation
//!
//! **IMPORTANT**: If an etcd lease is revoked (either manually or due to
//! network partition), all keys associated with that lease are automatically
//! deleted by etcd. This is an **unrecoverable** state in the current
//! implementation because:
//!
//! 1. The system does not track which keys were published under a lease
//! 2. When a lease is revoked, we create a new lease but cannot republish
//!    the deleted keys
//! 3. All peer registrations made with the old lease are permanently lost
//!
//! **Mitigation**: The keep-alive mechanism maintains the lease actively,
//! reducing the chance of expiration. However, extended network partitions
//! or manual lease revocation will result in lost registrations that require
//! application-level re-registration.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tonic::Code;

/// Result of checking lease validity.
///
/// Provides clear information about why a lease is valid or invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseValidityState {
    /// Lease is valid with the specified remaining TTL in seconds
    Valid { remaining_ttl: i64 },
    /// Lease has expired (TTL <= minimum threshold)
    Expired,
    /// Lease was not found on the etcd server
    NotFound,
    /// Failed to check lease validity (network error, etc.)
    CheckFailed(String),
}

impl LeaseValidityState {
    /// Returns true if the lease is valid and can be reused.
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        matches!(self, LeaseValidityState::Valid { .. })
    }
}

/// State tracking for an etcd lease.
#[derive(Debug)]
pub struct LeaseState {
    /// Current lease ID, if one exists
    lease_id: Option<i64>,
    /// When the lease was created
    created_at: Option<Instant>,
    /// Lease TTL duration
    ttl: Duration,
}

impl LeaseState {
    /// Create a new lease state with the specified TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            lease_id: None,
            created_at: None,
            ttl,
        }
    }

    /// Get the current lease ID if one exists.
    pub fn lease_id(&self) -> Option<i64> {
        self.lease_id
    }

    /// Get the lease TTL.
    #[allow(dead_code)]
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Ensure a valid lease exists, reusing the current one if still valid
    /// or creating a new one if expired/not found.
    ///
    /// # Strategy
    ///
    /// 1. If we have a lease ID, check if it's still valid (TTL > 1/3 remaining)
    /// 2. If valid, return the existing lease ID
    /// 3. If invalid or not found, create a new lease
    ///
    /// This allows us to survive transient disconnections without losing
    /// our registrations, while still creating a new lease if needed.
    pub async fn ensure_lease(&mut self, client: &mut etcd_client::Client) -> Result<i64> {
        // Try to reuse existing lease if it's still valid
        if let Some(lease_id) = self.lease_id {
            match self.check_lease_validity(client, lease_id).await {
                LeaseValidityState::Valid { remaining_ttl } => {
                    tracing::debug!(
                        "Reusing existing lease ID: {} (remaining TTL: {}s)",
                        lease_id,
                        remaining_ttl
                    );
                    return Ok(lease_id);
                }
                LeaseValidityState::Expired => {
                    tracing::debug!("Existing lease {} expired, creating new lease", lease_id);
                }
                LeaseValidityState::NotFound => {
                    // CRITICAL: When a lease is not found (revoked), all keys associated
                    // with it are already deleted by etcd. Creating a new lease will NOT
                    // restore those keys. This is an unrecoverable state - the caller
                    // must re-register all instances.
                    tracing::warn!(
                        "Existing lease {} not found on server (revoked). All keys associated \
                         with this lease have been deleted. Creating new lease - caller must \
                         re-register instances.",
                        lease_id
                    );
                }
                LeaseValidityState::CheckFailed(err) => {
                    tracing::warn!(
                        "Failed to check lease {} validity: {}, creating new lease",
                        lease_id,
                        err
                    );
                }
            }
        }

        // Create new lease
        self.create_new_lease(client).await
    }

    /// Check if a lease is still valid (has > 1/3 of TTL remaining).
    ///
    /// Returns a `LeaseValidityState` that provides clear information about
    /// the lease status.
    async fn check_lease_validity(
        &self,
        client: &mut etcd_client::Client,
        lease_id: i64,
    ) -> LeaseValidityState {
        // Try to get lease TTL
        let resp = match client.lease_time_to_live(lease_id, None).await {
            Ok(resp) => resp,
            Err(e) => {
                // Use structured error matching instead of fragile string matching
                match e {
                    etcd_client::Error::GRpcStatus(status) => {
                        match status.code() {
                            Code::NotFound => {
                                // Lease not found on the server
                                return LeaseValidityState::NotFound;
                            }
                            _ => {
                                // Other gRPC errors
                                return LeaseValidityState::CheckFailed(format!(
                                    "gRPC error: {} (code: {:?})",
                                    status.message(),
                                    status.code()
                                ));
                            }
                        }
                    }
                    _ => {
                        // Non-gRPC errors (transport, IO, etc.)
                        return LeaseValidityState::CheckFailed(e.to_string());
                    }
                }
            }
        };

        let remaining_ttl = resp.ttl();

        // TTL of 0 or negative means the lease is already gone
        if remaining_ttl <= 0 {
            return LeaseValidityState::NotFound;
        }

        // Consider lease valid if it has more than 1/3 of original TTL remaining
        let min_ttl = (self.ttl.as_secs() as i64) / 3;

        if remaining_ttl > min_ttl {
            LeaseValidityState::Valid { remaining_ttl }
        } else {
            LeaseValidityState::Expired
        }
    }

    /// Create a new lease with the configured TTL.
    async fn create_new_lease(&mut self, client: &mut etcd_client::Client) -> Result<i64> {
        let ttl_secs = self.ttl.as_secs() as i64;

        let resp = client
            .lease_grant(ttl_secs, None)
            .await
            .context("Failed to create new lease")?;

        let lease_id = resp.id();

        tracing::info!("Created new lease ID: {} (TTL: {}s)", lease_id, ttl_secs);

        self.lease_id = Some(lease_id);
        self.created_at = Some(Instant::now());

        Ok(lease_id)
    }

    /// Clear the current lease state (e.g., after failed reconnection).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.lease_id = None;
        self.created_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lease_state_creation() {
        let ttl = Duration::from_secs(60);
        let state = LeaseState::new(ttl);

        assert_eq!(state.lease_id(), None);
        assert_eq!(state.ttl(), ttl);
    }

    #[test]
    fn test_lease_state_clear() {
        let mut state = LeaseState::new(Duration::from_secs(60));
        state.lease_id = Some(12345);
        state.created_at = Some(Instant::now());

        state.clear();

        assert_eq!(state.lease_id(), None);
        assert_eq!(state.created_at, None);
    }
}
