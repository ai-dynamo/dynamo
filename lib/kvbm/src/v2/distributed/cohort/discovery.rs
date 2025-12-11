// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader discovery mechanisms for cohort workers.
//!
//! This module provides traits and implementations for discovering the leader node
//! in a distributed cohort. Workers use discovery to find and connect to the leader.

use anyhow::Result;
use async_trait::async_trait;
use dynamo_am::api::client::WorkerAddress;

/// Trait for discovering the leader's address.
///
/// Implementations provide different strategies for finding the leader,
/// such as static configuration, environment variables, or service discovery.
#[async_trait]
pub trait LeaderDiscovery: Send + Sync {
    /// Discover the leader's address.
    ///
    /// This method should return the WorkerAddress of the leader,
    /// which can then be used to establish a connection.
    ///
    /// # Returns
    /// WorkerAddress containing the transport locators for the leader
    ///
    /// # Errors
    /// Returns error if discovery fails or leader address cannot be determined
    async fn discover_leader(&self) -> Result<WorkerAddress>;
}

/// Static leader discovery using a pre-configured address.
///
/// This implementation is primarily useful for testing and simple deployments
/// where the leader address is known at startup time.
///
/// # Example
/// ```no_run
/// use dynamo_am::api::client::WorkerAddress;
/// use dynamo_kvbm::v2::distributed::cohort::StaticLeaderDiscovery;
///
/// let address = WorkerAddress::from_endpoint("tcp://127.0.0.1:5555");
/// let discovery = StaticLeaderDiscovery::new(address);
/// ```
#[derive(Debug, Clone)]
pub struct StaticLeaderDiscovery {
    address: WorkerAddress,
}

impl StaticLeaderDiscovery {
    /// Create a new static discovery with the given leader address.
    ///
    /// # Arguments
    /// * `address` - The WorkerAddress of the leader node
    pub fn new(address: WorkerAddress) -> Self {
        Self { address }
    }

    /// Get the configured leader address.
    pub fn address(&self) -> &WorkerAddress {
        &self.address
    }
}

#[async_trait]
impl LeaderDiscovery for StaticLeaderDiscovery {
    async fn discover_leader(&self) -> Result<WorkerAddress> {
        Ok(self.address.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_discovery() {
        let address = WorkerAddress::from_endpoint("tcp://127.0.0.1:5555");
        let discovery = StaticLeaderDiscovery::new(address.clone());

        let discovered = discovery.discover_leader().await.unwrap();
        assert_eq!(discovered.primary_endpoint(), address.primary_endpoint());
    }

    #[tokio::test]
    async fn test_static_discovery_with_ipc() {
        let address = WorkerAddress::from_endpoint("ipc:///tmp/leader.sock");
        let discovery = StaticLeaderDiscovery::new(address.clone());

        let discovered = discovery.discover_leader().await.unwrap();
        assert_eq!(
            discovered.primary_endpoint(),
            Some("ipc:///tmp/leader.sock")
        );
    }
}
