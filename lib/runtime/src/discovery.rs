// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, transports::etcd};

pub use etcd::Lease;

pub struct DiscoveryClient {
    namespace: String,
    etcd_client: etcd::Client,
}

impl DiscoveryClient {
    /// Create a new [`DiscoveryClient`]
    ///
    /// This will establish a connection to the etcd server, create a primary lease,
    /// and spawn a task to keep the lease alive and tie the lifetime of the [`Runtime`]
    /// to the lease.
    ///
    /// If the lease expires, the [`Runtime`] will be shutdown.
    /// If the [`Runtime`] is shutdown, the lease will be revoked.
    pub(crate) fn new(namespace: String, etcd_client: etcd::Client) -> Self {
        DiscoveryClient {
            namespace,
            etcd_client,
        }
    }

    /// Get the primary lease ID
    pub fn primary_lease_id(&self) -> u64 {
        self.etcd_client.lease_id()
    }

    /// Create a [`Lease`] with a given time-to-live (TTL).
    /// This [`Lease`] will be tied to the [`crate::Runtime`], but has its own independent [`crate::CancellationToken`].
    pub async fn create_lease(&self, ttl: u64) -> Result<Lease> {
        self.etcd_client.create_lease(ttl).await
    }
}
