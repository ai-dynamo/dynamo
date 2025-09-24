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
    pub fn primary_lease_id(&self) -> i64 {
        self.etcd_client.lease_id()
    }

    /// Create a [`Lease`] with a given time-to-live (TTL).
    /// This [`Lease`] will be tied to the [`crate::Runtime`], but has its own independent [`crate::CancellationToken`].
    pub async fn create_lease(&self, ttl: i64) -> Result<Lease> {
        self.etcd_client.create_lease(ttl).await
    }

    // the following two commented out codes are not implemented, but are placeholders for proposed ectd usage patterns

    // /// Create an ephemeral key/value pair tied to a lease_id.
    // /// This is an atomic create. If the key already exists, this will fail.
    // /// The [`etcd_client::KeyValue`] will be removed when the lease expires or is revoked.
    // pub async fn create_ephemerial_key(&self, key: &str, value: &str, lease_id: i64) -> Result<()> {
    //     // self.etcd_client.create_ephemeral_key(key, value, lease_id).await
    //     unimplemented!()
    // }

    // /// Create a shared [`etcd_client::KeyValue`] which behaves similar to a C++ `std::shared_ptr` or a
    // /// Rust [std::sync::Arc]. Instead of having one owner of the lease, multiple owners participate in
    // /// maintaining the lease. In this manner, when the last member of the group sharing the lease is gone,
    // /// the lease will be expired.
    // ///
    // /// Implementation notes: At the time of writing, it is unclear if we have atomics that control leases,
    // /// so in our initial implementation, the last member of the group will not revoke the lease, so the object
    // /// will live for upto the TTL after the last member is gone.
    // ///
    // /// Notes
    // /// -----
    // ///
    // /// - Multiple members sharing the lease and contributing to the heartbeat might cause some overheads.
    // ///   The implementation will try to randomize the heartbeat intervals to avoid thundering herd problem,
    // ///   and with any luck, the heartbeat watchers will be able to detect when if a external member triggered
    // ///   the heartbeat checking this interval and skip unnecessary heartbeat messages.
    // ///
    // /// A new lease will be created for this object. If you wish to add an object to a shared group s
    // ///
    // /// The [`etcd_client::KeyValue`] will be removed when the lease expires or is revoked.
    // pub async fn create_shared_key(&self, key: &str, value: &str, lease_id: i64) -> Result<()> {
    //     // self.etcd_client.create_ephemeral_key(key, value, lease_id).await
    //     unimplemented!()
    // }
}
