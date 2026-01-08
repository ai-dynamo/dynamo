// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry hub (server-side) implementation.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use tracing::{debug, warn};

use super::codec::{OffloadStatus, QueryType, RegistryCodec, ResponseType};
use super::hub_transport::{ClientId, HubMessage, HubTransport};
use super::key::RegistryKey;
use super::lease::{LeaseManager, LeaseStats};
use super::metadata::RegistryMetadata;
use super::storage::Storage;
use super::value::RegistryValue;

/// Statistics for the registry hub.
#[derive(Debug, Clone, Default)]
pub struct HubStats {
    pub queries_received: u64,
    pub registrations_received: u64,
    pub entries_stored: u64,
    pub lease_stats: LeaseStats,
}

/// Configuration for the registry hub.
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Lease TTL for can_offload claims.
    pub lease_ttl: Duration,
    /// Interval for cleaning up expired leases.
    pub lease_cleanup_interval: Duration,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            lease_ttl: Duration::from_secs(30),
            lease_cleanup_interval: Duration::from_secs(5),
        }
    }
}

/// Registry hub server.
///
/// Handles incoming queries and registrations from clients.
/// Now includes lease-based claiming to prevent race conditions.
pub struct RegistryHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V>,
    C: RegistryCodec<K, V, M>,
{
    storage: S,
    codec: C,
    lease_manager: Arc<LeaseManager<K>>,
    queries_received: AtomicU64,
    registrations_received: AtomicU64,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, S, C> RegistryHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V>,
    C: RegistryCodec<K, V, M>,
{
    /// Create a new registry hub with the given storage and codec.
    pub fn new(storage: S, codec: C) -> Self {
        Self::with_config(storage, codec, HubConfig::default())
    }

    /// Create a new registry hub with custom configuration.
    pub fn with_config(storage: S, codec: C, config: HubConfig) -> Self {
        Self {
            storage,
            codec,
            lease_manager: Arc::new(LeaseManager::new(config.lease_ttl)),
            queries_received: AtomicU64::new(0),
            registrations_received: AtomicU64::new(0),
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the lease manager.
    pub fn lease_manager(&self) -> &Arc<LeaseManager<K>> {
        &self.lease_manager
    }

    /// Get current statistics.
    pub fn stats(&self) -> HubStats {
        HubStats {
            queries_received: self.queries_received.load(Ordering::Relaxed),
            registrations_received: self.registrations_received.load(Ordering::Relaxed),
            entries_stored: self.storage.len() as u64,
            lease_stats: self.lease_manager.stats(),
        }
    }

    /// Get the number of entries in storage.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Insert directly into storage (for testing/seeding).
    pub fn storage_insert(&self, key: K, value: V) {
        self.storage.insert(key, value);
    }

    /// Generate a client ID from ClientId bytes.
    fn client_id_to_u64(client: &ClientId) -> u64 {
        let bytes = client.as_bytes();
        if bytes.len() >= 8 {
            u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8]))
        } else {
            // Hash short client IDs
            let mut hash: u64 = 0;
            for (i, &b) in bytes.iter().enumerate() {
                hash ^= (b as u64) << ((i % 8) * 8);
            }
            hash
        }
    }

    /// Process a single message.
    async fn process_message<T: HubTransport>(
        &self,
        transport: &mut T,
        message: HubMessage,
    ) -> Result<()> {
        match message {
            HubMessage::Query { client, data } => {
                self.queries_received.fetch_add(1, Ordering::Relaxed);

                let client_id = Self::client_id_to_u64(&client);
                let response = self.handle_query(&data, client_id);
                transport.respond(&client, &response).await?;
            }
            HubMessage::Publish { data } => {
                self.registrations_received.fetch_add(1, Ordering::Relaxed);
                self.handle_registration(&data);
            }
        }
        Ok(())
    }

    /// Handle a query message.
    fn handle_query(&self, data: &[u8], client_id: u64) -> Vec<u8> {
        let mut response_buf = Vec::new();

        let Some(query) = self.codec.decode_query(data) else {
            warn!("Failed to decode query");
            return response_buf;
        };

        match query {
            QueryType::CanOffload(keys) => {
                let statuses: Vec<_> = keys
                    .iter()
                    .map(|k| {
                        // Check if already stored
                        if self.storage.contains(k) {
                            return OffloadStatus::AlreadyStored;
                        }

                        // Try to acquire a lease
                        match self.lease_manager.try_acquire(*k, client_id) {
                            Some(_) => OffloadStatus::Granted,
                            None => OffloadStatus::Leased,
                        }
                    })
                    .collect();

                debug!(
                    num_keys = keys.len(),
                    granted = statuses
                        .iter()
                        .filter(|s| **s == OffloadStatus::Granted)
                        .count(),
                    already_stored = statuses
                        .iter()
                        .filter(|s| **s == OffloadStatus::AlreadyStored)
                        .count(),
                    leased = statuses
                        .iter()
                        .filter(|s| **s == OffloadStatus::Leased)
                        .count(),
                    "Processed can_offload query"
                );

                if let Err(e) = self
                    .codec
                    .encode_response(&ResponseType::CanOffload(statuses), &mut response_buf)
                {
                    warn!("Failed to encode response: {}", e);
                }
            }
            QueryType::Match(keys) => {
                let entries: Vec<_> = keys
                    .iter()
                    .filter_map(|k| self.storage.get(k).map(|v| (*k, v, M::default())))
                    .collect();

                debug!(
                    requested = keys.len(),
                    matched = entries.len(),
                    "Processed match query"
                );

                if let Err(e) = self
                    .codec
                    .encode_response(&ResponseType::Match(entries), &mut response_buf)
                {
                    warn!("Failed to encode response: {}", e);
                }
            }
        }

        response_buf
    }

    /// Handle a registration message.
    fn handle_registration(&self, data: &[u8]) {
        let Some(entries) = self.codec.decode_register(data) else {
            warn!("Failed to decode registration");
            return;
        };

        let count = entries.len();
        for (key, value, _metadata) in entries {
            // Release any lease for this key
            self.lease_manager.release(&key);
            // Store the entry
            self.storage.insert(key, value);
        }

        debug!(
            registered = count,
            total_entries = self.storage.len(),
            "Processed registration"
        );
    }

    /// Serve requests until an error occurs or shutdown.
    ///
    /// This is the main loop for the hub.
    pub async fn serve<T: HubTransport>(&self, transport: &mut T) -> Result<()> {
        tracing::info!(transport = transport.name(), "Registry hub starting");

        loop {
            match transport.recv().await {
                Ok(message) => {
                    if let Err(e) = self.process_message(transport, message).await {
                        warn!(error = %e, "Error processing message");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Transport error, shutting down");
                    return Err(e);
                }
            }
        }
    }

    /// Process a single message (for testing or custom loops).
    pub async fn process_one<T: HubTransport>(&self, transport: &mut T) -> Result<()> {
        let message = transport.recv().await?;
        self.process_message(transport, message).await
    }
}

/// Create a simple hub with HashMap storage and binary codec.
pub fn simple_hub<K, V, M>()
-> RegistryHub<K, V, M, super::storage::HashMapStorage<K, V>, super::codec::BinaryCodec<K, V, M>>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
{
    RegistryHub::new(
        super::storage::HashMapStorage::new(),
        super::codec::BinaryCodec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::hub_transport::InProcessHubTransport;
    use crate::block_manager::distributed::registry::core::{
        BinaryCodec, HashMapStorage, NoMetadata,
    };

    #[tokio::test]
    async fn test_hub_can_offload() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);

        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let hub = RegistryHub::new(storage, codec);

        let (mut transport, handle) = InProcessHubTransport::new();

        // Spawn hub to handle one request
        let hub_task = tokio::spawn(async move {
            hub.process_one(&mut transport).await.unwrap();
            hub
        });

        // Client query
        let client_codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let mut query_buf = Vec::new();
        client_codec
            .encode_query(&QueryType::CanOffload(vec![1, 2, 3, 4]), &mut query_buf)
            .unwrap();

        let response = handle.request(&query_buf).await.unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> =
            client_codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses.len(), 4);
                assert_eq!(statuses[0], OffloadStatus::AlreadyStored); // 1 exists
                assert_eq!(statuses[1], OffloadStatus::AlreadyStored); // 2 exists
                assert_eq!(statuses[2], OffloadStatus::Granted); // 3 doesn't exist
                assert_eq!(statuses[3], OffloadStatus::Granted); // 4 doesn't exist
            }
            _ => panic!("Wrong response type"),
        }

        let hub = hub_task.await.unwrap();
        assert_eq!(hub.stats().queries_received, 1);
    }

    #[tokio::test]
    async fn test_hub_lease_conflict() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let hub = RegistryHub::new(storage, codec);

        // First client acquires lease for key 1
        hub.lease_manager.try_acquire(1, 100);

        // Create transport with different client ID
        let (mut transport, handle) = InProcessHubTransport::new();

        let hub_task = tokio::spawn(async move {
            hub.process_one(&mut transport).await.unwrap();
            hub
        });

        let client_codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let mut query_buf = Vec::new();
        client_codec
            .encode_query(&QueryType::CanOffload(vec![1, 2]), &mut query_buf)
            .unwrap();

        let response = handle.request(&query_buf).await.unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> =
            client_codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses.len(), 2);
                // Key 1 should be Leased (held by client 100)
                assert_eq!(statuses[0], OffloadStatus::Leased);
                // Key 2 should be Granted (new lease for this client)
                assert_eq!(statuses[1], OffloadStatus::Granted);
            }
            _ => panic!("Wrong response type"),
        }

        hub_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_hub_registration_releases_lease() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let hub = RegistryHub::new(storage, codec);

        // Client acquires lease
        hub.lease_manager.try_acquire(1, 100);
        assert!(hub.lease_manager.is_leased(&1));

        let (mut transport, handle) = InProcessHubTransport::new();

        let hub_task = tokio::spawn(async move {
            hub.process_one(&mut transport).await.unwrap();
            hub
        });

        // Register the key
        let client_codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let mut reg_buf = Vec::new();
        client_codec
            .encode_register(&[(1, 100, NoMetadata)], &mut reg_buf)
            .unwrap();

        handle.publish(&reg_buf).unwrap();

        let hub = hub_task.await.unwrap();

        // Lease should be released and key stored
        assert!(!hub.lease_manager.is_leased(&1));
        assert!(hub.storage.contains(&1));
    }

    #[tokio::test]
    async fn test_hub_match() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);
        storage.insert(3, 300);

        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let hub = RegistryHub::new(storage, codec);

        let (mut transport, handle) = InProcessHubTransport::new();

        let hub_task = tokio::spawn(async move {
            hub.process_one(&mut transport).await.unwrap();
            hub
        });

        let client_codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let mut query_buf = Vec::new();
        client_codec
            .encode_query(&QueryType::Match(vec![1, 2, 5]), &mut query_buf)
            .unwrap();

        let response = handle.request(&query_buf).await.unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> =
            client_codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::Match(entries) => {
                assert_eq!(entries.len(), 2); // Only 1 and 2 exist
                assert_eq!(entries[0], (1, 100, NoMetadata));
                assert_eq!(entries[1], (2, 200, NoMetadata));
            }
            _ => panic!("Wrong response type"),
        }

        hub_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_hub_registration() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let hub = RegistryHub::new(storage, codec);

        assert!(hub.is_empty());

        let (mut transport, handle) = InProcessHubTransport::new();

        let hub_task = tokio::spawn(async move {
            hub.process_one(&mut transport).await.unwrap();
            hub
        });

        let client_codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();
        let mut reg_buf = Vec::new();
        client_codec
            .encode_register(&[(1, 100, NoMetadata), (2, 200, NoMetadata)], &mut reg_buf)
            .unwrap();

        handle.publish(&reg_buf).unwrap();

        let hub = hub_task.await.unwrap();
        assert_eq!(hub.len(), 2);
        assert_eq!(hub.stats().registrations_received, 1);
    }
}
