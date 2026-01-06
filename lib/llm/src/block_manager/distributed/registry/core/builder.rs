// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder patterns for registry components.
//!
//! Provides ergonomic construction of registry clients and hubs with
//! sensible defaults and fluent configuration.

use std::marker::PhantomData;
use std::time::Duration;

use anyhow::Result;

use super::codec::{BinaryCodec, RegistryCodec};
use super::hub::RegistryHub;
use super::hub_transport::HubTransport;
use super::key::RegistryKey;
use super::metadata::RegistryMetadata;
use super::registry::RegistryClient;
use super::storage::Storage;
use super::transport::RegistryTransport;
use super::value::RegistryValue;

/// Builder for constructing a `RegistryClient`.
///
/// # Example
///
/// ```ignore
/// use registry::core::{ClientBuilder, BinaryCodec, NoMetadata};
///
/// let client = ClientBuilder::new(transport, BinaryCodec::new())
///     .batch_size(50)
///     .batch_timeout(Duration::from_millis(20))
///     .build();
/// ```
pub struct ClientBuilder<K, V, M, T, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
    C: RegistryCodec<K, V, M>,
{
    transport: T,
    codec: C,
    batch_size: usize,
    batch_timeout: Duration,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, T, C> ClientBuilder<K, V, M, T, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
    C: RegistryCodec<K, V, M>,
{
    /// Create a new client builder with the given transport and codec.
    pub fn new(transport: T, codec: C) -> Self {
        Self {
            transport,
            codec,
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
            _phantom: PhantomData,
        }
    }

    /// Set the batch size for registrations.
    ///
    /// When this many registrations are pending, they are automatically flushed.
    /// Default: 100
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the batch timeout for registrations.
    ///
    /// Pending registrations are flushed after this duration even if
    /// the batch size hasn't been reached. Default: 10ms
    pub fn batch_timeout(mut self, timeout: Duration) -> Self {
        self.batch_timeout = timeout;
        self
    }

    /// Build the registry client.
    pub fn build(self) -> RegistryClient<K, V, M, T, C> {
        RegistryClient::new(self.transport, self.codec)
            .with_batch_size(self.batch_size)
            .with_batch_timeout(self.batch_timeout)
    }
}

/// Builder for constructing a `RegistryHub`.
///
/// # Example
///
/// ```ignore
/// use registry::core::{HubBuilder, BinaryCodec, HashMapStorage, NoMetadata, HubConfig};
///
/// let hub = HubBuilder::new(storage, BinaryCodec::new())
///     .lease_ttl(Duration::from_secs(60))
///     .build();
/// ```
pub struct HubBuilder<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V>,
    C: RegistryCodec<K, V, M>,
{
    storage: S,
    codec: C,
    lease_ttl: Duration,
    lease_cleanup_interval: Duration,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, S, C> HubBuilder<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V>,
    C: RegistryCodec<K, V, M>,
{
    /// Create a new hub builder with the given storage and codec.
    pub fn new(storage: S, codec: C) -> Self {
        Self {
            storage,
            codec,
            lease_ttl: Duration::from_secs(30),
            lease_cleanup_interval: Duration::from_secs(5),
            _phantom: PhantomData,
        }
    }

    /// Set the lease TTL for can_offload claims.
    ///
    /// Leases expire after this duration if not converted to registrations.
    /// Default: 30 seconds
    pub fn lease_ttl(mut self, ttl: Duration) -> Self {
        self.lease_ttl = ttl;
        self
    }

    /// Set the lease cleanup interval.
    ///
    /// Expired leases are cleaned up at this interval.
    /// Default: 5 seconds
    pub fn lease_cleanup_interval(mut self, interval: Duration) -> Self {
        self.lease_cleanup_interval = interval;
        self
    }

    /// Build the registry hub.
    pub fn build(self) -> RegistryHub<K, V, M, S, C> {
        use super::hub::HubConfig;
        let config = HubConfig {
            lease_ttl: self.lease_ttl,
            lease_cleanup_interval: self.lease_cleanup_interval,
        };
        RegistryHub::with_config(self.storage, self.codec, config)
    }

    /// Build and serve the hub with the given transport.
    ///
    /// This consumes the builder and starts serving requests.
    pub async fn serve<T: HubTransport>(self, transport: &mut T) -> Result<()> {
        let hub = self.build();
        hub.serve(transport).await
    }
}

/// Convenience function to create a client builder with binary codec.
///
/// # Example
///
/// ```ignore
/// let client = client(transport)
///     .batch_size(50)
///     .build();
/// ```
pub fn client<K, V, M, T>(transport: T) -> ClientBuilder<K, V, M, T, BinaryCodec<K, V, M>>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    T: RegistryTransport,
{
    ClientBuilder::new(transport, BinaryCodec::new())
}

/// Convenience function to create a hub builder with binary codec.
///
/// # Example
///
/// ```ignore
/// let hub = hub(storage)
///     .lease_ttl(Duration::from_secs(60))
///     .build();
/// ```
pub fn hub<K, V, M, S>(storage: S) -> HubBuilder<K, V, M, S, BinaryCodec<K, V, M>>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V>,
{
    HubBuilder::new(storage, BinaryCodec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::{
        HashMapStorage, InProcessTransport, NoMetadata, OffloadStatus, QueryType, Registry,
        ResponseType,
    };

    #[tokio::test]
    async fn test_client_builder() {
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let (transport, _rx) = InProcessTransport::new(move |data| {
            let query = codec.decode_query(data);
            match query {
                Some(QueryType::CanOffload(keys)) => {
                    let statuses: Vec<_> = keys.iter().map(|_| OffloadStatus::Granted).collect();
                    let mut buf = Vec::new();
                    codec
                        .encode_response(&ResponseType::CanOffload(statuses), &mut buf)
                        .unwrap();
                    buf
                }
                _ => Vec::new(),
            }
        });

        let built_client: RegistryClient<u64, u64, NoMetadata, _, _> = client(transport)
            .batch_size(50)
            .batch_timeout(Duration::from_millis(20))
            .build();

        let result = built_client.can_offload(&[1, 2, 3]).await.unwrap();
        assert_eq!(result.can_offload.len(), 3);
    }

    #[test]
    fn test_hub_builder() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();

        let built_hub: RegistryHub<u64, u64, NoMetadata, _, _> = hub(storage)
            .lease_ttl(Duration::from_secs(60))
            .lease_cleanup_interval(Duration::from_secs(10))
            .build();

        assert!(built_hub.is_empty());

        // Verify lease TTL was set
        assert_eq!(built_hub.lease_manager().ttl(), Duration::from_secs(60));
    }

    #[test]
    fn test_builder_with_custom_codec() {
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        let built_hub = HubBuilder::new(storage, codec)
            .lease_ttl(Duration::from_secs(120))
            .build();

        assert_eq!(built_hub.lease_manager().ttl(), Duration::from_secs(120));
    }
}
