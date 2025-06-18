// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{transports::etcd, Result, DistributedRuntime};

pub use etcd::{Lease, PrefixWatcher};
pub use etcd_client::{GetResponse, GetOptions, PutOptions, DeleteOptions, KeyValue};

/// Storage handle that provides etcd operations scoped to a specific key
/// We are borrowing from distributed runtime, hence the lifetime parameter
pub struct Storage<'a> {
    client: &'a etcd::Client,
    key: String,
}

impl<'a> Storage<'a> {
    /// Create a new lease with specified TTL
    pub async fn create_lease(&self, ttl: i64) -> Result<Lease> {
        self.client.create_lease(ttl).await
    }

    /// Revoke a lease
    pub async fn revoke_lease(&self, lease_id: i64) -> Result<()> {
        self.client.revoke_lease(lease_id).await
    }

    /// Atomically create only if key doesn't exist
    pub async fn create(&self, value: Vec<u8>, lease_id: Option<i64>) -> Result<()> {
        self.client.kv_create(self.key.clone(), value, lease_id).await
    }

    /// Create or validate existing value matches
    pub async fn create_or_validate(&self, value: Vec<u8>, lease_id: Option<i64>) -> Result<()> {
        self.client.kv_create_or_validate(self.key.clone(), value, lease_id).await
    }

    /// Put a value (create or overwrite)
    pub async fn put(&self, value: Vec<u8>, lease_id: Option<i64>) -> Result<()> {
        self.client.kv_put(&self.key, value, lease_id).await
    }

    /// Put with custom options
    pub async fn put_with_options(&self, value: Vec<u8>, options: Option<PutOptions>) -> Result<etcd_client::PutResponse> {
        self.client.kv_put_with_options(&self.key, value, options).await
    }

    /// Get by exact key
    pub async fn get(&self) -> Result<Vec<KeyValue>> {
        self.client.kv_get(self.key.as_bytes(), None).await
    }

    /// Get with options
    pub async fn get_with_options(&self, options: Option<GetOptions>) -> Result<Vec<KeyValue>> {
        self.client.kv_get(self.key.as_bytes(), options).await
    }

    /// Get all with prefix
    pub async fn get_prefix(&self) -> Result<Vec<KeyValue>> {
        self.client.kv_get_prefix(&self.key).await
    }

    /// Delete and return count
    pub async fn delete(&self, options: Option<DeleteOptions>) -> Result<i64> {
        self.client.kv_delete(self.key.as_bytes(), options).await
    }

    /// Get and watch prefix
    pub async fn watch_prefix(&self) -> Result<PrefixWatcher> {
        self.client.kv_get_and_watch_prefix(&self.key).await
    }

    /// Get the key this storage is scoped to
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// Minimal trait for entities that have etcd storage
pub trait DiscoveryClient: crate::traits::DistributedRuntimeProvider {
    /// Get the etcd key for this entity
    fn etcd_key(&self) -> String;

    /// Get a storage handle for this entity's etcd operations
    fn storage(&self) -> Result<Storage> {
        let client = self.drt()
            .etcd_client_internal()
            .ok_or_else(|| anyhow::anyhow!("etcd client not available"))?;

        Ok(Storage {
            client,
            key: self.etcd_key(),
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Runtime, DistributedRuntime, distributed::DistributedConfig};
    use tokio;

    // Test entity that implements DiscoveryClient
    struct TestEntity {
        key: String,
        drt: DistributedRuntime,
    }

    impl crate::traits::DistributedRuntimeProvider for TestEntity {
        fn drt(&self) -> &DistributedRuntime {
            &self.drt
        }
    }

    impl DiscoveryClient for TestEntity {
        fn etcd_key(&self) -> String {
            self.key.clone()
        }
    }

    // Helper to check if ETCD is available
    async fn is_etcd_available() -> bool {
        // Try to connect to default ETCD endpoint
        etcd_client::Client::connect(["localhost:2379"], None).await.is_ok()
    }

    // Helper to create test runtime with ETCD
    async fn create_test_runtime() -> Result<DistributedRuntime> {
        let runtime = Runtime::from_current()?;
        let mut config = DistributedConfig::from_settings(false);
        // Ensure we're using localhost:2379 for tests
        config.etcd_config.etcd_url = vec!["http://localhost:2379".to_string()];
        DistributedRuntime::new(runtime, config).await
    }

    #[tokio::test]
    async fn test_storage_put_get() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let entity = TestEntity {
            key: format!("test/storage/{}", uuid::Uuid::new_v4()),
            drt,
        };

        let storage = entity.storage()?;

        // Test put and get
        let test_data = b"hello world".to_vec();
        storage.put(test_data.clone(), None).await?;

        let values = storage.get().await?;
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].value(), &test_data);

        // Cleanup
        storage.delete(None).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_create_atomic() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let entity = TestEntity {
            key: format!("test/storage/{}", uuid::Uuid::new_v4()),
            drt,
        };

        let storage = entity.storage()?;

        // First create should succeed
        let test_data = b"first".to_vec();
        storage.create(test_data.clone(), None).await?;

        // Second create should fail (key exists)
        let result = storage.create(b"second".to_vec(), None).await;
        assert!(result.is_err());

        // Verify original value unchanged
        let values = storage.get().await?;
        assert_eq!(values[0].value(), &test_data);

        // Cleanup
        storage.delete(None).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_create_or_validate() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let entity = TestEntity {
            key: format!("test/storage/{}", uuid::Uuid::new_v4()),
            drt,
        };

        let storage = entity.storage()?;
        let test_data = b"consistent".to_vec();

        // First call creates
        storage.create_or_validate(test_data.clone(), None).await?;

        // Second call with same data succeeds
        storage.create_or_validate(test_data.clone(), None).await?;

        // Call with different data fails
        let result = storage.create_or_validate(b"different".to_vec(), None).await;
        assert!(result.is_err());

        // Cleanup
        storage.delete(None).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_lease_operations() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let entity = TestEntity {
            key: format!("test/storage/{}", uuid::Uuid::new_v4()),
            drt,
        };

        let storage = entity.storage()?;

        // Create a lease
        let lease = storage.create_lease(5).await?; // 5 second TTL

        // Put with lease
        storage.put(b"leased_data".to_vec(), Some(lease.id())).await?;

        // Verify data exists
        let values = storage.get().await?;
        assert_eq!(values.len(), 1);

        // Revoke lease (should delete the key)
        storage.revoke_lease(lease.id()).await?;

        // Give etcd a moment to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify data is gone
        let values = storage.get().await?;
        assert_eq!(values.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_prefix_operations() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let prefix = format!("test/prefix/{}", uuid::Uuid::new_v4());

        // Create multiple entities with same prefix
        let entities: Vec<TestEntity> = (0..3)
            .map(|i| TestEntity {
                key: format!("{}/item_{}", prefix, i),
                drt: drt.clone(),
            })
            .collect();

        // Put data for each entity
        for (i, entity) in entities.iter().enumerate() {
            let storage = entity.storage()?;
            storage.put(format!("data_{}", i).into_bytes(), None).await?;
        }

        // Use first entity to get all with prefix
        let base_entity = TestEntity {
            key: prefix.clone(),
            drt: drt.clone(),
        };
        let storage = base_entity.storage()?;
        let values = storage.get_prefix().await?;

        assert_eq!(values.len(), 3);

        // Cleanup
        for entity in &entities {
            entity.storage()?.delete(None).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_delete_operations() -> Result<()> {
        if !is_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return Ok(());
        }

        let drt = create_test_runtime().await?;
        let entity = TestEntity {
            key: format!("test/storage/{}", uuid::Uuid::new_v4()),
            drt,
        };

        let storage = entity.storage()?;

        // Put some data
        storage.put(b"to_delete".to_vec(), None).await?;

        // Delete and check count
        let count = storage.delete(None).await?;
        assert_eq!(count, 1);

        // Delete again should return 0
        let count = storage.delete(None).await?;
        assert_eq!(count, 0);

        Ok(())
    }
}
