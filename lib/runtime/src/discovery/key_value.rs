// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{DiscoveryClient, DiscoveryEvent, DiscoveryInstance, DiscoveryKey, DiscoveryStream};
use crate::Result;
use crate::storage::key_value_store::{Key, KeyValueStoreManager, WatchEvent};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashSet;
use std::sync::Arc;

const DISCOVERY_BUCKET_PREFIX: &str = "disco";

/// Discovery client implementation backed by a key-value store
///
/// # Design
/// This implementation uses a **bucket-per-endpoint** strategy:
/// - Each endpoint gets its own bucket: `dynamo/discovery/{namespace}/{component}/{endpoint}`
/// - Keys within each bucket are just the `instance_id`
pub struct KeyValueDiscoveryClient {
    instance_id: u64,
    store: Arc<KeyValueStoreManager>,
}

impl KeyValueDiscoveryClient {
    /// Create a new KeyValueDiscoveryClient
    ///
    /// # Arguments
    /// * `instance_id` - Unique identifier for this worker instance (typically connection_id/lease_id)
    /// * `store` - The key-value store manager to use for storage
    pub fn new(instance_id: u64, store: Arc<KeyValueStoreManager>) -> Self {
        Self { instance_id, store }
    }

    /// Generate bucket name for a discovery key
    /// Each endpoint gets its own bucket: dynamo/discovery/{namespace}/{component}/{endpoint}
    fn bucket_name_for_key(key: &DiscoveryKey) -> String {
        match key {
            DiscoveryKey::Endpoint {
                namespace,
                component,
                endpoint,
            } => {
                format!(
                    "{}/{}/{}/{}",
                    DISCOVERY_BUCKET_PREFIX, namespace, component, endpoint
                )
            }
        }
    }

    /// Generate a storage key for an instance (just the instance_id within the bucket)
    fn instance_key(instance_id: u64) -> Key {
        Key::new(&instance_id.to_string())
    }

    /// Parse a stored instance from the key-value store
    fn parse_instance(_key_str: &str, value: &[u8]) -> Result<DiscoveryInstance> {
        let instance: DiscoveryInstance = serde_json::from_slice(value)?;
        Ok(instance)
    }
}

#[async_trait]
impl DiscoveryClient for KeyValueDiscoveryClient {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn serve(&self, key: DiscoveryKey) -> Result<DiscoveryInstance> {
        let instance = match &key {
            DiscoveryKey::Endpoint {
                namespace,
                component,
                endpoint,
            } => DiscoveryInstance::Endpoint {
                namespace: namespace.clone(),
                component: component.clone(),
                endpoint: endpoint.clone(),
                instance_id: self.instance_id,
            },
        };

        // Store the instance in the key-value store
        let bucket_name = Self::bucket_name_for_key(&key);
        let storage_key = Self::instance_key(self.instance_id);
        let json = serde_json::to_string(&instance)?;

        let bucket = self
            .store
            .get_or_create_bucket(&bucket_name, None)
            .await
            .map_err(|e| crate::error!("Failed to get discovery bucket: {}", e))?;

        bucket
            .insert(&storage_key, &json, 0)
            .await
            .map_err(|e| crate::error!("Failed to insert discovery instance: {}", e))?;

        tracing::debug!(
            instance_id = %self.instance_id,
            bucket = %bucket_name,
            key = %storage_key,
            "Registered discovery instance"
        );

        Ok(instance)
    }

    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream> {
        let store = self.store.clone();
        let bucket_name = Self::bucket_name_for_key(&key);

        // This cancel token is to satisfy the method args, but it isn't used
        let cancel_token = crate::CancellationToken::new();

        let stream = async_stream::stream! {
            let (_watch_task, mut rx) = store.watch(
                &bucket_name,
                None,  // No TTL
                cancel_token
            );

            // Track known instances to handle deduplication
            let mut known_instances: HashSet<u64> = HashSet::new();

            while let Some(event) = rx.recv().await {
                match event {
                    WatchEvent::Put(kv) => {
                        let key_str = kv.key_str();
                        match Self::parse_instance(key_str, kv.value()) {
                            Ok(instance) => {
                                let instance_id = match &instance {
                                    DiscoveryInstance::Endpoint { instance_id, .. } => *instance_id,
                                };
                                if known_instances.insert(instance_id) {
                                    tracing::debug!(
                                        instance_id = %instance_id,
                                        key = %key_str,
                                        bucket = %bucket_name,
                                        "Discovery instance added"
                                    );
                                    yield Ok(DiscoveryEvent::Added(instance));
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    key = %key_str,
                                    error = %e,
                                    "Failed to parse discovery instance"
                                );
                            }
                        }
                    }
                    WatchEvent::Delete(kv) => {
                        let key_str = kv.key_str();
                        match Self::parse_instance(key_str, kv.value()) {
                            Ok(instance) => {
                                let instance_id = match &instance {
                                    DiscoveryInstance::Endpoint { instance_id, .. } => *instance_id,
                                };
                                if known_instances.remove(&instance_id) {
                                    tracing::debug!(
                                        instance_id = %instance_id,
                                        key = %key_str,
                                        bucket = %bucket_name,
                                        "Discovery instance removed"
                                    );
                                    yield Ok(DiscoveryEvent::Removed(instance_id));
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    key = %key_str,
                                    error = %e,
                                    "Failed to parse discovery instance from delete event"
                                );
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::key_value_store::MemoryStore;

    #[tokio::test]
    async fn test_key_value_discovery_serve_and_list() {
        let store = Arc::new(KeyValueStoreManager::memory());
        let client = KeyValueDiscoveryClient::new(1, store);

        let key = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
        };

        // Register an instance
        let instance = client.serve(key.clone()).await.unwrap();

        match instance {
            DiscoveryInstance::Endpoint {
                namespace,
                component,
                endpoint,
                instance_id,
            } => {
                assert_eq!(namespace, "test-ns");
                assert_eq!(component, "test-comp");
                assert_eq!(endpoint, "test-ep");
                assert_eq!(instance_id, 1);
            }
        }

        // Watch should emit the instance
        let mut stream = client.list_and_watch(key).await.unwrap();
        let event = stream.next().await.unwrap().unwrap();

        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { instance_id, .. }) => {
                assert_eq!(instance_id, 1);
            }
            _ => panic!("Expected Added event for instance-1"),
        }
    }

    #[tokio::test]
    async fn test_key_value_discovery_multiple_instances() {
        let store = Arc::new(KeyValueStoreManager::memory());
        let client1 = KeyValueDiscoveryClient::new(1, store.clone());
        let client2 = KeyValueDiscoveryClient::new(2, store.clone());

        let key = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "test-ep".to_string(),
        };

        // Start watching
        let mut stream = client1.list_and_watch(key.clone()).await.unwrap();

        // Register first instance
        client1.serve(key.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { instance_id, .. }) => {
                assert_eq!(instance_id, 1);
            }
            _ => panic!("Expected Added event for instance-1"),
        }

        // Register second instance
        client2.serve(key.clone()).await.unwrap();

        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { instance_id, .. }) => {
                assert_eq!(instance_id, 2);
            }
            _ => panic!("Expected Added event for instance-2"),
        }
    }

    #[tokio::test]
    async fn test_key_value_discovery_prefix_filtering() {
        let store = Arc::new(KeyValueStoreManager::memory());
        let client = KeyValueDiscoveryClient::new(1, store.clone());

        let key1 = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "ep1".to_string(),
        };

        let key2 = DiscoveryKey::Endpoint {
            namespace: "test-ns".to_string(),
            component: "test-comp".to_string(),
            endpoint: "ep2".to_string(),
        };

        // Register instances for both endpoints
        client.serve(key1.clone()).await.unwrap();
        client.serve(key2.clone()).await.unwrap();

        // Watch only ep1
        let mut stream = client.list_and_watch(key1).await.unwrap();

        // Should only get ep1 instance
        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint { endpoint, .. }) => {
                assert_eq!(endpoint, "ep1");
            }
            _ => panic!("Expected Added event for ep1"),
        }

        // Should not get ep2 instance (would timeout if we tried to wait for it)
    }
}
