// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::storage::key_value_store::{KeyValueStoreManager, WatchEvent};
use crate::{CancellationToken, Result};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

use super::{DiscoveryClient, DiscoveryEvent, DiscoveryInstance, DiscoveryKey, DiscoverySpec, DiscoveryStream};

const INSTANCES_BUCKET: &str = "v1/instances";
const MODEL_CARDS_BUCKET: &str = "v1/mdc";

/// Discovery client implementation backed by a KeyValueStore
pub struct KVStoreDiscoveryClient {
    store: Arc<KeyValueStoreManager>,
    cancel_token: CancellationToken,
}

impl KVStoreDiscoveryClient {
    pub fn new(store: KeyValueStoreManager, cancel_token: CancellationToken) -> Self {
        Self {
            store: Arc::new(store),
            cancel_token,
        }
    }

    /// Build the key path for an endpoint
    fn endpoint_key(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> String {
        format!("{}/{}/{}/{}/{:016x}", INSTANCES_BUCKET, namespace, component, endpoint, instance_id)
    }

    /// Build the key path for a model card
    fn model_card_key(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> String {
        format!("{}/{}/{}/{}/{:016x}", MODEL_CARDS_BUCKET, namespace, component, endpoint, instance_id)
    }

    /// Extract prefix for querying based on discovery key
    fn key_prefix(key: &DiscoveryKey) -> String {
        match key {
            DiscoveryKey::AllEndpoints => INSTANCES_BUCKET.to_string(),
            DiscoveryKey::NamespacedEndpoints { namespace } => {
                format!("{}/{}", INSTANCES_BUCKET, namespace)
            }
            DiscoveryKey::ComponentEndpoints { namespace, component } => {
                format!("{}/{}/{}", INSTANCES_BUCKET, namespace, component)
            }
            DiscoveryKey::Endpoint { namespace, component, endpoint } => {
                format!("{}/{}/{}/{}", INSTANCES_BUCKET, namespace, component, endpoint)
            }
            DiscoveryKey::AllModelCards => MODEL_CARDS_BUCKET.to_string(),
            DiscoveryKey::NamespacedModelCards { namespace } => {
                format!("{}/{}", MODEL_CARDS_BUCKET, namespace)
            }
            DiscoveryKey::ComponentModelCards { namespace, component } => {
                format!("{}/{}/{}", MODEL_CARDS_BUCKET, namespace, component)
            }
            DiscoveryKey::EndpointModelCards { namespace, component, endpoint } => {
                format!("{}/{}/{}/{}", MODEL_CARDS_BUCKET, namespace, component, endpoint)
            }
        }
    }

    /// Check if a key matches the given discovery key filter
    fn matches_prefix(key_str: &str, prefix: &str) -> bool {
        key_str.starts_with(prefix)
    }

    /// Parse and deserialize a discovery instance from KV store entry
    fn parse_instance(value: &[u8]) -> Result<DiscoveryInstance> {
        let instance: DiscoveryInstance = serde_json::from_slice(value)?;
        Ok(instance)
    }
}

#[async_trait]
impl DiscoveryClient for KVStoreDiscoveryClient {
    fn instance_id(&self) -> u64 {
        self.store.connection_id()
    }

    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance_id = self.instance_id();
        let instance = spec.with_instance_id(instance_id);

        let (bucket_name, key_path) = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                let key = Self::endpoint_key(
                    &inst.namespace,
                    &inst.component,
                    &inst.endpoint,
                    inst.instance_id,
                );
                (INSTANCES_BUCKET, key)
            }
            DiscoveryInstance::ModelCard {
                namespace,
                component,
                endpoint,
                instance_id,
                ..
            } => {
                let key = Self::model_card_key(namespace, component, endpoint, *instance_id);
                (MODEL_CARDS_BUCKET, key)
            }
        };

        // Serialize the instance
        let instance_json = serde_json::to_vec(&instance)?;

        // Store in the KV store with no TTL (instances persist until explicitly removed)
        let bucket = self
            .store
            .get_or_create_bucket(bucket_name, None)
            .await?;
        let key = crate::storage::key_value_store::Key::from_raw(key_path);
        
        // Use revision 0 for initial registration
        bucket.insert(&key, instance_json.into(), 0).await?;

        Ok(instance)
    }

    async fn list(&self, key: DiscoveryKey) -> Result<Vec<DiscoveryInstance>> {
        let prefix = Self::key_prefix(&key);
        let bucket_name = if prefix.starts_with(INSTANCES_BUCKET) {
            INSTANCES_BUCKET
        } else {
            MODEL_CARDS_BUCKET
        };

        // Get bucket - if it doesn't exist, return empty list
        let Some(bucket) = self.store.get_bucket(bucket_name).await? else {
            return Ok(Vec::new());
        };

        // Get all entries from the bucket
        let entries = bucket.entries().await?;

        // Filter by prefix and deserialize
        let mut instances = Vec::new();
        for (key_str, value) in entries {
            if Self::matches_prefix(&key_str, &prefix) {
                match Self::parse_instance(&value) {
                    Ok(instance) => instances.push(instance),
                    Err(e) => {
                        tracing::warn!(key = %key_str, error = %e, "Failed to parse discovery instance");
                    }
                }
            }
        }

        Ok(instances)
    }

    async fn list_and_watch(&self, key: DiscoveryKey) -> Result<DiscoveryStream> {
        let prefix = Self::key_prefix(&key);
        let bucket_name = if prefix.starts_with(INSTANCES_BUCKET) {
            INSTANCES_BUCKET
        } else {
            MODEL_CARDS_BUCKET
        };

        // Use the KeyValueStoreManager's watch mechanism
        let (_, mut rx) = self.store.clone().watch(
            bucket_name,
            None, // No TTL
            self.cancel_token.clone(),
        );

        // Create a stream that filters and transforms WatchEvents to DiscoveryEvents
        let stream = async_stream::stream! {
            while let Some(event) = rx.recv().await {
                let discovery_event = match event {
                    WatchEvent::Put(kv) => {
                        // Check if this key matches our prefix
                        if !Self::matches_prefix(kv.key_str(), &prefix) {
                            continue;
                        }

                        match Self::parse_instance(kv.value()) {
                            Ok(instance) => Some(DiscoveryEvent::Added(instance)),
                            Err(e) => {
                                tracing::warn!(
                                    key = %kv.key_str(),
                                    error = %e,
                                    "Failed to parse discovery instance from watch event"
                                );
                                None
                            }
                        }
                    }
                    WatchEvent::Delete(kv) => {
                        // Check if this key matches our prefix
                        if !Self::matches_prefix(kv.key_str(), &prefix) {
                            continue;
                        }

                        // Extract instance_id from the deleted entry
                        match Self::parse_instance(kv.value()) {
                            Ok(instance) => Some(DiscoveryEvent::Removed(instance.instance_id())),
                            Err(e) => {
                                tracing::warn!(
                                    key = %kv.key_str(),
                                    error = %e,
                                    "Failed to parse instance_id from deleted discovery instance"
                                );
                                None
                            }
                        }
                    }
                };

                if let Some(event) = discovery_event {
                    yield Ok(event);
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;

    #[tokio::test]
    async fn test_kv_store_discovery_register_endpoint() {
        let store = KeyValueStoreManager::memory();
        let cancel_token = CancellationToken::new();
        let client = KVStoreDiscoveryClient::new(store, cancel_token);

        let spec = DiscoverySpec::Endpoint {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
        };

        let instance = client.register(spec).await.unwrap();
        
        match instance {
            DiscoveryInstance::Endpoint(inst) => {
                assert_eq!(inst.namespace, "test");
                assert_eq!(inst.component, "comp1");
                assert_eq!(inst.endpoint, "ep1");
            }
            _ => panic!("Expected Endpoint instance"),
        }
    }

    #[tokio::test]
    async fn test_kv_store_discovery_list() {
        let store = KeyValueStoreManager::memory();
        let cancel_token = CancellationToken::new();
        let client = KVStoreDiscoveryClient::new(store, cancel_token);

        // Register multiple endpoints
        let spec1 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
        };
        client.register(spec1).await.unwrap();

        let spec2 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep2".to_string(),
            transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
        };
        client.register(spec2).await.unwrap();

        let spec3 = DiscoverySpec::Endpoint {
            namespace: "ns2".to_string(),
            component: "comp2".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
        };
        client.register(spec3).await.unwrap();

        // List all endpoints
        let all = client.list(DiscoveryKey::AllEndpoints).await.unwrap();
        assert_eq!(all.len(), 3);

        // List namespaced endpoints
        let ns1 = client
            .list(DiscoveryKey::NamespacedEndpoints {
                namespace: "ns1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(ns1.len(), 2);

        // List component endpoints
        let comp1 = client
            .list(DiscoveryKey::ComponentEndpoints {
                namespace: "ns1".to_string(),
                component: "comp1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(comp1.len(), 2);
    }

    #[tokio::test]
    async fn test_kv_store_discovery_watch() {
        let store = KeyValueStoreManager::memory();
        let cancel_token = CancellationToken::new();
        let client = Arc::new(KVStoreDiscoveryClient::new(store, cancel_token.clone()));

        // Start watching before registering
        let mut stream = client
            .list_and_watch(DiscoveryKey::AllEndpoints)
            .await
            .unwrap();

        let client_clone = client.clone();
        let register_task = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            let spec = DiscoverySpec::Endpoint {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: "ep1".to_string(),
                transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
            };
            client_clone.register(spec).await.unwrap();
        });

        // Wait for the added event
        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(instance) => {
                match instance {
                    DiscoveryInstance::Endpoint(inst) => {
                        assert_eq!(inst.namespace, "test");
                        assert_eq!(inst.component, "comp1");
                        assert_eq!(inst.endpoint, "ep1");
                    }
                    _ => panic!("Expected Endpoint instance"),
                }
            }
            _ => panic!("Expected Added event"),
        }

        register_task.await.unwrap();
        cancel_token.cancel();
    }
}

