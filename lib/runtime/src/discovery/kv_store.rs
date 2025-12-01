// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, DiscoveryStream,
};
use crate::storage::kv;

const INSTANCES_BUCKET: &str = "v1/instances";
const MODELS_BUCKET: &str = "v1/mdc";

/// Discovery implementation backed by a kv::Store
pub struct KVStoreDiscovery {
    store: Arc<kv::Manager>,
    cancel_token: CancellationToken,
}

impl KVStoreDiscovery {
    pub fn new(store: kv::Manager, cancel_token: CancellationToken) -> Self {
        Self {
            store: Arc::new(store),
            cancel_token,
        }
    }

    /// Build the key path for an endpoint (relative to bucket, not absolute)
    fn endpoint_key(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> String {
        format!("{}/{}/{}/{:x}", namespace, component, endpoint, instance_id)
    }

    /// Build the key path for a model (relative to bucket, not absolute)
    fn model_key(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> String {
        format!("{}/{}/{}/{:x}", namespace, component, endpoint, instance_id)
    }

    /// Extract prefix for querying based on discovery query
    fn query_prefix(query: &DiscoveryQuery) -> String {
        match query {
            DiscoveryQuery::AllEndpoints => INSTANCES_BUCKET.to_string(),
            DiscoveryQuery::NamespacedEndpoints { namespace } => {
                format!("{}/{}", INSTANCES_BUCKET, namespace)
            }
            DiscoveryQuery::ComponentEndpoints {
                namespace,
                component,
            } => {
                format!("{}/{}/{}", INSTANCES_BUCKET, namespace, component)
            }
            DiscoveryQuery::Endpoint {
                namespace,
                component,
                endpoint,
            } => {
                format!(
                    "{}/{}/{}/{}",
                    INSTANCES_BUCKET, namespace, component, endpoint
                )
            }
            DiscoveryQuery::AllModels => MODELS_BUCKET.to_string(),
            DiscoveryQuery::NamespacedModels { namespace } => {
                format!("{}/{}", MODELS_BUCKET, namespace)
            }
            DiscoveryQuery::ComponentModels {
                namespace,
                component,
            } => {
                format!("{}/{}/{}", MODELS_BUCKET, namespace, component)
            }
            DiscoveryQuery::EndpointModels {
                namespace,
                component,
                endpoint,
            } => {
                format!("{}/{}/{}/{}", MODELS_BUCKET, namespace, component, endpoint)
            }
        }
    }

    /// Strip bucket prefix from a key if present, returning the relative path within the bucket
    /// For example: "v1/instances/ns/comp/ep" -> "ns/comp/ep"
    /// Or if already relative: "ns/comp/ep" -> "ns/comp/ep"
    fn strip_bucket_prefix<'a>(key: &'a str, bucket_name: &str) -> &'a str {
        // Try to strip "bucket_name/" from the beginning
        if let Some(stripped) = key.strip_prefix(bucket_name) {
            // Strip the leading slash if present
            stripped.strip_prefix('/').unwrap_or(stripped)
        } else {
            // Key is already relative to bucket
            key
        }
    }

    /// Check if a key matches the given prefix, handling both absolute and relative key formats
    /// This works regardless of whether keys include the bucket prefix (etcd) or not (memory)
    fn matches_prefix(key_str: &str, prefix: &str, bucket_name: &str) -> bool {
        // Normalize both the key and prefix to relative paths (without bucket prefix)
        let relative_key = Self::strip_bucket_prefix(key_str, bucket_name);
        let relative_prefix = Self::strip_bucket_prefix(prefix, bucket_name);

        // Empty prefix matches everything in the bucket
        if relative_prefix.is_empty() {
            return true;
        }

        // Check if the relative key starts with the relative prefix
        relative_key.starts_with(relative_prefix)
    }

    /// Parse and deserialize a discovery instance from KV store entry
    fn parse_instance(value: &[u8]) -> Result<DiscoveryInstance> {
        let instance: DiscoveryInstance = serde_json::from_slice(value)?;
        Ok(instance)
    }
}

#[async_trait]
impl Discovery for KVStoreDiscovery {
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
                tracing::debug!(
                    "KVStoreDiscovery::register: Registering endpoint instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                    inst.instance_id,
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    key
                );
                (INSTANCES_BUCKET, key)
            }
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                ..
            } => {
                let key = Self::model_key(namespace, component, endpoint, *instance_id);
                tracing::debug!(
                    "KVStoreDiscovery::register: Registering model instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                    instance_id,
                    namespace,
                    component,
                    endpoint,
                    key
                );
                (MODELS_BUCKET, key)
            }
        };

        // Serialize the instance
        let instance_json = serde_json::to_vec(&instance)?;
        tracing::debug!(
            "KVStoreDiscovery::register: Serialized instance to {} bytes for key={}",
            instance_json.len(),
            key_path
        );

        // Store in the KV store with no TTL (instances persist until explicitly removed)
        tracing::debug!(
            "KVStoreDiscovery::register: Getting/creating bucket={} for key={}",
            bucket_name,
            key_path
        );
        let bucket = self.store.get_or_create_bucket(bucket_name, None).await?;
        let key = kv::Key::new(key_path.clone());

        tracing::debug!(
            "KVStoreDiscovery::register: Inserting into bucket={}, key={}",
            bucket_name,
            key_path
        );
        // Use revision 0 for initial registration
        let outcome = bucket.insert(&key, instance_json.into(), 0).await?;
        tracing::debug!(
            "KVStoreDiscovery::register: Successfully registered instance_id={}, key={}, outcome={:?}",
            instance_id,
            key_path,
            outcome
        );

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let (bucket_name, key_path) = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                let key = Self::endpoint_key(
                    &inst.namespace,
                    &inst.component,
                    &inst.endpoint,
                    inst.instance_id,
                );
                tracing::debug!(
                    "Unregistering endpoint instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                    inst.instance_id,
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    key
                );
                (INSTANCES_BUCKET, key)
            }
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                instance_id,
                ..
            } => {
                let key = Self::model_key(namespace, component, endpoint, *instance_id);
                tracing::debug!(
                    "Unregistering model instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                    instance_id,
                    namespace,
                    component,
                    endpoint,
                    key
                );
                (MODELS_BUCKET, key)
            }
        };

        // Get the bucket - if it doesn't exist, the instance is already removed from the KV store
        let Some(bucket) = self.store.get_bucket(bucket_name).await? else {
            tracing::warn!(
                "Bucket {} does not exist, instance already removed",
                bucket_name
            );
            return Ok(());
        };

        let key = kv::Key::new(key_path.clone());

        // Delete the entry from the bucket
        bucket.delete(&key).await?;

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        let prefix = Self::query_prefix(&query);
        let bucket_name = if prefix.starts_with(INSTANCES_BUCKET) {
            INSTANCES_BUCKET
        } else {
            MODELS_BUCKET
        };

        // Get bucket - if it doesn't exist, return empty list
        let Some(bucket) = self.store.get_bucket(bucket_name).await? else {
            return Ok(Vec::new());
        };

        // Get all entries from the bucket
        let entries = bucket.entries().await?;

        // Filter by prefix and deserialize
        let mut instances = Vec::new();
        for (key, value) in entries {
            if Self::matches_prefix(key.as_ref(), &prefix, bucket_name) {
                match Self::parse_instance(&value) {
                    Ok(instance) => instances.push(instance),
                    Err(e) => {
                        tracing::warn!(%key, error = %e, "Failed to parse discovery instance");
                    }
                }
            }
        }

        Ok(instances)
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        let prefix = Self::query_prefix(&query);
        let bucket_name = if prefix.starts_with(INSTANCES_BUCKET) {
            INSTANCES_BUCKET
        } else {
            MODELS_BUCKET
        };

        tracing::trace!(
            "KVStoreDiscovery::list_and_watch: Starting watch for query={:?}, prefix={}, bucket={}",
            query,
            prefix,
            bucket_name
        );

        // Use the provided cancellation token, or fall back to the default token
        let cancel_token = cancel_token.unwrap_or_else(|| self.cancel_token.clone());

        // Use the kv::Manager's watch mechanism
        let (_, mut rx) = self.store.clone().watch(
            bucket_name,
            None, // No TTL
            cancel_token,
        );

        // Create a stream that filters and transforms WatchEvents to DiscoveryEvents
        let stream = async_stream::stream! {
            while let Some(event) = rx.recv().await {
                let discovery_event = match event {
                    kv::WatchEvent::Put(kv) => {
                        // Check if this key matches our prefix
                        if !Self::matches_prefix(kv.key_str(), &prefix, bucket_name) {
                            continue;
                        }

                        match Self::parse_instance(kv.value()) {
                            Ok(instance) => {
                                Some(DiscoveryEvent::Added(instance))
                            },
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
                    kv::WatchEvent::Delete(kv) => {
                        let key_str = kv.as_ref();
                        // Check if this key matches our prefix
                        if !Self::matches_prefix(key_str, &prefix, bucket_name) {
                            continue;
                        }

                        // Extract instance_id from the key path, not the value
                        // Delete events have empty values in etcd, so we parse the instance_id from the key
                        // Key format: "v1/instances/namespace/component/endpoint/{instance_id:x}"
                        let key_parts: Vec<&str> = key_str.split('/').collect();
                        match key_parts.last() {
                            Some(instance_id_hex) => {
                                match u64::from_str_radix(instance_id_hex, 16) {
                                    Ok(instance_id) => {
                                        Some(DiscoveryEvent::Removed(instance_id))
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            key = %key_str,
                                            error = %e,
                                            "Failed to parse instance_id hex from deleted key"
                                        );
                                        None
                                    }
                                }
                            }
                            None => {
                                tracing::warn!(
                                    key = %key_str,
                                    "Delete event key has no path components"
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
        let store = kv::Manager::memory();
        let cancel_token = CancellationToken::new();
        let client = KVStoreDiscovery::new(store, cancel_token);

        let spec = DiscoverySpec::Endpoint {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
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
        let store = kv::Manager::memory();
        let cancel_token = CancellationToken::new();
        let client = KVStoreDiscovery::new(store, cancel_token);

        // Register multiple endpoints
        let spec1 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec1).await.unwrap();

        let spec2 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep2".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec2).await.unwrap();

        let spec3 = DiscoverySpec::Endpoint {
            namespace: "ns2".to_string(),
            component: "comp2".to_string(),
            endpoint: "ep1".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec3).await.unwrap();

        // List all endpoints
        let all = client.list(DiscoveryQuery::AllEndpoints).await.unwrap();
        assert_eq!(all.len(), 3);

        // List namespaced endpoints
        let ns1 = client
            .list(DiscoveryQuery::NamespacedEndpoints {
                namespace: "ns1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(ns1.len(), 2);

        // List component endpoints
        let comp1 = client
            .list(DiscoveryQuery::ComponentEndpoints {
                namespace: "ns1".to_string(),
                component: "comp1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(comp1.len(), 2);
    }

    #[tokio::test]
    async fn test_kv_store_discovery_watch() {
        let store = kv::Manager::memory();
        let cancel_token = CancellationToken::new();
        let client = Arc::new(KVStoreDiscovery::new(store, cancel_token.clone()));

        // Start watching before registering
        let mut stream = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();

        let client_clone = client.clone();
        let register_task = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            let spec = DiscoverySpec::Endpoint {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: "ep1".to_string(),
                transport: TransportType::Nats("nats://localhost:4222".to_string()),
            };
            client_clone.register(spec).await.unwrap();
        });

        // Wait for the added event
        let event = stream.next().await.unwrap().unwrap();
        match event {
            DiscoveryEvent::Added(instance) => match instance {
                DiscoveryInstance::Endpoint(inst) => {
                    assert_eq!(inst.namespace, "test");
                    assert_eq!(inst.component, "comp1");
                    assert_eq!(inst.endpoint, "ep1");
                }
                _ => panic!("Expected Endpoint instance"),
            },
            _ => panic!("Expected Added event"),
        }

        register_task.await.unwrap();
        cancel_token.cancel();
    }
}
