// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    DiscoverySpec, DiscoveryStream, EndpointInstanceId, EventChannelInstanceId,
    ModelCardInstanceId,
};
use crate::storage::kv;

const INSTANCES_BUCKET: &str = "v1/instances";
const MODELS_BUCKET: &str = "v1/mdc";
const EVENT_CHANNELS_BUCKET: &str = "v1/event_channels";

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

    /// Build the key path for an event channel relative to bucket, not absolute)
    fn event_channel_key(
        namespace: &str,
        component: &str,
        topic: &str,
        instance_id: u64,
    ) -> String {
        format!("{}/{}/{}/{:x}", namespace, component, topic, instance_id)
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
            DiscoveryQuery::EventChannels(query) => {
                let mut path = EVENT_CHANNELS_BUCKET.to_string();
                if let Some(ns) = &query.namespace {
                    path.push('/');
                    path.push_str(ns);
                    if let Some(comp) = &query.component {
                        path.push('/');
                        path.push_str(comp);
                        if let Some(topic) = &query.topic {
                            path.push('/');
                            path.push_str(topic);
                        }
                    }
                }
                path
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

    fn parse_instance_id_from_key(key_str: &str, bucket_name: &str) -> Option<DiscoveryInstanceId> {
        let relative_key = Self::strip_bucket_prefix(key_str, bucket_name);
        let key_parts: Vec<&str> = relative_key.split('/').collect();

        // EventChannels need 4 parts (namespace/component/topic/instance_id)
        // Endpoints/Models need at least 4 parts
        let min_parts = 4;
        if key_parts.len() < min_parts {
            tracing::warn!(
                key = %key_str,
                relative_key = %relative_key,
                actual_parts = key_parts.len(),
                expected_min = min_parts,
                bucket = bucket_name,
                "Delete/resync key doesn't have enough parts"
            );
            return None;
        }

        let namespace = key_parts[0].to_string();
        let component = key_parts[1].to_string();

        if bucket_name == EVENT_CHANNELS_BUCKET {
            let topic = key_parts[2].to_string();
            let instance_id_hex = key_parts[3];
            let instance_id = match u64::from_str_radix(instance_id_hex, 16) {
                Ok(instance_id) => instance_id,
                Err(e) => {
                    tracing::warn!(
                        key = %key_str,
                        error = %e,
                        instance_id_hex = %instance_id_hex,
                        "Failed to parse event channel instance_id hex"
                    );
                    return None;
                }
            };
            return Some(DiscoveryInstanceId::EventChannel(EventChannelInstanceId {
                namespace,
                component,
                topic,
                instance_id,
            }));
        }

        let endpoint = key_parts[2].to_string();
        let instance_id_hex = key_parts[3];
        let instance_id = match u64::from_str_radix(instance_id_hex, 16) {
            Ok(instance_id) => instance_id,
            Err(e) => {
                tracing::warn!(
                    key = %key_str,
                    error = %e,
                    instance_id_hex = %instance_id_hex,
                    "Failed to parse instance_id hex from key"
                );
                return None;
            }
        };

        if bucket_name == INSTANCES_BUCKET {
            Some(DiscoveryInstanceId::Endpoint(EndpointInstanceId {
                namespace,
                component,
                endpoint,
                instance_id,
            }))
        } else {
            let model_suffix = key_parts.get(4).map(|s| s.to_string());
            Some(DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace,
                component,
                endpoint,
                instance_id,
                model_suffix,
            }))
        }
    }

    fn discovery_events_from_watch_event(
        event: kv::WatchEvent,
        prefix: &str,
        bucket_name: &str,
        known_instances: &mut HashMap<DiscoveryInstanceId, DiscoveryInstance>,
    ) -> Vec<DiscoveryEvent> {
        match event {
            kv::WatchEvent::Put(kv) => {
                if !Self::matches_prefix(kv.key_str(), prefix, bucket_name) {
                    return vec![];
                }

                match Self::parse_instance(kv.value()) {
                    Ok(instance) => {
                        known_instances.insert(instance.id(), instance.clone());
                        vec![DiscoveryEvent::Added(instance)]
                    }
                    Err(e) => {
                        tracing::warn!(
                            key = %kv.key_str(),
                            error = %e,
                            "Failed to parse discovery instance from watch event"
                        );
                        vec![]
                    }
                }
            }
            kv::WatchEvent::Delete(kv) => {
                let key_str = kv.as_ref();
                if !Self::matches_prefix(key_str, prefix, bucket_name) {
                    return vec![];
                }

                let Some(id) = Self::parse_instance_id_from_key(key_str, bucket_name) else {
                    return vec![];
                };

                known_instances.remove(&id);
                tracing::debug!(
                    "KVStoreDiscovery::list_and_watch: Emitting Removed event for {:?}, key={}",
                    id,
                    key_str
                );
                vec![DiscoveryEvent::Removed(id)]
            }
            kv::WatchEvent::Resync(snapshot) => {
                let mut next_instances = HashMap::<DiscoveryInstanceId, DiscoveryInstance>::new();
                let mut events = Vec::new();

                for (key, value) in snapshot {
                    let key_str = key.as_ref();
                    if !Self::matches_prefix(key_str, prefix, bucket_name) {
                        continue;
                    }

                    match Self::parse_instance(value.as_ref()) {
                        Ok(instance) => {
                            next_instances.insert(instance.id(), instance.clone());
                            events.push(DiscoveryEvent::Added(instance));
                        }
                        Err(e) => {
                            tracing::warn!(
                                key = %key_str,
                                error = %e,
                                "Failed to parse discovery instance from resync event"
                            );
                        }
                    }
                }

                for id in known_instances.keys() {
                    if !next_instances.contains_key(id) {
                        events.push(DiscoveryEvent::Removed(id.clone()));
                    }
                }

                tracing::info!(
                    old_count = known_instances.len(),
                    new_count = next_instances.len(),
                    emitted_events = events.len(),
                    "KVStoreDiscovery::list_and_watch resynced discovery state"
                );

                *known_instances = next_instances;
                events
            }
        }
    }
}

#[async_trait]
impl Discovery for KVStoreDiscovery {
    fn instance_id(&self) -> u64 {
        self.store.connection_id()
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
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
                model_suffix,
                ..
            } => {
                let mut key = Self::model_key(namespace, component, endpoint, *instance_id);

                // If there's a model_suffix (e.g., for LoRA adapters), append it after the instance_id
                // Key format: {namespace}/{component}/{endpoint}/{instance_id:x}/{model_suffix}
                if let Some(suffix) = model_suffix
                    && !suffix.is_empty()
                {
                    key = format!("{}/{}", key, suffix);
                    tracing::debug!(
                        "KVStoreDiscovery::register: Registering LoRA model with suffix={}, instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                        suffix,
                        instance_id,
                        namespace,
                        component,
                        endpoint,
                        key
                    );
                }

                // Log for base models (no suffix or empty suffix)
                if model_suffix.as_ref().is_none_or(|s| s.is_empty()) {
                    tracing::debug!(
                        "KVStoreDiscovery::register: Registering base model instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                        instance_id,
                        namespace,
                        component,
                        endpoint,
                        key
                    );
                }
                (MODELS_BUCKET, key)
            }
            DiscoveryInstance::EventChannel {
                namespace,
                component,
                topic,
                instance_id,
                ..
            } => {
                let key = Self::event_channel_key(namespace, component, topic, *instance_id);
                // TODO: bis - remove this info log
                tracing::info!(
                    "KVStoreDiscovery::register: EventChannel bucket={}, key={}",
                    EVENT_CHANNELS_BUCKET,
                    key
                );
                tracing::debug!(
                    "KVStoreDiscovery::register: Registering event channel instance_id={}, namespace={}, component={}, topic={}, key={}",
                    instance_id,
                    namespace,
                    component,
                    topic,
                    key
                );
                (EVENT_CHANNELS_BUCKET, key)
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
                model_suffix,
                ..
            } => {
                let mut key = Self::model_key(namespace, component, endpoint, *instance_id);

                // If there's a model_suffix (e.g., for LoRA adapters), append it after the instance_id
                if let Some(suffix) = model_suffix
                    && !suffix.is_empty()
                {
                    key = format!("{}/{}", key, suffix);
                    tracing::debug!(
                        "KVStoreDiscovery::unregister: Unregistering LoRA model with suffix={}, instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                        suffix,
                        instance_id,
                        namespace,
                        component,
                        endpoint,
                        key
                    );
                }

                // Log for base models (no suffix or empty suffix)
                if model_suffix.as_ref().is_none_or(|s| s.is_empty()) {
                    tracing::debug!(
                        "Unregistering base model instance_id={}, namespace={}, component={}, endpoint={}, key={}",
                        instance_id,
                        namespace,
                        component,
                        endpoint,
                        key
                    );
                }
                (MODELS_BUCKET, key)
            }
            DiscoveryInstance::EventChannel {
                namespace,
                component,
                topic,
                instance_id,
                ..
            } => {
                let key = Self::event_channel_key(namespace, component, topic, *instance_id);
                tracing::debug!(
                    "KVStoreDiscovery::unregister: Unregistering event channel instance_id={}, namespace={}, component={}, topic={}, key={}",
                    instance_id,
                    namespace,
                    component,
                    topic,
                    key
                );
                (EVENT_CHANNELS_BUCKET, key)
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
        } else if prefix.starts_with(EVENT_CHANNELS_BUCKET) {
            EVENT_CHANNELS_BUCKET
        } else {
            MODELS_BUCKET
        };

        // Get bucket - if it doesn't exist, return empty list
        let Some(bucket) = self.store.get_bucket(bucket_name).await? else {
            tracing::debug!(
                "KVStoreDiscovery::list: bucket missing for query={:?}, prefix={}, bucket={}",
                query,
                prefix,
                bucket_name
            );
            return Ok(Vec::new());
        };

        // Get all entries from the bucket
        let entries = bucket.entries().await?;
        tracing::debug!(
            "KVStoreDiscovery::list: query={:?}, prefix={}, bucket={}, entries={}",
            query,
            prefix,
            bucket_name,
            entries.len()
        );

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
        } else if prefix.starts_with(EVENT_CHANNELS_BUCKET) {
            EVENT_CHANNELS_BUCKET
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
            let mut known_instances = HashMap::<DiscoveryInstanceId, DiscoveryInstance>::new();

            while let Some(event) = rx.recv().await {
                let discovery_events = Self::discovery_events_from_watch_event(
                    event,
                    &prefix,
                    bucket_name,
                    &mut known_instances,
                );

                for event in discovery_events {
                    yield Ok(event);
                }
            }
        };
        Ok(Box::pin(stream))
    }

    fn shutdown(&self) {
        self.store.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;

    fn endpoint_instance(instance_id: u64) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(crate::component::Instance {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
            instance_id,
            transport: TransportType::Nats("nats://127.0.0.1:4222".to_string()),
            device_type: None,
        })
    }

    fn endpoint_kv(instance_id: u64) -> kv::KeyValue {
        let instance = endpoint_instance(instance_id);
        kv::KeyValue::new(
            kv::Key::new(format!(
                "{}/{}/{}/{:x}",
                "ns", "component", "endpoint", instance_id
            )),
            serde_json::to_vec(&instance).unwrap().into(),
        )
    }

    #[test]
    fn test_resync_removes_missing_discovery_instances() {
        let prefix = format!("{}/{}/{}", INSTANCES_BUCKET, "ns", "component");
        let mut known_instances = HashMap::new();

        let first = endpoint_instance(1);
        let second = endpoint_instance(2);
        known_instances.insert(first.id(), first);
        known_instances.insert(second.id(), second.clone());

        let mut snapshot = HashMap::new();
        let second_kv = endpoint_kv(2);
        snapshot.insert(
            kv::Key::new(second_kv.key()),
            second_kv.value().to_vec().into(),
        );

        let events = KVStoreDiscovery::discovery_events_from_watch_event(
            kv::WatchEvent::Resync(snapshot),
            &prefix,
            INSTANCES_BUCKET,
            &mut known_instances,
        );

        assert!(events.contains(&DiscoveryEvent::Added(second)));
        assert!(events.contains(&DiscoveryEvent::Removed(endpoint_instance(1).id())));
        assert_eq!(known_instances.len(), 1);
        assert!(known_instances.contains_key(&endpoint_instance(2).id()));
    }

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
            device_type: None,
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
            device_type: None,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec1).await.unwrap();

        let spec2 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            device_type: None,
            endpoint: "ep2".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec2).await.unwrap();

        let spec3 = DiscoverySpec::Endpoint {
            namespace: "ns2".to_string(),
            device_type: None,
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
                device_type: None,
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
