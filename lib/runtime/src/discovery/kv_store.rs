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
    DiscoverySpec, DiscoveryStream, EndpointInstanceId, EventChannelInstanceId, EventScope,
    EventSourceInstanceId, ModelCardInstanceId, encode_event_segment,
};
use crate::storage::kv;

const INSTANCES_BUCKET: &str = "v1/instances";
const MODELS_BUCKET: &str = "v1/mdc";
const EVENT_CHANNELS_BUCKET: &str = "v1/event_channels";
const EVENT_SOURCES_BUCKET: &str = "v1/event_sources";

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
    fn endpoint_key(instance: &crate::component::Instance) -> String {
        instance.endpoint_instance_id().to_path()
    }

    /// Build the key path for a model (relative to bucket, not absolute)
    fn model_key(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> String {
        format!("{}/{}/{}/{:x}", namespace, component, endpoint, instance_id)
    }

    /// Build the key path for an event channel relative to bucket, not absolute)
    fn event_channel_key(scope: &EventScope, topic: &str, instance_id: u64) -> String {
        format!(
            "{}/topic/{}/{:x}",
            scope.path_prefix(),
            encode_event_segment(topic),
            instance_id
        )
    }

    /// Build the key path for an event source relative to its bucket.
    fn event_source_key(scope: &EventScope, topic: &str, publisher_id: u64) -> String {
        EventSourceInstanceId {
            scope: scope.clone(),
            topic: topic.to_string(),
            publisher_id,
        }
        .to_path()
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
                if let Some(scope) = &query.scope {
                    path.push('/');
                    path.push_str(&scope.path_prefix());
                    if let Some(topic) = &query.topic {
                        path.push_str("/topic/");
                        path.push_str(&encode_event_segment(topic));
                    }
                }
                path
            }
            DiscoveryQuery::EventSources(query) => {
                let mut path = EVENT_SOURCES_BUCKET.to_string();
                if let Some(scope) = &query.scope {
                    path.push('/');
                    path.push_str(&scope.path_prefix());
                    if let Some(topic) = &query.topic {
                        path.push_str("/topic/");
                        path.push_str(&encode_event_segment(topic));
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

        relative_key == relative_prefix
            || relative_key
                .strip_prefix(relative_prefix)
                .is_some_and(|suffix| suffix.starts_with('/'))
    }

    fn bucket_for_prefix(prefix: &str) -> &'static str {
        if prefix == INSTANCES_BUCKET
            || prefix
                .strip_prefix(INSTANCES_BUCKET)
                .is_some_and(|suffix| suffix.starts_with('/'))
        {
            INSTANCES_BUCKET
        } else if prefix == EVENT_CHANNELS_BUCKET
            || prefix
                .strip_prefix(EVENT_CHANNELS_BUCKET)
                .is_some_and(|suffix| suffix.starts_with('/'))
        {
            EVENT_CHANNELS_BUCKET
        } else if prefix == EVENT_SOURCES_BUCKET
            || prefix
                .strip_prefix(EVENT_SOURCES_BUCKET)
                .is_some_and(|suffix| suffix.starts_with('/'))
        {
            EVENT_SOURCES_BUCKET
        } else {
            MODELS_BUCKET
        }
    }

    /// Parse and deserialize a discovery instance from KV store entry
    fn parse_instance(value: &[u8]) -> Result<DiscoveryInstance> {
        let instance: DiscoveryInstance = serde_json::from_slice(value)?;
        Ok(instance)
    }

    fn parse_instance_id_from_key(key_str: &str, bucket_name: &str) -> Option<DiscoveryInstanceId> {
        let relative_key = Self::strip_bucket_prefix(key_str, bucket_name);
        let parsed = match bucket_name {
            INSTANCES_BUCKET => {
                EndpointInstanceId::from_path(relative_key).map(DiscoveryInstanceId::Endpoint)
            }
            MODELS_BUCKET => {
                ModelCardInstanceId::from_path(relative_key).map(DiscoveryInstanceId::Model)
            }
            EVENT_CHANNELS_BUCKET => EventChannelInstanceId::from_path(relative_key)
                .map(DiscoveryInstanceId::EventChannel),
            EVENT_SOURCES_BUCKET => {
                EventSourceInstanceId::from_path(relative_key).map(DiscoveryInstanceId::EventSource)
            }
            _ => {
                tracing::warn!(
                    key = %key_str,
                    bucket = bucket_name,
                    "Unknown discovery bucket for delete/resync key"
                );
                return None;
            }
        };

        parsed
            .inspect_err(|err| {
                tracing::warn!(
                    key = %key_str,
                    relative_key = %relative_key,
                    bucket = bucket_name,
                    error = %err,
                    "Failed to parse discovery instance id from key"
                );
            })
            .ok()
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

                for (key, value) in snapshot {
                    let key_str = key.as_ref();
                    if !Self::matches_prefix(key_str, prefix, bucket_name) {
                        continue;
                    }

                    match Self::parse_instance(value.as_ref()) {
                        Ok(instance) => {
                            next_instances.insert(instance.id(), instance);
                        }
                        Err(e) => {
                            tracing::warn!(
                                key = %key_str,
                                error = %e,
                                "Failed to parse discovery instance from resync event"
                            );
                            // The key is still present in the authoritative snapshot; keep
                            // the previous value if only this local parse failed.
                            if let Some(id) = Self::parse_instance_id_from_key(key_str, bucket_name)
                                && let Some(existing) = known_instances.get(&id)
                            {
                                next_instances.insert(id, existing.clone());
                            }
                        }
                    }
                }

                let mut events = Vec::new();
                for id in known_instances.keys() {
                    if !next_instances.contains_key(id) {
                        events.push(DiscoveryEvent::Removed(id.clone()));
                    }
                }

                for (id, instance) in &next_instances {
                    if known_instances.get(id) != Some(instance) {
                        // Added is an upsert event here: a resync can discover
                        // either a new instance or changed data for an existing id.
                        events.push(DiscoveryEvent::Added(instance.clone()));
                    }
                }

                tracing::warn!(
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
        let instance = spec.into_instance(self.instance_id());
        let instance_id = instance.instance_id();
        let is_event_source = matches!(&instance, DiscoveryInstance::EventSource { .. });

        let (bucket_name, key_path) = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                let key = Self::endpoint_key(inst);
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
                scope,
                topic,
                instance_id,
                ..
            } => {
                let key = Self::event_channel_key(scope, topic, *instance_id);
                // TODO: bis - remove this info log
                tracing::info!(
                    "KVStoreDiscovery::register: EventChannel bucket={}, key={}",
                    EVENT_CHANNELS_BUCKET,
                    key
                );
                tracing::debug!(
                    "KVStoreDiscovery::register: Registering event channel instance_id={}, scope={:?}, topic={}, key={}",
                    instance_id,
                    scope,
                    topic,
                    key
                );
                (EVENT_CHANNELS_BUCKET, key)
            }
            DiscoveryInstance::EventSource {
                scope,
                topic,
                publisher_id,
                ..
            } => {
                let key = Self::event_source_key(scope, topic, *publisher_id);
                tracing::debug!(
                    "KVStoreDiscovery::register: Registering event source publisher_id={}, scope={:?}, topic={}, key={}",
                    publisher_id,
                    scope,
                    topic,
                    key
                );
                (EVENT_SOURCES_BUCKET, key)
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

        if is_event_source && let Some(existing) = bucket.get(&key).await? {
            let existing: DiscoveryInstance = serde_json::from_slice(existing.as_ref())?;
            if existing == instance {
                return Ok(existing);
            }
            anyhow::bail!(
                "Event source incarnation '{}' cannot change its descriptor",
                key_path
            );
        }

        tracing::debug!(
            "KVStoreDiscovery::register: Inserting into bucket={}, key={}",
            bucket_name,
            key_path
        );
        // Use revision 0 for initial registration
        let outcome = match bucket.insert(&key, instance_json.into(), 0).await {
            Ok(outcome) => outcome,
            Err(error) if is_event_source => {
                let Some(existing) = bucket.get(&key).await? else {
                    return Err(error.into());
                };
                let existing: DiscoveryInstance = serde_json::from_slice(existing.as_ref())?;
                if existing == instance {
                    return Ok(existing);
                }
                anyhow::bail!(
                    "Event source incarnation '{}' cannot change its descriptor",
                    key_path
                );
            }
            Err(error) => return Err(error.into()),
        };
        tracing::debug!(
            "KVStoreDiscovery::register: Registration insert completed instance_id={}, key={}, outcome={:?}",
            instance_id,
            key_path,
            outcome
        );

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let (bucket_name, key_path) = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                let key = Self::endpoint_key(inst);
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
                scope,
                topic,
                instance_id,
                ..
            } => {
                let key = Self::event_channel_key(scope, topic, *instance_id);
                tracing::debug!(
                    "KVStoreDiscovery::unregister: Unregistering event channel instance_id={}, scope={:?}, topic={}, key={}",
                    instance_id,
                    scope,
                    topic,
                    key
                );
                (EVENT_CHANNELS_BUCKET, key)
            }
            DiscoveryInstance::EventSource {
                scope,
                topic,
                publisher_id,
                ..
            } => {
                let key = Self::event_source_key(scope, topic, *publisher_id);
                tracing::debug!(
                    "KVStoreDiscovery::unregister: Unregistering event source publisher_id={}, scope={:?}, topic={}, key={}",
                    publisher_id,
                    scope,
                    topic,
                    key
                );
                (EVENT_SOURCES_BUCKET, key)
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
        let bucket_name = Self::bucket_for_prefix(&prefix);

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
        let bucket_name = Self::bucket_for_prefix(&prefix);

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
    use crate::discovery::{EventChannelQuery, EventSourceQuery, EventTransport};
    use crate::protocols::EndpointId;

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
        let third = endpoint_instance(3);
        known_instances.insert(first.id(), first);
        known_instances.insert(second.id(), second.clone());

        let mut snapshot = HashMap::new();
        let second_kv = endpoint_kv(2);
        snapshot.insert(
            kv::Key::new(second_kv.key()),
            second_kv.value().to_vec().into(),
        );
        let third_kv = endpoint_kv(3);
        snapshot.insert(
            kv::Key::new(third_kv.key()),
            third_kv.value().to_vec().into(),
        );

        let events = KVStoreDiscovery::discovery_events_from_watch_event(
            kv::WatchEvent::Resync(snapshot),
            &prefix,
            INSTANCES_BUCKET,
            &mut known_instances,
        );

        assert!(!events.contains(&DiscoveryEvent::Added(second)));
        assert_eq!(
            events,
            vec![
                DiscoveryEvent::Removed(endpoint_instance(1).id()),
                DiscoveryEvent::Added(third),
            ]
        );
        assert_eq!(known_instances.len(), 2);
        assert!(known_instances.contains_key(&endpoint_instance(2).id()));
        assert!(known_instances.contains_key(&endpoint_instance(3).id()));
    }

    #[test]
    fn test_resync_retains_known_instance_on_parse_failure() {
        let prefix = format!("{}/{}/{}", INSTANCES_BUCKET, "ns", "component");
        let mut known_instances = HashMap::new();

        let first = endpoint_instance(1);
        known_instances.insert(first.id(), first.clone());

        let mut snapshot = HashMap::new();
        snapshot.insert(
            kv::Key::new(format!("ns/component/endpoint/{:x}", 1)),
            bytes::Bytes::from_static(b"not json"),
        );

        let events = KVStoreDiscovery::discovery_events_from_watch_event(
            kv::WatchEvent::Resync(snapshot),
            &prefix,
            INSTANCES_BUCKET,
            &mut known_instances,
        );

        assert!(events.is_empty());
        assert_eq!(known_instances.len(), 1);
        assert_eq!(known_instances.get(&first.id()), Some(&first));
    }

    #[test]
    fn test_matches_prefix_requires_path_boundary() {
        let prefix = format!("{}/{}/{}", INSTANCES_BUCKET, "ns", "component");

        assert!(KVStoreDiscovery::matches_prefix(
            "ns/component/endpoint/1",
            &prefix,
            INSTANCES_BUCKET
        ));
        assert!(KVStoreDiscovery::matches_prefix(
            "ns/component",
            &prefix,
            INSTANCES_BUCKET
        ));
        assert!(!KVStoreDiscovery::matches_prefix(
            "ns/component2/endpoint/1",
            &prefix,
            INSTANCES_BUCKET
        ));
    }

    #[test]
    fn test_bucket_for_prefix_requires_path_boundary() {
        assert_eq!(
            KVStoreDiscovery::bucket_for_prefix("v1/instances/ns/component"),
            INSTANCES_BUCKET
        );
        assert_eq!(
            KVStoreDiscovery::bucket_for_prefix("v1/event_channels/ns/component/topic"),
            EVENT_CHANNELS_BUCKET
        );
        assert_eq!(
            KVStoreDiscovery::bucket_for_prefix("v1/event_sources/ns/component/topic"),
            EVENT_SOURCES_BUCKET
        );
        assert_eq!(
            KVStoreDiscovery::bucket_for_prefix("v1/instances2/ns/component"),
            MODELS_BUCKET
        );
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
    async fn event_channel_keys_and_queries_preserve_exact_endpoint_scope() {
        let store = kv::Manager::memory();
        let client = KVStoreDiscovery::new(store, CancellationToken::new());
        let endpoint_a = EndpointId {
            namespace: "ns/one".to_string(),
            component: "worker.component".to_string(),
            name: "a/*".to_string(),
        };
        let endpoint_b = EndpointId {
            name: "b/>".to_string(),
            ..endpoint_a.clone()
        };

        for (publisher_id, endpoint) in [(1, endpoint_a.clone()), (2, endpoint_b.clone())] {
            client
                .register(DiscoverySpec::EventChannel {
                    scope: EventScope::Endpoint { endpoint },
                    topic: "kv/events".to_string(),
                    publisher_id,
                    transport: EventTransport::zmq(format!(
                        "tcp://127.0.0.1:{}",
                        5000 + publisher_id
                    )),
                })
                .await
                .unwrap();
        }

        let mut a = client
            .list(DiscoveryQuery::EventChannels(
                EventChannelQuery::endpoint_topic(endpoint_a.clone(), "kv/events"),
            ))
            .await
            .unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].instance_id(), 1);
        client.unregister(a.pop().unwrap()).await.unwrap();
        assert!(
            client
                .list(DiscoveryQuery::EventChannels(
                    EventChannelQuery::endpoint_topic(endpoint_a, "kv/events"),
                ))
                .await
                .unwrap()
                .is_empty()
        );

        let b = client
            .list(DiscoveryQuery::EventChannels(
                EventChannelQuery::endpoint_topic(endpoint_b, "kv/events"),
            ))
            .await
            .unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].instance_id(), 2);
    }

    async fn assert_event_source_lifecycle(store: kv::Manager) {
        let client = KVStoreDiscovery::new(store, CancellationToken::new());
        let endpoint = EndpointId {
            namespace: "ns/one".to_string(),
            component: "worker.component".to_string(),
            name: "decode/*".to_string(),
        };
        let query = DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
            endpoint.clone(),
            "kv/events",
        ));
        let spec = |publisher_id, worker_id| DiscoverySpec::EventSource {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv/events".to_string(),
            publisher_id,
            metadata: serde_json::json!({"worker_id": worker_id, "dp_rank": 0}),
        };

        let first = client.register(spec(100, 7)).await.unwrap();
        assert_eq!(client.register(spec(100, 7)).await.unwrap(), first);
        assert!(client.register(spec(100, 8)).await.is_err());
        assert_eq!(
            client.list(query.clone()).await.unwrap(),
            vec![first.clone()]
        );

        let second = client.register(spec(205, 7)).await.unwrap();
        assert_eq!(client.list(query.clone()).await.unwrap().len(), 2);

        client.unregister(first).await.unwrap();
        assert_eq!(client.list(query).await.unwrap(), vec![second]);
    }

    #[tokio::test]
    async fn event_source_lifecycle_round_trips_through_memory_kv_discovery() {
        assert_event_source_lifecycle(kv::Manager::memory()).await;
    }

    #[tokio::test]
    async fn event_source_lifecycle_round_trips_through_file_kv_discovery() {
        let tempdir = tempfile::tempdir().unwrap();
        let store_cancel = CancellationToken::new();
        let store = kv::Manager::file(store_cancel.clone(), tempdir.path());
        assert_event_source_lifecycle(store).await;
        store_cancel.cancel();
    }

    #[tokio::test]
    async fn event_source_watch_removes_exact_publisher_incarnation() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        let endpoint = EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "decode".to_string(),
        };
        let query = DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
            endpoint.clone(),
            "kv-events",
        ));
        let mut stream = client.list_and_watch(query, None).await.unwrap();
        let spec = |publisher_id| DiscoverySpec::EventSource {
            scope: EventScope::Endpoint {
                endpoint: endpoint.clone(),
            },
            topic: "kv-events".to_string(),
            publisher_id,
            metadata: serde_json::json!({"dp_rank": 0}),
        };

        let first = client.register(spec(100)).await.unwrap();
        let second = client.register(spec(205)).await.unwrap();
        let mut added = std::collections::HashSet::new();
        for _ in 0..2 {
            let DiscoveryEvent::Added(instance) = stream.next().await.unwrap().unwrap() else {
                panic!("expected source addition");
            };
            added.insert(instance.id());
        }
        assert_eq!(
            added,
            std::collections::HashSet::from([first.id(), second.id()])
        );

        client.unregister(first).await.unwrap();
        let removed = tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                if let DiscoveryEvent::Removed(id) = stream.next().await.unwrap().unwrap() {
                    break id;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(
            removed,
            DiscoveryInstanceId::EventSource(EventSourceInstanceId {
                scope: EventScope::Endpoint { endpoint },
                topic: "kv-events".to_string(),
                publisher_id: 100,
            })
        );
        assert_eq!(
            client
                .list(DiscoveryQuery::EventSources(EventSourceQuery::all()))
                .await
                .unwrap(),
            vec![second]
        );
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
