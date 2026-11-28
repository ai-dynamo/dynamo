// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use super::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    DiscoverySpec, DiscoveryStream, EndpointInstanceId, EventChannelInstanceId, EventScope,
    EventSourceInstanceId, ModelCardInstanceId, ModelRegistrationIdentity,
    classify_discovery_change, encode_event_segment, extract_model_registration_identity,
    model_with_updated_taints, resync_discovery_events, validate_event_source_reregistration,
    validate_model_reregistration,
};
use crate::storage::kv;

const INSTANCES_BUCKET: &str = "v1/instances";
const MODELS_BUCKET: &str = "v1/mdc";
const EVENT_CHANNELS_BUCKET: &str = "v1/event_channels";
const EVENT_SOURCES_BUCKET: &str = "v1/event_sources";
/// One record per model card, keyed `{ns}/{comp}/{ep}/{identity_digest}/{instance_id:x}[/{suffix}]`
/// with the card's identity as the value, so the conflict check in `Discovery::register` reads
/// one key per distinct identity instead of every card. A record is per card, not shared per
/// identity, so it ends with its worker's lease as the card does. It is written before the card
/// and deleted after it: a card without its record would be invisible to other workers' checks.
/// The name does not start with `v1/mdc`, so readers of the cards never see these records.
const MODEL_IDENTITIES_BUCKET: &str = "v1/model_identities";
/// One lock per model card key, held by `register_internal` and `unregister` for that card in
/// this process. A card key holds its worker's instance ID, so only this process writes it.
/// Holding the lock across the card and identity-record writes stops one call from deleting an
/// identity record that another call for the same card has reused.
static MODEL_CARD_LOCKS: std::sync::LazyLock<
    parking_lot::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
> = std::sync::LazyLock::new(Default::default);
const UPDATE_MODEL_TAINTS_MAX_ATTEMPTS: usize = 8;

async fn update_model_taints_in_bucket(
    bucket: &dyn kv::Bucket,
    key: &kv::Key,
    target_id: &DiscoveryInstanceId,
    taints: &HashSet<String>,
) -> Result<()> {
    for _ in 0..UPDATE_MODEL_TAINTS_MAX_ATTEMPTS {
        let existing_json = bucket
            .get(key)
            .await?
            .ok_or_else(|| kv::StoreError::MissingKey(key.to_string()))?;
        let existing: DiscoveryInstance = serde_json::from_slice(&existing_json)?;
        if &existing.id() != target_id {
            anyhow::bail!(
                "model discovery record {target_id:?} contains mismatched identity {:?}",
                existing.id()
            )
        }
        let candidate = model_with_updated_taints(&existing, taints.clone())?;
        if candidate == existing {
            return Ok(());
        }

        let candidate_json = serde_json::to_vec(&candidate)?.into();
        match bucket
            .compare_and_replace(key, existing_json, candidate_json)
            .await
        {
            Ok(_) => return Ok(()),
            Err(kv::StoreError::Retry) => continue,
            Err(error) => return Err(error.into()),
        }
    }

    Err(kv::StoreError::Retry.into())
}

/// Holds one entry of [`MODEL_CARD_LOCKS`]; the last holder removes the entry.
struct ModelCardLock {
    key: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl ModelCardLock {
    async fn acquire(key: &str) -> Self {
        let lock = MODEL_CARD_LOCKS
            .lock()
            .entry(key.to_string())
            .or_default()
            .clone();
        Self {
            key: key.to_string(),
            guard: Some(lock.lock_owned().await),
        }
    }
}

impl Drop for ModelCardLock {
    fn drop(&mut self) {
        self.guard.take();
        let mut locks = MODEL_CARD_LOCKS.lock();
        if locks
            .get(&self.key)
            .is_some_and(|lock| Arc::strong_count(lock) == 1)
        {
            locks.remove(&self.key);
        }
    }
}

/// The distinct identities in the model identity index under `endpoint_path`
/// (`{ns}/{comp}/{ep}`). On a store that ranges keys in order (etcd) this reads one entry per
/// distinct identity plus one, however many workers share the endpoint.
async fn model_identities_in_bucket(
    bucket: &dyn kv::Bucket,
    endpoint_path: &str,
) -> Result<Vec<ModelRegistrationIdentity>> {
    let endpoint_prefix = format!("{endpoint_path}/");
    let mut identities = Vec::new();
    let mut start = endpoint_prefix.clone();
    loop {
        let mut records: Vec<(String, bytes::Bytes)> = bucket
            .first_entry_under(endpoint_path, &start)
            .await?
            .into_iter()
            .filter_map(|(key, value)| {
                let key =
                    KVStoreDiscovery::strip_bucket_prefix(key.as_ref(), MODEL_IDENTITIES_BUCKET);
                (key.starts_with(&endpoint_prefix) && key >= start.as_str())
                    .then(|| (key.to_string(), value))
            })
            .collect();
        if records.is_empty() {
            return Ok(identities);
        }
        records.sort_by(|a, b| a.0.cmp(&b.0));

        let mut last_digest = String::new();
        for (key, value) in records {
            let digest = key[endpoint_prefix.len()..]
                .split('/')
                .next()
                .unwrap_or_default();
            if digest == last_digest {
                continue;
            }
            last_digest = digest.to_string();
            identities.push(
                serde_json::from_slice(&value)
                    .with_context(|| format!("invalid model identity record '{key}'"))?,
            );
        }
        // '0' is the byte after '/', so the next read starts past every record of this digest.
        start = format!("{endpoint_prefix}{last_digest}0");
    }
}

/// Discovery implementation backed by a kv::Store
pub struct KVStoreDiscovery {
    store: Arc<kv::Manager>,
    cancel_token: CancellationToken,
    is_model_identity_index_enabled: bool,
}

impl KVStoreDiscovery {
    pub fn new(store: kv::Manager, cancel_token: CancellationToken) -> Self {
        Self {
            store: Arc::new(store),
            cancel_token,
            is_model_identity_index_enabled: crate::config::env_is_truthy(
                crate::config::environment_names::discovery::DYN_MODEL_IDENTITY_INDEX,
            ),
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

    /// The key and value of a model card's record in [`MODEL_IDENTITIES_BUCKET`], or `None` for
    /// other instances.
    fn model_identity_record(
        instance: &DiscoveryInstance,
    ) -> Result<Option<(kv::Key, ModelRegistrationIdentity)>> {
        let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            instance_id,
            card_json,
            model_suffix,
            ..
        } = instance
        else {
            return Ok(None);
        };
        let mut identity = extract_model_registration_identity(card_json, model_suffix.as_deref())?;
        // Compatibility ignores alias order and repeats, so equal identities share one digest.
        identity.aliases.sort();
        identity.aliases.dedup();
        // Peers on other versions read this digest, so it must not depend on the process.
        let digest = blake3::hash(&serde_json::to_vec(&identity)?).to_hex();
        let mut key = format!("{namespace}/{component}/{endpoint}/{digest}/{instance_id:x}");
        if let Some(suffix) = model_suffix
            && !suffix.is_empty()
        {
            key = format!("{key}/{suffix}");
        }
        Ok(Some((kv::Key::new(key), identity)))
    }

    /// Best-effort removal of the identity record a failed registration created. A failed card
    /// write may still have committed, so the record is removed only when the card is absent or
    /// holds another identity, which has its own record.
    async fn remove_unused_identity_record(
        record: Option<(Box<dyn kv::Bucket>, kv::Key)>,
        cards: &dyn kv::Bucket,
        card_key: &kv::Key,
    ) {
        let Some((identities, key)) = record else {
            return;
        };
        let unused = match cards.get(card_key).await {
            Ok(None) => true,
            Ok(Some(card)) => serde_json::from_slice::<DiscoveryInstance>(&card)
                .ok()
                .and_then(|card| Self::model_identity_record(&card).ok().flatten())
                .is_some_and(|(card_record_key, _)| card_record_key != key),
            Err(error) => {
                tracing::warn!(%card_key, %error, "Cannot tell whether a model card was written; keeping its identity record");
                false
            }
        };
        if unused && let Err(error) = identities.delete(&key).await {
            tracing::warn!(%key, %error, "Failed to remove the identity record of an unregistered model");
        }
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
        is_established: bool,
    ) -> Vec<DiscoveryEvent> {
        match event {
            kv::WatchEvent::Put(kv) => {
                if !Self::matches_prefix(kv.key_str(), prefix, bucket_name) {
                    return vec![];
                }

                match Self::parse_instance(kv.value()) {
                    Ok(instance) => {
                        let id = instance.id();
                        match classify_discovery_change(known_instances.get(&id), &instance) {
                            Ok(Some(event)) => {
                                known_instances.insert(id, instance);
                                vec![event]
                            }
                            Ok(None) => vec![],
                            Err(error) => {
                                tracing::error!(
                                    key = %kv.key_str(),
                                    ?id,
                                    %error,
                                    "Rejecting immutable discovery model-card mutation"
                                );
                                vec![]
                            }
                        }
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

                let old_count = known_instances.len();
                let events = resync_discovery_events(known_instances, next_instances);

                if is_established {
                    tracing::warn!(
                        prefix,
                        old_count,
                        new_count = known_instances.len(),
                        emitted_events = events.len(),
                        "KVStoreDiscovery::list_and_watch resynced discovery state"
                    );
                } else {
                    tracing::debug!(
                        prefix,
                        count = known_instances.len(),
                        "KVStoreDiscovery::list_and_watch established from the initial snapshot"
                    );
                }

                events
            }
        }
    }
}

#[async_trait]
impl Discovery for KVStoreDiscovery {
    async fn check_connection(&self) -> Result<()> {
        anyhow::ensure!(
            !self.cancel_token.is_cancelled(),
            "discovery is shutting down"
        );
        self.store.check_connection().await?;
        Ok(())
    }

    fn instance_id(&self) -> u64 {
        self.store.connection_id()
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        self.register_instance(spec.into_instance(self.instance_id()))
            .await
    }

    async fn register_instance(&self, instance: DiscoveryInstance) -> Result<DiscoveryInstance> {
        let instance_id = instance.instance_id();
        let is_event_source = matches!(&instance, DiscoveryInstance::EventSource { .. });
        let is_model = matches!(&instance, DiscoveryInstance::Model { .. });

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
            validate_event_source_reregistration(&existing, &instance)?;
            return Ok(existing);
        }

        let _card_lock = if is_model {
            Some(ModelCardLock::acquire(&key_path).await)
        } else {
            None
        };

        // The identity record goes first, so other workers' conflict checks see this card.
        let mut created_identity_record = None;
        if let Some((record_key, identity)) = Self::model_identity_record(&instance)? {
            let identities = self
                .store
                .get_or_create_bucket(MODEL_IDENTITIES_BUCKET, None)
                .await?;
            let outcome = match identities
                .insert(&record_key, serde_json::to_vec(&identity)?.into(), 0)
                .await
            {
                Ok(outcome) => outcome,
                Err(error) => {
                    Self::remove_unused_identity_record(
                        Some((identities, record_key)),
                        bucket.as_ref(),
                        &key,
                    )
                    .await;
                    return Err(error.into());
                }
            };
            if matches!(outcome, kv::StoreOutcome::Created(_)) {
                created_identity_record = Some((identities, record_key));
            }
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
                validate_event_source_reregistration(&existing, &instance)?;
                return Ok(existing);
            }
            Err(error) => {
                Self::remove_unused_identity_record(created_identity_record, bucket.as_ref(), &key)
                    .await;
                return Err(error.into());
            }
        };
        tracing::debug!(
            "KVStoreDiscovery::register: Registration insert completed instance_id={}, key={}, outcome={:?}",
            instance_id,
            key_path,
            outcome
        );

        if is_model && matches!(outcome, kv::StoreOutcome::Exists(_)) {
            let replay = async {
                let existing = bucket.get(&key).await?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "model discovery record disappeared during same-ID registration replay"
                    )
                })?;
                let existing: DiscoveryInstance = serde_json::from_slice(existing.as_ref())?;
                validate_model_reregistration(&existing, &instance)?;
                Ok(existing)
            }
            .await;
            if replay.is_err() {
                Self::remove_unused_identity_record(created_identity_record, bucket.as_ref(), &key)
                    .await;
            }
            return replay;
        }

        Ok(instance)
    }

    async fn endpoint_model_identities(
        &self,
        namespace: &str,
        component: &str,
        endpoint: &str,
    ) -> Result<Vec<ModelRegistrationIdentity>> {
        let mut identities = match self.store.get_bucket(MODEL_IDENTITIES_BUCKET).await? {
            Some(bucket) => {
                model_identities_in_bucket(
                    bucket.as_ref(),
                    &format!("{namespace}/{component}/{endpoint}"),
                )
                .await?
            }
            None => Vec::new(),
        };
        // Older workers can still write cards without identity records during a rolling upgrade.
        if !self.is_model_identity_index_enabled {
            let cards = self
                .list(DiscoveryQuery::EndpointModels {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    endpoint: endpoint.to_string(),
                })
                .await?;
            for identity in super::distinct_model_identities(&cards)? {
                if !identities.contains(&identity) {
                    identities.push(identity);
                }
            }
        }
        Ok(identities)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let bucket = self
            .store
            .get_bucket(MODELS_BUCKET)
            .await?
            .ok_or_else(|| anyhow::anyhow!("model discovery bucket is not registered"))?;
        let key = kv::Key::new(id.to_path());
        let target_id = DiscoveryInstanceId::Model(id);

        update_model_taints_in_bucket(bucket.as_ref(), &key, &target_id, &taints).await
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

        let _card_lock = if matches!(&instance, DiscoveryInstance::Model { .. }) {
            Some(ModelCardLock::acquire(&key_path).await)
        } else {
            None
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

        // Callers may pass a model card without its JSON (LoRA detach does), so take the
        // identity from the stored card when it still exists.
        let identity_record = match Self::model_identity_record(&instance) {
            Ok(record) => record,
            Err(_) => match bucket.get(&key).await? {
                Some(stored) => serde_json::from_slice::<DiscoveryInstance>(&stored)
                    .ok()
                    .and_then(|stored| Self::model_identity_record(&stored).ok().flatten()),
                None => None,
            },
        };

        let identities = if matches!(&instance, DiscoveryInstance::Model { .. }) {
            self.store.get_bucket(MODEL_IDENTITIES_BUCKET).await?
        } else {
            None
        };
        let mut record_keys = Vec::new();
        if let Some((record_key, _)) = identity_record {
            record_keys.push(record_key);
        } else if let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            instance_id,
            model_suffix,
            ..
        } = &instance
            && let Some(identities) = &identities
        {
            // A previous detach may have deleted the card before identity deletion failed.
            let endpoint_path = format!("{namespace}/{component}/{endpoint}");
            let prefix = format!("{endpoint_path}/");
            let mut card_tail = format!("{instance_id:x}");
            if let Some(suffix) = model_suffix
                && !suffix.is_empty()
            {
                card_tail = format!("{card_tail}/{suffix}");
            }
            for record_key in identities.entries_under(&endpoint_path).await?.into_keys() {
                let relative =
                    Self::strip_bucket_prefix(record_key.as_ref(), MODEL_IDENTITIES_BUCKET);
                if relative
                    .strip_prefix(&prefix)
                    .and_then(|path| path.split_once('/'))
                    .is_some_and(|(_, tail)| tail == card_tail)
                {
                    record_keys.push(kv::Key::new(relative.to_string()));
                }
            }
        }

        // Delete the entry from the bucket
        match bucket.delete(&key).await {
            Ok(()) => {}
            Err(kv::StoreError::MissingKey(_))
                if matches!(&instance, DiscoveryInstance::Model { .. }) => {}
            Err(error) => return Err(error.into()),
        }

        // After the card, so the card is never visible without its identity record.
        if let Some(identities) = identities {
            for record_key in record_keys {
                match identities.delete(&record_key).await {
                    Ok(()) | Err(kv::StoreError::MissingKey(_)) => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }

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

        // Read only the query's keys where the store can scope the read (etcd); the prefix
        // filter below still applies to stores that return the whole bucket.
        let entries = bucket
            .entries_under(Self::strip_bucket_prefix(&prefix, bucket_name))
            .await?;
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

        // Watch only the query's keys where the store can scope the watch (etcd), so this
        // watcher neither snapshots nor receives changes from the rest of the bucket.
        let (_, mut rx) = self
            .store
            .clone()
            .watch(
                bucket_name,
                Self::strip_bucket_prefix(&prefix, bucket_name),
                None, // No TTL
                cancel_token,
            )
            .await?;

        // Create a stream that filters and transforms WatchEvents to DiscoveryEvents
        let stream = async_stream::stream! {
            let mut known_instances = HashMap::<DiscoveryInstanceId, DiscoveryInstance>::new();
            // The first storage event is the establishment snapshot.
            let mut is_established = false;

            while let Some(event) = rx.recv().await {
                let discovery_events = Self::discovery_events_from_watch_event(
                    event,
                    &prefix,
                    bucket_name,
                    &mut known_instances,
                    is_established,
                );
                is_established = true;

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
    use crate::discovery::startup_contract as contract;
    use crate::discovery::{
        EventChannelQuery, EventSourceQuery, EventTransport, ModelTaintsUpdate,
    };
    use crate::protocols::EndpointId;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn endpoint_instance(instance_id: u64) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(crate::component::Instance {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
            instance_id,
            transport: TransportType::Nats("nats://127.0.0.1:4222".to_string()),
            device_type: None,
            request_plane_codec: None,
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

    fn resync_ids(event: &DiscoveryEvent) -> HashSet<DiscoveryInstanceId> {
        let DiscoveryEvent::Resync(instances) = event else {
            panic!("expected a resync event, got {event:?}");
        };
        instances.iter().map(DiscoveryInstance::id).collect()
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
            true,
        );

        assert!(!events.contains(&DiscoveryEvent::Added(second)));
        assert_eq!(events.len(), 3);
        assert_eq!(
            events[..2],
            [
                DiscoveryEvent::Removed(endpoint_instance(1).id()),
                DiscoveryEvent::Added(third),
            ]
        );
        assert_eq!(
            resync_ids(&events[2]),
            HashSet::from([endpoint_instance(2).id(), endpoint_instance(3).id()])
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
            true,
        );

        assert_eq!(events.len(), 1);
        assert_eq!(resync_ids(&events[0]), HashSet::from([first.id()]));
        assert_eq!(known_instances.len(), 1);
        assert_eq!(known_instances.get(&first.id()), Some(&first));
    }

    #[test]
    fn resync_changed_model_taints_emits_scoped_event() {
        let prefix = format!("{}/{}/{}/{}", MODELS_BUCKET, "ns", "worker", "generate");
        let old = model_spec("first").into_instance(7);
        let updated = model_spec("second").into_instance(7);
        let DiscoveryInstanceId::Model(id) = updated.id() else {
            unreachable!()
        };
        let mut known_instances = HashMap::from([(old.id(), old)]);
        let snapshot = HashMap::from([(
            kv::Key::new(id.to_path()),
            serde_json::to_vec(&updated).unwrap().into(),
        )]);

        let events = KVStoreDiscovery::discovery_events_from_watch_event(
            kv::WatchEvent::Resync(snapshot),
            &prefix,
            MODELS_BUCKET,
            &mut known_instances,
            true,
        );

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id: id.clone(),
                taints: vec![
                    "dynamo.topology/zone=west".to_string(),
                    "second".to_string(),
                ],
            })
        );
        assert_eq!(
            resync_ids(&events[1]),
            HashSet::from([DiscoveryInstanceId::Model(id.clone())])
        );
        assert_eq!(
            known_instances.get(&DiscoveryInstanceId::Model(id)),
            Some(&updated)
        );
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
        contract::expect_empty_snapshot(&mut stream).await;
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
            request_plane_codec: None,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec1).await.unwrap();

        let spec2 = DiscoverySpec::Endpoint {
            namespace: "ns1".to_string(),
            component: "comp1".to_string(),
            device_type: None,
            request_plane_codec: None,
            endpoint: "ep2".to_string(),
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        };
        client.register(spec2).await.unwrap();

        let spec3 = DiscoverySpec::Endpoint {
            namespace: "ns2".to_string(),
            device_type: None,
            request_plane_codec: None,
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
    async fn hosted_instances_keep_their_own_ids() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        let own = client.instance_id();
        let mut hosted = Vec::new();
        for instance_id in [own.wrapping_add(1), own.wrapping_add(2)] {
            let DiscoveryInstance::Endpoint(mut instance) = endpoint_instance(instance_id) else {
                unreachable!()
            };
            instance.transport =
                TransportType::Direct(format!("http://engine-{instance_id}:30001"));
            hosted.push(
                client
                    .register_instance(DiscoveryInstance::Endpoint(instance))
                    .await
                    .unwrap(),
            );
        }
        let query = DiscoveryQuery::Endpoint {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
        };
        let ids = |listed: Vec<DiscoveryInstance>| -> HashSet<u64> {
            listed.iter().map(DiscoveryInstance::instance_id).collect()
        };
        assert_eq!(
            ids(client.list(query.clone()).await.unwrap()),
            HashSet::from([own.wrapping_add(1), own.wrapping_add(2)])
        );

        client.unregister(hosted.remove(0)).await.unwrap();
        assert_eq!(
            ids(client.list(query).await.unwrap()),
            HashSet::from([own.wrapping_add(2)])
        );
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
        contract::expect_empty_snapshot(&mut stream).await;

        let client_clone = client.clone();
        let register_task = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            let spec = DiscoverySpec::Endpoint {
                device_type: None,
                request_plane_codec: None,
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

    #[tokio::test]
    async fn watch_reports_an_unregister_that_follows_establishment() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        let instance = client
            .register(DiscoverySpec::Endpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep".to_string(),
                device_type: None,
                request_plane_codec: None,
                transport: TransportType::Nats("nats://127.0.0.1:4222".to_string()),
            })
            .await
            .unwrap();

        let mut stream = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();
        client.unregister(instance.clone()).await.unwrap();

        let added = tokio::time::timeout(tokio::time::Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(added, DiscoveryEvent::Added(instance.clone()));
        assert_eq!(
            contract::next(&mut stream).await,
            DiscoveryEvent::Resync(vec![instance.clone()])
        );

        let removed = tokio::time::timeout(tokio::time::Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            removed,
            DiscoveryEvent::Removed(instance.id()),
            "an unregister after establishment must reach the stream"
        );
    }

    #[tokio::test]
    async fn watch_reports_a_resync_after_the_backend_lags() {
        use crate::storage::kv::Bucket;

        let store = kv::Manager::memory();
        let client = KVStoreDiscovery::new(store.clone(), CancellationToken::new());
        let mut stream = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();
        // Consume the startup snapshot so the resync asserted below can only come from the lag.
        contract::expect_empty_snapshot(&mut stream).await;

        let instance = client
            .register(DiscoverySpec::Endpoint {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep".to_string(),
                device_type: None,
                request_plane_codec: None,
                transport: TransportType::Nats("nats://127.0.0.1:4222".to_string()),
            })
            .await
            .unwrap();
        assert_eq!(
            client.list(DiscoveryQuery::AllEndpoints).await.unwrap(),
            vec![instance.clone()]
        );
        client.unregister(instance).await.unwrap();

        // The memory store has one change buffer for all buckets. These writes push the register
        // and the unregister out of the buffer before the watch task reads them: on the
        // current-thread test runtime that task cannot run before this test yields.
        let filler = store.get_or_create_bucket("filler", None).await.unwrap();
        let key = kv::Key::new("key".to_string());
        for revision in 1..=(kv::MEMORY_EVENT_BUFFER_CAPACITY + 1) as u64 {
            filler.insert(&key, "value".into(), revision).await.unwrap();
        }

        let event = tokio::time::timeout(tokio::time::Duration::from_secs(1), stream.next())
            .await
            .expect("the stream must report the resync")
            .unwrap()
            .unwrap();
        assert_eq!(event, DiscoveryEvent::Resync(vec![]));
    }

    #[tokio::test]
    async fn memory_backend_keeps_the_startup_contract() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        contract::check(&client).await;
    }

    #[tokio::test]
    async fn file_backend_keeps_the_startup_contract() {
        let cancel_token = CancellationToken::new();
        let root = tempfile::tempdir().unwrap();
        let store = kv::Manager::file(cancel_token.clone(), root.path());
        let client = KVStoreDiscovery::new(store, cancel_token.clone());
        contract::check(&client).await;
        cancel_token.cancel();
    }

    fn model_spec(taint: &str) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({
                "display_name": "model",
                "runtime_config": {
                    "taints": [taint, "dynamo.topology/zone=west"],
                    "topology_domains": {"zone": "west"}
                }
            }),
            model_suffix: None,
        }
    }

    struct AlwaysConflictingBucket {
        value: bytes::Bytes,
        compare_attempts: AtomicUsize,
    }

    #[async_trait]
    impl kv::Bucket for AlwaysConflictingBucket {
        async fn insert(
            &self,
            _key: &kv::Key,
            _value: bytes::Bytes,
            _revision: u64,
        ) -> Result<kv::StoreOutcome, kv::StoreError> {
            unreachable!("insert is not used by this test")
        }

        async fn get(&self, _key: &kv::Key) -> Result<Option<bytes::Bytes>, kv::StoreError> {
            Ok(Some(self.value.clone()))
        }

        async fn compare_and_replace(
            &self,
            _key: &kv::Key,
            _expected: bytes::Bytes,
            _value: bytes::Bytes,
        ) -> Result<kv::StoreOutcome, kv::StoreError> {
            self.compare_attempts.fetch_add(1, Ordering::Relaxed);
            Err(kv::StoreError::Retry)
        }

        async fn delete(&self, _key: &kv::Key) -> Result<(), kv::StoreError> {
            unreachable!("delete is not used by this test")
        }

        async fn watch(
            &self,
        ) -> Result<Pin<Box<dyn futures::Stream<Item = kv::WatchEvent> + Send + '_>>, kv::StoreError>
        {
            unreachable!("watch is not used by this test")
        }

        async fn entries(&self) -> Result<HashMap<kv::Key, bytes::Bytes>, kv::StoreError> {
            unreachable!("entries is not used by this test")
        }
    }

    #[tokio::test]
    async fn model_taint_update_stops_after_bounded_conflicts() {
        let existing = model_spec("first").into_instance(7);
        let target_id = existing.id();
        let DiscoveryInstanceId::Model(id) = &target_id else {
            unreachable!()
        };
        let key = kv::Key::new(id.to_path());
        let bucket = AlwaysConflictingBucket {
            value: serde_json::to_vec(&existing).unwrap().into(),
            compare_attempts: AtomicUsize::new(0),
        };

        let error = update_model_taints_in_bucket(
            &bucket,
            &key,
            &target_id,
            &HashSet::from(["second".to_string()]),
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error.downcast_ref::<kv::StoreError>(),
            Some(kv::StoreError::Retry)
        ));
        assert_eq!(
            bucket.compare_attempts.load(Ordering::Relaxed),
            UPDATE_MODEL_TAINTS_MAX_ATTEMPTS
        );
    }

    #[tokio::test]
    async fn model_taint_updates_replace_existing_value_and_emit_scoped_event() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        let query = DiscoveryQuery::EndpointModels {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
        };
        let mut stream = client.list_and_watch(query.clone(), None).await.unwrap();
        contract::expect_empty_snapshot(&mut stream).await;

        client.register(model_spec("first")).await.unwrap();
        let DiscoveryEvent::Added(first) = stream.next().await.unwrap().unwrap() else {
            panic!("expected initial model addition");
        };

        let DiscoveryInstanceId::Model(id) = first.id() else {
            unreachable!()
        };
        for taint in ["second", "third"] {
            client
                .update_model_taints(id.clone(), HashSet::from([taint.to_string()]))
                .await
                .unwrap();
            let event = tokio::time::timeout(tokio::time::Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(
                event,
                DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                    id: id.clone(),
                    taints: vec!["dynamo.topology/zone=west".to_string(), taint.to_string(),],
                })
            );
        }

        let replayed = client.register(model_spec("first")).await.unwrap();
        let DiscoveryInstance::Model { card_json, .. } = &replayed else {
            panic!("expected model instance");
        };
        let replayed_taints = card_json["runtime_config"]["taints"].as_array().unwrap();
        assert!(replayed_taints.contains(&serde_json::json!("third")));
        assert!(!replayed_taints.contains(&serde_json::json!("first")));
        assert!(
            tokio::time::timeout(tokio::time::Duration::from_millis(50), stream.next())
                .await
                .is_err()
        );

        client
            .update_model_taints(id, HashSet::from(["third".to_string()]))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(tokio::time::Duration::from_millis(50), stream.next())
                .await
                .is_err()
        );

        let listed = client.list(query).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id(), first.id());
        let DiscoveryInstance::Model { card_json, .. } = &listed[0] else {
            panic!("expected model instance");
        };
        let taints = card_json["runtime_config"]["taints"].as_array().unwrap();
        assert!(taints.contains(&serde_json::json!("third")));
        assert!(!taints.contains(&serde_json::json!("second")));
    }

    fn named_model_spec(name: &str, source_path: &str) -> DiscoverySpec {
        DiscoverySpec::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            card_json: serde_json::json!({"display_name": name, "source_path": source_path}),
            model_suffix: None,
        }
    }

    /// A bucket that ranges its keys in order, as etcd does, and counts the reads.
    #[derive(Default)]
    struct OrderedBucket {
        entries: std::collections::BTreeMap<String, bytes::Bytes>,
        reads: AtomicUsize,
    }

    #[async_trait]
    impl kv::Bucket for OrderedBucket {
        async fn insert(
            &self,
            _key: &kv::Key,
            _value: bytes::Bytes,
            _revision: u64,
        ) -> Result<kv::StoreOutcome, kv::StoreError> {
            unreachable!("insert is not used by this test")
        }

        async fn get(&self, _key: &kv::Key) -> Result<Option<bytes::Bytes>, kv::StoreError> {
            unreachable!("get is not used by this test")
        }

        async fn compare_and_replace(
            &self,
            _key: &kv::Key,
            _expected: bytes::Bytes,
            _value: bytes::Bytes,
        ) -> Result<kv::StoreOutcome, kv::StoreError> {
            unreachable!("compare_and_replace is not used by this test")
        }

        async fn delete(&self, _key: &kv::Key) -> Result<(), kv::StoreError> {
            unreachable!("delete is not used by this test")
        }

        async fn watch(
            &self,
        ) -> Result<Pin<Box<dyn futures::Stream<Item = kv::WatchEvent> + Send + '_>>, kv::StoreError>
        {
            unreachable!("watch is not used by this test")
        }

        async fn entries(&self) -> Result<HashMap<kv::Key, bytes::Bytes>, kv::StoreError> {
            unreachable!("the identity lookup must not read every record")
        }

        async fn first_entry_under(
            &self,
            sub: &str,
            start: &str,
        ) -> Result<HashMap<kv::Key, bytes::Bytes>, kv::StoreError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let prefix = format!("{sub}/");
            Ok(self
                .entries
                .range(start.to_string()..)
                .take_while(|(key, _)| key.starts_with(&prefix))
                .take(1)
                .map(|(key, value)| {
                    (
                        kv::Key::new(format!("{MODEL_IDENTITIES_BUCKET}/{key}")),
                        value.clone(),
                    )
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn identity_lookup_reads_one_record_per_distinct_identity() {
        let mut bucket = OrderedBucket::default();
        for instance_id in 1..=64u64 {
            let name = if instance_id % 2 == 0 { "a" } else { "b" };
            let instance =
                named_model_spec(name, &format!("/models/{name}")).into_instance(instance_id);
            let (key, identity) = KVStoreDiscovery::model_identity_record(&instance)
                .unwrap()
                .unwrap();
            bucket.entries.insert(
                key.to_string(),
                serde_json::to_vec(&identity).unwrap().into(),
            );
        }
        let sibling = KVStoreDiscovery::model_identity_record(
            &named_model_spec("c", "/models/c").into_instance(1),
        )
        .unwrap()
        .unwrap()
        .1;
        for key in ["ns/worker/generate-2/00/1", "ns/worker/generatex/00/1"] {
            bucket.entries.insert(
                key.to_string(),
                serde_json::to_vec(&sibling).unwrap().into(),
            );
        }

        let identities = model_identities_in_bucket(&bucket, "ns/worker/generate")
            .await
            .unwrap();

        let mut names: Vec<_> = identities
            .iter()
            .map(|identity| identity.display_name.as_str())
            .collect();
        names.sort();
        assert_eq!(names, ["a", "b"]);
        // One read per distinct identity and one that finds none, for 64 cards.
        assert_eq!(bucket.reads.load(Ordering::Relaxed), 3);
    }

    async fn identity_record_count(client: &KVStoreDiscovery) -> usize {
        client
            .store
            .get_bucket(MODEL_IDENTITIES_BUCKET)
            .await
            .unwrap()
            .unwrap()
            .entries()
            .await
            .unwrap()
            .len()
    }

    #[tokio::test]
    async fn register_rejects_legacy_cards_with_or_without_an_identity_index() {
        for has_index in [false, true] {
            let mut client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
            client.is_model_identity_index_enabled = false;
            let legacy = named_model_spec("a", "/models/a").into_instance(client.instance_id() + 1);
            let cards = client
                .store
                .get_or_create_bucket(MODELS_BUCKET, None)
                .await
                .unwrap();
            let key = kv::Key::new(KVStoreDiscovery::model_key(
                "ns",
                "worker",
                "generate",
                legacy.instance_id(),
            ));
            cards
                .insert(&key, serde_json::to_vec(&legacy).unwrap().into(), 0)
                .await
                .unwrap();
            if has_index {
                let (record_key, identity) = KVStoreDiscovery::model_identity_record(
                    &named_model_spec("b", "/models/b").into_instance(client.instance_id()),
                )
                .unwrap()
                .unwrap();
                client
                    .store
                    .get_or_create_bucket(MODEL_IDENTITIES_BUCKET, None)
                    .await
                    .unwrap()
                    .insert(
                        &record_key,
                        serde_json::to_vec(&identity).unwrap().into(),
                        0,
                    )
                    .await
                    .unwrap();
            }

            let error = client
                .register(named_model_spec("b", "/models/b"))
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("a different model 'a' is already registered there"),
                "unexpected error: {error:#}"
            );

            // Once the old card has drained, index-only registration still checks new identities.
            cards.delete(&key).await.unwrap();
            client.is_model_identity_index_enabled = true;
            client
                .register(named_model_spec("b", "/models/b"))
                .await
                .unwrap();
            assert!(
                client
                    .register(named_model_spec("c", "/models/c"))
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn model_identity_records_follow_their_cards() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());

        let mut model = client
            .register(named_model_spec("a", "/models/a"))
            .await
            .unwrap();
        assert_eq!(identity_record_count(&client).await, 1);
        let identities = client
            .endpoint_model_identities("ns", "worker", "generate")
            .await
            .unwrap();
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].display_name, "a");

        // A same-ID replay with another identity is rejected and leaves no record behind.
        assert!(
            client
                .register_internal(named_model_spec("b", "/models/b"))
                .await
                .is_err()
        );
        assert_eq!(identity_record_count(&client).await, 1);

        // LoRA detach unregisters without the card JSON; the stored card names the record.
        let DiscoveryInstance::Model { card_json, .. } = &mut model else {
            unreachable!()
        };
        *card_json = serde_json::Value::Null;
        client.unregister(model).await.unwrap();
        assert_eq!(identity_record_count(&client).await, 0);

        // A record with no card: only a check that reads the index sees model 'c'.
        let (key, identity) = KVStoreDiscovery::model_identity_record(
            &named_model_spec("c", "/models/c").into_instance(client.instance_id() + 1),
        )
        .unwrap()
        .unwrap();
        client
            .store
            .get_bucket(MODEL_IDENTITIES_BUCKET)
            .await
            .unwrap()
            .unwrap()
            .insert(&key, serde_json::to_vec(&identity).unwrap().into(), 0)
            .await
            .unwrap();
        let error = client
            .register(named_model_spec("d", "/models/d"))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("a different model 'c' is already registered there"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn null_card_unregister_recovers_identity_cleanup_after_card_deletion() {
        for suffix in [None, Some("adapter")] {
            let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
            let mut spec = named_model_spec(
                if suffix.is_some() { "adapter" } else { "base" },
                "/models/a",
            );
            let DiscoverySpec::Model { model_suffix, .. } = &mut spec else {
                unreachable!()
            };
            *model_suffix = suffix.map(str::to_string);
            let mut model = client.register(spec).await.unwrap();
            let (record_key, _) = KVStoreDiscovery::model_identity_record(&model)
                .unwrap()
                .unwrap();
            let identities = client
                .store
                .get_bucket(MODEL_IDENTITIES_BUCKET)
                .await
                .unwrap()
                .unwrap();
            let mut other_spec = named_model_spec(
                if suffix.is_some() { "base" } else { "adapter" },
                "/models/a",
            );
            let DiscoverySpec::Model { model_suffix, .. } = &mut other_spec else {
                unreachable!()
            };
            *model_suffix = if suffix.is_some() {
                None
            } else {
                Some("adapter".to_string())
            };
            let other = client.register(other_spec).await.unwrap();
            let (other_key, _) = KVStoreDiscovery::model_identity_record(&other)
                .unwrap()
                .unwrap();

            let cards = client
                .store
                .get_bucket(MODELS_BUCKET)
                .await
                .unwrap()
                .unwrap();
            let mut card_key =
                KVStoreDiscovery::model_key("ns", "worker", "generate", client.instance_id());
            if let Some(suffix) = suffix {
                card_key = format!("{card_key}/{suffix}");
            }
            // The earlier detach removed the card but left its identity record.
            cards.delete(&kv::Key::new(card_key)).await.unwrap();
            let DiscoveryInstance::Model { card_json, .. } = &mut model else {
                unreachable!()
            };
            *card_json = serde_json::Value::Null;
            client.unregister(model.clone()).await.unwrap();
            client.unregister(model).await.unwrap();
            assert!(identities.get(&record_key).await.unwrap().is_none());
            assert!(identities.get(&other_key).await.unwrap().is_some());
        }
    }

    #[tokio::test]
    async fn failed_registration_keeps_the_identity_record_of_a_written_card() {
        let store = kv::Manager::memory();
        let cards = store
            .get_or_create_bucket(MODELS_BUCKET, None)
            .await
            .unwrap();
        let identities = store
            .get_or_create_bucket(MODEL_IDENTITIES_BUCKET, None)
            .await
            .unwrap();
        let instance = named_model_spec("a", "/models/a").into_instance(1);
        let (record_key, identity) = KVStoreDiscovery::model_identity_record(&instance)
            .unwrap()
            .unwrap();
        identities
            .insert(
                &record_key,
                serde_json::to_vec(&identity).unwrap().into(),
                0,
            )
            .await
            .unwrap();
        let created = || async {
            let identities = store.get_bucket(MODEL_IDENTITIES_BUCKET).await.unwrap();
            Some((identities.unwrap(), record_key.clone()))
        };

        // The card write committed although the caller saw an error.
        let card_key = kv::Key::new(KVStoreDiscovery::model_key("ns", "worker", "generate", 1));
        cards
            .insert(&card_key, serde_json::to_vec(&instance).unwrap().into(), 0)
            .await
            .unwrap();
        KVStoreDiscovery::remove_unused_identity_record(created().await, cards.as_ref(), &card_key)
            .await;
        assert!(identities.get(&record_key).await.unwrap().is_some());

        cards.delete(&card_key).await.unwrap();
        KVStoreDiscovery::remove_unused_identity_record(created().await, cards.as_ref(), &card_key)
            .await;
        assert!(identities.get(&record_key).await.unwrap().is_none());
    }

    /// Registers `concurrent_spec` through `concurrent` while `inner` is inside
    /// `register_internal`, after its write: an incompatible registration that passed its first
    /// check before this one was written.
    struct RacedDiscovery {
        inner: KVStoreDiscovery,
        concurrent: KVStoreDiscovery,
        concurrent_spec: DiscoverySpec,
    }

    #[async_trait]
    impl Discovery for RacedDiscovery {
        fn instance_id(&self) -> u64 {
            self.inner.instance_id()
        }

        async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
            let instance = self.inner.register_internal(spec).await?;
            self.concurrent
                .register_internal(self.concurrent_spec.clone())
                .await?;
            Ok(instance)
        }

        async fn endpoint_model_identities(
            &self,
            namespace: &str,
            component: &str,
            endpoint: &str,
        ) -> Result<Vec<ModelRegistrationIdentity>> {
            self.inner
                .endpoint_model_identities(namespace, component, endpoint)
                .await
        }

        async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
            self.inner.unregister(instance).await
        }

        async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
            self.inner.list(query).await
        }

        async fn list_and_watch(
            &self,
            query: DiscoveryQuery,
            cancel_token: Option<CancellationToken>,
        ) -> Result<DiscoveryStream> {
            self.inner.list_and_watch(query, cancel_token).await
        }
    }

    #[tokio::test]
    async fn register_rolls_back_after_a_concurrent_conflicting_registration() {
        let tempdir = tempfile::tempdir().unwrap();
        let store_cancel = CancellationToken::new();
        let _stop_stores = store_cancel.clone().drop_guard();
        let worker = || {
            KVStoreDiscovery::new(
                kv::Manager::file(store_cancel.clone(), tempdir.path()),
                CancellationToken::new(),
            )
        };
        let raced = RacedDiscovery {
            inner: worker(),
            concurrent: worker(),
            concurrent_spec: named_model_spec("a", "/models/a"),
        };

        let error = raced
            .register(named_model_spec("b", "/models/b"))
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains(
                "Cannot register model 'b' on endpoint 'ns/worker/generate': a different model 'a' is already registered there"
            ),
            "unexpected error: {error:#}"
        );
        let models = raced
            .list(DiscoveryQuery::EndpointModels {
                namespace: "ns".to_string(),
                component: "worker".to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].instance_id(), raced.concurrent.instance_id());
        let identities = raced
            .endpoint_model_identities("ns", "worker", "generate")
            .await
            .unwrap();
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].display_name, "a");
        store_cancel.cancel();
    }

    #[tokio::test]
    async fn register_checks_every_identity_in_the_index() {
        // B is compatible with A by name and C by source, but A and C conflict.
        let cards = [("a", "/mount/a"), ("a", "/mount/b"), ("b", "/mount/b")];
        for order in [[0, 1, 2], [2, 1, 0]] {
            let tempdir = tempfile::tempdir().unwrap();
            let store_cancel = CancellationToken::new();
            let _stop_stores = store_cancel.clone().drop_guard();
            for (position, index) in order.into_iter().enumerate() {
                let client = KVStoreDiscovery::new(
                    kv::Manager::file(store_cancel.clone(), tempdir.path()),
                    CancellationToken::new(),
                );
                let (name, source) = cards[index];
                let result = client.register(named_model_spec(name, source)).await;
                assert_eq!(
                    result.is_ok(),
                    position < 2,
                    "card {index} at position {position}: {result:?}"
                );
            }
            store_cancel.cancel();
        }
    }
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod etcd_tests {
    use super::*;
    use crate::Runtime;
    use crate::component::TransportType;
    use crate::transports::etcd as etcd_transport;
    use futures::StreamExt;

    fn endpoint(namespace: &str, component: &str, endpoint: &str) -> DiscoverySpec {
        DiscoverySpec::Endpoint {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            device_type: None,
            request_plane_codec: None,
            transport: TransportType::Nats("nats://localhost:4222".to_string()),
        }
    }

    /// The sorted `namespace/component/endpoint` names of endpoint instances.
    fn endpoint_names(instances: &[DiscoveryInstance]) -> Vec<String> {
        let mut names: Vec<_> = instances
            .iter()
            .map(|instance| match instance {
                DiscoveryInstance::Endpoint(inst) => {
                    format!("{}/{}/{}", inst.namespace, inst.component, inst.endpoint)
                }
                other => panic!("expected an endpoint, got {other:?}"),
            })
            .collect();
        names.sort();
        names
    }

    /// On etcd, the identity lookup returns every distinct identity on the endpoint and none
    /// from a sibling endpoint (`ep` and `ep-2`), and unregistering removes the records.
    #[test]
    fn identity_lookup_covers_only_its_endpoint() {
        let rt = Runtime::single_threaded().unwrap();
        let rt_clone = rt.clone();

        rt_clone.primary().block_on(async move {
            let etcd_client =
                etcd_transport::Client::new(etcd_transport::ClientOptions::default(), rt)
                    .await
                    .unwrap();
            let mut client =
                KVStoreDiscovery::new(kv::Manager::etcd(etcd_client), CancellationToken::new());
            client.is_model_identity_index_enabled = true;
            let ns = format!(
                "identity-{}",
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            );
            let model = |endpoint: &str, card_json: serde_json::Value, suffix: Option<&str>| {
                DiscoverySpec::Model {
                    namespace: ns.clone(),
                    component: "comp".to_string(),
                    endpoint: endpoint.to_string(),
                    card_json,
                    model_suffix: suffix.map(str::to_string),
                }
            };
            let mut registered = Vec::new();
            for spec in [
                model(
                    "ep",
                    serde_json::json!({"display_name": "base", "source_path": "/m/base"}),
                    None,
                ),
                model(
                    "ep",
                    serde_json::json!({
                        "display_name": "adapter",
                        "source_path": "/m/base",
                        "lora": {"name": "adapter"},
                    }),
                    Some("adapter"),
                ),
                model(
                    "ep-2",
                    serde_json::json!({"display_name": "other", "source_path": "/m/other"}),
                    None,
                ),
            ] {
                registered.push(client.register(spec).await.unwrap());
            }

            let names = |identities: Vec<ModelRegistrationIdentity>| {
                let mut names: Vec<_> = identities
                    .into_iter()
                    .map(|identity| identity.display_name)
                    .collect();
                names.sort();
                names
            };
            assert_eq!(
                names(
                    client
                        .endpoint_model_identities(&ns, "comp", "ep")
                        .await
                        .unwrap()
                ),
                ["adapter", "base"]
            );

            for instance in registered {
                client.unregister(instance).await.unwrap();
            }
            for endpoint in ["ep", "ep-2"] {
                assert!(
                    client
                        .endpoint_model_identities(&ns, "comp", endpoint)
                        .await
                        .unwrap()
                        .is_empty()
                );
            }
        });
    }

    /// On etcd, a scoped namespaced list and a scoped component watch still return every
    /// instance the query matches, next to a sibling namespace (`x-1` and `x-10`). The
    /// bucket-level test in `storage/kv/etcd.rs` shows the reads themselves are scoped.
    #[test]
    fn scoped_queries_return_every_matching_instance() {
        let rt = Runtime::single_threaded().unwrap();
        let rt_clone = rt.clone();

        rt_clone.primary().block_on(async move {
            let etcd_client =
                etcd_transport::Client::new(etcd_transport::ClientOptions::default(), rt)
                    .await
                    .unwrap();
            let client =
                KVStoreDiscovery::new(kv::Manager::etcd(etcd_client), CancellationToken::new());
            let run = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
            let (ns1, ns10) = (format!("scope-{run}-1"), format!("scope-{run}-10"));
            for (ns, comp, ep) in [
                (&ns1, "comp", "ep-a"),
                (&ns1, "comp", "ep-b"),
                (&ns1, "other", "ep-a"),
                (&ns10, "comp", "ep-a"),
            ] {
                client.register(endpoint(ns, comp, ep)).await.unwrap();
            }

            let namespaced = client
                .list(DiscoveryQuery::NamespacedEndpoints {
                    namespace: ns1.clone(),
                })
                .await
                .unwrap();
            assert_eq!(
                endpoint_names(&namespaced),
                [
                    format!("{ns1}/comp/ep-a"),
                    format!("{ns1}/comp/ep-b"),
                    format!("{ns1}/other/ep-a")
                ]
            );

            let mut stream = client
                .list_and_watch(
                    DiscoveryQuery::ComponentEndpoints {
                        namespace: ns1.clone(),
                        component: "comp".to_string(),
                    },
                    None,
                )
                .await
                .unwrap();
            let mut initial = Vec::new();
            loop {
                match stream.next().await.unwrap().unwrap() {
                    DiscoveryEvent::Added(instance) => initial.push(instance),
                    DiscoveryEvent::Resync(_) => break,
                    other => panic!("unexpected startup event {other:?}"),
                }
            }
            assert_eq!(
                endpoint_names(&initial),
                [format!("{ns1}/comp/ep-a"), format!("{ns1}/comp/ep-b")]
            );

            client
                .register(endpoint(&ns10, "comp", "ep-late"))
                .await
                .unwrap();
            client
                .register(endpoint(&ns1, "comp", "ep-late"))
                .await
                .unwrap();
            let next = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
                .await
                .expect("no event within 10 s")
                .unwrap()
                .unwrap();
            let DiscoveryEvent::Added(instance) = next else {
                panic!("expected an added endpoint, got {next:?}");
            };
            assert_eq!(endpoint_names(&[instance]), [format!("{ns1}/comp/ep-late")]);
        });
    }
}
