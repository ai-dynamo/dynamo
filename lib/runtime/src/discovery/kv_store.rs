// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio::sync::{OnceCell, broadcast, oneshot};
use tokio_util::sync::CancellationToken;

use super::{
    ClaimCloseOutcome, ClaimEvent, ClaimOutcome, ClaimPayload, ClaimPayloadFuture, Discovery,
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoverySpec,
    DiscoveryStream, EndpointInstanceId, EventChannelInstanceId, ModelCardInstanceId,
};
use crate::storage::kv;

const INSTANCES_BUCKET: &str = "v1/instances";
const MODELS_BUCKET: &str = "v1/mdc";
const EVENT_CHANNELS_BUCKET: &str = "v1/event_channels";
const CLAIMS_BUCKET: &str = "v1/claims";
const CLAIM_CREATE_ATTEMPTS: usize = 3;
const CLAIM_WATCH_RECONNECT_BACKOFF: Duration = Duration::from_millis(250);

/// Discovery implementation backed by a kv::Store
pub struct KVStoreDiscovery {
    store: Arc<kv::Manager>,
    cancel_token: CancellationToken,
    claims: ClaimState,
}

/// Process-local invalidation relay for the shared claims bucket.
///
/// One backend watcher serves all affinity coordinators attached to this discovery
/// instance. `Put` events never populate caches. `Delete(key)` evicts one entry, while
/// watcher loss, restart, or subscriber lag produces `Reset` so coordinators clear all
/// entries rather than retain potentially stale bindings.
///
/// TODO: `Bucket::watch` cannot yet surface etcd reconnect/compaction or FileStore
/// overflow errors, so those hidden backend failures cannot be converted into `Reset`.
struct ClaimState {
    events: broadcast::Sender<ClaimEvent>,
    watcher_started: OnceCell<()>,
    memory_warning_emitted: AtomicBool,
    #[cfg(test)]
    watcher_probe: ClaimWatcherProbe,
}

impl ClaimState {
    fn new() -> Self {
        let (events, _) = broadcast::channel(1024);
        Self {
            events,
            watcher_started: OnceCell::new(),
            memory_warning_emitted: AtomicBool::new(false),
            #[cfg(test)]
            watcher_probe: ClaimWatcherProbe::new(),
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<ClaimEvent> {
        self.events.subscribe()
    }

    fn warn_if_memory(&self, is_memory: bool) {
        if !is_memory || self.memory_warning_emitted.swap(true, Ordering::Relaxed) {
            return;
        }

        tracing::warn!(
            "session affinity claims use MemoryStore and coordinate only within this process/store"
        );
    }
}

#[cfg(test)]
struct ClaimWatcherProbe {
    start_count: Arc<std::sync::atomic::AtomicUsize>,
    active_count: Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
impl ClaimWatcherProbe {
    fn new() -> Self {
        Self {
            start_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            active_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    fn record_start(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        self.start_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.active_count.clone()
    }

    fn starts(&self) -> usize {
        self.start_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn active(&self) -> usize {
        self.active_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
struct ClaimWatcherActiveGuard(Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl ClaimWatcherActiveGuard {
    fn new(active_count: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        active_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(active_count)
    }
}

#[cfg(test)]
impl Drop for ClaimWatcherActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl KVStoreDiscovery {
    pub fn new(store: kv::Manager, cancel_token: CancellationToken) -> Self {
        Self {
            store: Arc::new(store),
            cancel_token,
            claims: ClaimState::new(),
        }
    }

    async fn ensure_claim_watcher(&self) -> Result<()> {
        self.claims
            .watcher_started
            .get_or_try_init(|| async {
                let (ready_tx, ready_rx) = oneshot::channel();
                let store = self.store.clone();
                let cancel_token = self.cancel_token.clone();
                let claim_events = self.claims.events.clone();
                #[cfg(test)]
                let active_count = self.claims.watcher_probe.record_start();

                tokio::spawn(async move {
                    #[cfg(test)]
                    let _active_guard = ClaimWatcherActiveGuard::new(active_count);
                    Self::run_claim_watcher(store, cancel_token, claim_events, ready_tx).await;
                });

                ready_rx
                    .await
                    .context("claim watcher stopped before startup completed")?
                    .map_err(anyhow::Error::msg)
            })
            .await?;
        Ok(())
    }

    async fn run_claim_watcher(
        store: Arc<kv::Manager>,
        cancel_token: CancellationToken,
        claim_events: broadcast::Sender<ClaimEvent>,
        ready_tx: oneshot::Sender<std::result::Result<(), String>>,
    ) {
        let mut ready_tx = Some(ready_tx);

        loop {
            if cancel_token.is_cancelled() {
                if let Some(ready_tx) = ready_tx.take() {
                    let _ = ready_tx.send(Err("claim watcher startup was cancelled".to_string()));
                }
                let _ = claim_events.send(ClaimEvent::Reset);
                return;
            }

            let bucket = match store.get_or_create_bucket(CLAIMS_BUCKET, None).await {
                Ok(bucket) => bucket,
                Err(err) => {
                    if let Some(ready_tx) = ready_tx.take() {
                        let _ = ready_tx.send(Err(err.to_string()));
                        return;
                    }
                    tracing::error!(error = %err, "failed to reconnect session claim watcher");
                    let _ = claim_events.send(ClaimEvent::Reset);
                    if Self::wait_for_claim_watcher_retry(&cancel_token).await {
                        return;
                    }
                    continue;
                }
            };

            let mut stream = match bucket.watch().await {
                Ok(stream) => stream,
                Err(err) => {
                    if let Some(ready_tx) = ready_tx.take() {
                        let _ = ready_tx.send(Err(err.to_string()));
                        return;
                    }
                    tracing::error!(error = %err, "failed to reconnect session claim watch stream");
                    let _ = claim_events.send(ClaimEvent::Reset);
                    if Self::wait_for_claim_watcher_retry(&cancel_token).await {
                        return;
                    }
                    continue;
                }
            };

            if let Some(ready_tx) = ready_tx.take() {
                let _ = ready_tx.send(Ok(()));
            } else {
                let _ = claim_events.send(ClaimEvent::Reset);
            }

            loop {
                let event = tokio::select! {
                    _ = cancel_token.cancelled() => {
                        let _ = claim_events.send(ClaimEvent::Reset);
                        return;
                    }
                    event = stream.next() => event,
                };

                let Some(event) = event else {
                    tracing::warn!(
                        "session claim watch stream ended; clearing local affinity caches"
                    );
                    let _ = claim_events.send(ClaimEvent::Reset);
                    break;
                };

                if let Some(event) = Self::claim_event_from_watch_event(event) {
                    let _ = claim_events.send(event);
                }
            }

            if Self::wait_for_claim_watcher_retry(&cancel_token).await {
                return;
            }
        }
    }

    async fn wait_for_claim_watcher_retry(cancel_token: &CancellationToken) -> bool {
        tokio::select! {
            _ = cancel_token.cancelled() => true,
            _ = tokio::time::sleep(CLAIM_WATCH_RECONNECT_BACKOFF) => false,
        }
    }

    fn claim_event_from_watch_event(event: kv::WatchEvent) -> Option<ClaimEvent> {
        match event {
            kv::WatchEvent::Delete(key) => {
                let key = Self::strip_bucket_prefix(key.as_ref(), CLAIMS_BUCKET).to_string();
                Some(ClaimEvent::Delete(key))
            }
            kv::WatchEvent::Resync(_) => Some(ClaimEvent::Reset),
            kv::WatchEvent::Put(_) => None,
        }
    }

    fn warn_if_memory_claims(&self) {
        self.claims.warn_if_memory(self.store.is_memory());
    }

    fn parse_claim(value: &[u8]) -> Result<ClaimPayload> {
        serde_json::from_slice(value).context("failed to deserialize session affinity claim")
    }

    async fn create_or_get_in_bucket(
        bucket: &dyn kv::Bucket,
        key: &kv::Key,
        proposed_payload: &mut ClaimPayloadFuture<'_>,
    ) -> Result<ClaimOutcome> {
        if let Some(payload) = bucket.get(key).await? {
            return Ok(ClaimOutcome::Existing(Self::parse_claim(&payload)?));
        }

        let proposed_payload = proposed_payload.as_mut().await?;
        let proposed_bytes = serde_json::to_vec(&proposed_payload)?;

        for attempt in 0..CLAIM_CREATE_ATTEMPTS {
            match bucket.insert(key, proposed_bytes.clone().into(), 0).await? {
                kv::StoreOutcome::Created(_) => {
                    return Ok(ClaimOutcome::Created(proposed_payload));
                }
                kv::StoreOutcome::Exists(_) => {
                    if let Some(payload) = bucket.get(key).await? {
                        return Ok(ClaimOutcome::Existing(Self::parse_claim(&payload)?));
                    }

                    if attempt + 1 == CLAIM_CREATE_ATTEMPTS {
                        anyhow::bail!(
                            "session affinity claim disappeared after {CLAIM_CREATE_ATTEMPTS} competing insert attempts"
                        );
                    }
                }
            }
        }

        unreachable!("claim creation loop always returns")
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

        relative_key == relative_prefix
            || relative_key
                .strip_prefix(relative_prefix)
                .is_some_and(|suffix| suffix.starts_with('/'))
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
                        }
                    }
                }

                let mut events = Vec::new();
                for (id, instance) in &next_instances {
                    if known_instances.get(id) != Some(instance) {
                        events.push(DiscoveryEvent::Added(instance.clone()));
                    }
                }

                for id in known_instances.keys() {
                    if !next_instances.contains_key(id) {
                        events.push(DiscoveryEvent::Removed(id.clone()));
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

    async fn create_or_get_claim(
        &self,
        key: &str,
        proposed_payload: &mut ClaimPayloadFuture<'_>,
    ) -> Result<ClaimOutcome> {
        self.warn_if_memory_claims();
        self.ensure_claim_watcher().await?;

        let bucket = self.store.get_or_create_bucket(CLAIMS_BUCKET, None).await?;
        let key = kv::Key::new(key.to_string());

        Self::create_or_get_in_bucket(bucket.as_ref(), &key, proposed_payload).await
    }

    async fn close_claim(&self, key: &str) -> Result<ClaimCloseOutcome> {
        self.warn_if_memory_claims();
        self.ensure_claim_watcher().await?;

        let Some(bucket) = self.store.get_bucket(CLAIMS_BUCKET).await? else {
            return Ok(ClaimCloseOutcome::Closed);
        };
        let key = kv::Key::new(key.to_string());

        match bucket.delete(&key).await {
            Ok(()) | Err(kv::StoreError::MissingBucket(_) | kv::StoreError::MissingKey(_)) => {
                Ok(ClaimCloseOutcome::Closed)
            }
            Err(err) => Err(err.into()),
        }
    }

    fn subscribe_claim_events(&self) -> Option<broadcast::Receiver<ClaimEvent>> {
        Some(self.claims.subscribe())
    }

    fn shutdown(&self) {
        self.store.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        assert!(events.contains(&DiscoveryEvent::Added(third)));
        assert!(events.contains(&DiscoveryEvent::Removed(endpoint_instance(1).id())));
        assert_eq!(known_instances.len(), 2);
        assert!(known_instances.contains_key(&endpoint_instance(2).id()));
        assert!(known_instances.contains_key(&endpoint_instance(3).id()));
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
    fn test_claim_resync_resets_claim_caches() {
        let mut snapshot = HashMap::new();
        snapshot.insert(
            kv::Key::new("scope/session".to_string()),
            serde_json::to_vec(&payload(1)).unwrap().into(),
        );

        assert_eq!(
            KVStoreDiscovery::claim_event_from_watch_event(kv::WatchEvent::Resync(snapshot)),
            Some(ClaimEvent::Reset)
        );
    }

    fn payload(worker_id: u64) -> ClaimPayload {
        serde_json::json!({"worker_id": worker_id, "dp_rank": 0})
    }

    struct DisappearingBucket {
        insert_calls: AtomicUsize,
        create_on_call: Option<usize>,
    }

    struct InsertBarrierBucket {
        inner: Box<dyn kv::Bucket>,
        barrier: tokio::sync::Barrier,
    }

    #[async_trait]
    impl kv::Bucket for InsertBarrierBucket {
        async fn insert(
            &self,
            key: &kv::Key,
            value: bytes::Bytes,
            revision: u64,
        ) -> std::result::Result<kv::StoreOutcome, kv::StoreError> {
            self.barrier.wait().await;
            self.inner.insert(key, value, revision).await
        }

        async fn get(
            &self,
            key: &kv::Key,
        ) -> std::result::Result<Option<bytes::Bytes>, kv::StoreError> {
            self.inner.get(key).await
        }

        async fn delete(&self, key: &kv::Key) -> std::result::Result<(), kv::StoreError> {
            self.inner.delete(key).await
        }

        async fn watch(
            &self,
        ) -> std::result::Result<
            Pin<Box<dyn futures::Stream<Item = kv::WatchEvent> + Send + '_>>,
            kv::StoreError,
        > {
            self.inner.watch().await
        }

        async fn entries(
            &self,
        ) -> std::result::Result<HashMap<kv::Key, bytes::Bytes>, kv::StoreError> {
            self.inner.entries().await
        }
    }

    #[async_trait]
    impl kv::Bucket for DisappearingBucket {
        async fn insert(
            &self,
            _key: &kv::Key,
            _value: bytes::Bytes,
            _revision: u64,
        ) -> std::result::Result<kv::StoreOutcome, kv::StoreError> {
            let call = self.insert_calls.fetch_add(1, Ordering::Relaxed);
            Ok(if self.create_on_call == Some(call) {
                kv::StoreOutcome::Created(1)
            } else {
                kv::StoreOutcome::Exists(1)
            })
        }

        async fn get(
            &self,
            _key: &kv::Key,
        ) -> std::result::Result<Option<bytes::Bytes>, kv::StoreError> {
            Ok(None)
        }

        async fn delete(&self, _key: &kv::Key) -> std::result::Result<(), kv::StoreError> {
            Ok(())
        }

        async fn watch(
            &self,
        ) -> std::result::Result<
            Pin<Box<dyn futures::Stream<Item = kv::WatchEvent> + Send + '_>>,
            kv::StoreError,
        > {
            Ok(Box::pin(futures::stream::pending()))
        }

        async fn entries(
            &self,
        ) -> std::result::Result<HashMap<kv::Key, bytes::Bytes>, kv::StoreError> {
            Ok(HashMap::new())
        }
    }

    #[tokio::test]
    async fn existing_claim_does_not_poll_proposal() {
        let client = KVStoreDiscovery::new(kv::Manager::memory(), CancellationToken::new());
        let mut first: ClaimPayloadFuture<'_> = Box::pin(async { Ok(payload(7)) });
        assert_eq!(
            client
                .create_or_get_claim("scope/session", &mut first)
                .await
                .unwrap(),
            ClaimOutcome::Created(payload(7))
        );

        let polled = Arc::new(AtomicBool::new(false));
        let proposal_polled = polled.clone();
        let mut second: ClaimPayloadFuture<'_> = Box::pin(async move {
            proposal_polled.store(true, Ordering::Relaxed);
            Ok(payload(8))
        });
        assert_eq!(
            client
                .create_or_get_claim("scope/session", &mut second)
                .await
                .unwrap(),
            ClaimOutcome::Existing(payload(7))
        );
        assert!(!polled.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn competing_claims_return_one_created_winner() {
        let store = kv::Manager::memory();
        let bucket = Arc::new(InsertBarrierBucket {
            inner: store
                .get_or_create_bucket(CLAIMS_BUCKET, None)
                .await
                .unwrap(),
            barrier: tokio::sync::Barrier::new(8),
        });
        let key = Arc::new(kv::Key::new("scope/race".to_string()));
        let mut tasks = Vec::new();
        for worker_id in 0..8 {
            let bucket = bucket.clone();
            let key = key.clone();
            tasks.push(tokio::spawn(async move {
                let mut proposal: ClaimPayloadFuture<'_> =
                    Box::pin(async move { Ok(payload(worker_id)) });
                KVStoreDiscovery::create_or_get_in_bucket(
                    bucket.as_ref(),
                    key.as_ref(),
                    &mut proposal,
                )
                .await
                .unwrap()
            }));
        }

        let outcomes = futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, ClaimOutcome::Created(_)))
                .count(),
            1
        );
        let winner = match outcomes
            .iter()
            .find(|outcome| matches!(outcome, ClaimOutcome::Created(_)))
            .unwrap()
        {
            ClaimOutcome::Created(payload) => payload,
            _ => unreachable!(),
        };
        assert!(outcomes.iter().all(|outcome| match outcome {
            ClaimOutcome::Created(payload) | ClaimOutcome::Existing(payload) => payload == winner,
            ClaimOutcome::Unsupported => false,
        }));
    }

    #[tokio::test]
    async fn claim_disappearance_retries_and_is_bounded() {
        let key = kv::Key::new("scope/disappearing".to_string());
        let recovering = DisappearingBucket {
            insert_calls: AtomicUsize::new(0),
            create_on_call: Some(1),
        };
        let mut proposal: ClaimPayloadFuture<'_> = Box::pin(async { Ok(payload(7)) });
        assert_eq!(
            KVStoreDiscovery::create_or_get_in_bucket(&recovering, &key, &mut proposal)
                .await
                .unwrap(),
            ClaimOutcome::Created(payload(7))
        );
        assert_eq!(recovering.insert_calls.load(Ordering::Relaxed), 2);

        let exhausting = DisappearingBucket {
            insert_calls: AtomicUsize::new(0),
            create_on_call: None,
        };
        let mut proposal: ClaimPayloadFuture<'_> = Box::pin(async { Ok(payload(8)) });
        let error = KVStoreDiscovery::create_or_get_in_bucket(&exhausting, &key, &mut proposal)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("3 competing insert attempts"));
        assert_eq!(exhausting.insert_calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn claim_watcher_ignores_put_emits_delete_and_close_is_idempotent() {
        let cancel = CancellationToken::new();
        let client = KVStoreDiscovery::new(kv::Manager::memory(), cancel.clone());
        let mut events = client.subscribe_claim_events().unwrap();
        let mut proposal: ClaimPayloadFuture<'_> = Box::pin(async { Ok(payload(7)) });
        client
            .create_or_get_claim("scope/close", &mut proposal)
            .await
            .unwrap();

        assert_eq!(client.claims.watcher_probe.starts(), 1);
        assert_eq!(client.claims.watcher_probe.active(), 1);

        assert_eq!(
            client.close_claim("scope/close").await.unwrap(),
            ClaimCloseOutcome::Closed
        );
        assert_eq!(
            events.recv().await.unwrap(),
            ClaimEvent::Delete("scope/close".to_string())
        );
        assert_eq!(
            client.close_claim("scope/close").await.unwrap(),
            ClaimCloseOutcome::Closed
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn claim_watcher_stops_on_cancellation() {
        let store = Arc::new(kv::Manager::memory());
        let cancel = CancellationToken::new();
        let (events, _) = broadcast::channel(16);
        let (ready_tx, ready_rx) = oneshot::channel();
        let watcher = tokio::spawn(KVStoreDiscovery::run_claim_watcher(
            store,
            cancel.clone(),
            events,
            ready_tx,
        ));
        ready_rx.await.unwrap().unwrap();

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), watcher)
            .await
            .expect("claim watcher did not stop after cancellation")
            .unwrap();
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
