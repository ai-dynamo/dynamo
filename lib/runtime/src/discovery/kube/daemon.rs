// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryMetadata,
    reconcile_discovery_snapshot,
};
use anyhow::Result;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    Api, Client as KubeClient,
    runtime::{WatchStreamExt, reflector, watcher, watcher::Config},
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Notify, RwLock, broadcast};

use super::crd::DynamoWorkerMetadata;
use super::utils::{KubeDiscoveryMode, PodInfo, extract_endpoint_info, extract_ready_containers};

#[derive(Clone)]
struct CachedCrMetadata {
    metadata: Arc<DiscoveryMetadata>,
    generation: i64,
    uid: Option<String>,
    /// UID of the Pod that owns this CR (from `metadata.ownerReferences`).
    /// Required for incarnation-safe join: a new Pod with the same name must
    /// not match a CR whose owner is a different (old) Pod UID.
    owner_pod_uid: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CrRevision {
    generation: i64,
    uid: Option<String>,
}

struct JoinTable {
    /// cr_name → (instance_id, pod_uid). Pod UID carried so every join also
    /// validates the CR owner, preventing a new incarnation from matching an
    /// old incarnation's metadata before GC removes the stale CR.
    left: HashMap<String, (u64, String)>,
    right: HashMap<String, CachedCrMetadata>,
    known: HashMap<u64, Arc<DiscoveryMetadata>>,
    revisions: HashMap<u64, CrRevision>,
}

impl JoinTable {
    fn new() -> Self {
        Self {
            left: HashMap::new(),
            right: HashMap::new(),
            known: HashMap::new(),
            revisions: HashMap::new(),
        }
    }

    fn apply_readiness_scan(
        &mut self,
        new_left: HashMap<String, (u64, String)>,
    ) -> Vec<DiscoveryEvent> {
        let mut events = Vec::new();

        // Departed: cr_key gone entirely, or same cr_key but pod UID changed (new incarnation).
        let departed: Vec<(String, u64)> = self
            .left
            .iter()
            .filter(|(k, (_, old_uid))| {
                new_left
                    .get(*k)
                    .map(|(_, new_uid)| new_uid != old_uid)
                    .unwrap_or(true)
            })
            .map(|(k, (id, _))| (k.clone(), *id))
            .collect();

        for (cr_key, instance_id) in departed {
            self.left.remove(&cr_key);
            if let Some(metadata) = self.known.remove(&instance_id) {
                self.revisions.remove(&instance_id);
                for instance in metadata.get_all() {
                    events.push(DiscoveryEvent::Removed(instance.id()));
                }
                tracing::info!(
                    cr_key = %cr_key,
                    instance_id = format!("{instance_id:x}"),
                    "Pod no longer ready, removed from known"
                );
            }
        }

        // Arrived: new cr_key, or same cr_key with changed pod UID (departed processing already
        // removed the old entry from self.left, so !contains_key catches both cases).
        let arrived: Vec<(String, u64, String)> = new_left
            .iter()
            .filter(|(k, _)| !self.left.contains_key(*k))
            .map(|(k, (id, pod_uid))| (k.clone(), *id, pod_uid.clone()))
            .collect();

        for (cr_key, instance_id, pod_uid) in arrived {
            self.left
                .insert(cr_key.clone(), (instance_id, pod_uid.clone()));
            if let Some(cached) = self.right.get(&cr_key) {
                if cached.owner_pod_uid.as_deref() != Some(&pod_uid) {
                    tracing::debug!(
                        cr_key = %cr_key,
                        pod_uid = %pod_uid,
                        cr_owner = ?cached.owner_pod_uid,
                        "Pod UID does not match CR owner, not joining"
                    );
                    continue;
                }
                for instance in cached.metadata.get_all() {
                    events.push(DiscoveryEvent::Added(instance));
                }
                self.known.insert(instance_id, cached.metadata.clone());
                self.revisions.insert(
                    instance_id,
                    CrRevision {
                        generation: cached.generation,
                        uid: cached.uid.clone(),
                    },
                );
                tracing::info!(
                    cr_key = %cr_key,
                    instance_id = format!("{instance_id:x}"),
                    "Pod became ready, joined with existing CR"
                );
            }
        }

        events
    }

    fn apply_cr_scan(
        &mut self,
        new_right: HashMap<String, CachedCrMetadata>,
    ) -> Vec<DiscoveryEvent> {
        let mut events = Vec::new();

        let removed: Vec<String> = self
            .right
            .keys()
            .filter(|k| !new_right.contains_key(*k))
            .cloned()
            .collect();

        for cr_key in removed {
            self.right.remove(&cr_key);
            if let Some((instance_id, _)) = self.left.get(&cr_key) {
                let instance_id = *instance_id;
                if let Some(metadata) = self.known.remove(&instance_id) {
                    self.revisions.remove(&instance_id);
                    for instance in metadata.get_all() {
                        events.push(DiscoveryEvent::Removed(instance.id()));
                    }
                    tracing::info!(
                        cr_key = %cr_key,
                        instance_id = format!("{instance_id:x}"),
                        "CR removed, evicted from known"
                    );
                }
            }
        }

        for (cr_key, new_cached) in &new_right {
            let new_revision = CrRevision {
                generation: new_cached.generation,
                uid: new_cached.uid.clone(),
            };
            // insert returns the old entry so we can detect owner UID changes
            let old_right = self.right.insert(cr_key.clone(), new_cached.clone());

            let (instance_id, pod_uid) = match self.left.get(cr_key) {
                Some((id, uid)) => (*id, uid.clone()),
                None => continue,
            };

            let uid_matches = new_cached.owner_pod_uid.as_deref() == Some(&pod_uid);
            let old_uid_matched =
                old_right.as_ref().and_then(|o| o.owner_pod_uid.as_deref()) == Some(&pod_uid);

            if !uid_matches {
                if old_uid_matched {
                    // CR owner changed away from the current pod → evict
                    if let Some(metadata) = self.known.remove(&instance_id) {
                        self.revisions.remove(&instance_id);
                        for instance in metadata.get_all() {
                            events.push(DiscoveryEvent::Removed(instance.id()));
                        }
                        tracing::info!(
                            cr_key = %cr_key,
                            instance_id = format!("{instance_id:x}"),
                            pod_uid = %pod_uid,
                            cr_owner = ?new_cached.owner_pod_uid,
                            "CR owner changed, evicted from known"
                        );
                    }
                } else {
                    tracing::debug!(
                        cr_key = %cr_key,
                        pod_uid = %pod_uid,
                        cr_owner = ?new_cached.owner_pod_uid,
                        "Pod UID does not match CR owner, skipping join"
                    );
                }
                continue;
            }

            let old_revision = self.revisions.get(&instance_id).cloned();

            match self.known.entry(instance_id) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if old_revision.as_ref() != Some(&new_revision) {
                        let old_metadata = e.get().clone();
                        e.insert(new_cached.metadata.clone());
                        self.revisions.insert(instance_id, new_revision);
                        let old_flat: HashMap<DiscoveryInstanceId, DiscoveryInstance> =
                            old_metadata
                                .get_all()
                                .into_iter()
                                .map(|i| (i.id(), i))
                                .collect();
                        let new_flat: HashMap<DiscoveryInstanceId, DiscoveryInstance> = new_cached
                            .metadata
                            .get_all()
                            .into_iter()
                            .map(|i| (i.id(), i))
                            .collect();
                        let (diff, _) = reconcile_discovery_snapshot(&old_flat, new_flat);
                        events.extend(diff);
                        tracing::debug!(
                            cr_key = %cr_key,
                            instance_id = format!("{instance_id:x}"),
                            "CR updated for ready pod"
                        );
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    for instance in new_cached.metadata.get_all() {
                        events.push(DiscoveryEvent::Added(instance));
                    }
                    e.insert(new_cached.metadata.clone());
                    self.revisions.insert(instance_id, new_revision);
                    tracing::info!(
                        cr_key = %cr_key,
                        instance_id = format!("{instance_id:x}"),
                        "CR arrived for ready pod, added to known"
                    );
                }
            }
        }

        events
    }
}

enum DiscoverySource {
    EndpointSlice(reflector::Store<EndpointSlice>),
    Pod(reflector::Store<Pod>),
}

impl DiscoverySource {
    async fn new(pod_info: &PodInfo, kube_client: KubeClient, notify: Arc<Notify>) -> Self {
        let labels = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        match pod_info.mode {
            KubeDiscoveryMode::Pod => {
                let api: Api<EndpointSlice> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();

                tracing::info!("Daemon watching EndpointSlices (pod mode)");

                let stream = reflector(writer, watcher(api, labels))
                    .default_backoff()
                    .for_each(move |res| {
                        match res {
                            Ok(watcher::Event::Apply(_))
                            | Ok(watcher::Event::Delete(_))
                            | Ok(watcher::Event::InitDone) => {
                                notify.notify_one();
                            }
                            Ok(watcher::Event::Init) | Ok(watcher::Event::InitApply(_)) => {}
                            Err(e) => {
                                tracing::warn!("EndpointSlice reflector error: {e}");
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                Self::EndpointSlice(reader)
            }

            KubeDiscoveryMode::Container => {
                let api: Api<Pod> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();

                tracing::info!("Daemon watching Pods (container mode)");

                let stream = reflector(writer, watcher(api, labels))
                    .default_backoff()
                    .for_each(move |res| {
                        match res {
                            Ok(watcher::Event::Apply(_))
                            | Ok(watcher::Event::Delete(_))
                            | Ok(watcher::Event::InitDone) => {
                                notify.notify_one();
                            }
                            Ok(watcher::Event::Init) | Ok(watcher::Event::InitApply(_)) => {}
                            Err(e) => {
                                tracing::warn!("Pod reflector error: {e}");
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                Self::Pod(reader)
            }
        }
    }

    fn ready_entries(&self) -> HashMap<String, (u64, String)> {
        match self {
            Self::EndpointSlice(reader) => reader
                .state()
                .iter()
                .flat_map(|s| extract_endpoint_info(s.as_ref()))
                .map(|(id, key, pod_uid)| (key, (id, pod_uid)))
                .collect(),
            Self::Pod(reader) => reader
                .state()
                .iter()
                .flat_map(|p| extract_ready_containers(p.as_ref()))
                .map(|(id, key, pod_uid)| (key, (id, pod_uid)))
                .collect(),
        }
    }
}

/// Discovers and aggregates metadata from DynamoWorkerMetadata CRs in the cluster.
#[derive(Clone)]
pub(super) struct DiscoveryDaemon {
    kube_client: KubeClient,
    pod_info: PodInfo,
    cancel_token: CancellationToken,
}

impl DiscoveryDaemon {
    pub fn new(
        kube_client: KubeClient,
        pod_info: PodInfo,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        Ok(Self {
            kube_client,
            pod_info,
            cancel_token,
        })
    }

    pub async fn run(
        self,
        list_state: Arc<RwLock<HashMap<u64, Arc<DiscoveryMetadata>>>>,
        event_tx: broadcast::Sender<DiscoveryEvent>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        let es_notify = Arc::new(Notify::new());
        let cr_notify = Arc::new(Notify::new());

        let source =
            DiscoverySource::new(&self.pod_info, self.kube_client.clone(), es_notify.clone()).await;

        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
        let (cr_reader, cr_writer) = reflector::store();

        tracing::info!(
            "Daemon watching DynamoWorkerMetadata CRs in namespace: {}",
            self.pod_info.pod_namespace
        );

        let cr_notify_clone = cr_notify.clone();
        let cr_reflector_stream = reflector(cr_writer, watcher(metadata_crs, Config::default()))
            .default_backoff()
            .for_each(move |res| {
                match res {
                    Ok(watcher::Event::Apply(_))
                    | Ok(watcher::Event::Delete(_))
                    | Ok(watcher::Event::InitDone) => {
                        cr_notify_clone.notify_one();
                    }
                    Ok(watcher::Event::Init) | Ok(watcher::Event::InitApply(_)) => {}
                    Err(e) => {
                        tracing::warn!("DynamoWorkerMetadata CR reflector error: {e}");
                    }
                }
                futures::future::ready(())
            });
        tokio::spawn(cr_reflector_stream);

        let mut join_table = JoinTable::new();
        let mut valid_cr_cache: HashMap<String, CachedCrMetadata> = HashMap::new();

        loop {
            tokio::select! {
                biased;
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("Discovery daemon received cancellation");
                    break;
                }
                _ = es_notify.notified() => {
                    tracing::trace!("Readiness store updated, scanning");
                    let new_left = source.ready_entries();
                    let events = join_table.apply_readiness_scan(new_left);
                    if !events.is_empty() {
                        let mut state = list_state.write().await;
                        *state = join_table.known.clone();
                        for event in &events {
                            event_tx.send(event.clone()).ok();
                        }
                    }
                }
                _ = cr_notify.notified() => {
                    tracing::trace!("CR store updated, scanning");
                    let new_right = scan_cr_store(&cr_reader, &mut valid_cr_cache);
                    let events = join_table.apply_cr_scan(new_right);
                    if !events.is_empty() {
                        let mut state = list_state.write().await;
                        *state = join_table.known.clone();
                        for event in &events {
                            event_tx.send(event.clone()).ok();
                        }
                    }
                }
            }
        }

        tracing::info!("Discovery daemon stopped");
        Ok(())
    }
}

fn scan_cr_store(
    cr_reader: &reflector::Store<DynamoWorkerMetadata>,
    valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
) -> HashMap<String, CachedCrMetadata> {
    let cr_state = cr_reader.state();
    let mut new_right: HashMap<String, CachedCrMetadata> = HashMap::new();
    let mut observed: HashSet<String> = HashSet::new();

    for arc_cr in cr_state.iter() {
        let Some(cr_name) = arc_cr.metadata.name.as_ref() else {
            continue;
        };
        let generation = arc_cr.metadata.generation.unwrap_or(0);
        let uid = arc_cr.metadata.uid.clone();
        let resource_version = arc_cr
            .metadata
            .resource_version
            .as_deref()
            .unwrap_or("unknown");

        observed.insert(cr_name.clone());

        let owner_pod_uid = arc_cr
            .metadata
            .owner_references
            .as_ref()
            .and_then(|refs| refs.iter().find(|o| o.kind == "Pod"))
            .map(|o| o.uid.clone());

        if arc_cr.spec.data.is_null() {
            tracing::debug!(
                cr_name = %cr_name,
                uid = %uid.as_deref().unwrap_or("unknown"),
                resource_version = %resource_version,
                generation,
                managed_fields = ?managed_fields_summary(arc_cr.as_ref()),
                "DynamoWorkerMetadata CR has null spec.data; reusing last valid metadata if available"
            );
            if let Some(cached) =
                cached_metadata_for_invalid_cr(cr_name, uid.as_deref(), valid_cr_cache)
            {
                new_right.insert(cr_name.clone(), cached.clone());
            }
            continue;
        }

        match super::crd::deserialize_metadata(arc_cr.spec.data.clone()) {
            Ok(metadata) => {
                tracing::trace!("Loaded metadata from CR '{cr_name}'");
                let cached = CachedCrMetadata {
                    metadata: Arc::new(metadata),
                    generation,
                    uid,
                    owner_pod_uid,
                };
                valid_cr_cache.insert(cr_name.clone(), cached.clone());
                new_right.insert(cr_name.clone(), cached);
            }
            Err(e) => {
                tracing::warn!(
                    cr_name = %cr_name,
                    uid = %uid.as_deref().unwrap_or("unknown"),
                    resource_version = %resource_version,
                    generation,
                    managed_fields = ?managed_fields_summary(arc_cr.as_ref()),
                    error = %e,
                    "Failed to deserialize metadata from DynamoWorkerMetadata CR"
                );
                if let Some(cached) =
                    cached_metadata_for_invalid_cr(cr_name, uid.as_deref(), valid_cr_cache)
                {
                    new_right.insert(cr_name.clone(), cached.clone());
                }
            }
        }
    }

    valid_cr_cache.retain(|cr_name, _| observed.contains(cr_name));

    tracing::trace!(
        "CR scan: {} valid entries from {} observed CRs",
        new_right.len(),
        observed.len()
    );

    new_right
}

fn cached_metadata_for_invalid_cr<'a>(
    cr_key: &str,
    uid: Option<&str>,
    valid_cr_cache: &'a HashMap<String, CachedCrMetadata>,
) -> Option<&'a CachedCrMetadata> {
    let cached = valid_cr_cache.get(cr_key)?;
    if cached.uid.as_deref() == uid {
        Some(cached)
    } else {
        None
    }
}

fn managed_fields_summary(cr: &DynamoWorkerMetadata) -> Option<String> {
    let managed_fields = cr.metadata.managed_fields.as_ref()?;

    if managed_fields.is_empty() {
        return None;
    }

    Some(
        managed_fields
            .iter()
            .map(|entry| {
                let manager = entry.manager.as_deref().unwrap_or("unknown");
                let operation = entry.operation.as_deref().unwrap_or("unknown");
                let api_version = entry.api_version.as_deref().unwrap_or("unknown");
                let subresource = entry
                    .subresource
                    .as_deref()
                    .filter(|subresource| !subresource.is_empty())
                    .unwrap_or("-");
                let time = entry
                    .time
                    .as_ref()
                    .map(|time| time.0.to_rfc3339())
                    .unwrap_or_else(|| "unknown".to_string());

                format!("{manager}/{operation}/{api_version}/subresource={subresource}/time={time}")
            })
            .collect::<Vec<_>>()
            .join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Instance, TransportType};
    use crate::discovery::DiscoveryEvent;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ManagedFieldsEntry;

    const TEST_POD_UID: &str = "pod-uid-test";

    fn make_cached(uid: &str, generation: i64) -> CachedCrMetadata {
        CachedCrMetadata {
            metadata: Arc::new(DiscoveryMetadata::new()),
            generation,
            uid: Some(uid.to_string()),
            owner_pod_uid: Some(TEST_POD_UID.to_string()),
        }
    }

    fn make_cached_with_endpoint(uid: &str, generation: i64) -> CachedCrMetadata {
        let mut meta = DiscoveryMetadata::new();
        meta.register_endpoint(DiscoveryInstance::Endpoint(Instance {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: "ep".to_string(),
            instance_id: 99,
            transport: TransportType::Tcp("127.0.0.1:1234".to_string()),
            device_type: None,
            request_plane_codec: None,
        }))
        .unwrap();
        CachedCrMetadata {
            metadata: Arc::new(meta),
            generation,
            uid: Some(uid.to_string()),
            owner_pod_uid: Some(TEST_POD_UID.to_string()),
        }
    }

    fn readiness(entries: &[(&str, u64)]) -> HashMap<String, (u64, String)> {
        entries
            .iter()
            .map(|(k, id)| (k.to_string(), (*id, TEST_POD_UID.to_string())))
            .collect()
    }

    #[test]
    fn join_table_detects_cr_recreated_with_same_generation() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));

        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));
        assert!(table.known.contains_key(&1u64));

        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-2", 1),
        )]));
        // UID change is detected (revision updated) even though same-content metadata
        // produces no events.
        assert_eq!(
            table.revisions[&1u64].uid.as_deref(),
            Some("uid-2"),
            "UID change must be detected even at same generation"
        );
    }

    #[test]
    fn join_table_removes_immediately_when_pod_not_ready() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));
        assert!(table.known.contains_key(&1u64));

        let events = table.apply_readiness_scan(HashMap::new());
        assert!(!table.known.contains_key(&1u64));
        assert!(table.revisions.is_empty());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DiscoveryEvent::Removed(_)))
        );
    }

    #[test]
    fn join_table_adds_when_cr_arrives_after_pod_ready() {
        let mut table = JoinTable::new();

        let events = table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));
        assert!(events.is_empty(), "no CR yet, should have no events");
        assert!(!table.known.contains_key(&1u64));

        let events = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));
        assert!(!events.is_empty());
        assert!(table.known.contains_key(&1u64));
    }

    #[test]
    fn join_table_evicts_when_cr_removed() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));
        assert!(table.known.contains_key(&1u64));

        let events = table.apply_cr_scan(HashMap::new());
        assert!(!table.known.contains_key(&1u64));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DiscoveryEvent::Removed(_)))
        );
    }

    #[test]
    fn join_table_no_change_on_same_revision() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));

        let events = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1", 1),
        )]));
        assert!(events.is_empty());
    }

    #[test]
    fn cached_metadata_for_invalid_cr_reuses_same_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1", 7));

        let cached = cached_metadata_for_invalid_cr("worker-a", Some("uid-1"), &cache)
            .expect("cache should be reused for the same CR UID");

        assert_eq!(cached.generation, 7);
    }

    #[test]
    fn cached_metadata_for_invalid_cr_rejects_recreated_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1", 7));

        assert!(cached_metadata_for_invalid_cr("worker-a", Some("uid-2"), &cache).is_none());
    }

    #[test]
    fn join_requires_matching_pod_uid() {
        // Pod U2 arrives while old CR still has owner=U1 — must not join.
        let mut table = JoinTable::new();

        let new_left: HashMap<String, (u64, String)> =
            HashMap::from([("worker-0".to_string(), (1u64, "pod-uid-U2".to_string()))]);
        table.apply_readiness_scan(new_left);

        let mut cr = make_cached_with_endpoint("cr-uid-1", 1);
        cr.owner_pod_uid = Some("pod-uid-U1".to_string()); // old owner
        let events = table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr)]));

        assert!(
            events.is_empty(),
            "stale CR owner must not join new pod incarnation"
        );
        assert!(!table.known.contains_key(&1u64));
    }

    #[test]
    fn join_succeeds_when_pod_uid_matches_cr_owner() {
        let mut table = JoinTable::new();

        let new_left: HashMap<String, (u64, String)> =
            HashMap::from([("worker-0".to_string(), (1u64, "pod-uid-U1".to_string()))]);
        table.apply_readiness_scan(new_left);

        let mut cr = make_cached_with_endpoint("cr-uid-1", 1);
        cr.owner_pod_uid = Some("pod-uid-U1".to_string());
        let events = table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr)]));

        assert!(
            !events.is_empty(),
            "matching UIDs must produce Added events"
        );
        assert!(table.known.contains_key(&1u64));
    }

    #[test]
    fn new_pod_replaces_old_pod_after_uid_change() {
        // Full incarnation cycle: U1 joins, U2 replaces, then U2's CR arrives.
        let mut table = JoinTable::new();

        // U1 ready + CR owner U1 → joined
        let mut cr_u1 = make_cached_with_endpoint("cr-uid-1", 1);
        cr_u1.owner_pod_uid = Some("pod-uid-U1".to_string());
        table.apply_readiness_scan(HashMap::from([(
            "worker-0".to_string(),
            (1u64, "pod-uid-U1".to_string()),
        )]));
        table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr_u1.clone())]));
        assert!(table.known.contains_key(&1u64), "U1 should be in known");

        // U2 replaces U1 in readiness (EndpointSlice updated)
        let events = table.apply_readiness_scan(HashMap::from([(
            "worker-0".to_string(),
            (1u64, "pod-uid-U2".to_string()),
        )]));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DiscoveryEvent::Removed(_))),
            "U1 departure must emit Removed"
        );
        assert!(!table.known.contains_key(&1u64), "U1 should be evicted");

        // Old CR still present (GC hasn't run) — must not rejoin U2
        let events = table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr_u1.clone())]));
        assert!(
            events.is_empty(),
            "old CR must not rejoin new pod incarnation"
        );

        // New CR with owner U2 arrives → U2 joins
        let mut cr_u2 = make_cached_with_endpoint("cr-uid-2", 1);
        cr_u2.owner_pod_uid = Some("pod-uid-U2".to_string());
        let events = table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr_u2)]));
        assert!(
            events.iter().any(|e| matches!(e, DiscoveryEvent::Added(_))),
            "U2 + matching CR must produce Added"
        );
        assert!(table.known.contains_key(&1u64), "U2 should be in known");
    }

    #[test]
    fn cr_owner_change_evicts_joined_pod() {
        // CR owner changes in-place while pod U1 is still ready → evict U1.
        let mut table = JoinTable::new();

        let mut cr = make_cached_with_endpoint("cr-uid-1", 1);
        cr.owner_pod_uid = Some("pod-uid-U1".to_string());
        table.apply_readiness_scan(HashMap::from([(
            "worker-0".to_string(),
            (1u64, "pod-uid-U1".to_string()),
        )]));
        table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr)]));
        assert!(table.known.contains_key(&1u64));

        // CR updated with new owner U2 (in-place, same CR object, different owner)
        let mut cr_new_owner = make_cached_with_endpoint("cr-uid-1", 2);
        cr_new_owner.owner_pod_uid = Some("pod-uid-U2".to_string());
        let events = table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr_new_owner)]));

        assert!(
            events
                .iter()
                .any(|e| matches!(e, DiscoveryEvent::Removed(_))),
            "CR owner change must evict the joined pod"
        );
        assert!(!table.known.contains_key(&1u64));
    }

    #[test]
    fn managed_fields_summary_names_field_managers() {
        let mut cr = DynamoWorkerMetadata::new(
            "worker-a",
            super::super::crd::DynamoWorkerMetadataSpec::new(serde_json::Value::Null),
        );
        cr.metadata.managed_fields = Some(vec![ManagedFieldsEntry {
            manager: Some("dynamo-worker".to_string()),
            operation: Some("Apply".to_string()),
            api_version: Some("nvidia.com/v1alpha1".to_string()),
            ..Default::default()
        }]);

        let summary = managed_fields_summary(&cr).expect("managed fields should produce a summary");

        assert!(summary.contains("dynamo-worker/Apply/nvidia.com/v1alpha1"));
    }

    #[test]
    fn managed_fields_summary_returns_none_without_field_managers() {
        let cr = DynamoWorkerMetadata::new(
            "worker-a",
            super::super::crd::DynamoWorkerMetadataSpec::new(serde_json::Value::Null),
        );

        assert!(managed_fields_summary(&cr).is_none());
    }
}
