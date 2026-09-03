// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{DiscoveryMetadata, MetadataSnapshot};
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
use tokio::sync::Notify;

use super::crd::DynamoWorkerMetadata;
use super::utils::{KubeDiscoveryMode, PodInfo, extract_endpoint_info, extract_ready_containers};

#[derive(Clone)]
struct CachedCrMetadata {
    metadata: Arc<DiscoveryMetadata>,
    generation: i64,
    uid: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CrRevision {
    generation: i64,
    uid: Option<String>,
}

struct JoinTable {
    left: HashMap<String, u64>,
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

    fn apply_readiness_scan(&mut self, new_left: HashMap<String, u64>) -> bool {
        let mut changed = false;

        let departed: Vec<(String, u64)> = self
            .left
            .iter()
            .filter(|(k, _)| !new_left.contains_key(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        for (cr_key, instance_id) in departed {
            self.left.remove(&cr_key);
            if self.known.remove(&instance_id).is_some() {
                self.revisions.remove(&instance_id);
                tracing::info!(
                    cr_key = %cr_key,
                    instance_id = format!("{instance_id:x}"),
                    "Pod no longer ready, removed from known"
                );
                changed = true;
            }
        }

        let arrived: Vec<(String, u64)> = new_left
            .iter()
            .filter(|(k, _)| !self.left.contains_key(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        for (cr_key, instance_id) in arrived {
            self.left.insert(cr_key.clone(), instance_id);
            if let Some(cached) = self.right.get(&cr_key) {
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
                changed = true;
            }
        }

        changed
    }

    fn apply_cr_scan(&mut self, new_right: HashMap<String, CachedCrMetadata>) -> bool {
        let mut changed = false;

        let removed: Vec<String> = self
            .right
            .keys()
            .filter(|k| !new_right.contains_key(*k))
            .cloned()
            .collect();

        for cr_key in removed {
            self.right.remove(&cr_key);
            if let Some(&instance_id) = self.left.get(&cr_key)
                && self.known.remove(&instance_id).is_some()
            {
                self.revisions.remove(&instance_id);
                tracing::info!(
                    cr_key = %cr_key,
                    instance_id = format!("{instance_id:x}"),
                    "CR removed, evicted from known"
                );
                changed = true;
            }
        }

        for (cr_key, new_cached) in &new_right {
            let new_revision = CrRevision {
                generation: new_cached.generation,
                uid: new_cached.uid.clone(),
            };
            self.right.insert(cr_key.clone(), new_cached.clone());

            let Some(&instance_id) = self.left.get(cr_key) else {
                continue;
            };

            let old_revision = self.revisions.get(&instance_id).cloned();

            match self.known.entry(instance_id) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if old_revision.as_ref() != Some(&new_revision) {
                        e.insert(new_cached.metadata.clone());
                        self.revisions.insert(instance_id, new_revision);
                        tracing::debug!(
                            cr_key = %cr_key,
                            instance_id = format!("{instance_id:x}"),
                            "CR updated for ready pod"
                        );
                        changed = true;
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(new_cached.metadata.clone());
                    self.revisions.insert(instance_id, new_revision);
                    tracing::info!(
                        cr_key = %cr_key,
                        instance_id = format!("{instance_id:x}"),
                        "CR arrived for ready pod, added to known"
                    );
                    changed = true;
                }
            }
        }

        changed
    }

    fn to_snapshot(&self, sequence: u64) -> MetadataSnapshot {
        MetadataSnapshot {
            instances: self.known.clone(),
            generations: self
                .revisions
                .iter()
                .map(|(id, rev)| (*id, rev.generation))
                .collect(),
            sequence,
            timestamp: std::time::Instant::now(),
        }
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
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "EndpointSlice reflector updated"
                                );
                                notify.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("EndpointSlice reflector error: {e}");
                                notify.notify_one();
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
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "Pod reflector updated"
                                );
                                notify.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("Pod reflector error: {e}");
                                notify.notify_one();
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                Self::Pod(reader)
            }
        }
    }

    fn ready_entries(&self) -> HashMap<String, u64> {
        match self {
            Self::EndpointSlice(reader) => reader
                .state()
                .iter()
                .flat_map(|s| extract_endpoint_info(s.as_ref()))
                .map(|(id, key)| (key, id))
                .collect(),
            Self::Pod(reader) => reader
                .state()
                .iter()
                .flat_map(|p| extract_ready_containers(p.as_ref()))
                .map(|(id, key)| (key, id))
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
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
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
            .touched_objects()
            .for_each(move |res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            cr_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "DynamoWorkerMetadata CR reflector updated"
                        );
                        cr_notify_clone.notify_one();
                    }
                    Err(e) => {
                        tracing::warn!("DynamoWorkerMetadata CR reflector error: {e}");
                        cr_notify_clone.notify_one();
                    }
                }
                futures::future::ready(())
            });
        tokio::spawn(cr_reflector_stream);

        let mut sequence = 0u64;
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
                    if join_table.apply_readiness_scan(new_left) {
                        sequence += 1;
                        if watch_tx.send(Arc::new(join_table.to_snapshot(sequence))).is_err() {
                            tracing::debug!("No watch subscribers, daemon stopping");
                            break;
                        }
                    }
                }
                _ = cr_notify.notified() => {
                    tracing::trace!("CR store updated, scanning");
                    let new_right = scan_cr_store(&cr_reader, &mut valid_cr_cache);
                    if join_table.apply_cr_scan(new_right) {
                        sequence += 1;
                        if watch_tx.send(Arc::new(join_table.to_snapshot(sequence))).is_err() {
                            tracing::debug!("No watch subscribers, daemon stopping");
                            break;
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
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ManagedFieldsEntry;

    fn make_cached(uid: &str, generation: i64) -> CachedCrMetadata {
        CachedCrMetadata {
            metadata: Arc::new(DiscoveryMetadata::new()),
            generation,
            uid: Some(uid.to_string()),
        }
    }

    #[test]
    fn join_table_detects_cr_recreated_with_same_generation() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(HashMap::from([("worker-a".to_string(), 1u64)]));

        let changed = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));
        assert!(changed);
        assert!(table.known.contains_key(&1u64));

        let changed = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-2", 1),
        )]));
        assert!(
            changed,
            "UID change must be detected even at same generation"
        );
        assert_eq!(table.revisions[&1u64].uid.as_deref(), Some("uid-2"));
    }

    #[test]
    fn join_table_removes_immediately_when_pod_not_ready() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(HashMap::from([("worker-a".to_string(), 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));
        assert!(table.known.contains_key(&1u64));

        let changed = table.apply_readiness_scan(HashMap::new());
        assert!(changed);
        assert!(!table.known.contains_key(&1u64));
        assert!(table.revisions.is_empty());
    }

    #[test]
    fn join_table_adds_when_cr_arrives_after_pod_ready() {
        let mut table = JoinTable::new();

        let changed = table.apply_readiness_scan(HashMap::from([("worker-a".to_string(), 1u64)]));
        assert!(!changed, "no CR yet, should not enter known");
        assert!(!table.known.contains_key(&1u64));

        let changed = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));
        assert!(changed);
        assert!(table.known.contains_key(&1u64));
    }

    #[test]
    fn join_table_evicts_when_cr_removed() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(HashMap::from([("worker-a".to_string(), 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));
        assert!(table.known.contains_key(&1u64));

        let changed = table.apply_cr_scan(HashMap::new());
        assert!(changed);
        assert!(!table.known.contains_key(&1u64));
    }

    #[test]
    fn join_table_no_change_on_same_revision() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(HashMap::from([("worker-a".to_string(), 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));

        let changed = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached("uid-1", 1),
        )]));
        assert!(!changed);
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
