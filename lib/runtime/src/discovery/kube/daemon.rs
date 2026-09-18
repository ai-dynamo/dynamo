// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{DiscoveryEvent, DiscoveryMetadata};
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
use tokio::sync::{RwLock, broadcast, mpsc};

use super::crd::DynamoWorkerMetadata;
use super::utils::{KubeDiscoveryMode, PodInfo, extract_endpoint_info, extract_ready_containers};

mod state;

use state::{BatchChanges, CachedCrMetadata, JoinTable, ReadinessIndex, ReadyEntry, StateChange};

const SOURCE_CHANNEL_CAPACITY: usize = 1024;

#[derive(Debug, PartialEq, Eq)]
enum ReadinessEvent {
    Apply {
        object_key: String,
        entries: Vec<(String, ReadyEntry)>,
    },
    Delete {
        object_key: String,
    },
    Rebuild,
}

enum CrEvent {
    Apply(DynamoWorkerMetadata),
    Delete(DynamoWorkerMetadata),
    Rebuild,
}

enum DiscoverySource {
    EndpointSlice(reflector::Store<EndpointSlice>),
    Pod(reflector::Store<Pod>),
}

fn endpoint_slice_update(slice: &EndpointSlice) -> Option<(String, Vec<(String, ReadyEntry)>)> {
    let object_key = slice.metadata.name.clone()?;
    let entries = extract_endpoint_info(slice)
        .into_iter()
        .map(|(instance_id, cr_key, pod_uid)| (cr_key, ReadyEntry::new(instance_id, pod_uid)))
        .collect();
    Some((object_key, entries))
}

fn pod_update(pod: &Pod) -> Option<(String, Vec<(String, ReadyEntry)>)> {
    let object_key = pod.metadata.name.clone()?;
    let entries = extract_ready_containers(pod)
        .into_iter()
        .map(|(instance_id, cr_key, pod_uid)| (cr_key, ReadyEntry::new(instance_id, pod_uid)))
        .collect();
    Some((object_key, entries))
}

fn endpoint_slice_event(event: watcher::Event<EndpointSlice>) -> Option<ReadinessEvent> {
    match event {
        watcher::Event::Apply(slice) => {
            endpoint_slice_update(&slice).map(|(object_key, entries)| ReadinessEvent::Apply {
                object_key,
                entries,
            })
        }
        watcher::Event::Delete(slice) => slice
            .metadata
            .name
            .map(|object_key| ReadinessEvent::Delete { object_key }),
        watcher::Event::InitDone => Some(ReadinessEvent::Rebuild),
        watcher::Event::Init | watcher::Event::InitApply(_) => None,
    }
}

fn pod_event(event: watcher::Event<Pod>) -> Option<ReadinessEvent> {
    match event {
        watcher::Event::Apply(pod) => {
            pod_update(&pod).map(|(object_key, entries)| ReadinessEvent::Apply {
                object_key,
                entries,
            })
        }
        watcher::Event::Delete(pod) => pod
            .metadata
            .name
            .map(|object_key| ReadinessEvent::Delete { object_key }),
        watcher::Event::InitDone => Some(ReadinessEvent::Rebuild),
        watcher::Event::Init | watcher::Event::InitApply(_) => None,
    }
}

fn cr_event(event: watcher::Event<DynamoWorkerMetadata>) -> Option<CrEvent> {
    match event {
        watcher::Event::Apply(cr) => Some(CrEvent::Apply(cr)),
        watcher::Event::Delete(cr) => Some(CrEvent::Delete(cr)),
        watcher::Event::InitDone => Some(CrEvent::Rebuild),
        watcher::Event::Init | watcher::Event::InitApply(_) => None,
    }
}

impl DiscoverySource {
    fn new(
        pod_info: &PodInfo,
        kube_client: KubeClient,
        events: mpsc::Sender<ReadinessEvent>,
    ) -> Self {
        let labels = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        match pod_info.mode {
            KubeDiscoveryMode::Pod => {
                let api: Api<EndpointSlice> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                tracing::info!("Daemon watching EndpointSlices (pod mode)");

                let stream = reflector(writer, watcher(api, labels)).default_backoff();
                tokio::spawn(async move {
                    tokio::pin!(stream);
                    while let Some(res) = stream.next().await {
                        match res {
                            Ok(event) => {
                                if let Some(event) = endpoint_slice_event(event)
                                    && events.send(event).await.is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("EndpointSlice reflector error: {e}");
                            }
                        }
                    }
                });

                Self::EndpointSlice(reader)
            }
            KubeDiscoveryMode::Container => {
                let api: Api<Pod> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                tracing::info!("Daemon watching Pods (container mode)");

                let stream = reflector(writer, watcher(api, labels)).default_backoff();
                tokio::spawn(async move {
                    tokio::pin!(stream);
                    while let Some(res) = stream.next().await {
                        match res {
                            Ok(event) => {
                                if let Some(event) = pod_event(event)
                                    && events.send(event).await.is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Pod reflector error: {e}");
                            }
                        }
                    }
                });

                Self::Pod(reader)
            }
        }
    }

    fn rebuild_index(&self) -> ReadinessIndex {
        let mut index = ReadinessIndex::default();
        match self {
            Self::EndpointSlice(reader) => {
                for slice in reader.state() {
                    if let Some((object_key, entries)) = endpoint_slice_update(slice.as_ref()) {
                        index.replace_object(object_key, entries);
                    }
                }
            }
            Self::Pod(reader) => {
                for pod in reader.state() {
                    if let Some((object_key, entries)) = pod_update(pod.as_ref()) {
                        index.replace_object(object_key, entries);
                    }
                }
            }
        }
        index
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

        let (readiness_tx, mut readiness_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);
        let source = DiscoverySource::new(&self.pod_info, self.kube_client.clone(), readiness_tx);

        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
        let (cr_reader, cr_writer) = reflector::store();
        let (cr_tx, mut cr_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);

        tracing::info!(
            "Daemon watching DynamoWorkerMetadata CRs in namespace: {}",
            self.pod_info.pod_namespace
        );

        let cr_reflector_stream =
            reflector(cr_writer, watcher(metadata_crs, Config::default())).default_backoff();
        tokio::spawn(async move {
            tokio::pin!(cr_reflector_stream);
            while let Some(res) = cr_reflector_stream.next().await {
                match res {
                    Ok(event) => {
                        if let Some(event) = cr_event(event)
                            && cr_tx.send(event).await.is_err()
                        {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("DynamoWorkerMetadata CR reflector error: {e}");
                    }
                }
            }
        });

        let mut join_table = JoinTable::new();
        let mut readiness_index = ReadinessIndex::default();
        let mut valid_cr_cache: HashMap<String, CachedCrMetadata> = HashMap::new();

        loop {
            let mut changes = BatchChanges::default();

            tokio::select! {
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("Discovery daemon received cancellation");
                    break;
                }
                event = readiness_rx.recv() => {
                    let Some(event) = event else {
                        anyhow::bail!("Readiness reflector stream stopped");
                    };
                    apply_readiness_event(
                        event,
                        &source,
                        &mut readiness_index,
                        &mut join_table,
                        &mut changes,
                    );
                }
                event = cr_rx.recv() => {
                    let Some(event) = event else {
                        anyhow::bail!("DynamoWorkerMetadata reflector stream stopped");
                    };
                    apply_cr_event(
                        event,
                        &cr_reader,
                        &mut valid_cr_cache,
                        &mut join_table,
                        &mut changes,
                    );
                }
            }

            let publication = changes.finish(&join_table);
            if !publication.state_changes.is_empty() || !publication.events.is_empty() {
                let mut state = list_state.write().await;
                for change in publication.state_changes {
                    match change {
                        StateChange::Upsert(instance_id, metadata) => {
                            state.insert(instance_id, metadata);
                        }
                        StateChange::Remove(instance_id) => {
                            state.remove(&instance_id);
                        }
                    }
                }
                for event in publication.events {
                    event_tx.send(event).ok();
                }
            }
        }

        tracing::info!("Discovery daemon stopped");
        Ok(())
    }
}

fn apply_readiness_event(
    event: ReadinessEvent,
    source: &DiscoverySource,
    readiness_index: &mut ReadinessIndex,
    join_table: &mut JoinTable,
    changes: &mut BatchChanges,
) {
    let affected = match event {
        ReadinessEvent::Apply {
            object_key,
            entries,
        } => readiness_index.replace_object(object_key, entries),
        ReadinessEvent::Delete { object_key } => readiness_index.remove_object(&object_key),
        ReadinessEvent::Rebuild => {
            let next = source.rebuild_index();
            let resolved = next.resolved_entries();
            join_table.replace_readiness(resolved, changes);
            *readiness_index = next;
            return;
        }
    };

    for cr_key in affected {
        let ready = readiness_index.resolved(&cr_key);
        join_table.set_readiness(cr_key, ready, changes);
    }
}

fn apply_cr_event(
    event: CrEvent,
    cr_reader: &reflector::Store<DynamoWorkerMetadata>,
    valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
    join_table: &mut JoinTable,
    changes: &mut BatchChanges,
) {
    match event {
        CrEvent::Apply(cr) => {
            if let Some((cr_key, cached)) = read_cr_object(&cr, valid_cr_cache) {
                join_table.set_cr(cr_key, cached, changes);
            }
        }
        CrEvent::Delete(cr) => {
            let Some(cr_key) = cr.metadata.name else {
                return;
            };
            valid_cr_cache.remove(&cr_key);
            join_table.set_cr(cr_key, None, changes);
        }
        CrEvent::Rebuild => {
            let next = scan_cr_store(cr_reader, valid_cr_cache);
            join_table.replace_crs(next, changes);
        }
    }
}

fn scan_cr_store(
    cr_reader: &reflector::Store<DynamoWorkerMetadata>,
    valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
) -> HashMap<String, CachedCrMetadata> {
    let cr_state = cr_reader.state();
    let mut new_right: HashMap<String, CachedCrMetadata> = HashMap::new();
    let mut observed: HashSet<String> = HashSet::new();

    for cr in cr_state {
        if let Some((cr_name, cached)) = read_cr_object(cr.as_ref(), valid_cr_cache) {
            observed.insert(cr_name.clone());
            if let Some(cached) = cached {
                new_right.insert(cr_name, cached);
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

fn read_cr_object(
    cr: &DynamoWorkerMetadata,
    valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
) -> Option<(String, Option<CachedCrMetadata>)> {
    let cr_name = cr.metadata.name.clone()?;
    let generation = cr.metadata.generation.unwrap_or(0);
    let uid = cr.metadata.uid.clone();
    let resource_version = cr.metadata.resource_version.as_deref().unwrap_or("unknown");
    let owner_pod_uid = cr
        .metadata
        .owner_references
        .as_ref()
        .and_then(|refs| refs.iter().find(|owner| owner.kind == "Pod"))
        .map(|owner| owner.uid.clone());

    if cr.spec.data.is_null() {
        tracing::debug!(
            cr_name,
            uid = %uid.as_deref().unwrap_or("unknown"),
            resource_version,
            generation,
            managed_fields = ?managed_fields_summary(cr),
            "DynamoWorkerMetadata CR has null spec.data; reusing last valid metadata if available"
        );
        let cached = cached_metadata_for_invalid_cr(
            &cr_name,
            uid.as_deref(),
            owner_pod_uid.as_deref(),
            valid_cr_cache,
        )
        .cloned();
        if cached.is_none() {
            valid_cr_cache.remove(&cr_name);
        }
        return Some((cr_name, cached));
    }

    match super::crd::deserialize_metadata(cr.spec.data.clone()) {
        Ok(metadata) => {
            tracing::trace!("Loaded metadata from CR '{cr_name}'");
            let cached = CachedCrMetadata {
                metadata: Arc::new(metadata),
                uid,
                owner_pod_uid,
            };
            valid_cr_cache.insert(cr_name.clone(), cached.clone());
            Some((cr_name, Some(cached)))
        }
        Err(error) => {
            tracing::warn!(
                cr_name,
                uid = %uid.as_deref().unwrap_or("unknown"),
                resource_version,
                generation,
                managed_fields = ?managed_fields_summary(cr),
                %error,
                "Failed to deserialize metadata from DynamoWorkerMetadata CR"
            );
            let cached = cached_metadata_for_invalid_cr(
                &cr_name,
                uid.as_deref(),
                owner_pod_uid.as_deref(),
                valid_cr_cache,
            )
            .cloned();
            if cached.is_none() {
                valid_cr_cache.remove(&cr_name);
            }
            Some((cr_name, cached))
        }
    }
}

fn cached_metadata_for_invalid_cr<'a>(
    cr_key: &str,
    uid: Option<&str>,
    owner_pod_uid: Option<&str>,
    valid_cr_cache: &'a HashMap<String, CachedCrMetadata>,
) -> Option<&'a CachedCrMetadata> {
    let cached = valid_cr_cache.get(cr_key)?;
    if cached.uid.as_deref() == uid && cached.owner_pod_uid.as_deref() == owner_pod_uid {
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
    use crate::discovery::{DiscoveryEvent, DiscoveryInstance};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ManagedFieldsEntry, OwnerReference};

    const TEST_POD_UID: &str = "pod-uid-test";

    fn make_cached(uid: &str) -> CachedCrMetadata {
        CachedCrMetadata {
            metadata: Arc::new(DiscoveryMetadata::new()),
            uid: Some(uid.to_string()),
            owner_pod_uid: Some(TEST_POD_UID.to_string()),
        }
    }

    fn make_cached_with_endpoint(uid: &str) -> CachedCrMetadata {
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
            make_cached_with_endpoint("uid-1"),
        )]));
        assert!(table.known.contains_key(&1u64));

        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-2"),
        )]));
        assert_eq!(table.cr_uid("worker-a"), Some("uid-2"));
    }

    #[test]
    fn join_table_removes_immediately_when_pod_not_ready() {
        let mut table = JoinTable::new();

        table.apply_readiness_scan(readiness(&[("worker-a", 1u64)]));
        table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1"),
        )]));
        assert!(table.known.contains_key(&1u64));

        let events = table.apply_readiness_scan(HashMap::new());
        assert!(!table.known.contains_key(&1u64));
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
            make_cached_with_endpoint("uid-1"),
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
            make_cached_with_endpoint("uid-1"),
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
            make_cached_with_endpoint("uid-1"),
        )]));

        let events = table.apply_cr_scan(HashMap::from([(
            "worker-a".to_string(),
            make_cached_with_endpoint("uid-1"),
        )]));
        assert!(events.is_empty());
    }

    #[test]
    fn cached_metadata_for_invalid_cr_reuses_same_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1"));

        let cached =
            cached_metadata_for_invalid_cr("worker-a", Some("uid-1"), Some(TEST_POD_UID), &cache)
                .expect("cache should be reused for the same CR and owner UIDs");

        assert_eq!(cached.uid.as_deref(), Some("uid-1"));
    }

    #[test]
    fn cached_metadata_for_invalid_cr_rejects_recreated_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1"));

        assert!(
            cached_metadata_for_invalid_cr("worker-a", Some("uid-2"), Some(TEST_POD_UID), &cache,)
                .is_none()
        );
    }

    #[test]
    fn cached_metadata_for_invalid_cr_rejects_new_pod_owner() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1"));

        assert!(
            cached_metadata_for_invalid_cr("worker-a", Some("uid-1"), Some("new-pod-uid"), &cache,)
                .is_none()
        );
    }

    #[test]
    fn invalid_cr_owner_change_discards_cached_metadata() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), make_cached("uid-1"));

        let mut cr = DynamoWorkerMetadata::new(
            "worker-a",
            super::super::crd::DynamoWorkerMetadataSpec::new(serde_json::Value::Null),
        );
        cr.metadata.uid = Some("uid-1".to_string());
        cr.metadata.owner_references = Some(vec![OwnerReference {
            api_version: "v1".to_string(),
            kind: "Pod".to_string(),
            name: "worker-a".to_string(),
            uid: "new-pod-uid".to_string(),
            block_owner_deletion: None,
            controller: Some(true),
        }]);

        let (cr_key, cached) =
            read_cr_object(&cr, &mut cache).expect("named CR should be processed");
        assert_eq!(cr_key, "worker-a");
        assert!(cached.is_none());
        assert!(!cache.contains_key("worker-a"));
    }

    #[test]
    fn relist_events_only_wake_on_init_done() {
        assert!(endpoint_slice_event(watcher::Event::Init).is_none());
        assert!(pod_event(watcher::Event::Init).is_none());

        let slice = EndpointSlice {
            metadata: Default::default(),
            address_type: "IPv4".to_string(),
            endpoints: Vec::new(),
            ports: None,
        };
        assert!(endpoint_slice_event(watcher::Event::InitApply(slice)).is_none());
        assert_eq!(
            endpoint_slice_event(watcher::Event::InitDone),
            Some(ReadinessEvent::Rebuild)
        );
        assert!(matches!(
            cr_event(watcher::Event::InitDone),
            Some(CrEvent::Rebuild)
        ));
    }

    #[test]
    fn join_requires_matching_pod_uid() {
        // Pod U2 arrives while old CR still has owner=U1 — must not join.
        let mut table = JoinTable::new();

        let new_left: HashMap<String, (u64, String)> =
            HashMap::from([("worker-0".to_string(), (1u64, "pod-uid-U2".to_string()))]);
        table.apply_readiness_scan(new_left);

        let mut cr = make_cached_with_endpoint("cr-uid-1");
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

        let mut cr = make_cached_with_endpoint("cr-uid-1");
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
        let mut cr_u1 = make_cached_with_endpoint("cr-uid-1");
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
        let mut cr_u2 = make_cached_with_endpoint("cr-uid-2");
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

        let mut cr = make_cached_with_endpoint("cr-uid-1");
        cr.owner_pod_uid = Some("pod-uid-U1".to_string());
        table.apply_readiness_scan(HashMap::from([(
            "worker-0".to_string(),
            (1u64, "pod-uid-U1".to_string()),
        )]));
        table.apply_cr_scan(HashMap::from([("worker-0".to_string(), cr)]));
        assert!(table.known.contains_key(&1u64));

        // CR updated with new owner U2 (in-place, same CR object, different owner)
        let mut cr_new_owner = make_cached_with_endpoint("cr-uid-1");
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
