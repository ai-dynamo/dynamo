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
use tokio::task::JoinHandle;

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
    /// Spawns the readiness reflector task and returns its handle alongside the
    /// store. The caller owns the task's lifetime: abort and await the handle on
    /// every exit path so the underlying Kubernetes watch is released instead of
    /// running detached for the life of the runtime (see issue #13874).
    fn new(
        pod_info: &PodInfo,
        kube_client: KubeClient,
        events: mpsc::Sender<ReadinessEvent>,
    ) -> (Self, JoinHandle<()>) {
        let labels = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        match pod_info.mode {
            KubeDiscoveryMode::Pod => {
                let api: Api<EndpointSlice> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                tracing::info!("Daemon watching EndpointSlices (pod mode)");

                let stream = reflector(writer, watcher(api, labels)).default_backoff();
                let handle = tokio::spawn(async move {
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

                (Self::EndpointSlice(reader), handle)
            }
            KubeDiscoveryMode::Container => {
                let api: Api<Pod> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                tracing::info!("Daemon watching Pods (container mode)");

                let stream = reflector(writer, watcher(api, labels)).default_backoff();
                let handle = tokio::spawn(async move {
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

                (Self::Pod(reader), handle)
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

        let (readiness_tx, readiness_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);
        let (source, readiness_handle) =
            DiscoverySource::new(&self.pod_info, self.kube_client.clone(), readiness_tx);

        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
        let (cr_reader, cr_writer) = reflector::store();
        let (cr_tx, cr_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);

        tracing::info!(
            "Daemon watching DynamoWorkerMetadata CRs in namespace: {}",
            self.pod_info.pod_namespace
        );

        let cr_reflector_stream =
            reflector(cr_writer, watcher(metadata_crs, Config::default())).default_backoff();
        let cr_handle = tokio::spawn(async move {
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

        let outcome = run_discovery_loop(
            self.cancel_token.clone(),
            readiness_rx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
            ReflectorTasks::new(readiness_handle, cr_handle),
        )
        .await;

        tracing::info!("Discovery daemon stopped");
        outcome
    }
}

/// Owns both reflector task handles for the lifetime of one `run()` call.
///
/// `run()`'s two explicit exit points call [`ReflectorTasks::stop`], which
/// aborts and awaits both handles so a returning `run()` means the underlying
/// Kubernetes watches have actually stopped (issue #13874). But a plain local
/// `JoinHandle` only detaches its task when dropped -- it does not abort it --
/// so if `run()`'s own future were ever dropped or panicked after spawning the
/// reflectors but before reaching `stop`, the handles would be dropped without
/// either task being told to stop, recreating the leak outside those two exit
/// points. Wrapping both handles in this guard closes that gap: its `Drop`
/// aborts whatever handles `stop` did not already take, so even an abnormal
/// drop of `run()`'s future schedules cancellation instead of leaking silently.
struct ReflectorTasks {
    readiness_handle: Option<JoinHandle<()>>,
    cr_handle: Option<JoinHandle<()>>,
}

impl ReflectorTasks {
    fn new(readiness_handle: JoinHandle<()>, cr_handle: JoinHandle<()>) -> Self {
        Self {
            readiness_handle: Some(readiness_handle),
            cr_handle: Some(cr_handle),
        }
    }

    /// Aborts both reflector tasks and waits for them to actually finish.
    /// `abort` only requests cancellation -- it takes effect the next time the
    /// target task reaches an `.await` point -- so this does not return until
    /// both tasks have, releasing whatever Kubernetes watch stream and
    /// reflector writer each one held (issue #13874).
    async fn stop(mut self) {
        if let Some(handle) = self.readiness_handle.take() {
            handle.abort();
            join_ignoring_cancellation(handle, "readiness").await;
        }
        if let Some(handle) = self.cr_handle.take() {
            handle.abort();
            join_ignoring_cancellation(handle, "DynamoWorkerMetadata").await;
        }
    }
}

impl Drop for ReflectorTasks {
    fn drop(&mut self) {
        // Reached only when `stop` never ran to completion. A synchronous
        // `Drop` cannot await task completion, but aborting still schedules
        // cancellation so the task stops holding its Kubernetes watch instead
        // of running detached for the life of the runtime.
        if let Some(handle) = self.readiness_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.cr_handle.take() {
            handle.abort();
        }
    }
}

/// Awaits an aborted reflector handle, distinguishing the expected
/// cancellation error from a genuine panic. A reflector task can already have
/// panicked before `abort` is called; silently discarding every `JoinError`
/// would let that race hide the panic behind an apparently clean shutdown.
async fn join_ignoring_cancellation(handle: JoinHandle<()>, reflector_name: &str) {
    match handle.await {
        Ok(()) => {}
        Err(error) if error.is_cancelled() => {}
        Err(error) => {
            tracing::error!("{reflector_name} reflector task panicked: {error}");
        }
    }
}

/// Runs the discovery event loop until cancellation or either reflector
/// channel closes, then stops both reflector tasks before returning.
///
/// Split out of `DiscoveryDaemon::run` so tests can drive it directly with
/// synthetic reflector tasks and channels instead of a real Kubernetes client
/// (issue #13874).
#[allow(clippy::too_many_arguments)]
async fn run_discovery_loop(
    cancel_token: CancellationToken,
    mut readiness_rx: mpsc::Receiver<ReadinessEvent>,
    mut cr_rx: mpsc::Receiver<CrEvent>,
    source: DiscoverySource,
    cr_reader: reflector::Store<DynamoWorkerMetadata>,
    list_state: Arc<RwLock<HashMap<u64, Arc<DiscoveryMetadata>>>>,
    event_tx: broadcast::Sender<DiscoveryEvent>,
    reflector_tasks: ReflectorTasks,
) -> Result<()> {
    let mut join_table = JoinTable::new();
    let mut readiness_index = ReadinessIndex::default();
    let mut valid_cr_cache: HashMap<String, CachedCrMetadata> = HashMap::new();

    // The loop itself yields the daemon's outcome via `break` instead of
    // returning directly, so every exit path -- cancellation or either
    // reflector channel closing -- reaches the cleanup below before this
    // function returns. See issue #13874: without this, an early `bail!`
    // from inside the loop would skip stopping the two reflector tasks.
    let outcome: Result<()> = loop {
        let mut changes = BatchChanges::default();

        tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::info!("Discovery daemon received cancellation");
                break Ok(());
            }
            event = readiness_rx.recv() => {
                let Some(event) = event else {
                    break Err(anyhow::anyhow!("Readiness reflector stream stopped"));
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
                    break Err(anyhow::anyhow!("DynamoWorkerMetadata reflector stream stopped"));
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
    };

    reflector_tasks.stop().await;
    outcome
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
    use std::sync::atomic::{AtomicBool, Ordering};

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

    /// Sets its flag on drop, including when the future holding it is aborted
    /// rather than run to completion -- unlike checking that a stop function
    /// merely returns, this distinguishes an implementation that awaits both
    /// handles from one that only calls `abort` (which schedules cancellation
    /// but does not itself wait for the task's drop glue to run).
    struct SetOnDrop(Arc<AtomicBool>);

    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Spawns a task that runs forever until aborted, setting `done` (via
    /// `SetOnDrop`) only once its future is actually dropped. Stands in for a
    /// reflector loop without needing a real Kubernetes client.
    ///
    /// Only returns once the task is confirmed running: a task aborted before
    /// its first poll never runs any of its body -- including constructing the
    /// `SetOnDrop` guard -- so it would never set `done` either. Waiting for
    /// this signal means callers only ever abort a real in-flight task,
    /// matching what happens to the reflector loops this stands in for.
    async fn spawn_never_ending(done: Arc<AtomicBool>) -> JoinHandle<()> {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let _guard = SetOnDrop(done);
            let _ = started_tx.send(());
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
        started_rx
            .await
            .expect("spawned task must start before its handle is used");
        handle
    }

    /// Builds the non-Kubernetes-backed pieces `run_discovery_loop` needs: an
    /// empty `DiscoverySource`/CR store pair, fresh channels, list state, and
    /// an event bus. Tests drive the loop's exit paths by cancelling
    /// `cancel_token` or dropping the returned senders, then assert on the
    /// synthetic reflector tasks' `AtomicBool` flags to prove both were
    /// actually stopped, not just asked to stop.
    #[allow(clippy::type_complexity)]
    fn discovery_loop_test_fixture() -> (
        CancellationToken,
        mpsc::Sender<ReadinessEvent>,
        mpsc::Receiver<ReadinessEvent>,
        mpsc::Sender<CrEvent>,
        mpsc::Receiver<CrEvent>,
        DiscoverySource,
        reflector::Store<DynamoWorkerMetadata>,
        Arc<RwLock<HashMap<u64, Arc<DiscoveryMetadata>>>>,
        broadcast::Sender<DiscoveryEvent>,
    ) {
        let cancel_token = CancellationToken::new();
        let (readiness_tx, readiness_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);
        let (cr_tx, cr_rx) = mpsc::channel(SOURCE_CHANNEL_CAPACITY);
        let (source_reader, _source_writer) = reflector::store::<EndpointSlice>();
        let source = DiscoverySource::EndpointSlice(source_reader);
        let (cr_reader, _cr_writer) = reflector::store::<DynamoWorkerMetadata>();
        let list_state = Arc::new(RwLock::new(HashMap::new()));
        let (event_tx, _event_rx) = broadcast::channel(16);
        (
            cancel_token,
            readiness_tx,
            readiness_rx,
            cr_tx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
        )
    }

    /// Regression test for issue #13874.
    #[tokio::test]
    async fn reflector_tasks_stop_awaits_both_aborted_handles() {
        let readiness_done = Arc::new(AtomicBool::new(false));
        let cr_done = Arc::new(AtomicBool::new(false));
        let readiness_handle = spawn_never_ending(readiness_done.clone()).await;
        let cr_handle = spawn_never_ending(cr_done.clone()).await;

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ReflectorTasks::new(readiness_handle, cr_handle).stop(),
        )
        .await
        .expect("stop must return once both tasks are aborted, not hang");

        assert!(
            readiness_done.load(Ordering::SeqCst),
            "readiness task must have actually finished, not just been asked to abort"
        );
        assert!(
            cr_done.load(Ordering::SeqCst),
            "CR task must have actually finished, not just been asked to abort"
        );
    }

    /// Companion regression test for issue #13874: proves `run_discovery_loop`
    /// itself -- not just the extracted `ReflectorTasks::stop` helper -- stops
    /// both reflector tasks on its cancellation exit path. Mutation-tested by
    /// removing the `reflector_tasks.stop().await` call from
    /// `run_discovery_loop`: without it, this test fails because the flags are
    /// still false by the time the loop returns.
    #[tokio::test]
    async fn run_discovery_loop_stops_both_reflectors_on_cancellation() {
        let readiness_done = Arc::new(AtomicBool::new(false));
        let cr_done = Arc::new(AtomicBool::new(false));
        let readiness_handle = spawn_never_ending(readiness_done.clone()).await;
        let cr_handle = spawn_never_ending(cr_done.clone()).await;

        let (
            cancel_token,
            _readiness_tx,
            readiness_rx,
            _cr_tx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
        ) = discovery_loop_test_fixture();
        cancel_token.cancel();

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_discovery_loop(
                cancel_token,
                readiness_rx,
                cr_rx,
                source,
                cr_reader,
                list_state,
                event_tx,
                ReflectorTasks::new(readiness_handle, cr_handle),
            ),
        )
        .await
        .expect("run_discovery_loop must return once cancelled, not hang");

        assert!(outcome.is_ok(), "cancellation must be a clean exit");
        assert!(
            readiness_done.load(Ordering::SeqCst),
            "readiness reflector must actually be stopped, not just asked to stop"
        );
        assert!(
            cr_done.load(Ordering::SeqCst),
            "CR reflector must actually be stopped, not just asked to stop"
        );
    }

    /// Companion regression test for issue #13874: the readiness reflector
    /// channel closing (its sender dropped, mirroring the reflector task
    /// exiting) must also stop both reflectors before `run_discovery_loop`
    /// returns.
    #[tokio::test]
    async fn run_discovery_loop_stops_both_reflectors_when_readiness_channel_closes() {
        let readiness_done = Arc::new(AtomicBool::new(false));
        let cr_done = Arc::new(AtomicBool::new(false));
        let readiness_handle = spawn_never_ending(readiness_done.clone()).await;
        let cr_handle = spawn_never_ending(cr_done.clone()).await;

        let (
            cancel_token,
            readiness_tx,
            readiness_rx,
            _cr_tx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
        ) = discovery_loop_test_fixture();
        drop(readiness_tx);

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_discovery_loop(
                cancel_token,
                readiness_rx,
                cr_rx,
                source,
                cr_reader,
                list_state,
                event_tx,
                ReflectorTasks::new(readiness_handle, cr_handle),
            ),
        )
        .await
        .expect("run_discovery_loop must return once the readiness channel closes, not hang");

        assert!(
            outcome.is_err(),
            "a closed readiness channel must be reported as a failure"
        );
        assert!(
            readiness_done.load(Ordering::SeqCst),
            "readiness reflector must actually be stopped, not just asked to stop"
        );
        assert!(
            cr_done.load(Ordering::SeqCst),
            "CR reflector must actually be stopped, not just asked to stop"
        );
    }

    /// Companion regression test for issue #13874: the CR reflector channel
    /// closing must also stop both reflectors before `run_discovery_loop`
    /// returns.
    #[tokio::test]
    async fn run_discovery_loop_stops_both_reflectors_when_cr_channel_closes() {
        let readiness_done = Arc::new(AtomicBool::new(false));
        let cr_done = Arc::new(AtomicBool::new(false));
        let readiness_handle = spawn_never_ending(readiness_done.clone()).await;
        let cr_handle = spawn_never_ending(cr_done.clone()).await;

        let (
            cancel_token,
            _readiness_tx,
            readiness_rx,
            cr_tx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
        ) = discovery_loop_test_fixture();
        drop(cr_tx);

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_discovery_loop(
                cancel_token,
                readiness_rx,
                cr_rx,
                source,
                cr_reader,
                list_state,
                event_tx,
                ReflectorTasks::new(readiness_handle, cr_handle),
            ),
        )
        .await
        .expect("run_discovery_loop must return once the CR channel closes, not hang");

        assert!(
            outcome.is_err(),
            "a closed CR channel must be reported as a failure"
        );
        assert!(
            readiness_done.load(Ordering::SeqCst),
            "readiness reflector must actually be stopped, not just asked to stop"
        );
        assert!(
            cr_done.load(Ordering::SeqCst),
            "CR reflector must actually be stopped, not just asked to stop"
        );
    }

    /// Negative control for the three tests above: with no exit condition
    /// triggered (token not cancelled, both channels open), the loop must
    /// keep running -- and, since nothing asked either reflector to stop, both
    /// must still be running too. Without this control, the other three tests
    /// could pass vacuously if `run_discovery_loop` returned immediately
    /// regardless of input.
    #[tokio::test]
    async fn run_discovery_loop_keeps_running_without_an_exit_condition() {
        let readiness_done = Arc::new(AtomicBool::new(false));
        let cr_done = Arc::new(AtomicBool::new(false));
        let readiness_handle = spawn_never_ending(readiness_done.clone()).await;
        let cr_handle = spawn_never_ending(cr_done.clone()).await;

        let (
            cancel_token,
            _readiness_tx,
            readiness_rx,
            _cr_tx,
            cr_rx,
            source,
            cr_reader,
            list_state,
            event_tx,
        ) = discovery_loop_test_fixture();

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            run_discovery_loop(
                cancel_token,
                readiness_rx,
                cr_rx,
                source,
                cr_reader,
                list_state,
                event_tx,
                ReflectorTasks::new(readiness_handle, cr_handle),
            ),
        )
        .await;

        assert!(
            result.is_err(),
            "run_discovery_loop must not exit while no exit condition has fired"
        );
        assert!(
            !readiness_done.load(Ordering::SeqCst),
            "readiness reflector must still be running, not already stopped"
        );
        assert!(
            !cr_done.load(Ordering::SeqCst),
            "CR reflector must still be running, not already stopped"
        );
    }
}
