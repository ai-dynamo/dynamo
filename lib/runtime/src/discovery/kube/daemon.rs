// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{DiscoveryMetadata, MetadataSnapshot};
use anyhow::Result;
use futures::StreamExt;
use futures::stream::BoxStream;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    Api, Client as KubeClient, Resource,
    runtime::{WatchStreamExt, reflector, reflector::store::Writer, watcher, watcher::Config},
};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

use super::crd::DynamoWorkerMetadata;
use super::utils::{KubeDiscoveryMode, PodInfo, extract_endpoint_info, extract_ready_containers};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

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

struct AggregatedSnapshot {
    snapshot: MetadataSnapshot,
    revisions: HashMap<u64, CrRevision>,
}

fn snapshot_has_changes(
    snapshot: &MetadataSnapshot,
    previous_snapshot: &MetadataSnapshot,
    revisions: &HashMap<u64, CrRevision>,
    previous_revisions: &HashMap<u64, CrRevision>,
) -> bool {
    let metadata_changed = snapshot.has_changes_from(previous_snapshot);
    let revisions_changed = revisions != previous_revisions;
    if revisions_changed && !metadata_changed {
        tracing::debug!("DynamoWorkerMetadata CR identity changed");
    }
    metadata_changed || revisions_changed
}

/// An unstarted watch stream, before a reflector is attached to it.
type WatchStream<K> = BoxStream<'static, watcher::Result<watcher::Event<K>>>;

/// Drive `stream` into `writer`'s store until the stream ends or `token` fires.
///
/// The returned handle is what makes daemon shutdown observable: the task holds
/// a Kubernetes watch connection, so the daemon that started it has to be able
/// to stop it and wait for it.
fn spawn_reflector<K>(
    writer: Writer<K>,
    stream: WatchStream<K>,
    notify: Arc<Notify>,
    token: CancellationToken,
    kind: &'static str,
) -> JoinHandle<()>
where
    K: Resource + Clone + Send + Sync + 'static,
    K::DynamicType: Eq + Hash + Clone,
{
    tokio::spawn(async move {
        let reflector_stream = reflector(writer, stream)
            .default_backoff()
            .touched_objects()
            .for_each(move |res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            kind,
                            name = obj.meta().name.as_deref().unwrap_or("?"),
                            "reflector updated"
                        );
                        notify.notify_one();
                    }
                    Err(e) => {
                        tracing::warn!(kind, "reflector error: {e}");
                        notify.notify_one();
                    }
                }
                futures::future::ready(())
            });

        // A `watcher` stream is infinite and retries internally, so cancellation
        // is the only thing that ends this task on a healthy cluster.
        tokio::select! {
            _ = reflector_stream => {
                tracing::debug!(kind, "Reflector stream ended");
            }
            _ = token.cancelled() => {
                tracing::debug!(kind, "Reflector stopping on daemon shutdown");
            }
        }
    })
}

/// Readiness watch stream, tagged with the kind the daemon's mode implies.
///
/// Separating stream construction from the daemon loop keeps `kube::Client` out
/// of the loop's signature, so the loop's shutdown contract is testable.
enum ReadinessWatch {
    EndpointSlice(WatchStream<EndpointSlice>),
    Pod(WatchStream<Pod>),
}

impl ReadinessWatch {
    fn from_cluster(pod_info: &PodInfo, kube_client: KubeClient) -> Self {
        let labels = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        match pod_info.mode {
            KubeDiscoveryMode::Pod => {
                let api: Api<EndpointSlice> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                tracing::info!("Daemon watching EndpointSlices (pod mode)");
                Self::EndpointSlice(watcher(api, labels).boxed())
            }
            KubeDiscoveryMode::Container => {
                let api: Api<Pod> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                tracing::info!("Daemon watching Pods (container mode)");
                Self::Pod(watcher(api, labels).boxed())
            }
        }
    }
}

/// Readiness data source for the discovery daemon.
///
/// Pod mode watches EndpointSlices (one entry per ready pod).
/// Container mode watches Pods directly (one entry per ready container).
/// Both produce the same (instance_id, cr_key) tuples for snapshot correlation.
enum DiscoverySource {
    EndpointSlice(reflector::Store<EndpointSlice>),
    Pod(reflector::Store<Pod>),
}

impl DiscoverySource {
    fn new(
        watch: ReadinessWatch,
        notify: Arc<Notify>,
        token: CancellationToken,
    ) -> (Self, JoinHandle<()>) {
        match watch {
            ReadinessWatch::EndpointSlice(stream) => {
                let (reader, writer) = reflector::store();
                let handle = spawn_reflector(writer, stream, notify, token, "EndpointSlice");
                (Self::EndpointSlice(reader), handle)
            }

            ReadinessWatch::Pod(stream) => {
                let (reader, writer) = reflector::store();
                let handle = spawn_reflector(writer, stream, notify, token, "Pod");
                (Self::Pod(reader), handle)
            }
        }
    }

    fn ready_entries(&self) -> Vec<(u64, String)> {
        match self {
            Self::EndpointSlice(reader) => reader
                .state()
                .iter()
                .flat_map(|s| extract_endpoint_info(s.as_ref()))
                .collect(),
            Self::Pod(reader) => reader
                .state()
                .iter()
                .flat_map(|p| extract_ready_containers(p.as_ref()))
                .collect(),
        }
    }
}

/// Discovers and aggregates metadata from DynamoWorkerMetadata CRs in the cluster
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

    /// Run the discovery daemon.
    ///
    /// Watches a readiness source and DynamoWorkerMetadata CRs. An entry is
    /// included in the snapshot only if it appears ready AND has valid current
    /// or cached metadata from a matching CR.
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        let readiness = ReadinessWatch::from_cluster(&self.pod_info, self.kube_client.clone());

        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        tracing::info!(
            "Daemon watching DynamoWorkerMetadata CRs in namespace: {}",
            self.pod_info.pod_namespace
        );

        let cr_watch = watcher(metadata_crs, Config::default()).boxed();

        Self::run_daemon(
            &self.pod_info,
            &self.cancel_token,
            readiness,
            cr_watch,
            watch_tx,
        )
        .await
    }

    /// The daemon loop, over watch streams that have already been built.
    ///
    /// Returning from here means both reflector tasks have stopped, so the
    /// watches and task state they held are released.
    async fn run_daemon(
        pod_info: &PodInfo,
        cancel_token: &CancellationToken,
        readiness: ReadinessWatch,
        cr_watch: WatchStream<DynamoWorkerMetadata>,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        let notify = Arc::new(Notify::new());

        // Cancelled by the caller's token and by every exit below, so a reflector
        // cannot outlive the daemon that started it. It is a child rather than the
        // caller's own token so the receiver-dropped exit can stop the reflectors
        // without cancelling unrelated work that shares the caller's token.
        let reflector_token = cancel_token.child_token();

        // Readiness source — EndpointSlice or Pod depending on mode
        let (source, readiness_task) =
            DiscoverySource::new(readiness, notify.clone(), reflector_token.clone());

        // DynamoWorkerMetadata CR reflector
        let (cr_reader, cr_writer) = reflector::store();
        let cr_task = spawn_reflector(
            cr_writer,
            cr_watch,
            notify.clone(),
            reflector_token.clone(),
            "DynamoWorkerMetadata",
        );

        // Event-driven loop with debouncing
        let mut sequence = 0u64;
        let mut prev_snapshot = MetadataSnapshot::empty();
        let mut prev_revisions = HashMap::new();
        // Keeps transient invalid CR updates from looking like removals.
        let mut valid_cr_cache: HashMap<String, CachedCrMetadata> = HashMap::new();

        loop {
            tokio::select! {
                _ = notify.notified() => {
                    tokio::time::sleep(DEBOUNCE_DURATION).await;
                    let _ = timeout(Duration::ZERO, notify.notified()).await;

                    tracing::trace!("Debounce window elapsed, processing snapshot");

                    match Self::aggregate_snapshot(
                        pod_info,
                        &source,
                        &cr_reader,
                        &mut valid_cr_cache,
                        sequence,
                    )
                    .await
                    {
                        Ok(aggregated) => {
                            let AggregatedSnapshot { snapshot, revisions } = aggregated;
                            if snapshot_has_changes(
                                &snapshot,
                                &prev_snapshot,
                                &revisions,
                                &prev_revisions,
                            ) {
                                prev_snapshot = snapshot.clone();
                                prev_revisions = revisions;

                                if watch_tx.send(Arc::new(snapshot)).is_err() {
                                    tracing::debug!("No watch subscribers, daemon stopping");
                                    break;
                                }
                            }

                            sequence += 1;
                        }
                        Err(e) => {
                            tracing::error!("Failed to aggregate snapshot: {e}");
                        }
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::info!("Discovery daemon received cancellation");
                    break;
                }
            }
        }

        // Every exit above lands here, so both reflectors are signalled and waited
        // for whether the daemon was cancelled or lost its snapshot receiver.
        reflector_token.cancel();
        for (kind, task) in [
            ("readiness", readiness_task),
            ("DynamoWorkerMetadata", cr_task),
        ] {
            // A JoinError means the runtime is already tearing that task down, so
            // the watch is released either way; shutdown is not an error path.
            if let Err(e) = task.await {
                tracing::debug!(kind, "Reflector task did not exit cleanly: {e}");
            }
        }

        tracing::info!("Discovery daemon stopped");
        Ok(())
    }

    async fn aggregate_snapshot(
        pod_info: &PodInfo,
        source: &DiscoverySource,
        cr_reader: &reflector::Store<DynamoWorkerMetadata>,
        valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
        sequence: u64,
    ) -> Result<AggregatedSnapshot> {
        let start = std::time::Instant::now();

        let ready_entries = source.ready_entries();

        tracing::trace!(
            "Daemon found {} ready entries (mode={:?})",
            ready_entries.len(),
            pod_info.mode,
        );

        let cr_state = cr_reader.state();
        let mut cr_map: HashMap<String, CachedCrMetadata> = HashMap::new();
        let mut invalid_crs: HashMap<String, Option<String>> = HashMap::new();
        let mut observed_crs: HashSet<String> = HashSet::new();

        for arc_cr in cr_state.iter() {
            let Some(cr_name) = arc_cr.metadata.name.as_ref() else {
                continue;
            };

            observed_crs.insert(cr_name.clone());
            let generation = arc_cr.metadata.generation.unwrap_or(0);
            let uid = arc_cr.metadata.uid.clone();
            let resource_version = arc_cr
                .metadata
                .resource_version
                .as_deref()
                .unwrap_or("unknown");

            if arc_cr.spec.data.is_null() {
                tracing::debug!(
                    cr_name = %cr_name,
                    uid = %uid.as_deref().unwrap_or("unknown"),
                    resource_version = %resource_version,
                    generation,
                    managed_fields = ?managed_fields_summary(arc_cr.as_ref()),
                    "DynamoWorkerMetadata CR has null spec.data; reusing last valid metadata if available"
                );
                invalid_crs.insert(cr_name.clone(), uid);
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
                    cr_map.insert(cr_name.clone(), cached.clone());
                    valid_cr_cache.insert(cr_name.clone(), cached);
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
                    invalid_crs.insert(cr_name.clone(), uid);
                }
            }
        }

        valid_cr_cache.retain(|cr_name, _| observed_crs.contains(cr_name));

        tracing::trace!("Daemon loaded {} DynamoWorkerMetadata CRs", cr_map.len());

        let mut instances: HashMap<u64, Arc<DiscoveryMetadata>> = HashMap::new();
        let mut generations: HashMap<u64, i64> = HashMap::new();
        let mut revisions: HashMap<u64, CrRevision> = HashMap::new();

        for (instance_id, cr_key) in ready_entries {
            if let Some(cached) = cr_map.get(&cr_key) {
                instances.insert(instance_id, cached.metadata.clone());
                generations.insert(instance_id, cached.generation);
                revisions.insert(
                    instance_id,
                    CrRevision {
                        generation: cached.generation,
                        uid: cached.uid.clone(),
                    },
                );
                tracing::trace!(
                    "Included '{}' (instance_id={:x}, generation={}) in snapshot",
                    cr_key,
                    instance_id,
                    cached.generation
                );
            } else if let Some(uid) = invalid_crs.get(&cr_key) {
                if let Some(cached) =
                    cached_metadata_for_invalid_cr(&cr_key, uid.as_deref(), valid_cr_cache)
                {
                    instances.insert(instance_id, cached.metadata.clone());
                    generations.insert(instance_id, cached.generation);
                    revisions.insert(
                        instance_id,
                        CrRevision {
                            generation: cached.generation,
                            uid: cached.uid.clone(),
                        },
                    );
                    tracing::trace!(
                        "Included cached metadata for '{}' (instance_id={:x}, generation={}) because current CR data is not valid",
                        cr_key,
                        instance_id,
                        cached.generation
                    );
                } else {
                    tracing::trace!(
                        "Skipping '{}' (instance_id={:x}): DynamoWorkerMetadata CR data is not valid yet",
                        cr_key,
                        instance_id
                    );
                }
            } else {
                tracing::trace!(
                    "Skipping '{}' (instance_id={:x}): no DynamoWorkerMetadata CR found",
                    cr_key,
                    instance_id
                );
            }
        }

        let elapsed = start.elapsed();

        tracing::trace!(
            "Daemon snapshot complete (seq={}): {} instances in {:?}",
            sequence,
            instances.len(),
            elapsed
        );

        Ok(AggregatedSnapshot {
            snapshot: MetadataSnapshot {
                instances,
                generations,
                sequence,
                timestamp: std::time::Instant::now(),
            },
            revisions,
        })
    }
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
    use super::super::crd::build_cr;
    use super::super::utils::KubeDiscoveryTarget;
    use super::*;
    use futures::Stream;
    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ManagedFieldsEntry, ObjectMeta};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};

    /// Bound for every daemon shutdown assertion below. Generous relative to the
    /// work involved (nothing here talks to a network) so a loaded CI box does
    /// not turn a shutdown regression test into a flake.
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

    /// A watch stream that records when the task driving it dropped it.
    ///
    /// The reflector task owns its stream, so the flag flips exactly when that
    /// task's future is torn down — which is the property under test.
    struct MarkedStream<K> {
        inner: WatchStream<K>,
        finished: Arc<AtomicBool>,
    }

    impl<K> Stream for MarkedStream<K> {
        type Item = watcher::Result<watcher::Event<K>>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.get_mut().inner.as_mut().poll_next(cx)
        }
    }

    impl<K> Drop for MarkedStream<K> {
        fn drop(&mut self) {
            self.finished.store(true, Ordering::SeqCst);
        }
    }

    /// Emit `events`, then never end. Only cancellation can finish this stream,
    /// which is what a real `watcher` behaves like on a healthy cluster.
    fn synthetic_watch<K: Send + 'static>(
        events: Vec<watcher::Event<K>>,
    ) -> (WatchStream<K>, Arc<AtomicBool>) {
        let finished = Arc::new(AtomicBool::new(false));
        let stream = MarkedStream {
            inner: futures::stream::iter(events.into_iter().map(Ok))
                .chain(futures::stream::pending())
                .boxed(),
            finished: finished.clone(),
        };
        (stream.boxed(), finished)
    }

    /// kube stores are only populated by an Init/InitApply/InitDone sequence.
    fn initial_state<K>(objects: Vec<K>) -> Vec<watcher::Event<K>> {
        let mut events = vec![watcher::Event::Init];
        events.extend(objects.into_iter().map(watcher::Event::InitApply));
        events.push(watcher::Event::InitDone);
        events
    }

    fn test_pod_info(mode: KubeDiscoveryMode) -> PodInfo {
        let target = match mode {
            KubeDiscoveryMode::Pod => KubeDiscoveryTarget::Pod("worker-0".to_string()),
            KubeDiscoveryMode::Container => {
                KubeDiscoveryTarget::Container("worker-0".to_string(), "main".to_string())
            }
        };

        PodInfo {
            pod_name: "worker-0".to_string(),
            pod_namespace: "dynamo".to_string(),
            pod_uid: "pod-uid".to_string(),
            system_port: 9090,
            mode,
            target,
        }
    }

    fn ready_endpoint_slice(pod_name: &str) -> EndpointSlice {
        EndpointSlice {
            metadata: ObjectMeta {
                name: Some("dynamo-slice".to_string()),
                namespace: Some("dynamo".to_string()),
                ..Default::default()
            },
            address_type: "IPv4".to_string(),
            endpoints: vec![Endpoint {
                conditions: Some(EndpointConditions {
                    ready: Some(true),
                    ..Default::default()
                }),
                target_ref: Some(ObjectReference {
                    kind: Some("Pod".to_string()),
                    name: Some(pod_name.to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ports: None,
        }
    }

    fn worker_metadata_cr(cr_name: &str) -> DynamoWorkerMetadata {
        build_cr(cr_name, "worker-0", "pod-uid", &DiscoveryMetadata::new())
            .expect("empty discovery metadata should serialize into a CR")
    }

    fn spawn_daemon(
        pod_info: PodInfo,
        cancel_token: CancellationToken,
        readiness: ReadinessWatch,
        cr_watch: WatchStream<DynamoWorkerMetadata>,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> JoinHandle<Result<()>> {
        tokio::spawn(async move {
            DiscoveryDaemon::run_daemon(&pod_info, &cancel_token, readiness, cr_watch, watch_tx)
                .await
        })
    }

    async fn await_daemon(daemon: JoinHandle<Result<()>>) {
        timeout(SHUTDOWN_TIMEOUT, daemon)
            .await
            .expect("daemon should return once it is shutting down")
            .expect("daemon task should not panic")
            .expect("daemon should return Ok");
    }

    #[tokio::test]
    async fn cancellation_stops_both_reflectors_in_pod_mode() {
        let (readiness, readiness_finished) = synthetic_watch::<EndpointSlice>(vec![]);
        let (cr_watch, cr_finished) = synthetic_watch::<DynamoWorkerMetadata>(vec![]);
        let (watch_tx, _watch_rx) =
            tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        let cancel_token = CancellationToken::new();

        let daemon = spawn_daemon(
            test_pod_info(KubeDiscoveryMode::Pod),
            cancel_token.clone(),
            ReadinessWatch::EndpointSlice(readiness),
            cr_watch,
            watch_tx,
        );

        cancel_token.cancel();
        await_daemon(daemon).await;

        assert!(
            readiness_finished.load(Ordering::SeqCst),
            "EndpointSlice reflector should have finished before run returned"
        );
        assert!(
            cr_finished.load(Ordering::SeqCst),
            "CR reflector should have finished before run returned"
        );
    }

    #[tokio::test]
    async fn cancellation_stops_both_reflectors_in_container_mode() {
        let (readiness, readiness_finished) = synthetic_watch::<Pod>(vec![]);
        let (cr_watch, cr_finished) = synthetic_watch::<DynamoWorkerMetadata>(vec![]);
        let (watch_tx, _watch_rx) =
            tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        let cancel_token = CancellationToken::new();

        let daemon = spawn_daemon(
            test_pod_info(KubeDiscoveryMode::Container),
            cancel_token.clone(),
            ReadinessWatch::Pod(readiness),
            cr_watch,
            watch_tx,
        );

        cancel_token.cancel();
        await_daemon(daemon).await;

        assert!(
            readiness_finished.load(Ordering::SeqCst),
            "Pod reflector should have finished before run returned"
        );
        assert!(
            cr_finished.load(Ordering::SeqCst),
            "CR reflector should have finished before run returned"
        );
    }

    #[tokio::test]
    async fn lost_snapshot_receiver_stops_both_reflectors() {
        // The daemon only sends when the aggregated snapshot changed, so this
        // test has to drive a real ready entry and its CR through both streams
        // to reach the send-failed exit.
        let (readiness, readiness_finished) =
            synthetic_watch(initial_state(vec![ready_endpoint_slice("worker-0")]));
        let (cr_watch, cr_finished) =
            synthetic_watch(initial_state(vec![worker_metadata_cr("worker-0")]));
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        drop(watch_rx);

        let daemon = spawn_daemon(
            test_pod_info(KubeDiscoveryMode::Pod),
            CancellationToken::new(),
            ReadinessWatch::EndpointSlice(readiness),
            cr_watch,
            watch_tx,
        );

        await_daemon(daemon).await;

        assert!(
            readiness_finished.load(Ordering::SeqCst),
            "EndpointSlice reflector should have finished after the receiver was dropped"
        );
        assert!(
            cr_finished.load(Ordering::SeqCst),
            "CR reflector should have finished after the receiver was dropped"
        );
    }

    #[tokio::test]
    async fn running_daemon_keeps_its_reflectors_alive() {
        let (readiness, readiness_finished) =
            synthetic_watch(initial_state(vec![ready_endpoint_slice("worker-0")]));
        let (cr_watch, cr_finished) =
            synthetic_watch(initial_state(vec![worker_metadata_cr("worker-0")]));
        let (watch_tx, mut watch_rx) =
            tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        let cancel_token = CancellationToken::new();

        let daemon = spawn_daemon(
            test_pod_info(KubeDiscoveryMode::Pod),
            cancel_token.clone(),
            ReadinessWatch::EndpointSlice(readiness),
            cr_watch,
            watch_tx,
        );

        timeout(SHUTDOWN_TIMEOUT, watch_rx.changed())
            .await
            .expect("daemon should publish a snapshot")
            .expect("snapshot sender should still be open");
        assert_eq!(watch_rx.borrow_and_update().instances.len(), 1);

        assert!(
            !readiness_finished.load(Ordering::SeqCst),
            "EndpointSlice reflector must keep watching while the daemon runs"
        );
        assert!(
            !cr_finished.load(Ordering::SeqCst),
            "CR reflector must keep watching while the daemon runs"
        );

        cancel_token.cancel();
        await_daemon(daemon).await;
    }

    #[test]
    fn snapshot_detects_recreated_cr_with_same_generation() {
        let mut previous_snapshot = MetadataSnapshot::empty();
        previous_snapshot.generations.insert(1, 1);
        let current_snapshot = previous_snapshot.clone();
        let previous_revisions = HashMap::from([(
            1,
            CrRevision {
                generation: 1,
                uid: Some("uid-1".to_string()),
            },
        )]);
        let current_revisions = HashMap::from([(
            1,
            CrRevision {
                generation: 1,
                uid: Some("uid-2".to_string()),
            },
        )]);

        assert!(snapshot_has_changes(
            &current_snapshot,
            &previous_snapshot,
            &current_revisions,
            &previous_revisions,
        ));
    }

    fn cached_cr(uid: &str) -> CachedCrMetadata {
        CachedCrMetadata {
            metadata: Arc::new(DiscoveryMetadata::new()),
            generation: 7,
            uid: Some(uid.to_string()),
        }
    }

    #[test]
    fn cached_metadata_for_invalid_cr_reuses_same_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), cached_cr("uid-1"));

        let cached = cached_metadata_for_invalid_cr("worker-a", Some("uid-1"), &cache)
            .expect("cache should be reused for the same CR UID");

        assert_eq!(cached.generation, 7);
    }

    #[test]
    fn cached_metadata_for_invalid_cr_rejects_recreated_kube_object() {
        let mut cache = HashMap::new();
        cache.insert("worker-a".to_string(), cached_cr("uid-1"));

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
