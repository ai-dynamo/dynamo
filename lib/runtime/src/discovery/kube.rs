// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod crd;
mod daemon;
mod utils;

pub use crd::{DynamoWorkerMetadata, DynamoWorkerMetadataSpec};
// hash_pod_name/hash_container_name are used by C bindings and the Rust EPP
// for pod- and container-level worker ID mapping.
pub use utils::{hash_container_name, hash_pod_name};

use crd::{apply_cr, build_cr};
use daemon::{DaemonOutputs, DaemonState, DiscoveryDaemon, ListState};
use futures::future::BoxFuture;
use utils::{KubeDiscoveryMode, PodInfo};

use crate::CancellationToken;
use crate::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryMetadata,
    DiscoveryQuery, DiscoverySpec, DiscoveryStream, MAX_JSON_SAFE_PUBLISHER_ID,
    ModelCardInstanceId, reconcile_discovery_snapshot, resync_discovery_events,
};
use anyhow::Result;
use async_trait::async_trait;
use kube::{
    Api, Client as KubeClient,
    api::{DeleteParams, ListParams},
};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, broadcast, watch};

fn validate_kubernetes_publisher_id(publisher_id: u64) -> Result<()> {
    if publisher_id > MAX_JSON_SAFE_PUBLISHER_ID {
        anyhow::bail!(
            "Kubernetes discovery publisher ID {publisher_id} exceeds the JSON-safe maximum \
             {MAX_JSON_SAFE_PUBLISHER_ID}"
        );
    }

    Ok(())
}

/// Default bound on a single DynamoWorkerMetadata CR apply.
const DEFAULT_CR_APPLY_TIMEOUT: Duration = Duration::from_secs(30);
/// Environment variable overriding [`DEFAULT_CR_APPLY_TIMEOUT`], in seconds.
const DYN_KUBE_CR_APPLY_TIMEOUT_SECS: &str = "DYN_KUBE_CR_APPLY_TIMEOUT_SECS";
/// Attempts per CR write (one initial apply plus one retry).
const CR_APPLY_ATTEMPTS: u32 = 2;

/// Serializes DynamoWorkerMetadata CR writes without holding the metadata lock across the
/// network call.
///
/// Local mutations bump `generation` while holding the metadata write lock. `applied` is the
/// CR write lock and records the newest generation known to be persisted.
struct CrWriter {
    generation: AtomicU64,
    applied: Mutex<u64>,
    apply_timeout: Duration,
}

impl CrWriter {
    fn new(apply_timeout: Duration) -> Self {
        Self {
            generation: AtomicU64::new(0),
            applied: Mutex::new(0),
            apply_timeout,
        }
    }

    fn from_env() -> Self {
        let apply_timeout = std::env::var(DYN_KUBE_CR_APPLY_TIMEOUT_SECS)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|secs| *secs > 0)
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_CR_APPLY_TIMEOUT);
        Self::new(apply_timeout)
    }

    async fn persist_with_retry<P, Fut>(
        &self,
        snapshot: &DiscoveryMetadata,
        persist: &P,
    ) -> Result<()>
    where
        P: Fn(DiscoveryMetadata) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let mut last_error = None;
        for attempt in 1..=CR_APPLY_ATTEMPTS {
            let error =
                match tokio::time::timeout(self.apply_timeout, persist(snapshot.clone())).await {
                    Ok(Ok(())) => return Ok(()),
                    Ok(Err(e)) => e,
                    Err(_) => anyhow::anyhow!(
                        "timed out after {:?} applying DynamoWorkerMetadata CR",
                        self.apply_timeout
                    ),
                };
            tracing::warn!(attempt, "DynamoWorkerMetadata CR write failed: {error}");
            last_error = Some(error);
        }
        let error = last_error.expect("CR_APPLY_ATTEMPTS is non-zero");
        Err(anyhow::anyhow!(
            "DynamoWorkerMetadata CR write failed after {CR_APPLY_ATTEMPTS} attempts: {error}"
        ))
    }
}

/// Mutates the in-memory metadata, then persists it to the CR.
///
/// The metadata lock is held only for the in-memory mutation, never across the CR apply, so a
/// stalled apply cannot block other registrations. CR writes are serialized by
/// `writer.applied` and always apply the latest metadata, so an earlier mutation can never
/// overwrite a later one. A write whose generation was already covered by a newer persisted
/// snapshot is skipped. If the write fails, the local mutation is rolled back unless a newer
/// mutation landed meanwhile (its own CR write then carries this one too), and the original
/// snapshot is written back to the CR. Every fenced write bumps the CR's resourceVersion, so a
/// dropped apply still in flight is rejected with 409 rather than landing after the restore.
async fn mutate_and_persist<R, M, P, Fut>(
    metadata: &RwLock<DiscoveryMetadata>,
    writer: &CrWriter,
    mutate: M,
    persist: P,
) -> Result<R>
where
    M: FnOnce(&mut DiscoveryMetadata) -> Result<R>,
    P: Fn(DiscoveryMetadata) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let (result, original, generation) = {
        let mut guard = metadata.write().await;
        let original = guard.clone();
        let result = mutate(&mut guard)?;
        let generation = writer.generation.fetch_add(1, Ordering::SeqCst) + 1;
        (result, original, generation)
    };

    let mut applied = writer.applied.lock().await;
    if *applied >= generation {
        // A newer snapshot, which includes this mutation, is already persisted.
        return Ok(result);
    }
    let (snapshot, snapshot_generation) = {
        let guard = metadata.read().await;
        (guard.clone(), writer.generation.load(Ordering::SeqCst))
    };

    match writer.persist_with_retry(&snapshot, &persist).await {
        Ok(()) => {
            *applied = snapshot_generation;
            Ok(result)
        }
        Err(e) => {
            // Keep the CR write lock so the restore below cannot interleave with a newer write.
            let rolled_back = {
                let mut guard = metadata.write().await;
                if writer.generation.load(Ordering::SeqCst) == generation {
                    tracing::warn!(
                        "Failed to persist metadata to CR, rolling back local state: {e}"
                    );
                    *guard = original.clone();
                    true
                } else {
                    tracing::warn!(
                        "Failed to persist metadata to CR; a newer local update exists and will \
                         be persisted by its own write: {e}"
                    );
                    false
                }
            };
            if rolled_back {
                // A dropped apply may still land. This write bumps the CR's resourceVersion
                // (even if the CR already equals `original`), so older in-flight applies get 409.
                if let Err(restore_error) = writer.persist_with_retry(&original, &persist).await {
                    tracing::warn!(
                        "Failed to restore the CR after rollback; a stale apply may still land \
                         and leave the CR ahead of local state: {restore_error}"
                    );
                }
            }
            Err(e)
        }
    }
}

async fn update_model_taints_and_persist<P, Fut>(
    metadata: &Arc<RwLock<DiscoveryMetadata>>,
    writer: &Arc<CrWriter>,
    id: ModelCardInstanceId,
    taints: HashSet<String>,
    persist: P,
) -> Result<bool>
where
    P: Fn(DiscoveryMetadata) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let metadata = Arc::clone(metadata);
    let writer = Arc::clone(writer);
    // Once started, persistence and the matching local commit must outlive request cancellation.
    // Dropping the JoinHandle detaches this task instead of cancelling the remote-commit/local-
    // state critical section.
    tokio::spawn(async move {
        // Persist even a local no-op. This repairs an authoritative CR that may differ after an
        // earlier commit/ack ambiguity instead of trusting potentially stale local metadata.
        mutate_and_persist(
            &metadata,
            &writer,
            |m| m.update_model_taints(&id, taints),
            persist,
        )
        .await
    })
    .await
    .map_err(|error| anyhow::anyhow!("model taint persistence task failed: {error}"))?
}

/// Kubernetes-based discovery client
#[derive(Clone)]
pub struct KubeDiscoveryClient {
    instance_id: u64,
    metadata: Arc<RwLock<DiscoveryMetadata>>,
    list_state: ListState,
    event_tx: broadcast::Sender<DiscoveryEvent>,
    /// The daemon's readiness; `list` and `list_and_watch` wait for `Ready` on it.
    daemon_state: watch::Receiver<DaemonState>,
    kube_client: KubeClient,
    pod_info: PodInfo,
    cr_writer: Arc<CrWriter>,
}

impl KubeDiscoveryClient {
    /// Waits until the daemon holds its first complete view of the cluster.
    ///
    /// Fails when the daemon stopped or failed first, or when `cancel_token` fires.
    async fn await_daemon_ready(&self, cancel_token: Option<&CancellationToken>) -> Result<()> {
        let mut daemon_state = self.daemon_state.clone();
        let settled = daemon_state.wait_for(|state| *state != DaemonState::Pending);
        let state = match cancel_token {
            Some(token) => token.run_until_cancelled(settled).await.ok_or_else(|| {
                anyhow::anyhow!(
                    "the watch was cancelled before the Kubernetes discovery daemon was ready"
                )
            })?,
            None => settled.await,
        }
        .map_err(|_| {
            anyhow::anyhow!("the Kubernetes discovery daemon ended without reporting its state")
        })?;
        match &*state {
            DaemonState::Ready => Ok(()),
            DaemonState::Stopped => anyhow::bail!("the Kubernetes discovery daemon is stopped"),
            DaemonState::Failed(reason) => {
                anyhow::bail!("the Kubernetes discovery daemon failed: {reason}")
            }
            DaemonState::Pending => unreachable!("wait_for returns once the daemon left Pending"),
        }
    }

    /// Create a new Kubernetes discovery client
    ///
    /// # Arguments
    /// * `metadata` - Shared metadata store (also used by system server)
    /// * `cancel_token` - Cancellation token for shutdown
    pub async fn new(
        metadata: Arc<RwLock<DiscoveryMetadata>>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let pod_info = PodInfo::from_env()?;
        let instance_id = pod_info.target.instance_id();
        let cr_name = pod_info.target.cr_name();

        tracing::info!(
            "Initializing KubeDiscoveryClient: mode={:?}, target={:?}, cr_name={}, instance_id={:x}, namespace={}, pod_uid={}",
            pod_info.mode,
            pod_info.target,
            cr_name,
            instance_id,
            pod_info.pod_namespace,
            pod_info.pod_uid
        );

        let kube_client = KubeClient::try_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Kubernetes client: {}", e))?;

        // In container mode, delete any stale CR from a previous incarnation of this container.
        // In failover pods, the pod stays alive when a container crashes and restarts,
        // so the old CR persists. Deleting it ensures the daemon doesn't see stale data.
        // In pod mode this is unnecessary — pod restart creates a new pod (and new CR name).
        if pod_info.mode == KubeDiscoveryMode::Container {
            let cr_api: Api<DynamoWorkerMetadata> =
                Api::namespaced(kube_client.clone(), &pod_info.pod_namespace);
            match cr_api.delete(&cr_name, &DeleteParams::default()).await {
                Ok(_) => tracing::info!("Deleted stale CR: {}", cr_name),
                Err(kube::Error::Api(err_resp)) if err_resp.code == 404 => {
                    tracing::debug!("No stale CR to delete: {}", cr_name);
                }
                Err(e) => {
                    panic!(
                        "Failed to clear stale CR '{}': {} — cannot start with stale discovery state",
                        cr_name, e
                    );
                }
            }
        }

        let list_state: ListState = Arc::new(RwLock::new(HashMap::new()));
        let (event_tx, _) = broadcast::channel::<DiscoveryEvent>(4096);
        let (state_tx, daemon_state) = watch::channel(DaemonState::Pending);

        let daemon = DiscoveryDaemon::new(kube_client.clone(), pod_info.clone(), cancel_token)?;
        tokio::spawn(daemon.run(DaemonOutputs {
            list_state: list_state.clone(),
            event_tx: event_tx.clone(),
            state_tx,
        }));

        tracing::info!("Discovery daemon started");

        Ok(Self {
            instance_id,
            metadata,
            list_state,
            event_tx,
            daemon_state,
            kube_client,
            pod_info,
            cr_writer: Arc::new(CrWriter::from_env()),
        })
    }

    /// Builds the persist function: renders a metadata snapshot into the CR and applies it.
    fn cr_persister(
        &self,
    ) -> impl Fn(DiscoveryMetadata) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static {
        let kube_client = self.kube_client.clone();
        let namespace = self.pod_info.pod_namespace.clone();
        let cr_name = self.pod_info.target.cr_name();
        let pod_name = self.pod_info.pod_name.clone();
        let pod_uid = self.pod_info.pod_uid.clone();
        move |snapshot| {
            let kube_client = kube_client.clone();
            let namespace = namespace.clone();
            let cr_name = cr_name.clone();
            let pod_name = pod_name.clone();
            let pod_uid = pod_uid.clone();
            Box::pin(async move {
                let cr = build_cr(&cr_name, &pod_name, &pod_uid, &snapshot)?;
                apply_cr(&kube_client, &namespace, &cr).await
            })
        }
    }
}

#[async_trait]
impl Discovery for KubeDiscoveryClient {
    async fn check_connection(&self) -> Result<()> {
        // Check Kubernetes API connectivity and permission to list DynamoWorkerMetadata.
        // Request metadata for at most one object to keep the response small.
        // An empty collection is valid before registration; API errors propagate.
        // This does not verify discovery watch health or local cache synchronization.
        let api: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
        api.list_metadata(&ListParams::default().limit(1)).await?;
        Ok(())
    }

    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        match &spec {
            DiscoverySpec::EventChannel { publisher_id, .. }
            | DiscoverySpec::EventSource { publisher_id, .. } => {
                validate_kubernetes_publisher_id(*publisher_id)?;
            }
            _ => {}
        }
        let instance = spec.into_instance(self.instance_id());
        let instance_id = instance.instance_id();

        tracing::debug!(
            "Registering discovery instance: {:?}, instance_id={:x}",
            instance,
            instance_id
        );

        // Mutate local metadata under the lock, then persist a snapshot to the CR outside it.
        let registered_instance = mutate_and_persist(
            &self.metadata,
            &self.cr_writer,
            |metadata| match &instance {
                DiscoveryInstance::Endpoint(inst) => {
                    tracing::info!(
                        "Registering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                        inst.namespace,
                        inst.component,
                        inst.endpoint,
                        instance_id
                    );
                    metadata.register_endpoint(instance.clone())?;
                    Ok(instance.clone())
                }
                DiscoveryInstance::Model {
                    namespace,
                    component,
                    endpoint,
                    ..
                } => {
                    tracing::info!(
                        "Registering model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                        namespace,
                        component,
                        endpoint,
                        instance_id
                    );
                    metadata.register_model_card(instance.clone())
                }
                DiscoveryInstance::EventChannel { scope, topic, .. } => {
                    tracing::info!(
                        "Registering event channel: scope={:?}, topic={}, instance_id={:x}",
                        scope,
                        topic,
                        instance_id
                    );
                    metadata.register_event_channel(instance.clone())?;
                    Ok(instance.clone())
                }
                DiscoveryInstance::EventSource { scope, topic, .. } => {
                    tracing::info!(
                        "Registering event source: scope={:?}, topic={}, publisher_id={:x}",
                        scope,
                        topic,
                        instance_id
                    );
                    metadata.register_event_source(instance.clone())?;
                    Ok(instance.clone())
                }
            },
            self.cr_persister(),
        )
        .await?;

        tracing::debug!("Persisted metadata to DynamoWorkerMetadata CR");

        Ok(registered_instance)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let changed = update_model_taints_and_persist(
            &self.metadata,
            &self.cr_writer,
            id,
            taints,
            self.cr_persister(),
        )
        .await?;
        if !changed {
            return Ok(());
        }

        tracing::debug!("Persisted model taint update to DynamoWorkerMetadata CR");
        Ok(())
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let instance_id = instance.instance_id();

        // Mutate local metadata under the lock, then persist a snapshot to the CR outside it.
        mutate_and_persist(
            &self.metadata,
            &self.cr_writer,
            |metadata| {
            match &instance {
                DiscoveryInstance::Endpoint(inst) => {
                    tracing::info!(
                        "Unregistering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                        inst.namespace,
                        inst.component,
                        inst.endpoint,
                        instance_id
                    );
                    metadata.unregister_endpoint(&instance)?;
                }
                DiscoveryInstance::Model {
                    namespace,
                    component,
                    endpoint,
                    ..
                } => {
                    tracing::info!(
                        "Unregistering model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                        namespace,
                        component,
                        endpoint,
                        instance_id
                    );
                    metadata.unregister_model_card(&instance)?;
                }
                DiscoveryInstance::EventChannel { scope, topic, .. } => {
                    tracing::info!(
                        "Unregistering event channel: scope={:?}, topic={}, instance_id={:x}",
                        scope,
                        topic,
                        instance_id
                    );
                    metadata.unregister_event_channel(&instance)?;
                }
                DiscoveryInstance::EventSource { scope, topic, .. } => {
                    tracing::info!(
                        "Unregistering event source: scope={:?}, topic={}, publisher_id={:x}",
                        scope,
                        topic,
                        instance_id
                    );
                    metadata.unregister_event_source(&instance)?;
                }
            }
                Ok(())
            },
            self.cr_persister(),
        )
        .await?;

        tracing::debug!("Persisted metadata removal to DynamoWorkerMetadata CR");

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        tracing::debug!("KubeDiscoveryClient::list called with query={:?}", query);

        // Before the initial sync, list_state is empty whatever the cluster holds.
        self.await_daemon_ready(None).await?;
        let state = self.list_state.read().await;
        let instances: Vec<DiscoveryInstance> =
            state.values().flat_map(|m| m.filter(&query)).collect();

        tracing::info!(
            "KubeDiscoveryClient::list returning {} instances for query={:?}",
            instances.len(),
            query
        );

        Ok(instances)
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        use broadcast::error::RecvError;
        use tokio::sync::mpsc;

        tracing::info!(
            "KubeDiscoveryClient::list_and_watch started for query={:?}",
            query
        );

        // Before the initial sync, the snapshot would be a false empty set.
        self.await_daemon_ready(cancel_token.as_ref()).await?;
        let (out_tx, out_rx) = mpsc::unbounded_channel();
        let stream_id = uuid::Uuid::new_v4();

        // Acquire read lock, subscribe to broadcast, then read initial state.
        // The write lock (held by the daemon while updating list_state and sending events)
        // is mutually exclusive with our read lock, so no events can slip between
        // our subscription point and our initial state read.
        // This runs before the return, so a caller that lists afterwards cannot observe an
        // instance that a removal deletes before the subscription.
        let (initial_instances, mut broadcast_rx) = {
            let state = self.list_state.read().await;
            let rx = self.event_tx.subscribe();
            let initial = state
                .values()
                .flat_map(|m| m.filter(&query))
                .collect::<Vec<_>>();
            (initial, rx)
        };
        let list_state = self.list_state.clone();

        tokio::spawn(async move {
            tracing::debug!(
                stream_id = %stream_id,
                initial_count = initial_instances.len(),
                "Watch started for query={:?}",
                query
            );

            let mut known: HashMap<DiscoveryInstanceId, DiscoveryInstance> = initial_instances
                .iter()
                .map(|i| (i.id(), i.clone()))
                .collect();

            for instance in &initial_instances {
                tracing::info!(
                    stream_id = %stream_id,
                    instance_id = format!("{:x}", instance.instance_id()),
                    "Emitting initial Added event"
                );
                if out_tx
                    .send(Ok(DiscoveryEvent::Added(instance.clone())))
                    .is_err()
                {
                    return;
                }
            }
            if out_tx
                .send(Ok(DiscoveryEvent::Resync(initial_instances)))
                .is_err()
            {
                return;
            }

            loop {
                let recv_result = if let Some(ref token) = cancel_token {
                    tokio::select! {
                        result = broadcast_rx.recv() => result,
                        _ = token.cancelled() => {
                            tracing::info!(stream_id = %stream_id, "Watch cancelled via cancel token");
                            break;
                        }
                    }
                } else {
                    broadcast_rx.recv().await
                };

                match recv_result {
                    Ok(event) => {
                        let forwarded = match &event {
                            DiscoveryEvent::Added(instance) => {
                                if instance.matches(&query) {
                                    let id = instance.id();
                                    if known.get(&id) != Some(instance) {
                                        known.insert(id.clone(), instance.clone());
                                        Some(("added", id))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            DiscoveryEvent::Removed(id) => {
                                known.remove(id).is_some().then(|| ("removed", id.clone()))
                            }
                            DiscoveryEvent::ModelTaintsUpdated(update) => {
                                let id = DiscoveryInstanceId::Model(update.id.clone());
                                known
                                    .contains_key(&id)
                                    .then_some(("model_taints_updated", id))
                            }
                            // The daemon publishes incremental events only.
                            DiscoveryEvent::Resync(_) => None,
                        };
                        if let Some((event_kind, instance_id)) = forwarded {
                            tracing::info!(
                                stream_id = %stream_id,
                                event_kind,
                                ?instance_id,
                                "Emitting discovery event"
                            );
                            if out_tx.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(
                            stream_id = %stream_id,
                            dropped = n,
                            "Broadcast receiver lagged, reconciling from list_state"
                        );
                        let state = list_state.read().await;
                        let current: HashMap<DiscoveryInstanceId, DiscoveryInstance> = state
                            .values()
                            .flat_map(|m| m.filter(&query))
                            .map(|i| (i.id(), i))
                            .collect();
                        drop(state);
                        for event in resync_discovery_events(&mut known, current) {
                            if out_tx.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvError::Closed) => {
                        tracing::info!(
                            stream_id = %stream_id,
                            "Broadcast channel closed (daemon stopped)"
                        );
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(out_rx),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;
    use crate::discovery::startup_contract as contract;
    use crate::discovery::{EventScope, EventTransport, ModelTaintsUpdate};
    use crd::{CrApi, apply_cr_fenced};

    // Regression: suppressing NotFound reports healthy for a missing discovery
    // API, even though discovery cannot list or watch any workers.
    #[tokio::test]
    async fn connectivity_requires_resource_but_not_registered_workers() {
        use axum::{
            body::Body,
            http::{Request, Response, StatusCode},
        };
        use std::sync::atomic::{AtomicBool, Ordering};

        let available = Arc::new(AtomicBool::new(false));
        let resource_available = available.clone();
        let service = tower::service_fn(move |request: Request<kube::client::Body>| {
            assert_eq!(request.method(), "GET");
            assert_eq!(
                request.uri().path(),
                "/apis/nvidia.com/v1alpha1/namespaces/default/dynamoworkermetadatas"
            );
            assert!(
                request
                    .uri()
                    .query()
                    .unwrap()
                    .split('&')
                    .any(|part| part == "limit=1")
            );
            assert_eq!(
                request.headers()["accept"],
                "application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1"
            );
            let available = resource_available.load(Ordering::Acquire);
            async move {
                let response = if available {
                    Response::new(Body::from(
                        r#"{"apiVersion":"meta.k8s.io/v1","kind":"PartialObjectMetadataList","metadata":{"resourceVersion":"1"},"items":[]}"#,
                    ))
                } else {
                    Response::builder().status(StatusCode::NOT_FOUND).body(Body::from(
                        r#"{"kind":"Status","apiVersion":"v1","status":"Failure","reason":"NotFound","code":404,"message":"resource not found"}"#,
                    )).unwrap()
                };
                Ok::<_, std::convert::Infallible>(response)
            }
        });
        let discovery = KubeDiscoveryClient {
            instance_id: 1,
            metadata: Arc::new(RwLock::new(DiscoveryMetadata::new())),
            list_state: Arc::new(RwLock::new(HashMap::new())),
            event_tx: broadcast::channel(1).0,
            daemon_state: watch::channel(DaemonState::Pending).1,
            kube_client: KubeClient::new(service, "default"),
            cr_writer: Arc::new(CrWriter::from_env()),
            pod_info: PodInfo {
                pod_name: "worker".into(),
                pod_namespace: "default".into(),
                pod_uid: "uid".into(),
                system_port: 0,
                mode: KubeDiscoveryMode::Pod,
                target: utils::KubeDiscoveryTarget::Pod("worker".into()),
            },
        };
        assert!(discovery.check_connection().await.is_err());
        available.store(true, Ordering::Release);
        assert!(discovery.check_connection().await.is_ok());
        available.store(false, Ordering::Release);
        assert!(discovery.check_connection().await.is_err());
    }

    fn endpoint_instance(instance_id: u64, transport: &str) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(crate::component::Instance {
            namespace: "ns".to_string(),
            component: "component".to_string(),
            endpoint: "endpoint".to_string(),
            instance_id,
            transport: TransportType::Tcp(transport.to_string()),
            device_type: None,
            request_plane_codec: None,
        })
    }

    fn model_with_taint(taint: &str) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 7,
            card_json: serde_json::json!({
                "runtime_config": {"taints": [taint]}
            }),
            model_suffix: None,
        }
    }

    /// A client over `instances` with no cluster behind it, and the sender for its daemon
    /// state, which starts `Pending`.
    fn client_with(
        instances: &[DiscoveryInstance],
    ) -> (KubeDiscoveryClient, watch::Sender<DaemonState>) {
        let kube_client =
            KubeClient::try_from(kube::Config::new("http://127.0.0.1:1".parse().unwrap())).unwrap();
        let list_state = instances
            .iter()
            .map(|instance| {
                let mut metadata = DiscoveryMetadata::new();
                metadata.register_endpoint(instance.clone()).unwrap();
                (instance.instance_id(), Arc::new(metadata))
            })
            .collect();
        let (event_tx, _) = broadcast::channel(16);
        let (state_tx, daemon_state) = watch::channel(DaemonState::Pending);
        let client = KubeDiscoveryClient {
            instance_id: 1,
            metadata: Arc::new(RwLock::new(DiscoveryMetadata::new())),
            list_state: Arc::new(RwLock::new(list_state)),
            event_tx,
            daemon_state,
            kube_client,
            cr_writer: Arc::new(CrWriter::from_env()),
            pod_info: PodInfo {
                pod_name: "worker-a".to_string(),
                pod_namespace: "ns".to_string(),
                pod_uid: "pod-uid".to_string(),
                system_port: 0,
                mode: KubeDiscoveryMode::Pod,
                target: utils::KubeDiscoveryTarget::Pod("worker-a".to_string()),
            },
        };
        (client, state_tx)
    }

    #[tokio::test]
    async fn watch_reports_the_initial_set_as_added_events_then_one_resync() {
        let first = endpoint_instance(1, "127.0.0.1:8000");
        let second = endpoint_instance(2, "127.0.0.1:9000");
        let (client, state_tx) = client_with(&[first.clone(), second.clone()]);
        state_tx.send_replace(DaemonState::Ready);
        let mut events = client
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();

        let mut added = HashSet::new();
        for _ in 0..2 {
            let DiscoveryEvent::Added(instance) = contract::next(&mut events).await else {
                panic!("expected an initial Added event");
            };
            added.insert(instance.id());
        }
        assert_eq!(added, HashSet::from([first.id(), second.id()]));
        let DiscoveryEvent::Resync(snapshot) = contract::next(&mut events).await else {
            panic!("expected the establishment snapshot after the Added burst");
        };
        assert_eq!(
            snapshot
                .iter()
                .map(DiscoveryInstance::id)
                .collect::<HashSet<_>>(),
            HashSet::from([first.id(), second.id()])
        );

        let third = endpoint_instance(3, "127.0.0.1:9100");
        client
            .event_tx
            .send(DiscoveryEvent::Added(third.clone()))
            .unwrap();
        assert_eq!(
            contract::next(&mut events).await,
            DiscoveryEvent::Added(third)
        );
    }

    #[tokio::test]
    async fn list_and_watch_waits_for_ready_unless_cancelled_or_the_daemon_ended() {
        let (client, state_tx) = client_with(&[]);
        let client = Arc::new(client);
        let open_watch = |token: Option<CancellationToken>| {
            let client = client.clone();
            tokio::spawn(async move {
                client
                    .list_and_watch(DiscoveryQuery::AllEndpoints, token)
                    .await
            })
        };
        let finish = |task| tokio::time::timeout(std::time::Duration::from_secs(1), task);

        let token = CancellationToken::new();
        let cancelled = open_watch(Some(token.clone()));
        tokio::task::yield_now().await;
        assert!(
            !cancelled.is_finished(),
            "a pending daemon must hold the watch"
        );
        token.cancel();
        let Err(error) = finish(cancelled)
            .await
            .expect("cancellation must release the wait")
            .unwrap()
        else {
            panic!("a watch cancelled before Ready must fail");
        };
        assert!(
            error.to_string().contains("cancelled"),
            "unexpected error: {error}"
        );

        let waiting = open_watch(None);
        tokio::task::yield_now().await;
        assert!(
            !waiting.is_finished(),
            "a pending daemon must hold the watch"
        );
        state_tx.send_replace(DaemonState::Ready);
        let mut events = finish(waiting)
            .await
            .expect("Ready must release the wait")
            .unwrap()
            .unwrap();
        contract::expect_empty_snapshot(&mut events).await;

        for (state, expected) in [
            (
                DaemonState::Failed("reflector stopped".to_string()),
                "daemon failed: reflector stopped",
            ),
            (DaemonState::Stopped, "daemon is stopped"),
        ] {
            state_tx.send_replace(state);
            let Err(error) = client
                .list_and_watch(DiscoveryQuery::AllEndpoints, None)
                .await
            else {
                panic!("a daemon that ended before Ready must fail the watch");
            };
            assert!(
                error.to_string().contains(expected),
                "unexpected error: {error}"
            );
            let error = client
                .list(DiscoveryQuery::AllEndpoints)
                .await
                .expect_err("a daemon that ended before Ready must fail the list");
            assert!(
                error.to_string().contains(expected),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn publisher_ids_must_fit_kubernetes_json_safe_range() {
        assert!(validate_kubernetes_publisher_id(MAX_JSON_SAFE_PUBLISHER_ID).is_ok());
        assert!(validate_kubernetes_publisher_id(MAX_JSON_SAFE_PUBLISHER_ID + 1).is_err());
        assert!(validate_kubernetes_publisher_id(u64::MAX).is_err());
    }

    #[test]
    fn snapshot_diff_emits_updated_instance_when_transport_changes() {
        let original = endpoint_instance(1, "127.0.0.1:8000");
        let updated = endpoint_instance(1, "127.0.0.1:9000");
        let known = HashMap::from([(original.id(), original)]);
        let current = HashMap::from([(updated.id(), updated.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert_eq!(events, vec![DiscoveryEvent::Added(updated.clone())]);
        assert_eq!(reconciled.get(&updated.id()), Some(&updated));
    }

    #[test]
    fn snapshot_diff_ignores_same_id_event_channel_changes() {
        let event_channel = |endpoint: &str| DiscoveryInstance::EventChannel {
            scope: EventScope::Namespace {
                name: "ns".to_string(),
            },
            topic: "topic".to_string(),
            instance_id: 1,
            transport: EventTransport::zmq(endpoint),
        };
        let original = event_channel("tcp://127.0.0.1:8000");
        let updated = event_channel("tcp://127.0.0.1:9000");
        let known = HashMap::from([(original.id(), original.clone())]);
        let current = HashMap::from([(updated.id(), updated)]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert!(events.is_empty());
        assert_eq!(reconciled.get(&original.id()), Some(&original));
    }

    #[test]
    fn snapshot_diff_emits_added_and_removed_instances() {
        let removed_instance = endpoint_instance(1, "127.0.0.1:8000");
        let added_instance = endpoint_instance(2, "127.0.0.1:9000");
        let removed_id = removed_instance.id();
        let added_id = added_instance.id();
        let known = HashMap::from([(removed_id.clone(), removed_instance)]);
        let current = HashMap::from([(added_id.clone(), added_instance.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        assert_eq!(events.len(), 2);
        assert!(events.contains(&DiscoveryEvent::Removed(removed_id.clone())));
        assert!(events.contains(&DiscoveryEvent::Added(added_instance.clone())));
        assert!(!reconciled.contains_key(&removed_id));
        assert_eq!(reconciled.get(&added_id), Some(&added_instance));
    }
    #[test]
    fn changed_model_taints_emit_scoped_event() {
        let old = model_with_taint("old");
        let updated = model_with_taint("updated");
        let known = HashMap::from([(old.id(), old)]);
        let current = HashMap::from([(updated.id(), updated.clone())]);

        let (events, reconciled) = reconcile_discovery_snapshot(&known, current);

        let DiscoveryInstanceId::Model(id) = updated.id() else {
            unreachable!()
        };
        assert_eq!(
            events,
            vec![DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id,
                taints: vec!["updated".to_string()],
            })]
        );
        assert_eq!(reconciled.get(&updated.id()), Some(&updated));
    }

    #[tokio::test]
    async fn model_taint_persistence_completes_after_caller_cancellation() {
        let model = model_with_taint("old");
        let DiscoveryInstanceId::Model(id) = model.id() else {
            unreachable!()
        };
        let mut initial = DiscoveryMetadata::new();
        initial.register_model_card(model).unwrap();
        let metadata = Arc::new(RwLock::new(initial));
        let task_metadata = metadata.clone();
        let remote = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        let task_remote = remote.clone();
        let writer = Arc::new(CrWriter::new(Duration::from_secs(30)));
        let (remote_committed_tx, remote_committed_rx) = tokio::sync::oneshot::channel();
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let once = Arc::new(std::sync::Mutex::new(Some((remote_committed_tx, ack_rx))));

        let task = tokio::spawn(async move {
            update_model_taints_and_persist(
                &task_metadata,
                &writer,
                id,
                HashSet::from(["new".to_string()]),
                move |candidate| {
                    let task_remote = task_remote.clone();
                    let once = once.clone();
                    async move {
                        let (remote_committed_tx, ack_rx) = once.lock().unwrap().take().unwrap();
                        *task_remote.write().await = candidate;
                        remote_committed_tx.send(()).unwrap();
                        ack_rx.await.unwrap();
                        Ok(())
                    }
                },
            )
            .await
        });

        remote_committed_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        ack_tx.send(()).unwrap();

        let stored = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let stored = metadata.read().await.get_all_model_cards().pop().unwrap();
                let DiscoveryInstance::Model { card_json, .. } = &stored else {
                    unreachable!()
                };
                if card_json["runtime_config"]["taints"] == serde_json::json!(["new"]) {
                    break stored;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached persistence did not commit local metadata");
        let DiscoveryInstance::Model { card_json, .. } = stored else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["new"])
        );
        let remote = remote.read().await.get_all_model_cards().pop().unwrap();
        let DiscoveryInstance::Model { card_json, .. } = remote else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["new"])
        );
    }

    #[tokio::test]
    async fn local_noop_reapplies_authoritative_model_taints() {
        let local_model = model_with_taint("old");
        let DiscoveryInstanceId::Model(id) = local_model.id() else {
            unreachable!()
        };
        let mut initial = DiscoveryMetadata::new();
        initial.register_model_card(local_model).unwrap();
        let metadata = Arc::new(RwLock::new(initial));
        let persisted = Arc::new(RwLock::new(None));
        let task_persisted = persisted.clone();

        let writer = Arc::new(CrWriter::new(Duration::from_secs(30)));
        let changed = update_model_taints_and_persist(
            &metadata,
            &writer,
            id,
            HashSet::from(["old".to_string()]),
            move |candidate| {
                let task_persisted = task_persisted.clone();
                async move {
                    *task_persisted.write().await = Some(candidate);
                    Ok(())
                }
            },
        )
        .await
        .unwrap();

        assert!(!changed);
        let reapplied = persisted
            .read()
            .await
            .clone()
            .expect("no-op was not persisted");
        let DiscoveryInstance::Model { card_json, .. } =
            reapplied.get_all_model_cards().pop().unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            card_json["runtime_config"]["taints"],
            serde_json::json!(["old"])
        );
    }

    fn cr_endpoint_instance(endpoint: &str) -> DiscoveryInstance {
        DiscoverySpec::Endpoint {
            namespace: "ns".to_string(),
            component: "comp".to_string(),
            endpoint: endpoint.to_string(),
            transport: TransportType::Tcp("127.0.0.1:1".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
        .into_instance(1)
    }

    fn register(instance: DiscoveryInstance) -> impl FnOnce(&mut DiscoveryMetadata) -> Result<()> {
        move |m| m.register_endpoint(instance)
    }

    // The metadata lock must not be held across the CR apply: a second registration finishes
    // its in-memory update while the first apply is stalled.
    #[tokio::test]
    async fn stalled_cr_apply_does_not_block_other_registrations() {
        let metadata = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        let writer = Arc::new(CrWriter::new(Duration::from_secs(30)));
        let (stall_tx, stall_rx) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let stall = Arc::new(std::sync::Mutex::new(Some((started_tx, stall_rx))));

        let first = {
            let (metadata, writer) = (metadata.clone(), writer.clone());
            tokio::spawn(async move {
                mutate_and_persist(
                    &metadata,
                    &writer,
                    register(cr_endpoint_instance("first")),
                    move |_| {
                        let stall = stall.clone();
                        async move {
                            let (started_tx, stall_rx) = stall.lock().unwrap().take().unwrap();
                            started_tx.send(()).unwrap();
                            stall_rx.await.unwrap();
                            Ok(())
                        }
                    },
                )
                .await
            })
        };
        started_rx.await.unwrap();

        // The second mutation completes in memory while the first apply is in flight.
        let second = {
            let (metadata, writer) = (metadata.clone(), writer.clone());
            tokio::spawn(async move {
                mutate_and_persist(
                    &metadata,
                    &writer,
                    register(cr_endpoint_instance("second")),
                    |_| async { Ok(()) },
                )
                .await
            })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while metadata.read().await.get_all_endpoints().len() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second registration was blocked behind the stalled CR apply");
        assert!(!second.is_finished(), "second CR write must wait its turn");

        stall_tx.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    // Concurrent writers serialize and the last CR content reflects the latest metadata, even
    // when an earlier writer is slow.
    #[tokio::test]
    async fn final_cr_reflects_latest_metadata() {
        let metadata = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        let writer = Arc::new(CrWriter::new(Duration::from_secs(30)));
        let cr = Arc::new(std::sync::Mutex::new(None::<DiscoveryMetadata>));
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let gate = Arc::new(std::sync::Mutex::new(Some((started_tx, release_rx))));

        type Gate = Arc<
            std::sync::Mutex<
                Option<(
                    tokio::sync::oneshot::Sender<()>,
                    tokio::sync::oneshot::Receiver<()>,
                )>,
            >,
        >;
        let persist = |cr: Arc<std::sync::Mutex<Option<DiscoveryMetadata>>>, gate: Option<Gate>| {
            move |snapshot: DiscoveryMetadata| {
                let cr = cr.clone();
                let gate = gate.clone();
                async move {
                    let held = gate.and_then(|g| g.lock().unwrap().take());
                    if let Some((started_tx, release_rx)) = held {
                        started_tx.send(()).unwrap();
                        release_rx.await.unwrap();
                    }
                    *cr.lock().unwrap() = Some(snapshot);
                    Ok(())
                }
            }
        };

        let first = {
            let (metadata, writer) = (metadata.clone(), writer.clone());
            let persist = persist(cr.clone(), Some(gate));
            tokio::spawn(async move {
                mutate_and_persist(
                    &metadata,
                    &writer,
                    register(cr_endpoint_instance("a")),
                    persist,
                )
                .await
            })
        };
        started_rx.await.unwrap();
        let second = {
            let (metadata, writer) = (metadata.clone(), writer.clone());
            let persist = persist(cr.clone(), None);
            tokio::spawn(async move {
                mutate_and_persist(
                    &metadata,
                    &writer,
                    register(cr_endpoint_instance("b")),
                    persist,
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        release_tx.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();

        let persisted = cr.lock().unwrap().clone().expect("CR never written");
        assert_eq!(persisted.get_all_endpoints().len(), 2);
    }

    /// In-memory CR store with API-server semantics: a stale resourceVersion gets 409, no
    /// resourceVersion lands on an existing object, and an apply that changes nothing is a
    /// no-op that does not bump the resourceVersion.
    #[derive(Default)]
    struct FakeCrApi {
        state: std::sync::Mutex<(Option<DynamoWorkerMetadata>, u64)>,
        applies: AtomicU64,
        // The first `hang_first` applies never return; they land after `land_after`.
        hang_first: u64,
        land_after: Duration,
    }

    impl FakeCrApi {
        fn land(&self, cr: &DynamoWorkerMetadata) -> Result<()> {
            let mut st = self.state.lock().unwrap();
            if let Some(cur) = &st.0 {
                if cr.metadata.resource_version.is_some()
                    && cr.metadata.resource_version != cur.metadata.resource_version
                {
                    anyhow::bail!("409 Conflict: stale resourceVersion");
                }
                if cr.spec.data == cur.spec.data
                    && cr.metadata.annotations == cur.metadata.annotations
                {
                    return Ok(());
                }
            }
            st.1 += 1;
            let mut stored = cr.clone();
            stored.metadata.resource_version = Some(st.1.to_string());
            st.0 = Some(stored);
            Ok(())
        }

        fn data(&self) -> Option<serde_json::Value> {
            self.state
                .lock()
                .unwrap()
                .0
                .as_ref()
                .map(|c| c.spec.data.clone())
        }

        fn rv(&self) -> u64 {
            self.state.lock().unwrap().1
        }
    }

    #[async_trait]
    impl CrApi for Arc<FakeCrApi> {
        async fn get_cr(&self, _: &str) -> Result<Option<DynamoWorkerMetadata>> {
            Ok(self.state.lock().unwrap().0.clone())
        }

        async fn apply(&self, _: &str, cr: &DynamoWorkerMetadata) -> Result<()> {
            if self.applies.fetch_add(1, Ordering::SeqCst) < self.hang_first {
                let (api, cr, delay) = (self.clone(), cr.clone(), self.land_after);
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = api.land(&cr);
                });
                return std::future::pending().await;
            }
            self.land(cr)
        }
    }

    fn fenced_persist(
        api: &Arc<FakeCrApi>,
    ) -> impl Fn(DiscoveryMetadata) -> BoxFuture<'static, Result<()>> {
        let api = api.clone();
        move |snapshot| {
            let api = api.clone();
            Box::pin(async move {
                let cr = build_cr("cr", "pod", "uid", &snapshot)?;
                apply_cr_fenced(&api, &cr).await
            })
        }
    }

    fn seeded(api: FakeCrApi) -> Arc<FakeCrApi> {
        let cr = build_cr("cr", "pod", "uid", &DiscoveryMetadata::new()).unwrap();
        let api = Arc::new(api);
        api.land(&cr).unwrap();
        api
    }

    // A timed-out apply that actually landed is committed: the retry is at worst a redundant
    // write, and local state is kept.
    #[tokio::test(start_paused = true)]
    async fn timed_out_apply_that_landed_is_committed() {
        let api = seeded(FakeCrApi {
            hang_first: 1,
            ..Default::default()
        });
        let metadata = RwLock::new(DiscoveryMetadata::new());
        let writer = CrWriter::new(Duration::from_millis(50));

        mutate_and_persist(
            &metadata,
            &writer,
            register(cr_endpoint_instance("landed")),
            fenced_persist(&api),
        )
        .await
        .unwrap();

        assert_eq!(metadata.read().await.get_all_endpoints().len(), 1);
        assert_eq!(
            api.data().unwrap(),
            build_cr("cr", "pod", "uid", &*metadata.read().await)
                .unwrap()
                .spec
                .data
        );
    }

    // Every attempt times out and none has landed yet. The CR already equals the original, but
    // the restore must still be a write that bumps the resourceVersion, so the stale attempts
    // arriving afterwards get 409.
    #[tokio::test(start_paused = true)]
    async fn failed_writes_roll_back_and_fence_late_applies() {
        let api = seeded(FakeCrApi {
            hang_first: CR_APPLY_ATTEMPTS as u64,
            land_after: Duration::from_secs(5),
            ..Default::default()
        });
        let (original, rv_before) = (api.data().unwrap(), api.rv());
        let metadata = RwLock::new(DiscoveryMetadata::new());
        let writer = CrWriter::new(Duration::from_secs(1));

        let err = mutate_and_persist(
            &metadata,
            &writer,
            register(cr_endpoint_instance("lost")),
            fenced_persist(&api),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");

        assert!(metadata.read().await.get_all_endpoints().is_empty());
        assert_eq!(api.rv(), rv_before + 1, "restore must write");
        tokio::time::sleep(Duration::from_secs(30)).await;
        assert_eq!(api.data().unwrap(), original, "stale applies must get 409");
        assert_eq!(api.rv(), rv_before + 1);
    }
}
