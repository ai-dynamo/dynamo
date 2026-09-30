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
use utils::{KubeDiscoveryMode, PodInfo};

use crate::CancellationToken;
use crate::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryMetadata,
    DiscoveryQuery, DiscoverySpec, DiscoveryStream, MAX_JSON_SAFE_PUBLISHER_ID,
    ModelCardInstanceId, reconcile_discovery_snapshot, resync_discovery_events,
};
use anyhow::Result;
use async_trait::async_trait;
use kube::{Api, Client as KubeClient, api::DeleteParams};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, watch};

fn validate_kubernetes_publisher_id(publisher_id: u64) -> Result<()> {
    if publisher_id > MAX_JSON_SAFE_PUBLISHER_ID {
        anyhow::bail!(
            "Kubernetes discovery publisher ID {publisher_id} exceeds the JSON-safe maximum \
             {MAX_JSON_SAFE_PUBLISHER_ID}"
        );
    }

    Ok(())
}

async fn update_model_taints_and_persist<F, Fut>(
    metadata: &Arc<RwLock<DiscoveryMetadata>>,
    id: ModelCardInstanceId,
    taints: HashSet<String>,
    persist: F,
) -> Result<bool>
where
    F: FnOnce(DiscoveryMetadata) -> Fut + Send + 'static,
    Fut: Future<Output = Result<DiscoveryMetadata>> + Send + 'static,
{
    let metadata = Arc::clone(metadata);
    // Once started, persistence and the matching local commit must outlive request cancellation.
    // Dropping the JoinHandle detaches this task instead of cancelling the remote-commit/local-
    // state critical section.
    tokio::spawn(async move {
        let mut metadata = metadata.write().await;
        let mut candidate = metadata.clone();
        let changed = candidate.update_model_taints(&id, taints)?;

        // Persist even a local no-op. This repairs an authoritative CR that may differ after an
        // earlier commit/ack ambiguity instead of trusting potentially stale local metadata.
        let persisted = persist(candidate).await?;
        *metadata = persisted;
        Ok(changed)
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
        })
    }
}

#[async_trait]
impl Discovery for KubeDiscoveryClient {
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

        // Write to local metadata and persist to CR
        // IMPORTANT: Hold the write lock across the CR write to prevent race conditions
        let mut metadata = self.metadata.write().await;

        // Clone state for rollback in case CR persistence fails
        let original_state = metadata.clone();

        let registered_instance = match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Registering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    instance_id
                );
                metadata.register_endpoint(instance.clone())?;
                instance.clone()
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
                metadata.register_model_card(instance.clone())?
            }
            DiscoveryInstance::EventChannel { scope, topic, .. } => {
                tracing::info!(
                    "Registering event channel: scope={:?}, topic={}, instance_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.register_event_channel(instance.clone())?;
                instance.clone()
            }
            DiscoveryInstance::EventSource { scope, topic, .. } => {
                tracing::info!(
                    "Registering event source: scope={:?}, topic={}, publisher_id={:x}",
                    scope,
                    topic,
                    instance_id
                );
                metadata.register_event_source(instance.clone())?;
                instance.clone()
            }
        };

        // Build and apply the CR with the updated metadata
        // This persists the metadata to Kubernetes for other pods to discover
        let cr_name = self.pod_info.target.cr_name();
        let cr = build_cr(
            &cr_name,
            &self.pod_info.pod_name,
            &self.pod_info.pod_uid,
            &metadata,
        )?;

        if let Err(e) = apply_cr(&self.kube_client, &self.pod_info.pod_namespace, &cr).await {
            // Rollback local state on CR persistence failure
            tracing::warn!(
                "Failed to persist metadata to CR, rolling back local state: {}",
                e
            );
            *metadata = original_state;
            return Err(e);
        }

        tracing::debug!("Persisted metadata to DynamoWorkerMetadata CR");

        Ok(registered_instance)
    }

    async fn update_model_taints_internal(
        &self,
        id: ModelCardInstanceId,
        taints: HashSet<String>,
    ) -> Result<()> {
        let kube_client = self.kube_client.clone();
        let pod_namespace = self.pod_info.pod_namespace.clone();
        let cr_name = self.pod_info.target.cr_name();
        let pod_name = self.pod_info.pod_name.clone();
        let pod_uid = self.pod_info.pod_uid.clone();
        let changed = update_model_taints_and_persist(
            &self.metadata,
            id,
            taints,
            move |candidate| async move {
                let cr = build_cr(&cr_name, &pod_name, &pod_uid, &candidate)?;
                apply_cr(&kube_client, &pod_namespace, &cr).await?;
                Ok(candidate)
            },
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

        // Write to local metadata and persist to CR
        // IMPORTANT: Hold the write lock across the CR write to prevent race conditions
        let mut metadata = self.metadata.write().await;

        // Clone state for rollback in case CR persistence fails
        let original_state = metadata.clone();

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

        // Build and apply the CR with the updated metadata
        // This persists the removal to Kubernetes for other pods to see
        let cr_name = self.pod_info.target.cr_name();
        let cr = build_cr(
            &cr_name,
            &self.pod_info.pod_name,
            &self.pod_info.pod_uid,
            &metadata,
        )?;

        if let Err(e) = apply_cr(&self.kube_client, &self.pod_info.pod_namespace, &cr).await {
            // Rollback local state on CR persistence failure
            tracing::warn!(
                "Failed to persist metadata removal to CR, rolling back local state: {}",
                e
            );
            *metadata = original_state;
            return Err(e);
        }

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
        let (remote_committed_tx, remote_committed_rx) = tokio::sync::oneshot::channel();
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();

        let task = tokio::spawn(async move {
            update_model_taints_and_persist(
                &task_metadata,
                id,
                HashSet::from(["new".to_string()]),
                move |candidate| async move {
                    *task_remote.write().await = candidate.clone();
                    remote_committed_tx.send(()).unwrap();
                    ack_rx.await.unwrap();
                    Ok(candidate)
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

        let changed = update_model_taints_and_persist(
            &metadata,
            id,
            HashSet::from(["old".to_string()]),
            move |candidate| async move {
                *task_persisted.write().await = Some(candidate.clone());
                Ok(candidate)
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
}
