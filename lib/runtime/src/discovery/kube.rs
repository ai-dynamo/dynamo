// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod crd;
mod daemon;
mod utils;

pub use crd::{DynamoWorkerMetadata, DynamoWorkerMetadataSpec};
pub use utils::hash_pod_name;

use crd::{apply_cr, build_cr};
use daemon::DiscoveryDaemon;
use utils::PodInfo;

use crate::CancellationToken;
use crate::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryMetadata, DiscoveryQuery, DiscoverySpec,
    DiscoveryStream, MetadataSnapshot,
};
use anyhow::Result;
use async_trait::async_trait;
use kube::Client as KubeClient;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Kubernetes-based discovery client
#[derive(Clone)]
pub struct KubeDiscoveryClient {
    instance_id: u64,
    metadata: Arc<RwLock<DiscoveryMetadata>>,
    metadata_watch: tokio::sync::watch::Receiver<Arc<MetadataSnapshot>>,
    kube_client: KubeClient,
    pod_info: PodInfo,
}

impl KubeDiscoveryClient {
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
        let instance_id = hash_pod_name(&pod_info.pod_name);

        tracing::info!(
            "Initializing KubeDiscoveryClient: pod_name={}, instance_id={:x}, namespace={}, pod_uid={}",
            pod_info.pod_name,
            instance_id,
            pod_info.pod_namespace,
            pod_info.pod_uid
        );

        let kube_client = KubeClient::try_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Kubernetes client: {}", e))?;

        // Create watch channel with initial empty snapshot
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));

        // Create and spawn daemon
        let daemon = DiscoveryDaemon::new(kube_client.clone(), pod_info.clone(), cancel_token)?;

        tokio::spawn(async move {
            if let Err(e) = daemon.run(watch_tx).await {
                tracing::error!("Discovery daemon failed: {}", e);
            }
        });

        tracing::info!("Discovery daemon started");

        Ok(Self {
            instance_id,
            metadata,
            metadata_watch: watch_rx,
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

    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance_id = self.instance_id();
        let instance = spec.with_instance_id(instance_id);

        tracing::debug!(
            "Registering instance: {:?} with instance_id={:x}",
            instance,
            instance_id
        );

        // Write to local metadata and persist to CR
        // IMPORTANT: Hold the write lock across the CR write to prevent race conditions
        let mut metadata = self.metadata.write().await;

        // Clone state for rollback in case CR persistence fails
        let original_state = metadata.clone();

        match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Registering endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    instance_id
                );
                metadata.register_endpoint(instance.clone())?;
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
                metadata.register_model_card(instance.clone())?;
            }
        }

        // Build and apply the CR with the updated metadata
        // This persists the metadata to Kubernetes for other pods to discover
        let cr = build_cr(&self.pod_info.pod_name, &self.pod_info.pod_uid, &metadata)?;

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

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        let instance_id = self.instance_id();

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
        }

        // Build and apply the CR with the updated metadata
        // This persists the removal to Kubernetes for other pods to see
        let cr = build_cr(&self.pod_info.pod_name, &self.pod_info.pod_uid, &metadata)?;

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

        // Get current snapshot (may be empty if daemon hasn't fetched yet)
        let snapshot = self.metadata_watch.borrow().clone();

        tracing::debug!(
            "List using snapshot seq={} with {} instances",
            snapshot.sequence,
            snapshot.instances.len()
        );

        // Filter snapshot by query
        let instances = snapshot.filter(&query);

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
        use tokio::sync::mpsc;

        tracing::info!(
            "KubeDiscoveryClient::list_and_watch started for query={:?}",
            query
        );

        // Clone the watch receiver
        let mut watch_rx = self.metadata_watch.clone();

        // Create output stream
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        // Generate unique stream identifier for tracing
        let stream_id = uuid::Uuid::new_v4();

        // Spawn task to process snapshots
        tokio::spawn(async move {
            // Initialize known_instances from current snapshot state
            // This is critical: watch_rx.changed() only fires on FUTURE changes,
            // so we must capture the current state first to detect removals correctly
            let initial_snapshot = watch_rx.borrow_and_update().clone();

            let mut known_instances: HashSet<u64> = initial_snapshot
                .instances
                .iter()
                .filter_map(|(&instance_id, metadata)| {
                    let filtered = metadata.filter(&query);
                    if !filtered.is_empty() {
                        Some(instance_id)
                    } else {
                        None
                    }
                })
                .collect();

            tracing::debug!(
                stream_id = %stream_id,
                initial_instances = known_instances.len(),
                "Watch started for query={:?}",
                query
            );

            // Emit initial Added events for all existing instances (the "list" part of list_and_watch)
            for &instance_id in &known_instances {
                if let Some(metadata) = initial_snapshot.instances.get(&instance_id) {
                    let instances = metadata.filter(&query);
                    for instance in instances {
                        tracing::info!(
                            stream_id = %stream_id,
                            instance_id = format!("{:x}", instance.instance_id()),
                            "Emitting initial Added event"
                        );
                        if event_tx.send(Ok(DiscoveryEvent::Added(instance))).is_err() {
                            tracing::debug!(
                                stream_id = %stream_id,
                                "Watch receiver dropped during initial sync"
                            );
                            return;
                        }
                    }
                }
            }

            loop {
                tracing::trace!(
                    stream_id = %stream_id,
                    known_count = known_instances.len(),
                    "Watch loop waiting for changes"
                );

                // Wait for next snapshot or cancellation
                let watch_result = if let Some(ref token) = cancel_token {
                    tokio::select! {
                        result = watch_rx.changed() => result,
                        _ = token.cancelled() => {
                            tracing::info!(
                                stream_id = %stream_id,
                                "Watch cancelled via cancel token"
                            );
                            break;
                        }
                    }
                } else {
                    watch_rx.changed().await
                };

                match watch_result {
                    Ok(()) => {
                        // Get latest snapshot
                        let snapshot = watch_rx.borrow_and_update().clone();

                        tracing::debug!(
                            stream_id = %stream_id,
                            seq = snapshot.sequence,
                            snapshot_instances = snapshot.instances.len(),
                            known_instances = known_instances.len(),
                            "Watch received snapshot update"
                        );

                        // Filter snapshot by query
                        let current_instances: HashSet<u64> = snapshot
                            .instances
                            .iter()
                            .filter_map(|(&instance_id, metadata)| {
                                let filtered = metadata.filter(&query);
                                if !filtered.is_empty() {
                                    Some(instance_id)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        tracing::trace!(
                            stream_id = %stream_id,
                            current_ids = ?current_instances.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>(),
                            known_ids = ?known_instances.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>(),
                            "Comparing instance sets"
                        );

                        // Compute diff
                        let added: Vec<u64> = current_instances
                            .difference(&known_instances)
                            .copied()
                            .collect();

                        let removed: Vec<u64> = known_instances
                            .difference(&current_instances)
                            .copied()
                            .collect();

                        // Log diff results (even if empty, for debugging)
                        if added.is_empty() && removed.is_empty() {
                            tracing::debug!(
                                stream_id = %stream_id,
                                seq = snapshot.sequence,
                                current_count = current_instances.len(),
                                known_count = known_instances.len(),
                                "Watch snapshot received but no diff detected"
                            );
                        } else {
                            tracing::debug!(
                                stream_id = %stream_id,
                                seq = snapshot.sequence,
                                added = added.len(),
                                removed = removed.len(),
                                total = current_instances.len(),
                                "Watch detected changes"
                            );
                        }

                        // Emit Added events
                        for instance_id in added {
                            if let Some(metadata) = snapshot.instances.get(&instance_id) {
                                let instances = metadata.filter(&query);
                                for instance in instances {
                                    tracing::info!(
                                        stream_id = %stream_id,
                                        instance_id = format!("{:x}", instance.instance_id()),
                                        "Emitting Added event"
                                    );
                                    if event_tx.send(Ok(DiscoveryEvent::Added(instance))).is_err() {
                                        tracing::debug!(
                                            stream_id = %stream_id,
                                            "Watch receiver dropped"
                                        );
                                        return;
                                    }
                                }
                            }
                        }

                        // Emit Removed events
                        for instance_id in removed {
                            tracing::info!(
                                stream_id = %stream_id,
                                instance_id = format!("{:x}", instance_id),
                                "Emitting Removed event"
                            );
                            if event_tx
                                .send(Ok(DiscoveryEvent::Removed(instance_id)))
                                .is_err()
                            {
                                tracing::debug!(stream_id = %stream_id, "Watch receiver dropped");
                                return;
                            }
                        }

                        // Update known set
                        known_instances = current_instances;
                    }
                    Err(_) => {
                        tracing::info!(
                            stream_id = %stream_id,
                            "Watch channel closed (daemon stopped)"
                        );
                        break;
                    }
                }
            }
        });

        // Convert receiver to stream
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(event_rx);
        Ok(Box::pin(stream))
    }
}
