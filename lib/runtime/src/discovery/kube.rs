// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod daemon;
mod utils;

pub use utils::hash_pod_name;

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
use tokio::sync::{OnceCell, RwLock};
use tokio::task::JoinHandle;

/// Kubernetes-based discovery client
#[derive(Clone)]
pub struct KubeDiscoveryClient {
    instance_id: u64,
    metadata: Arc<RwLock<DiscoveryMetadata>>,
    kube_client: KubeClient,
    pod_info: PodInfo,
    cancel_token: CancellationToken,
    daemon_state: Arc<OnceCell<Arc<DaemonHandles>>>,
}

struct DaemonHandles {
    metadata_watch: tokio::sync::watch::Receiver<Arc<MetadataSnapshot>>,
    #[allow(dead_code)]
    daemon_handle: JoinHandle<()>,
}

impl DaemonHandles {
    fn receiver(&self) -> tokio::sync::watch::Receiver<Arc<MetadataSnapshot>> {
        self.metadata_watch.clone()
    }
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
            "Initializing KubeDiscoveryClient: pod_name={}, instance_id={:x}, namespace={}",
            pod_info.pod_name,
            instance_id,
            pod_info.pod_namespace
        );

        let kube_client = KubeClient::try_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Kubernetes client: {}", e))?;

        Ok(Self {
            instance_id,
            metadata,
            kube_client,
            pod_info,
            cancel_token,
            daemon_state: Arc::new(OnceCell::new()),
        })
    }

    async fn metadata_watch(&self) -> Result<tokio::sync::watch::Receiver<Arc<MetadataSnapshot>>> {
        let handles = self
            .daemon_state
            .get_or_try_init(|| {
                let instance_id = self.instance_id;
                let kube_client = self.kube_client.clone();
                let pod_info = self.pod_info.clone();
                let cancel_token = self.cancel_token.clone();

                async move {
                    let (watch_tx, watch_rx) =
                        tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
                    let daemon = DiscoveryDaemon::new(kube_client, pod_info, cancel_token)?;

                    let daemon_handle = tokio::spawn(async move {
                        if let Err(e) = daemon.run(watch_tx).await {
                            tracing::error!("Discovery daemon failed: {}", e);
                        }
                    });

                    tracing::info!(
                        "Discovery daemon started lazily for instance_id={:x}",
                        instance_id
                    );

                    Ok::<Arc<DaemonHandles>, anyhow::Error>(Arc::new(DaemonHandles {
                        metadata_watch: watch_rx,
                        daemon_handle,
                    }))
                }
            })
            .await?;

        Ok(handles.receiver())
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

        // Write to local metadata
        let mut metadata = self.metadata.write().await;
        match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Registered endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
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
                    "Registered model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    namespace,
                    component,
                    endpoint,
                    instance_id
                );
                metadata.register_model_card(instance.clone())?;
            }
        }

        Ok(instance)
    }

    async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
        // TODO: need to handle meta data change propagation to other pods
        // Current implementation delete the entry from local metadata but
        // it doesn't invalidate the cached service metadata on other pods
        let mut metadata = self.metadata.write().await;
        match &instance {
            DiscoveryInstance::Endpoint(_inst) => {
                metadata.unregister_endpoint(&instance)?;
            }
            DiscoveryInstance::Model { .. } => {
                metadata.unregister_model_card(&instance)?;
            }
        }

        Ok(())
    }

    async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        tracing::debug!("KubeDiscoveryClient::list called with query={:?}", query);

        // Ensure the daemon is running before accessing the snapshot
        let mut metadata_watch = self.metadata_watch().await?;

        // Check if we need to wait for initial snapshot (avoid holding borrow across await)
        let needs_wait = {
            let snapshot = metadata_watch.borrow();
            snapshot.sequence == 0 && snapshot.instances.is_empty()
        };

        // Wait for daemon to fetch at least one snapshot if this is the initial empty state
        if needs_wait {
            tracing::debug!("Waiting for initial discovery snapshot...");
            // Wait for first update with a timeout
            tokio::time::timeout(std::time::Duration::from_secs(10), metadata_watch.changed())
                .await
                .map_err(|_| anyhow::anyhow!("Timeout waiting for initial discovery snapshot"))?
                .map_err(|_| {
                    anyhow::anyhow!("Discovery daemon stopped before providing snapshot")
                })?;
        }

        // Get current snapshot
        let snapshot = metadata_watch.borrow().clone();

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

        // Clone the watch receiver (starts daemon lazily on demand)
        let mut watch_rx = self.metadata_watch().await?;

        // Create output stream
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        // Generate unique stream identifier for tracing
        let stream_id = uuid::Uuid::new_v4();

        // Spawn task to process snapshots
        tokio::spawn(async move {
            let mut known_instances = HashSet::<u64>::new();

            tracing::debug!(
                stream_id = %stream_id,
                "Watch started for query={:?}",
                query
            );

            loop {
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

                        // Compute diff
                        let added: Vec<u64> = current_instances
                            .difference(&known_instances)
                            .copied()
                            .collect();

                        let removed: Vec<u64> = known_instances
                            .difference(&current_instances)
                            .copied()
                            .collect();

                        // Only log if there are changes
                        if !added.is_empty() || !removed.is_empty() {
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
