// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{DiscoveryMetadata, MetadataSnapshot};
use anyhow::Result;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    Api, Client as KubeClient,
    runtime::{WatchStreamExt, reflector, watcher, watcher::Config},
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{Duration, timeout};

use super::crd::DynamoWorkerMetadata;
use super::utils::{PodInfo, extract_ready_containers};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

/// Discovers and aggregates metadata from DynamoWorkerMetadata CRs in the cluster
#[derive(Clone)]
pub(super) struct DiscoveryDaemon {
    kube_client: KubeClient,
    // This pod's info
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

    /// Run the discovery daemon
    ///
    /// Watches both Pods (to know which containers are ready) and
    /// DynamoWorkerMetadata CRs (to get the metadata for each container).
    /// A container is included in the snapshot only if:
    /// 1. Its `containerStatuses[].ready` is true
    /// 2. It has a corresponding DynamoWorkerMetadata CR (CR name = `{pod}-{container}`)
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        // Create notify for watch-driven updates (shared by both reflectors)
        let notify = Arc::new(Notify::new());

        // --- Pod Reflector ---
        let pods: Api<Pod> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        let (pod_reader, pod_writer) = reflector::store();

        let pod_watch_config = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        tracing::info!(
            "Daemon watching Pods with labels: nvidia.com/dynamo-discovery-backend=kubernetes, nvidia.com/dynamo-discovery-enabled=true"
        );

        let notify_pod = notify.clone();
        let pod_reflector_stream = reflector(pod_writer, watcher(pods, pod_watch_config))
            .default_backoff()
            .touched_objects()
            .for_each(move |res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            pod_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "Pod reflector updated"
                        );
                        notify_pod.notify_one();
                    }
                    Err(e) => {
                        tracing::warn!("Pod reflector error: {e}");
                        notify_pod.notify_one();
                    }
                }
                // for_each expects a Future; ready(()) is an immediately-complete one
                futures::future::ready(())
            });

        tokio::spawn(pod_reflector_stream);

        // --- DynamoWorkerMetadata CR Reflector ---
        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        let (cr_reader, cr_writer) = reflector::store();

        // Watch all DynamoWorkerMetadata CRs in the namespace
        let cr_watch_config = Config::default();

        tracing::info!(
            "Daemon watching DynamoWorkerMetadata CRs in namespace: {}",
            self.pod_info.pod_namespace
        );

        let notify_cr = notify.clone();
        let cr_reflector_stream = reflector(cr_writer, watcher(metadata_crs, cr_watch_config))
            .default_backoff()
            .touched_objects()
            .for_each(move |res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            cr_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "DynamoWorkerMetadata CR reflector updated"
                        );
                        notify_cr.notify_one();
                    }
                    Err(e) => {
                        tracing::warn!("DynamoWorkerMetadata CR reflector error: {e}");
                        notify_cr.notify_one();
                    }
                }
                // for_each expects a Future; ready(()) is an immediately-complete one
                futures::future::ready(())
            });

        tokio::spawn(cr_reflector_stream);

        // Event-driven loop with debouncing
        let mut sequence = 0u64;
        let mut prev_snapshot = MetadataSnapshot::empty();

        loop {
            tokio::select! {
                _ = notify.notified() => {
                    // Debounce: K8s can emit many events in quick succession
                    // Wait briefly to batch them into a single snapshot update.
                    tokio::time::sleep(DEBOUNCE_DURATION).await;

                    // Drain any permit that accumulated during the sleep
                    let _ = timeout(Duration::ZERO, notify.notified()).await;

                    tracing::trace!("Debounce window elapsed, processing snapshot");

                    match self.aggregate_snapshot(&pod_reader, &cr_reader, sequence).await {
                        Ok(snapshot) => {
                            if snapshot.has_changes_from(&prev_snapshot) {
                                prev_snapshot = snapshot.clone();

                                if watch_tx.send(Arc::new(snapshot)).is_err() {
                                    tracing::debug!("No watch subscribers, daemon stopping");
                                    break;
                                }
                            }

                            sequence += 1;
                        }
                        Err(e) => {
                            tracing::error!("Failed to aggregate snapshot: {e}");
                            // Continue on errors - don't crash daemon
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("Discovery daemon received cancellation");
                    break;
                }
            }
        }

        tracing::info!("Discovery daemon stopped");
        Ok(())
    }

    /// Aggregate metadata from Pods and DynamoWorkerMetadata CRs into a snapshot
    ///
    /// A container is included in the snapshot only if:
    /// 1. Its `containerStatuses[].ready` is true in the Pod status
    /// 2. It has a corresponding DynamoWorkerMetadata CR (CR name = `{pod_name}-{container_name}`)
    async fn aggregate_snapshot(
        &self,
        pod_reader: &reflector::Store<Pod>,
        cr_reader: &reflector::Store<DynamoWorkerMetadata>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        let start = std::time::Instant::now();

        // Extract ready containers from Pods: (instance_id, cr_name)
        let ready_containers: Vec<(u64, String)> = pod_reader
            .state()
            .iter()
            .flat_map(|pod| extract_ready_containers(pod.as_ref()))
            .collect();

        tracing::trace!(
            "Daemon found {} ready containers from Pods",
            ready_containers.len()
        );

        // Single read of CR state to extract metadata and generations atomically
        // We store (metadata, generation) tuples keyed by CR name
        let cr_state = cr_reader.state();
        let mut cr_map: HashMap<String, (Arc<DiscoveryMetadata>, i64)> = HashMap::new();

        for arc_cr in cr_state.iter() {
            let Some(cr_name) = arc_cr.metadata.name.as_ref() else {
                continue;
            };

            let generation = arc_cr.metadata.generation.unwrap_or(0);

            // Deserialize the data field to DiscoveryMetadata
            match serde_json::from_value::<DiscoveryMetadata>(arc_cr.spec.data.clone()) {
                Ok(metadata) => {
                    tracing::trace!("Loaded metadata from CR '{cr_name}'");
                    cr_map.insert(cr_name.clone(), (Arc::new(metadata), generation));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to deserialize metadata from CR '{}': {}",
                        cr_name,
                        e
                    );
                }
            }
        }

        tracing::trace!("Daemon loaded {} DynamoWorkerMetadata CRs", cr_map.len());

        // Correlate: ready container + CR exists = include in snapshot
        // Both instances and generations are keyed by instance_id with matching keys
        let mut instances: HashMap<u64, Arc<DiscoveryMetadata>> = HashMap::new();
        let mut generations: HashMap<u64, i64> = HashMap::new();

        for (inst_id, container_cr_name) in ready_containers {
            if let Some((metadata, generation)) = cr_map.get(&container_cr_name) {
                instances.insert(inst_id, metadata.clone());
                generations.insert(inst_id, *generation);
                tracing::trace!(
                    "Included container '{}' (instance_id={:x}, generation={}) in snapshot",
                    container_cr_name,
                    inst_id,
                    generation
                );
            } else {
                tracing::trace!(
                    "Skipping container '{}' (instance_id={:x}): no DynamoWorkerMetadata CR found",
                    container_cr_name,
                    inst_id
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

        Ok(MetadataSnapshot {
            instances,
            generations,
            sequence,
            timestamp: std::time::Instant::now(),
        })
    }
}
