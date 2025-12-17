// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CancellationToken;
use crate::discovery::{DiscoveryMetadata, MetadataSnapshot};
use anyhow::Result;
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    Api, Client as KubeClient,
    runtime::{WatchStreamExt, reflector, watcher, watcher::Config},
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};

use super::crd::DynamoWorkerMetadata;
use super::utils::{PodInfo, extract_endpoint_info};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

/// Discovers and aggregates metadata from DynamoWorkerMetadata CRs in the cluster
#[derive(Clone)]
pub(super) struct DiscoveryDaemon {
    /// Kubernetes client
    kube_client: KubeClient,
    /// This pod's info
    pod_info: PodInfo,
    /// Cancellation token for shutdown
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
    /// Watches both EndpointSlices (to know which pods are ready) and
    /// DynamoWorkerMetadata CRs (to get the metadata for each pod).
    /// A pod is included in the snapshot only if:
    /// 1. It appears as ready in an EndpointSlice
    /// 2. It has a corresponding DynamoWorkerMetadata CR
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        // Create notify for watch-driven updates (shared by both reflectors)
        let notify = Arc::new(Notify::new());

        // --- EndpointSlice Reflector ---
        let endpoint_slices: Api<EndpointSlice> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        let (ep_reader, ep_writer) = reflector::store();

        let ep_watch_config = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        tracing::info!(
            "Daemon watching EndpointSlices with labels: nvidia.com/dynamo-discovery-backend=kubernetes, nvidia.com/dynamo-discovery-enabled=true"
        );

        let notify_ep = notify.clone();
        let ep_reflector_stream = reflector(ep_writer, watcher(endpoint_slices, ep_watch_config))
            .default_backoff()
            .touched_objects()
            .for_each(move |res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            slice_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "EndpointSlice reflector updated"
                        );
                        notify_ep.notify_one();
                    }
                    Err(e) => {
                        tracing::warn!("EndpointSlice reflector error: {}", e);
                        notify_ep.notify_one();
                    }
                }
                futures::future::ready(())
            });

        tokio::spawn(ep_reflector_stream);

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
                        tracing::warn!("DynamoWorkerMetadata CR reflector error: {}", e);
                        notify_cr.notify_one();
                    }
                }
                futures::future::ready(())
            });

        tokio::spawn(cr_reflector_stream);

        // Event-driven loop with debouncing
        let mut sequence = 0u64;
        let mut prev_instance_ids: HashSet<u64> = HashSet::new();

        loop {
            tokio::select! {
                // Wait for notification from either reflector
                _ = notify.notified() => {
                    // Debounce: wait fixed duration to batch rapid events
                    tokio::time::sleep(DEBOUNCE_DURATION).await;

                    // Drain any permit that accumulated during the sleep
                    let _ = timeout(Duration::ZERO, notify.notified()).await;

                    tracing::trace!("Debounce window elapsed, processing snapshot");

                    match self.aggregate_snapshot(&ep_reader, &cr_reader, sequence).await {
                        Ok(snapshot) => {
                            // Compare instance IDs to detect changes
                            let current_instance_ids: HashSet<u64> =
                                snapshot.instances.keys().copied().collect();

                            let instances_changed = current_instance_ids != prev_instance_ids;

                            if instances_changed {
                                // Compute what was added and removed
                                let added: Vec<u64> = current_instance_ids
                                    .difference(&prev_instance_ids)
                                    .copied()
                                    .collect();

                                let removed: Vec<u64> = prev_instance_ids
                                    .difference(&current_instance_ids)
                                    .copied()
                                    .collect();

                                tracing::info!(
                                    "Daemon snapshot (seq={}): instances changed, total={}, added=[{}], removed=[{}]",
                                    sequence,
                                    current_instance_ids.len(),
                                    added.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>().join(", "),
                                    removed.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>().join(", ")
                                );

                                // Broadcast the snapshot (only when changed)
                                if watch_tx.send(Arc::new(snapshot)).is_err() {
                                    tracing::debug!("No watch subscribers, daemon stopping");
                                    break;
                                }

                                prev_instance_ids = current_instance_ids;
                            } else {
                                tracing::trace!(
                                    "Daemon snapshot (seq={}): no changes, {} instances",
                                    sequence,
                                    current_instance_ids.len()
                                );
                            }

                            sequence += 1;
                        }
                        Err(e) => {
                            tracing::error!("Failed to aggregate snapshot: {}", e);
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

    /// Aggregate metadata from EndpointSlices and DynamoWorkerMetadata CRs into a snapshot
    ///
    /// A pod is included in the snapshot only if:
    /// 1. It appears as ready in an EndpointSlice
    /// 2. It has a corresponding DynamoWorkerMetadata CR (CR name = pod name)
    async fn aggregate_snapshot(
        &self,
        ep_reader: &reflector::Store<EndpointSlice>,
        cr_reader: &reflector::Store<DynamoWorkerMetadata>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        let start = std::time::Instant::now();

        // Extract ready pods from EndpointSlices: (instance_id, pod_name, pod_ip, system_port)
        let ready_pods: Vec<(u64, String, String, u16)> = ep_reader
            .state()
            .iter()
            .flat_map(|arc_slice| extract_endpoint_info(arc_slice.as_ref()))
            .collect();

        tracing::trace!(
            "Daemon found {} ready pods from EndpointSlices",
            ready_pods.len()
        );

        // Build a map of CR name -> metadata for quick lookup
        let cr_map: HashMap<String, Arc<DiscoveryMetadata>> = cr_reader
            .state()
            .iter()
            .filter_map(|arc_cr| {
                let cr_name = arc_cr.metadata.name.as_ref()?;
                
                // Deserialize the data field to DiscoveryMetadata
                match serde_json::from_value::<DiscoveryMetadata>(arc_cr.spec.data.clone()) {
                    Ok(metadata) => {
                        tracing::trace!("Loaded metadata from CR '{}'", cr_name);
                        Some((cr_name.clone(), Arc::new(metadata)))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to deserialize metadata from CR '{}': {}",
                            cr_name,
                            e
                        );
                        None
                    }
                }
            })
            .collect();

        tracing::trace!(
            "Daemon loaded {} DynamoWorkerMetadata CRs",
            cr_map.len()
        );

        // Correlate: ready pod + CR exists = include in snapshot
        let mut instances: HashMap<u64, Arc<DiscoveryMetadata>> = HashMap::new();

        for (instance_id, pod_name, _pod_ip, _system_port) in ready_pods {
            // CR name is the pod name
            if let Some(metadata) = cr_map.get(&pod_name) {
                instances.insert(instance_id, metadata.clone());
                tracing::trace!(
                    "Included pod '{}' (instance_id={:x}) in snapshot",
                    pod_name,
                    instance_id
                );
            } else {
                tracing::trace!(
                    "Skipping pod '{}' (instance_id={:x}): no DynamoWorkerMetadata CR found",
                    pod_name,
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

        Ok(MetadataSnapshot {
            instances,
            sequence,
            timestamp: std::time::Instant::now(),
        })
    }
}
