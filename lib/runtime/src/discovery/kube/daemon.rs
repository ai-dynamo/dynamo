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
use tokio::time::{Duration, timeout};

use super::crd::DynamoWorkerMetadata;
use super::utils::{
    KubeDiscoveryGranularity, PodInfo, extract_endpoint_info, extract_ready_containers,
};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

/// Holds the readiness reflector store, abstracting over the K8s resource type.
/// Both variants produce `Vec<(instance_id, cr_correlation_key)>` via `ready_entries()`.
enum ReadinessReader {
    /// EndpointSlice-based: one entry per ready pod (pod granularity)
    EndpointSlice(reflector::Store<EndpointSlice>),
    /// Pod-based: one entry per ready container (container granularity)
    Pod(reflector::Store<Pod>),
}

impl ReadinessReader {
    /// Extract `(instance_id, cr_key)` tuples from the current reflector state.
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
    /// Watches a readiness source (EndpointSlices or Pods depending on granularity)
    /// and DynamoWorkerMetadata CRs. An entry is included in the snapshot only if:
    /// 1. It appears as ready in the readiness source
    /// 2. It has a corresponding DynamoWorkerMetadata CR (matched by CR name)
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        // Create notify for watch-driven updates (shared by all reflectors)
        let notify = Arc::new(Notify::new());

        // --- Readiness reflector: EndpointSlice or Pod depending on granularity ---
        let readiness_reader = match self.pod_info.granularity {
            KubeDiscoveryGranularity::Pod => {
                let api: Api<EndpointSlice> =
                    Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                let config = Config::default()
                    .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
                    .labels("nvidia.com/dynamo-discovery-enabled=true");

                tracing::info!(
                    "Daemon watching EndpointSlices (pod granularity) with labels: nvidia.com/dynamo-discovery-backend=kubernetes, nvidia.com/dynamo-discovery-enabled=true"
                );

                let notify_clone = notify.clone();
                let stream = reflector(writer, watcher(api, config))
                    .default_backoff()
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "EndpointSlice reflector updated"
                                );
                                notify_clone.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("EndpointSlice reflector error: {e}");
                                notify_clone.notify_one();
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                ReadinessReader::EndpointSlice(reader)
            }

            KubeDiscoveryGranularity::Container => {
                let api: Api<Pod> =
                    Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);
                let (reader, writer) = reflector::store();
                let config = Config::default()
                    .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
                    .labels("nvidia.com/dynamo-discovery-enabled=true");

                tracing::info!(
                    "Daemon watching Pods (container granularity) with labels: nvidia.com/dynamo-discovery-backend=kubernetes, nvidia.com/dynamo-discovery-enabled=true"
                );

                let notify_clone = notify.clone();
                let stream = reflector(writer, watcher(api, config))
                    .default_backoff()
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "Pod reflector updated"
                                );
                                notify_clone.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("Pod reflector error: {e}");
                                notify_clone.notify_one();
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                ReadinessReader::Pod(reader)
            }
        };

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

                    match self.aggregate_snapshot(&readiness_reader, &cr_reader, sequence).await {
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

    /// Aggregate metadata from the readiness source and DynamoWorkerMetadata CRs into a snapshot.
    ///
    /// An entry is included only if it appears ready AND has a matching CR (by name).
    async fn aggregate_snapshot(
        &self,
        readiness: &ReadinessReader,
        cr_reader: &reflector::Store<DynamoWorkerMetadata>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        let start = std::time::Instant::now();

        // Extract ready entries: (instance_id, cr_key)
        let ready_entries = readiness.ready_entries();

        tracing::trace!(
            "Daemon found {} ready entries (granularity={:?})",
            ready_entries.len(),
            self.pod_info.granularity,
        );

        // Single read of CR state to extract metadata and generations atomically
        let cr_state = cr_reader.state();
        let mut cr_map: HashMap<String, (Arc<DiscoveryMetadata>, i64)> = HashMap::new();

        for arc_cr in cr_state.iter() {
            let Some(cr_name) = arc_cr.metadata.name.as_ref() else {
                continue;
            };

            let generation = arc_cr.metadata.generation.unwrap_or(0);

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

        // Correlate: ready entry + CR exists = include in snapshot
        let mut instances: HashMap<u64, Arc<DiscoveryMetadata>> = HashMap::new();
        let mut generations: HashMap<u64, i64> = HashMap::new();

        for (instance_id, cr_key) in ready_entries {
            if let Some((metadata, generation)) = cr_map.get(&cr_key) {
                instances.insert(instance_id, metadata.clone());
                generations.insert(instance_id, *generation);
                tracing::trace!(
                    "Included '{}' (instance_id={:x}, generation={}) in snapshot",
                    cr_key,
                    instance_id,
                    generation
                );
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

        Ok(MetadataSnapshot {
            instances,
            generations,
            sequence,
            timestamp: std::time::Instant::now(),
        })
    }
}
