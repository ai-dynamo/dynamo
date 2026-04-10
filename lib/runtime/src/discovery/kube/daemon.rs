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
use super::utils::{KubeDiscoveryMode, PodInfo, extract_endpoint_info, extract_ready_containers};

const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

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
    async fn new(pod_info: &PodInfo, kube_client: KubeClient, notify: Arc<Notify>) -> Self {
        let labels = Config::default()
            .labels("nvidia.com/dynamo-discovery-backend=kubernetes")
            .labels("nvidia.com/dynamo-discovery-enabled=true");

        match pod_info.mode {
            KubeDiscoveryMode::Pod => {
                let api: Api<EndpointSlice> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();

                tracing::info!("Daemon watching EndpointSlices (pod mode)");

                let stream = reflector(writer, watcher(api, labels))
                    .default_backoff()
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "EndpointSlice reflector updated"
                                );
                                notify.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("EndpointSlice reflector error: {e}");
                                notify.notify_one();
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                Self::EndpointSlice(reader)
            }

            KubeDiscoveryMode::Container => {
                let api: Api<Pod> = Api::namespaced(kube_client, &pod_info.pod_namespace);
                let (reader, writer) = reflector::store();

                tracing::info!("Daemon watching Pods (container mode)");

                let stream = reflector(writer, watcher(api, labels))
                    .default_backoff()
                    .touched_objects()
                    .for_each(move |res| {
                        match res {
                            Ok(obj) => {
                                tracing::debug!(
                                    name = obj.metadata.name.as_deref().unwrap_or("?"),
                                    "Pod reflector updated"
                                );
                                notify.notify_one();
                            }
                            Err(e) => {
                                tracing::warn!("Pod reflector error: {e}");
                                notify.notify_one();
                            }
                        }
                        futures::future::ready(())
                    });
                tokio::spawn(stream);

                Self::Pod(reader)
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
    /// included in the snapshot only if it appears ready AND has a matching CR.
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        let notify = Arc::new(Notify::new());

        // Readiness source — EndpointSlice or Pod depending on mode
        let source =
            DiscoverySource::new(&self.pod_info, self.kube_client.clone(), notify.clone()).await;

        // DynamoWorkerMetadata CR reflector
        let metadata_crs: Api<DynamoWorkerMetadata> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        let (cr_reader, cr_writer) = reflector::store();
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
                    tokio::time::sleep(DEBOUNCE_DURATION).await;
                    let _ = timeout(Duration::ZERO, notify.notified()).await;

                    tracing::trace!("Debounce window elapsed, processing snapshot");

                    match self.aggregate_snapshot(&source, &cr_reader, sequence).await {
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

    async fn aggregate_snapshot(
        &self,
        source: &DiscoverySource,
        cr_reader: &reflector::Store<DynamoWorkerMetadata>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        let start = std::time::Instant::now();

        let ready_entries = source.ready_entries();

        tracing::trace!(
            "Daemon found {} ready entries (mode={:?})",
            ready_entries.len(),
            self.pod_info.mode,
        );

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
