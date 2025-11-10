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
use tokio::sync::RwLock;

use super::utils::{PodInfo, extract_endpoint_info, hash_pod_name};

const SNAPSHOT_POLL_INTERVAL_MS: u64 = 5000;
const MAX_CONCURRENT_FETCHES: usize = 20;
const METADATA_FETCH_TIMEOUT_SECS: u64 = 5;

/// Discovers and aggregates metadata from pods in the cluster
#[derive(Clone)]
pub(super) struct DiscoveryDaemon {
    /// Kubernetes client
    kube_client: KubeClient,
    /// HTTP client for fetching remote metadata
    http_client: reqwest::Client,
    /// Cache of remote pod metadata (instance_id -> metadata)
    cache: Arc<RwLock<HashMap<u64, Arc<DiscoveryMetadata>>>>,
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
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(METADATA_FETCH_TIMEOUT_SECS))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            kube_client,
            http_client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            pod_info,
            cancel_token,
        })
    }

    /// Run the discovery daemon
    pub async fn run(
        self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        tracing::info!("Discovery daemon starting");

        // Create reflector for ALL EndpointSlices in our namespace
        let endpoint_slices: Api<EndpointSlice> =
            Api::namespaced(self.kube_client.clone(), &self.pod_info.pod_namespace);

        let (reader, writer) = reflector::store();

        // Apply label selector to only watch discovery-enabled EndpointSlices
        let watch_config =
            Config::default().labels("nvidia.com/dynamo-discovery-backend=kubernetes");

        tracing::info!(
            "Daemon watching EndpointSlices with label: nvidia.com/dynamo-discovery-backend=kubernetes"
        );

        // Spawn reflector task (runs independently)
        let reflector_stream = reflector(writer, watcher(endpoint_slices, watch_config))
            .default_backoff()
            .touched_objects()
            .for_each(|res| {
                match res {
                    Ok(obj) => {
                        tracing::debug!(
                            slice_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "Daemon reflector updated EndpointSlice"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Daemon reflector error: {}", e);
                    }
                }
                futures::future::ready(())
            });

        tokio::spawn(reflector_stream);

        // Polling loop
        let mut sequence = 0u64;
        let mut prev_instance_ids: HashSet<u64> = HashSet::new();
        let mut interval =
            tokio::time::interval(std::time::Duration::from_millis(SNAPSHOT_POLL_INTERVAL_MS));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match self.aggregate_snapshot(&reader, sequence).await {
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

                                // Prune cache for removed instances
                                if !removed.is_empty() {
                                    self.prune_cache(&removed).await;
                                }

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

    /// Aggregate metadata from all pods into a snapshot
    async fn aggregate_snapshot(
        &self,
        reader: &reflector::Store<EndpointSlice>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        let start = std::time::Instant::now();

        // Extract ALL ready endpoints (instance_id, pod_name, pod_ip) directly from reflector
        let all_endpoints: Vec<(u64, String, String)> = reader
            .state()
            .iter()
            .flat_map(|arc_slice| extract_endpoint_info(arc_slice.as_ref()))
            .collect();

        tracing::trace!(
            "Daemon found {} ready endpoints to fetch",
            all_endpoints.len()
        );

        // Concurrent fetch: Fetch metadata for all endpoints in parallel
        let fetch_futures = all_endpoints
            .into_iter()
            .map(|(instance_id, pod_name, pod_ip)| {
                let daemon = self.clone();
                async move {
                    match daemon.fetch_metadata(&pod_name, &pod_ip).await {
                        Ok(metadata) => Some((instance_id, metadata)),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to fetch metadata for pod {} (instance_id={:x}): {}",
                                pod_name,
                                instance_id,
                                e
                            );
                            None
                        }
                    }
                }
            });

        // Execute fetches concurrently with bounded parallelism
        let results: Vec<_> = futures::stream::iter(fetch_futures)
            .buffer_unordered(MAX_CONCURRENT_FETCHES)
            .collect()
            .await;

        // Build the snapshot
        let instances: HashMap<u64, Arc<DiscoveryMetadata>> =
            results.into_iter().flatten().collect();

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

    /// Fetch metadata for a single pod (with caching)
    async fn fetch_metadata(&self, pod_name: &str, pod_ip: &str) -> Result<Arc<DiscoveryMetadata>> {
        let instance_id = hash_pod_name(pod_name);

        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&instance_id) {
                tracing::trace!(
                    "Cache hit for pod_name={}, instance_id={:x}",
                    pod_name,
                    instance_id
                );
                return Ok(cached.clone());
            }
        }

        // Cache miss: fetch from HTTP
        let url = format!("http://{}:{}/metadata", pod_ip, self.pod_info.system_port);

        tracing::debug!("Fetching metadata from {url}");

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch metadata from {}: {}", url, e))?;

        let metadata: DiscoveryMetadata = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse metadata from {}: {}", url, e))?;

        let metadata = Arc::new(metadata);

        // Cache it
        {
            let mut cache = self.cache.write().await;
            // Check again in case another task inserted while we were fetching
            if let Some(existing) = cache.get(&instance_id) {
                tracing::debug!(
                    "Another task cached metadata for instance_id={:x} while we were fetching",
                    instance_id
                );
                return Ok(existing.clone());
            }

            cache.insert(instance_id, metadata.clone());

            tracing::debug!(
                "Cached metadata for pod_name={}, instance_id={:x}",
                pod_name,
                instance_id
            );
        }

        Ok(metadata)
    }

    /// Prune cache entries for removed instances
    async fn prune_cache(&self, removed_ids: &[u64]) {
        let mut cache = self.cache.write().await;
        for id in removed_ids {
            if cache.remove(id).is_some() {
                tracing::debug!("Pruned cache for removed instance_id={:x}", id);
            }
        }
    }
}
