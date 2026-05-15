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

#[derive(Clone)]
struct CachedCrMetadata {
    metadata: Arc<DiscoveryMetadata>,
    generation: i64,
    uid: Option<String>,
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
    /// included in the snapshot only if it appears ready AND has valid current
    /// or cached metadata from a matching CR.
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
        // Keeps transient invalid CR updates from looking like removals.
        let mut valid_cr_cache: HashMap<String, CachedCrMetadata> = HashMap::new();

        loop {
            tokio::select! {
                _ = notify.notified() => {
                    tokio::time::sleep(DEBOUNCE_DURATION).await;
                    let _ = timeout(Duration::ZERO, notify.notified()).await;

                    tracing::trace!("Debounce window elapsed, processing snapshot");

                    match self
                        .aggregate_snapshot(&source, &cr_reader, &mut valid_cr_cache, sequence)
                        .await
                    {
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
        valid_cr_cache: &mut HashMap<String, CachedCrMetadata>,
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

            match serde_json::from_value::<DiscoveryMetadata>(arc_cr.spec.data.clone()) {
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

        for (instance_id, cr_key) in ready_entries {
            if let Some(cached) = cr_map.get(&cr_key) {
                instances.insert(instance_id, cached.metadata.clone());
                generations.insert(instance_id, cached.generation);
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

        Ok(MetadataSnapshot {
            instances,
            generations,
            sequence,
            timestamp: std::time::Instant::now(),
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
    use super::*;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ManagedFieldsEntry;

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
