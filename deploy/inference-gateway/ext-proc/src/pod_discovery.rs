// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pod discovery for standalone (raw-vLLM) mode, driven by the `InferencePool`.
//!
//! The pod label selector and HTTP target port come from the GAIE
//! [`InferencePool`](crate::inference_pool) this EPP backs — the same object the
//! gateway routes to — so EPP and gateway can never disagree about pool
//! membership. To pick up live selector/target-port edits without restarting any
//! watch, the reflector watches **all** pods in the namespace and filters them
//! in memory against the current [`PoolState`]; the pool watch just swaps the
//! filter.
//!
//! Pods are `Ready`-filtered (and excluded once terminating), so in-flight
//! rollouts and crash-looping pods receive no traffic.
//!
//! `worker_id = hash_pod_name(pod_name)`, so the IDs produced here line up with
//! whatever consumes them (the topology adapter and selector catalog).

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::discovery::hash_pod_name;
use k8s_openapi::api::core::v1::Pod;
use tokio::sync::watch;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::inference_pool::{PoolState, spawn_pool_watch};

/// A discovered, `Ready` raw vLLM worker normalized for selector registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawWorker {
    /// Stable hash of the pod name; the selector catalog key.
    pub worker_id: u64,
    /// Kubernetes pod name.
    pub pod_name: String,
    /// Pod IP.
    pub pod_ip: String,
    /// OpenAI HTTP inference endpoint, `http://<ip>:<target_port>`.
    pub http_endpoint: String,
    /// vLLM KV-event ZMQ PUB endpoint, `tcp://<ip>:<kv_event_port>`.
    pub kv_events_endpoint: String,
    /// Optional ZMQ REQ endpoint for live-stream gap replay.
    pub replay_endpoint: Option<String>,
    /// Routing identity fed to the selector's `stable_routing_id`. Currently the
    /// pod name, so it is NOT yet restart-stable: a Deployment pod restart yields
    /// a new name and a new identity. A truly stable source (StatefulSet ordinal
    /// or a pod label) is a follow-up.
    pub stable_routing_id: String,
}

/// Lock-free view over the `Ready` raw vLLM pods selected by the EPP's
/// `InferencePool`. Reads never touch the Kubernetes API.
#[derive(Clone)]
pub struct PodDiscovery {
    store: kube::runtime::reflector::Store<Pod>,
    pool_rx: watch::Receiver<Option<PoolState>>,
    kv_event_port: u16,
    replay_port: Option<u16>,
    changes: watch::Receiver<u64>,
}

impl PodDiscovery {
    /// Start the InferencePool watch and a namespace-wide pod reflector. Returns
    /// a readiness flag that flips once the initial pod LIST has populated the
    /// store. Pods become selectable only once the pool has also resolved.
    pub async fn spawn(cfg: &EppStandaloneConfig) -> Result<(Self, Arc<AtomicBool>)> {
        use futures::StreamExt;
        use kube::{Api, Client, runtime::reflector, runtime::watcher};

        let client = Client::try_default().await?;
        let namespace = cfg.namespace.clone();

        let (pool_rx, _pool_task) = spawn_pool_watch(
            client.clone(),
            namespace.clone(),
            cfg.inference_pool_name.clone(),
        )
        .await?;

        // Namespace-wide pod watch; membership is decided in memory by the pool
        // selector so selector edits never require re-spawning this watch.
        let pods: Api<Pod> = Api::namespaced(client, &namespace);
        let writer = reflector::store::Writer::default();
        let store = writer.as_reader();
        let ready = Arc::new(AtomicBool::new(false));
        let reflect = reflector::reflector(writer, watcher(pods, watcher::Config::default()));

        let (changes_tx, changes_rx) = watch::channel(0u64);

        tracing::info!(
            namespace = %namespace,
            pool = %cfg.inference_pool_name,
            kv_event_port = cfg.kv_event_port,
            "Starting namespace pod reflector for standalone mode"
        );

        // Pod reflector stream -> bump the change generation.
        let changes_for_pods = changes_tx.clone();
        tokio::spawn(async move {
            tokio::pin!(reflect);
            let mut generation = 0u64;
            while reflect.next().await.is_some() {
                generation = generation.wrapping_add(1);
                let _ = changes_for_pods.send(generation);
            }
            tracing::warn!("Raw-vLLM pod reflector stream ended unexpectedly");
        });

        // Pool changes also drive reconciliation (membership/target port may move).
        let mut pool_rx_for_changes = pool_rx.clone();
        tokio::spawn(async move {
            let mut generation = u64::MAX / 2; // distinct space from pod bumps
            while pool_rx_for_changes.changed().await.is_ok() {
                generation = generation.wrapping_add(1);
                let _ = changes_tx.send(generation);
            }
        });

        let store_for_wait = store.clone();
        let ready_for_wait = ready.clone();
        match tokio::time::timeout(Duration::from_secs(30), store_for_wait.wait_until_ready()).await
        {
            Ok(Ok(())) => {
                ready_for_wait.store(true, Ordering::Release);
                tracing::info!("Pod reflector initial LIST sync complete");
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Pod reflector writer dropped before initial LIST");
            }
            Err(_) => {
                tracing::warn!(
                    "Pod reflector initial LIST timed out after 30s; ready in background"
                );
                let store_bg = store.clone();
                let ready_bg = ready.clone();
                tokio::spawn(async move {
                    if store_bg.wait_until_ready().await.is_ok() {
                        ready_bg.store(true, Ordering::Release);
                        tracing::info!("Pod reflector became ready after startup timeout");
                    }
                });
            }
        }

        Ok((
            Self {
                store,
                pool_rx,
                kv_event_port: cfg.kv_event_port,
                replay_port: cfg.replay_port,
                changes: changes_rx,
            },
            ready,
        ))
    }

    /// All currently `Ready` workers selected by the pool, normalized for
    /// selector registration. Empty until the `InferencePool` has resolved.
    pub fn ready_workers(&self) -> Vec<RawWorker> {
        let pool = self.pool_rx.borrow().clone();
        let Some(pool) = pool else {
            return Vec::new();
        };
        self.store
            .state()
            .iter()
            .filter_map(|pod| raw_worker_from_pod(pod, &pool, self.kv_event_port, self.replay_port))
            .collect()
    }

    /// Worker IDs of all currently `Ready`, pool-selected workers.
    pub fn ready_worker_ids(&self) -> HashSet<u64> {
        self.ready_workers()
            .into_iter()
            .map(|w| w.worker_id)
            .collect()
    }

    /// Resolve a `worker_id` to its current `ip:port` HTTP endpoint, if the pod
    /// is still `Ready` and pool-selected. Short-circuits on the first match
    /// instead of building the full worker list.
    pub fn resolve_endpoint(&self, worker_id: u64) -> Option<String> {
        let pool = self.pool_rx.borrow().clone()?;
        self.store.state().iter().find_map(|pod| {
            let worker = raw_worker_from_pod(pod, &pool, self.kv_event_port, self.replay_port)?;
            (worker.worker_id == worker_id).then(|| strip_scheme(&worker.http_endpoint).to_string())
        })
    }

    /// Subscribe to change notifications (a generation counter) bumped on pod or
    /// pool changes, so a reconciler can re-sync.
    pub fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes.clone()
    }
}

fn strip_scheme(endpoint: &str) -> &str {
    endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint)
}

/// Return `true` iff the pod is `Ready` and not terminating. Mirrors llm-d's
/// `IsPodReady`: a pod with a deletion timestamp is excluded even if it still
/// reports `Ready=True`, so draining pods stop receiving traffic promptly.
fn pod_is_ready(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }
    pod.status
        .as_ref()
        .and_then(|s| s.conditions.as_ref())
        .map(|conds| {
            conds
                .iter()
                .any(|c| c.type_ == "Ready" && c.status == "True")
        })
        .unwrap_or(false)
}

/// Return `true` iff the pod carries every `match_labels` key with the equal
/// value (equality-based selector, matching `InferencePool.spec.selector`).
fn pod_matches(pod: &Pod, match_labels: &BTreeMap<String, String>) -> bool {
    let Some(labels) = pod.metadata.labels.as_ref() else {
        return match_labels.is_empty();
    };
    match_labels
        .iter()
        .all(|(k, v)| labels.get(k).map(|pv| pv == v).unwrap_or(false))
}

/// Build a [`RawWorker`] from a pod, or `None` if it is not `Ready`, not
/// pool-selected, or lacks an IP/name. Pure function — unit-testable.
fn raw_worker_from_pod(
    pod: &Pod,
    pool: &PoolState,
    kv_event_port: u16,
    replay_port: Option<u16>,
) -> Option<RawWorker> {
    if !pod_is_ready(pod) || !pod_matches(pod, &pool.match_labels) {
        return None;
    }
    let pod_name = pod.metadata.name.as_deref()?;
    let pod_ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    if pod_ip.is_empty() {
        return None;
    }

    Some(RawWorker {
        worker_id: hash_pod_name(pod_name),
        pod_name: pod_name.to_string(),
        pod_ip: pod_ip.to_string(),
        http_endpoint: format!("http://{pod_ip}:{}", pool.target_port),
        kv_events_endpoint: format!("tcp://{pod_ip}:{kv_event_port}"),
        replay_endpoint: replay_port.map(|p| format!("tcp://{pod_ip}:{p}")),
        stable_routing_id: pod_name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
    use kube::api::ObjectMeta;

    fn pool() -> PoolState {
        PoolState {
            match_labels: BTreeMap::from([("app".to_string(), "vllm-qwen".to_string())]),
            target_port: 8000,
        }
    }

    fn pod(name: &str, ip: Option<&str>, ready: Option<bool>, labels: &[(&str, &str)]) -> Pod {
        let conditions = ready.map(|r| {
            vec![PodCondition {
                type_: "Ready".to_string(),
                status: if r { "True" } else { "False" }.to_string(),
                ..Default::default()
            }]
        });
        let label_map = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(label_map),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(|s| s.to_string()),
                conditions,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ready_selected_pod_maps_to_worker() {
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            5557,
            Some(5560),
        )
        .expect("ready, selected pod should map");
        assert_eq!(w.worker_id, hash_pod_name("vllm-0"));
        assert_eq!(w.http_endpoint, "http://10.0.0.1:8000");
        assert_eq!(w.kv_events_endpoint, "tcp://10.0.0.1:5557");
        assert_eq!(w.replay_endpoint.as_deref(), Some("tcp://10.0.0.1:5560"));
    }

    #[test]
    fn pod_not_matching_selector_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "other-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "something-else")]
                ),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn not_ready_pod_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(false),
                    &[("app", "vllm-qwen")]
                ),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn terminating_pod_is_skipped() {
        let mut p = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        p.metadata.deletion_timestamp = Some(Time(k8s_openapi::chrono::Utc::now()));
        assert!(raw_worker_from_pod(&p, &pool(), 5557, None).is_none());
    }

    #[test]
    fn pod_without_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod("vllm-0", None, Some(true), &[("app", "vllm-qwen")]),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }
}
