// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes pod reflector for router-only ("on-ramp") mode.
//!
//! Unlike the Dynamo-mode reflector in `epp.rs` (which selects worker pods by
//! the operator-applied `nvidia.com/dynamo-*` labels and resolves the container
//! port *named* `http`), router-only mode fronts **raw `vllm serve` pods** that:
//!
//! * carry an arbitrary, user-chosen label set — so the selector is supplied
//!   verbatim via `DYN_EPP_POD_SELECTOR` (e.g. `app=vllm-qwen`); and
//! * expose their OpenAI HTTP API on a known **port number** (`DYN_EPP_TARGET_PORT`,
//!   the InferencePool `targetPort`) rather than a Dynamo-named container port.
//!
//! Endpoints are also **Ready-filtered**: we never hand the gateway a pod that
//! is not `Ready`, so in-flight rollouts and crash-looping pods don't receive
//! traffic.
//!
//! `worker_id` is `hash_pod_name(pod_name)` — identical to the Dynamo-mode
//! scheme — so the IDs produced here line up with the scheduler and with the
//! `KvRepublisher`'s per-pod event stamping.
//!
//! NOTE: this module is not yet wired into [`crate::epp::Router`]; the
//! router-only constructor that consumes it lands in a subsequent change.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::reflector::Store;

use dynamo_runtime::discovery::hash_pod_name;

/// How long to block startup waiting for the reflector's initial LIST before
/// continuing in the background (mirrors the Dynamo-mode reflector).
const INITIAL_SYNC_TIMEOUT: Duration = Duration::from_secs(30);

/// A label-selector-driven, Ready-filtered pod reflector for raw vLLM pods.
///
/// Provides lock-free reads of the current pod set; no K8s API calls on the
/// hot path. Endpoints are resolved as `<pod-ip>:<target_port>`.
pub struct RouterOnlyPodReflector {
    store: Store<Pod>,
    ready: Arc<AtomicBool>,
    target_port: u16,
}

impl RouterOnlyPodReflector {
    /// Start a background reflector watching pods that match `selector` in
    /// `k8s_namespace`, resolving endpoints on `target_port`.
    ///
    /// Returns once the initial LIST completes (or after [`INITIAL_SYNC_TIMEOUT`],
    /// after which it finishes in the background and [`is_ready`](Self::is_ready)
    /// flips to `true`).
    pub async fn spawn(k8s_namespace: &str, selector: &str, target_port: u16) -> Result<Self> {
        use futures::StreamExt;
        use kube::{Api, Client, runtime::reflector, runtime::watcher};

        let client = Client::try_default().await?;
        let pods: Api<Pod> = Api::namespaced(client, k8s_namespace);

        let writer = reflector::store::Writer::default();
        let store = writer.as_reader();
        let ready = Arc::new(AtomicBool::new(false));
        let watcher_config = watcher::Config::default().labels(selector);
        let reflect = reflector::reflector(writer, watcher(pods, watcher_config));

        tracing::info!(
            namespace = k8s_namespace,
            selector = selector,
            target_port,
            "Starting router-only pod reflector for raw vLLM endpoint resolution"
        );

        let store_for_wait = store.clone();
        tokio::spawn(async move {
            tokio::pin!(reflect);
            while reflect.next().await.is_some() {}
            tracing::warn!("Router-only pod reflector stream ended unexpectedly");
        });

        // Bounded wait for the initial LIST so the first request doesn't race an
        // empty cache; if it times out we keep waiting in the background.
        match tokio::time::timeout(INITIAL_SYNC_TIMEOUT, store_for_wait.wait_until_ready()).await {
            Ok(Ok(())) => {
                ready.store(true, Ordering::Release);
                tracing::info!("Router-only pod reflector initial LIST sync complete");
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    error = %e,
                    "Router-only pod reflector writer dropped before initial LIST; returning 503 until ready"
                );
            }
            Err(_) => {
                tracing::warn!(
                    "Router-only pod reflector initial LIST timed out after {}s; returning 503 until ready",
                    INITIAL_SYNC_TIMEOUT.as_secs()
                );
                let store_for_background_wait = store.clone();
                let ready_for_background_wait = ready.clone();
                tokio::spawn(async move {
                    match store_for_background_wait.wait_until_ready().await {
                        Ok(()) => {
                            ready_for_background_wait.store(true, Ordering::Release);
                            tracing::info!(
                                "Router-only pod reflector became ready after startup timeout"
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "Router-only pod reflector writer dropped while waiting; store stays not-ready"
                            );
                        }
                    }
                });
            }
        }

        Ok(Self {
            store,
            ready,
            target_port,
        })
    }

    /// Whether the initial LIST sync has completed.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Shared readiness flag, so callers (e.g. the gRPC health reporter) can
    /// gate SERVING on the reflector being usable.
    pub fn ready_flag(&self) -> Arc<AtomicBool> {
        self.ready.clone()
    }

    /// `(worker_id, "ip:port")` for every currently-Ready pod.
    pub fn ready_workers(&self) -> Vec<(u64, String)> {
        self.store
            .state()
            .iter()
            .filter(|pod| pod_is_ready(pod))
            .filter_map(|pod| {
                let name = pod.metadata.name.as_deref()?;
                let endpoint = pod_endpoint_by_port(pod, self.target_port)?;
                Some((hash_pod_name(name), endpoint))
            })
            .collect()
    }

    /// Worker IDs of all Ready pods — the admission set to register with the
    /// scheduler so it never selects a worker outside the live fleet.
    pub fn ready_worker_ids(&self) -> HashSet<u64> {
        self.ready_workers().into_iter().map(|(id, _)| id).collect()
    }

    /// Resolve a `worker_id` to its `ip:port`, if the pod is reflected and Ready.
    pub fn resolve_worker_endpoint(&self, worker_id: u64) -> Option<String> {
        for pod in self.store.state() {
            let Some(name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if hash_pod_name(name) == worker_id && pod_is_ready(&pod) {
                return pod_endpoint_by_port(&pod, self.target_port);
            }
        }
        None
    }

    /// Resolve any Ready pod's `ip:port` (for body-less requests such as
    /// `GET /v1/models`).
    pub fn resolve_any_worker_endpoint(&self) -> Option<String> {
        self.store
            .state()
            .iter()
            .filter(|pod| pod_is_ready(pod))
            .find_map(|pod| pod_endpoint_by_port(pod, self.target_port))
    }
}

/// A pod is routable only when its `Ready` condition is `True`.
fn pod_is_ready(pod: &Pod) -> bool {
    let Some(status) = pod.status.as_ref() else {
        return false;
    };
    status.conditions.as_ref().is_some_and(|conds| {
        conds
            .iter()
            .any(|c| c.type_ == "Ready" && c.status == "True")
    })
}

/// Resolve `<pod-ip>:<target_port>`. Unlike the Dynamo-mode resolver, the port
/// is the operator-independent `DYN_EPP_TARGET_PORT` number rather than a
/// container port *named* `http`. Returns `None` if the pod has no IP yet.
fn pod_endpoint_by_port(pod: &Pod, target_port: u16) -> Option<String> {
    let ip = pod.status.as_ref()?.pod_ip.as_ref()?;
    Some(format!("{ip}:{target_port}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use kube::api::ObjectMeta;

    fn pod(name: &str, ip: Option<&str>, ready: Option<&str>) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(str::to_string),
                conditions: ready.map(|s| {
                    vec![PodCondition {
                        type_: "Ready".to_string(),
                        status: s.to_string(),
                        ..Default::default()
                    }]
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ready_requires_ready_condition_true() {
        assert!(pod_is_ready(&pod("a", Some("10.0.0.1"), Some("True"))));
        assert!(!pod_is_ready(&pod("a", Some("10.0.0.1"), Some("False"))));
        assert!(!pod_is_ready(&pod("a", Some("10.0.0.1"), None)));
    }

    #[test]
    fn endpoint_is_ip_plus_target_port() {
        assert_eq!(
            pod_endpoint_by_port(&pod("a", Some("10.0.0.5"), Some("True")), 8000).as_deref(),
            Some("10.0.0.5:8000")
        );
        // No IP yet -> not resolvable.
        assert_eq!(
            pod_endpoint_by_port(&pod("a", None, Some("True")), 8000),
            None
        );
    }
}
