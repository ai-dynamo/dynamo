// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only watcher for the GAIE `InferencePool` this EPP backs.
//!
//! In standalone mode the `InferencePool` is the single source of truth for
//! which pods are candidate backends: the gateway routes to it, and the EPP
//! reads its `spec.selector` and target port to discover the same pods. This
//! watcher only ever *reads* the pool — it never writes status or manages the
//! object, so it coexists with the gateway provider's controller (which owns the
//! pool lifecycle) without conflict.
//!
//! The pool is watched as a dynamic object (no compiled CRD type), but the
//! contract is explicitly `inference.networking.k8s.io/v1`: the selector is read
//! from `spec.selector.matchLabels` and the target port from
//! `spec.targetPorts[].number`. Legacy `v1alpha2` pools use a different GVK and a
//! flat `spec.selector` map, so they never reach this watch or parser.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use kube::core::{ApiResource, DynamicObject, GroupVersionKind};
use kube::runtime::watcher;
use kube::{Api, Client};
use tokio::sync::watch;

const INFERENCE_POOL_GROUP: &str = "inference.networking.k8s.io";
const INFERENCE_POOL_VERSION: &str = "v1";
const INFERENCE_POOL_KIND: &str = "InferencePool";

/// The slice of `InferencePool` spec the EPP needs to discover pods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolState {
    /// `spec.selector.matchLabels` — the pod label selector (equality only).
    pub match_labels: BTreeMap<String, String>,
    /// The OpenAI HTTP target port resolved from the pool.
    pub target_port: u16,
}

/// Start a background watch of the named `InferencePool`. The returned receiver
/// carries the latest [`PoolState`] (or `None` until the pool is first observed,
/// or after it is deleted).
pub async fn spawn_pool_watch(
    client: Client,
    namespace: String,
    name: String,
) -> Result<(
    watch::Receiver<Option<PoolState>>,
    tokio::task::JoinHandle<()>,
)> {
    let gvk = GroupVersionKind::gvk(
        INFERENCE_POOL_GROUP,
        INFERENCE_POOL_VERSION,
        INFERENCE_POOL_KIND,
    );
    let ar = ApiResource::from_gvk(&gvk);
    let api: Api<DynamicObject> = Api::namespaced_with(client, &namespace, &ar);

    let (tx, rx) = watch::channel::<Option<PoolState>>(None);

    tracing::info!(
        namespace = %namespace,
        pool = %name,
        "Starting InferencePool watch (read-only) for standalone discovery"
    );

    let handle = tokio::spawn(async move {
        use futures::StreamExt;
        // `watch_object` tracks a single named object and yields its current
        // state as `Option`: `Some` while it exists, `None` once it is gone.
        // Crucially, deletions that happen while the watch is disconnected are
        // reported via `None` on the next relist (not a `Delete` event), so
        // stale `PoolState` can never survive a reconnect.
        let stream = watcher::watch_object(api, &name);
        tokio::pin!(stream);
        loop {
            match stream.next().await {
                Some(Ok(obj)) => publish_pool_state(obj, &name, &tx),
                Some(Err(e)) => {
                    tracing::warn!(error = %e, pool = %name, "InferencePool watch error; retrying");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                None => {
                    tracing::warn!(pool = %name, "InferencePool watch stream ended");
                    break;
                }
            }
        }
    });

    Ok((rx, handle))
}

/// Publish the latest [`PoolState`] derived from the observed pool object.
/// Sends `None` when the pool is absent or its spec cannot be parsed, so a
/// deleted or malformed pool always clears stale discovery state.
fn publish_pool_state(
    obj: Option<DynamicObject>,
    name: &str,
    tx: &watch::Sender<Option<PoolState>>,
) {
    let state = obj.as_ref().and_then(|o| parse_pool_state(&o.data));
    match (&obj, &state) {
        (Some(_), Some(s)) => {
            tracing::info!(
                pool = %name,
                target_port = s.target_port,
                labels = ?s.match_labels,
                "InferencePool resolved"
            );
        }
        (Some(_), None) => {
            tracing::warn!(
                pool = %name,
                "InferencePool present but unsupported (missing matchLabels/targetPort, or not exactly one targetPort); clearing discovery state"
            );
        }
        (None, _) => {
            tracing::warn!(pool = %name, "InferencePool absent; clearing discovery state");
        }
    }
    let _ = tx.send(state);
}

/// Extract a [`PoolState`] from an `InferencePool` `spec` JSON value. Returns
/// `None` if the selector is empty/missing, no target port can be resolved, or
/// the pool declares anything other than exactly one target port.
/// Pure function — unit-testable without a cluster.
fn parse_pool_state(data: &serde_json::Value) -> Option<PoolState> {
    let spec = data.get("spec")?;

    let labels_obj = spec.get("selector")?.get("matchLabels")?.as_object()?;
    let mut match_labels = BTreeMap::new();
    for (k, v) in labels_obj {
        if let Some(s) = v.as_str() {
            match_labels.insert(k.clone(), s.to_string());
        }
    }
    // An empty selector would match every pod in the namespace; refuse it.
    if match_labels.is_empty() {
        return None;
    }

    // v1: spec.targetPorts[].number. Exactly one target port is required.
    // Discovery keys each worker by `worker_id = hash_pod_name(pod_name)` (see
    // pod_discovery.rs), so a pod maps to a single endpoint. GAIE, by contrast,
    // treats every (podIP, port) as a distinct endpoint, so a multi-port pool
    // would collapse into colliding worker IDs and lose endpoints. Reject it
    // instead of silently keeping the first port; a live update into a multi-port
    // shape returns `None` here and clears discovery state.
    let target_ports = spec.get("targetPorts").and_then(|tp| tp.as_array())?;
    if target_ports.len() != 1 {
        return None;
    }
    let target_port_u64 = target_ports
        .first()
        .and_then(|p| p.get("number"))
        .and_then(|n| n.as_u64())?;
    let target_port = u16::try_from(target_port_u64).ok()?;
    if target_port == 0 {
        return None;
    }

    Some(PoolState {
        match_labels,
        target_port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_v1_target_ports() {
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": [{"number": 8000}]
            }
        });
        let s = parse_pool_state(&data).expect("v1 pool should parse");
        assert_eq!(s.target_port, 8000);
        assert_eq!(s.match_labels.get("app").unwrap(), "vllm-qwen");
    }

    #[test]
    fn parses_multiple_match_labels() {
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen", "role": "decode"}},
                "targetPorts": [{"number": 8000}]
            }
        });
        let s = parse_pool_state(&data).expect("v1 pool should parse");
        assert_eq!(s.target_port, 8000);
        assert_eq!(s.match_labels.len(), 2);
    }

    #[test]
    fn v1alpha2_target_port_number_is_rejected() {
        // Legacy `spec.targetPortNumber` is not part of the v1 contract.
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPortNumber": 8000
            }
        });
        assert!(parse_pool_state(&data).is_none());
    }

    #[test]
    fn empty_selector_is_rejected() {
        let data = json!({
            "spec": {"selector": {"matchLabels": {}}, "targetPorts": [{"number": 8000}]}
        });
        assert!(parse_pool_state(&data).is_none());
    }

    #[test]
    fn missing_target_port_is_rejected() {
        let data = json!({
            "spec": {"selector": {"matchLabels": {"app": "x"}}}
        });
        assert!(parse_pool_state(&data).is_none());
    }

    #[test]
    fn multi_port_pool_is_rejected() {
        // Workers are keyed by hash(pod_name), so a pod maps to one endpoint.
        // A multi-port pool would collapse into colliding worker IDs; reject it.
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": [{"number": 8000}, {"number": 8001}]
            }
        });
        assert!(parse_pool_state(&data).is_none());
    }

    #[test]
    fn empty_target_ports_is_rejected() {
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": []
            }
        });
        assert!(parse_pool_state(&data).is_none());
    }
}
