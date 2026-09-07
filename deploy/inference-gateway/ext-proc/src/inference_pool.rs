// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only watcher for the GAIE `inference.networking.k8s.io/v1` `InferencePool`
//! (not `v1alpha2`).
//!
//! Returns a [`PoolState`] containing the pool's `match_labels` and `target_port`.
//!
//! Currently supports only pools with at least one match label and exactly one
//! target port. Additional configurations may be supported in the future, for
//! example to enable data-parallel-aware routing.

use std::collections::BTreeMap;

use anyhow::Result;
use kube::core::{ApiResource, DynamicObject, GroupVersionKind};
use kube::runtime::{WatchStreamExt, watcher};
use kube::{Api, Client};
use tokio::sync::watch;

const INFERENCE_POOL_GROUP: &str = "inference.networking.k8s.io";
const INFERENCE_POOL_VERSION: &str = "v1";
const INFERENCE_POOL_KIND: &str = "InferencePool";

/// The slice of `InferencePool` spec the EPP needs to discover pods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolState {
    /// The pod label selector from InferencePool spec
    pub match_labels: BTreeMap<String, String>,
    /// The OpenAI HTTP target port resolved from the pool
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
        let stream = watcher::watch_object(api, &name).default_backoff();
        tokio::pin!(stream);
        loop {
            match stream.next().await {
                Some(Ok(obj)) => publish_pool_state(obj, &name, &tx),
                Some(Err(e)) => {
                    tracing::warn!(error = %e, pool = %name, "InferencePool watch error; retrying");
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

/// Derive the [`PoolState`] from the observed pool object and publish it **only
/// when it changes**. Non-existent or invalid spec both publish `None`, so a
/// deleted or invalid pool always clears stale discovery state.
fn publish_pool_state(
    obj: Option<DynamicObject>,
    name: &str,
    tx: &watch::Sender<Option<PoolState>>,
) {
    let (state, clear_reason) = match &obj {
        None => (None, Some("does not exist".to_string())),
        Some(o) => match parse_pool_state(&o.data) {
            Ok(s) => (Some(s), None),
            Err(reason) => (None, Some(reason)),
        },
    };

    // Only send a notification if the match labels or target port has changed.
    tx.send_if_modified(|current| {
        if *current == state {
            return false;
        }
        match &state {
            Some(s) => tracing::info!(
                pool = %name,
                target_port = s.target_port,
                labels = ?s.match_labels,
                "InferencePool resolved"
            ),
            None => tracing::warn!(
                pool = %name,
                reason = clear_reason.as_deref().unwrap_or("unknown"),
                "InferencePool is invalid, clearing discovery state"
            ),
        }
        *current = state;
        true
    });
}

/// Extract a [`PoolState`] from an `InferencePool` `spec`, or a message naming why it is invalid.
fn parse_pool_state(data: &serde_json::Value) -> std::result::Result<PoolState, String> {
    let spec = data
        .get("spec")
        .ok_or_else(|| "spec is missing".to_string())?;

    let labels_obj = spec
        .get("selector")
        .and_then(|s| s.get("matchLabels"))
        .and_then(|m| m.as_object())
        .ok_or_else(|| "spec.selector.matchLabels is missing or not a string map".to_string())?;
    let match_labels: BTreeMap<String, String> = labels_obj
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();

    if match_labels.is_empty() {
        return Err(
            "spec.selector.matchLabels is empty, needs at least one match label".to_string(),
        );
    }

    let target_ports = spec
        .get("targetPorts")
        .and_then(|tp| tp.as_array())
        .ok_or_else(|| "spec.targetPorts is missing or not an array".to_string())?;
    if target_ports.len() != 1 {
        return Err(format!(
            "expected exactly one targetPort, found {}",
            target_ports.len()
        ));
    }
    let target_port_u64 = target_ports[0]
        .get("number")
        .and_then(|n| n.as_u64())
        .ok_or_else(|| "targetPorts[0].number is missing or not an integer".to_string())?;
    let target_port = match u16::try_from(target_port_u64) {
        Ok(p) if p != 0 => p,
        _ => {
            return Err(format!(
                "targetPort {target_port_u64} is outside the valid 1-65535 range"
            ));
        }
    };

    Ok(PoolState {
        match_labels,
        target_port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A `DynamicObject` carrying `data` (spec/status live here for a CRD read as
    /// a dynamic object), enough to exercise `publish_pool_state`.
    fn dyn_obj(data: serde_json::Value) -> DynamicObject {
        DynamicObject {
            types: None,
            metadata: Default::default(),
            data,
        }
    }

    #[test]
    fn status_only_update_does_not_wake_discovery() {
        let (tx, mut rx) = watch::channel::<Option<PoolState>>(None);

        // First real spec resolves the pool -> receiver observes a change.
        publish_pool_state(
            Some(dyn_obj(json!({
                "spec": {
                    "selector": {"matchLabels": {"app": "vllm-qwen"}},
                    "targetPorts": [{"number": 8000}]
                },
                "status": {"conditions": [{"type": "Accepted", "status": "True"}]}
            }))),
            "pool",
            &tx,
        );
        assert!(rx.has_changed().unwrap());
        assert_eq!(rx.borrow_and_update().as_ref().unwrap().target_port, 8000);

        // Same spec, different status/metadata only -> derived state is identical,
        // so no notification (this is the churn we must suppress).
        publish_pool_state(
            Some(dyn_obj(json!({
                "metadata": {"resourceVersion": "12345"},
                "spec": {
                    "selector": {"matchLabels": {"app": "vllm-qwen"}},
                    "targetPorts": [{"number": 8000}]
                },
                "status": {"conditions": [{"type": "Accepted", "status": "False"}]}
            }))),
            "pool",
            &tx,
        );
        assert!(!rx.has_changed().unwrap());

        // A real target-port change wakes discovery again.
        publish_pool_state(
            Some(dyn_obj(json!({
                "spec": {
                    "selector": {"matchLabels": {"app": "vllm-qwen"}},
                    "targetPorts": [{"number": 9000}]
                }
            }))),
            "pool",
            &tx,
        );
        assert!(rx.has_changed().unwrap());
        assert_eq!(rx.borrow_and_update().as_ref().unwrap().target_port, 9000);
    }

    #[test]
    fn repeated_unparseable_pool_clears_once() {
        let (tx, mut rx) = watch::channel::<Option<PoolState>>(Some(PoolState {
            match_labels: BTreeMap::from([("app".to_string(), "vllm-qwen".to_string())]),
            target_port: 8000,
        }));

        // Pool edited into an unusable spec -> clears to None (one notification).
        publish_pool_state(Some(dyn_obj(json!({"spec": {}}))), "pool", &tx);
        assert!(rx.has_changed().unwrap());
        assert!(rx.borrow_and_update().is_none());

        // Still unusable on the next delivery -> already None, so no re-wake.
        publish_pool_state(Some(dyn_obj(json!({"spec": {}}))), "pool", &tx);
        assert!(!rx.has_changed().unwrap());
    }

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
    fn missing_spec_is_rejected() {
        assert_eq!(parse_pool_state(&json!({})).unwrap_err(), "spec is missing");
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
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "spec.targetPorts is missing or not an array"
        );
    }

    #[test]
    fn empty_selector_is_rejected() {
        let data = json!({
            "spec": {"selector": {"matchLabels": {}}, "targetPorts": [{"number": 8000}]}
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "spec.selector.matchLabels is empty, needs at least one match label"
        );
    }

    #[test]
    fn missing_selector_is_rejected() {
        let data = json!({
            "spec": {"targetPorts": [{"number": 8000}]}
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "spec.selector.matchLabels is missing or not a string map"
        );
    }

    #[test]
    fn missing_target_port_is_rejected() {
        let data = json!({
            "spec": {"selector": {"matchLabels": {"app": "x"}}}
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "spec.targetPorts is missing or not an array"
        );
    }

    #[test]
    fn multi_port_pool_is_rejected() {
        // Workers are keyed by hash(pod_name), so a pod maps to one endpoint;
        // multi-port pools (DP-aware routing) aren't in scope yet.
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": [{"number": 8000}, {"number": 8001}]
            }
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "expected exactly one targetPort, found 2"
        );
    }

    #[test]
    fn empty_target_ports_is_rejected() {
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": []
            }
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "expected exactly one targetPort, found 0"
        );
    }

    #[test]
    fn missing_target_port_number_is_rejected() {
        let data = json!({
            "spec": {
                "selector": {"matchLabels": {"app": "vllm-qwen"}},
                "targetPorts": [{"protocol": "TCP"}]
            }
        });
        assert_eq!(
            parse_pool_state(&data).unwrap_err(),
            "targetPorts[0].number is missing or not an integer"
        );
    }

    #[test]
    fn out_of_range_or_zero_target_port_is_rejected() {
        for port in [0u64, 70_000u64] {
            let data = json!({
                "spec": {
                    "selector": {"matchLabels": {"app": "vllm-qwen"}},
                    "targetPorts": [{"number": port}]
                }
            });
            assert_eq!(
                parse_pool_state(&data).unwrap_err(),
                format!("targetPort {port} is outside the valid 1-65535 range")
            );
        }
    }
}
