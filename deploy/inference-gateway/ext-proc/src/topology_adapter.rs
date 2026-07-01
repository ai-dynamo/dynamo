// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine topology adapter: reconcile discovered raw vLLM pods into the
//! selection-service worker catalog.
//!
//! Raw vLLM pods never register a `ModelRuntimeConfig` in any Dynamo discovery
//! plane, so the per-worker metadata the selector needs to admit a worker
//! (endpoint, block size, KV-event endpoints, DP size, capacity hints) is
//! produced here instead. For each `Ready` pod the [`PodDiscovery`]
//! surfaces, this adapter normalizes a [`WorkerRegistration`] from environment
//! defaults plus the pod's resolved endpoints, and hands the whole desired set
//! to the [`SelectionBackend`], which owns the actual-vs-desired diff:
//!
//! - the HTTP fleet applies the diff to every live selector replica (and
//!   bootstraps replicas that appear or restart), and
//! - the embedded backend applies it to the in-process core.
//!
//! The adapter owns `worker_id` generation (via the reflector's stable hash) and
//! applies the same id everywhere, so each replica's catalog agrees.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::epp_config::EppConfig;
use crate::pod_discovery::{PodDiscovery, RawWorker};
use crate::selection_backend::{SelectionBackend, WorkerRegistration};

/// Periodic re-reconcile interval. Bounds how long a selection-service restart
/// (or any transient upsert/delete failure) can leave a catalog stale while the
/// pod set is otherwise unchanged.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

/// Environment-derived defaults applied to every worker registration. Raw vLLM
/// pods carry no model card, so these come from the EPP's configuration.
#[derive(Debug, Clone)]
pub struct RegistrationDefaults {
    pub model_name: String,
    pub block_size: u32,
    pub data_parallel_size: u32,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
}

impl RegistrationDefaults {
    pub fn from_config(cfg: &EppConfig) -> Self {
        Self {
            model_name: cfg.model_name.clone(),
            block_size: cfg.block_size,
            data_parallel_size: cfg.data_parallel_size,
            total_kv_blocks: cfg.total_kv_blocks,
            max_num_batched_tokens: cfg.max_num_batched_tokens,
        }
    }
}

/// Background task that keeps the selector catalog in sync with the reflector.
pub struct TopologyAdapter {
    _task: tokio::task::JoinHandle<()>,
}

impl TopologyAdapter {
    /// Spawn the reconciliation loop. Performs an initial reconcile immediately,
    /// then re-reconciles whenever the reflector reports a pod change, the
    /// backend reports a selector-fleet change, or the periodic timer fires.
    pub fn spawn(
        reflector: Arc<PodDiscovery>,
        backend: Arc<dyn SelectionBackend>,
        defaults: RegistrationDefaults,
    ) -> Self {
        let task = tokio::spawn(async move {
            let mut pod_changes = reflector.subscribe_changes();
            let mut backend_changes = backend.subscribe_changes();
            let mut ticker = tokio::time::interval(RECONCILE_INTERVAL);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                reconcile_once(&reflector, backend.as_ref(), &defaults).await;
                // Re-reconcile on a pod change, a selector-fleet change (a
                // replica appeared/disappeared), OR a periodic tick. The timer
                // recovers a selection-service restart (which drops the catalog)
                // even when nothing else changes. Exit if the pod-change sender
                // drops (the reflector is gone).
                tokio::select! {
                    changed = pod_changes.changed() => {
                        if changed.is_err() {
                            tracing::warn!(
                                "Reflector change channel closed; topology adapter stopping"
                            );
                            break;
                        }
                    }
                    _ = backend_changes.changed() => {}
                    _ = ticker.tick() => {}
                }
            }
        });
        Self { _task: task }
    }
}

/// Run one reconcile pass: build the desired catalog from the Ready pods and
/// hand it to the backend, which owns the actual-vs-desired diff.
async fn reconcile_once(
    reflector: &PodDiscovery,
    backend: &dyn SelectionBackend,
    defaults: &RegistrationDefaults,
) {
    let desired: HashMap<u64, WorkerRegistration> = reflector
        .ready_workers()
        .iter()
        .map(|w| (w.worker_id, build_registration(w, defaults)))
        .collect();

    if let Err(e) = backend.reconcile(&desired).await {
        tracing::warn!(error = %e, "Selector reconcile failed; will retry on next change or tick");
    }
}

/// Normalize a discovered worker into a selector registration payload.
fn build_registration(w: &RawWorker, defaults: &RegistrationDefaults) -> WorkerRegistration {
    // Aggregated V1: a single dp_rank 0 maps to the pod's KV-event PUB socket.
    let mut kv_events_endpoints = HashMap::new();
    kv_events_endpoints.insert(0u32, w.kv_events_endpoint.clone());

    WorkerRegistration {
        worker_id: w.worker_id,
        model_name: defaults.model_name.clone(),
        endpoint: w.http_endpoint.clone(),
        block_size: defaults.block_size,
        data_parallel_size: defaults.data_parallel_size,
        kv_events_endpoints,
        replay_endpoint: w.replay_endpoint.clone(),
        total_kv_blocks: defaults.total_kv_blocks,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
        stable_routing_id: Some(w.stable_routing_id.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> RegistrationDefaults {
        RegistrationDefaults {
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            block_size: 16,
            data_parallel_size: 1,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: None,
        }
    }

    fn worker(id: u64, ip: &str) -> RawWorker {
        RawWorker {
            worker_id: id,
            pod_name: format!("vllm-{id}"),
            pod_ip: ip.to_string(),
            http_endpoint: format!("http://{ip}:8000"),
            kv_events_endpoint: format!("tcp://{ip}:5557"),
            replay_endpoint: None,
            stable_routing_id: format!("vllm-{id}"),
        }
    }

    #[test]
    fn registration_maps_env_and_endpoints() {
        let reg = build_registration(&worker(7, "10.0.0.1"), &defaults());
        assert_eq!(reg.worker_id, 7);
        assert_eq!(reg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(reg.endpoint, "http://10.0.0.1:8000");
        assert_eq!(reg.block_size, 16);
        assert_eq!(reg.data_parallel_size, 1);
        assert_eq!(
            reg.kv_events_endpoints.get(&0).unwrap(),
            "tcp://10.0.0.1:5557"
        );
        assert_eq!(reg.total_kv_blocks, Some(1000));
        assert_eq!(reg.stable_routing_id.as_deref(), Some("vllm-7"));
    }
}
