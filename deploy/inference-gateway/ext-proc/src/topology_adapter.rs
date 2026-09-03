// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconciles discovered vLLM pods with the selection-service worker catalog.
//!
//! The adapter converts each ready pod into a [`WorkerRegistration`] using its
//! resolved endpoints and configured defaults, then passes the desired worker
//! set to the [`Selector`].

use std::collections::HashMap;

use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::pod_discovery::{PodDiscovery, RawWorker};
use crate::selector::{RoleSelectors, Selector, WorkerRegistration};
use crate::worker_role::WorkerRole;

#[derive(Debug, Clone)]
pub struct RegistrationDefaults {
    pub model_name: String,
    pub block_size: u32,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
}

impl RegistrationDefaults {
    pub fn from_config(cfg: &EppStandaloneConfig) -> Self {
        Self::for_role(cfg, WorkerRole::Aggregated)
    }

    /// Catalog metadata for a worker in `role`.
    ///
    /// Capacity is per-role because disaggregated fleets size prefill and decode
    /// differently — shipped recipes differ by 8-16x on batched tokens — and that
    /// value is the denominator of the scheduler's busy test, so one shared
    /// number mis-sizes whichever role did not set it.
    pub fn for_role(cfg: &EppStandaloneConfig, role: WorkerRole) -> Self {
        Self {
            model_name: cfg.model_name.clone(),
            block_size: cfg.block_size,
            total_kv_blocks: cfg.total_kv_blocks,
            max_num_batched_tokens: cfg.max_num_batched_tokens_for(role),
        }
    }
}

/// Background task that keeps the selector catalog in sync with the reflector.
/// Dropping the adapter cancels the task so it stops promptly and releases its
/// `Selector`/`PodDiscovery` handles.
pub struct TopologyAdapter {
    cancel: CancellationToken,
}

impl TopologyAdapter {
    pub fn spawn(
        reflector: PodDiscovery,
        selectors: RoleSelectors,
        cfg: &EppStandaloneConfig,
    ) -> Self {
        // Resolved once: the defaults are per-role but static for the process.
        let defaults: Vec<(WorkerRole, RegistrationDefaults)> = selectors
            .each()
            .into_iter()
            .map(|(role, _)| (role, RegistrationDefaults::for_role(cfg, role)))
            .collect();

        let cancel = CancellationToken::new();
        let cancel_child = cancel.clone();
        tokio::spawn(async move {
            let mut pod_changes = reflector.subscribe_changes();
            loop {
                reconcile_once(&reflector, &selectors, &defaults).await;
                tokio::select! {
                    _ = cancel_child.cancelled() => break,
                    // Re-reconcile on a pod change. Exit if the sender drops
                    // (reflector gone).
                    changed = pod_changes.changed() => {
                        if changed.is_err() {
                            tracing::warn!(
                                "Reflector change channel closed; clearing selector topology"
                            );
                            // Every role's catalog, not just the serving one:
                            // a stale prefill catalog would outlive the watch.
                            for (role, selector) in selectors.each() {
                                if let Err(e) = selector.reconcile(&[]).await {
                                    tracing::warn!(
                                        error = %e,
                                        role = %role,
                                        "Failed to clear selector topology after reflector stopped"
                                    );
                                }
                            }
                            break;
                        }
                    }
                }
            }
        });
        Self { cancel }
    }
}

impl Drop for TopologyAdapter {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Run one reconcile pass: build each role's desired catalog from the Ready pods
/// and hand it to that role's selector, which owns the actual-vs-desired diff.
async fn reconcile_once(
    reflector: &PodDiscovery,
    selectors: &RoleSelectors,
    defaults: &[(WorkerRole, RegistrationDefaults)],
) {
    match selectors {
        RoleSelectors::Aggregated(selector) => {
            let Some((role, defaults)) = defaults.first() else {
                return;
            };
            let desired = registrations(reflector.ready_workers(), defaults);
            apply(selector, *role, desired).await;
        }
        RoleSelectors::Disaggregated { prefill, decode } => {
            // One snapshot for both roles. Reading them through two calls would
            // take two locks and observe two generations, and a role flip
            // between them would put one worker into both desired sets.
            let sets = reflector.ready_workers_by_role();
            for (role, workers) in [
                (WorkerRole::Prefill, sets.prefill),
                (WorkerRole::Decode, sets.decode),
            ] {
                let Some((_, role_defaults)) = defaults.iter().find(|(r, _)| *r == role) else {
                    continue;
                };
                let selector = match role {
                    WorkerRole::Prefill => prefill,
                    _ => decode,
                };
                apply(selector, role, registrations(workers, role_defaults)).await;
            }
        }
    }
}

fn registrations(
    workers: Vec<RawWorker>,
    defaults: &RegistrationDefaults,
) -> Vec<WorkerRegistration> {
    workers
        .into_iter()
        .map(|w| build_registration(w, defaults))
        .collect()
}

async fn apply(selector: &Selector, role: WorkerRole, desired: Vec<WorkerRegistration>) {
    if let Err(e) = selector.reconcile(&desired).await {
        tracing::warn!(
            error = %e,
            role = %role,
            "Selector reconcile failed; will retry on next change"
        );
    }
}

fn build_registration(w: RawWorker, defaults: &RegistrationDefaults) -> WorkerRegistration {
    let mut kv_events_endpoints = HashMap::new();
    // A decode worker has no endpoint: its `SelectionService` runs with KV
    // events off, so leaving the map empty is exactly what keeps the worker
    // schedulable there — `missing_schedulable_metadata` only demands an
    // endpoint per dp_rank when that instance consumes KV events.
    if let Some(endpoint) = w.kv_events_endpoint {
        kv_events_endpoints.insert(0u32, endpoint);
    }

    WorkerRegistration {
        worker_id: w.worker_id,
        model_name: defaults.model_name.clone(),
        endpoint: w.http_endpoint,
        block_size: defaults.block_size,
        kv_events_endpoints,
        replay_endpoint: w.replay_endpoint,
        total_kv_blocks: defaults.total_kv_blocks,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    fn config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            total_kv_blocks: Some(1000),
            ..EppStandaloneConfig::for_test()
        }
    }

    fn defaults() -> RegistrationDefaults {
        RegistrationDefaults {
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            block_size: 16,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: None,
        }
    }

    fn worker(id: u64, ip: &str) -> RawWorker {
        RawWorker {
            worker_id: id,
            pod_name: format!("vllm-{id}"),
            pod_ip: ip.to_string(),
            role: WorkerRole::Aggregated,
            http_endpoint: format!("http://{ip}:8000"),
            kv_events_endpoint: Some(format!("tcp://{ip}:5557")),
            replay_endpoint: None,
        }
    }

    #[test]
    fn registration_maps_env_and_endpoints() {
        let reg = build_registration(worker(7, "10.0.0.1"), &defaults());
        assert_eq!(reg.worker_id, 7);
        assert_eq!(reg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(reg.endpoint, "http://10.0.0.1:8000");
        assert_eq!(reg.block_size, 16);
        assert_eq!(
            reg.kv_events_endpoints.get(&0).unwrap(),
            "tcp://10.0.0.1:5557"
        );
        assert_eq!(reg.total_kv_blocks, Some(1000));
    }

    #[tokio::test]
    async fn channel_close_clears_selector_topology() {
        let selector = Arc::new(
            Selector::new(
                &config(),
                dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry::default(),
            )
            .await
            .expect("selector should build"),
        );
        let (discovery, changes_tx) = PodDiscovery::for_test(vec![worker(7, "10.0.0.1")]);
        let adapter = TopologyAdapter::spawn(
            discovery,
            RoleSelectors::Aggregated(selector.clone()),
            &config(),
        );

        tokio::time::timeout(Duration::from_secs(1), async {
            while !selector.any_ready().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("initial topology was not reconciled");

        // There is no unseen generation when the sole sender closes.
        drop(changes_tx);

        tokio::time::timeout(Duration::from_secs(1), async {
            while selector.any_ready().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal empty topology was not reconciled");

        drop(adapter);
    }

    // --- disaggregated fan-out ---------------------------------------------

    fn disagg_config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            topology_mode: crate::epp_standalone_config::EppTopologyMode::Disaggregated,
            model_name: "test-model".to_string(),
            ..EppStandaloneConfig::for_test()
        }
    }

    fn role_worker(id: u64, ip: &str, role: WorkerRole) -> RawWorker {
        RawWorker {
            worker_id: id,
            pod_name: format!("w-{id}"),
            pod_ip: ip.to_string(),
            role,
            http_endpoint: format!("http://{ip}:8000"),
            // Mirrors what discovery produces: only prefill carries an endpoint.
            kv_events_endpoint: (role != WorkerRole::Decode).then(|| format!("tcp://{ip}:5557")),
            replay_endpoint: None,
        }
    }

    /// Two real `SelectionService`s configured exactly as production does.
    async fn role_selectors(cfg: &EppStandaloneConfig) -> RoleSelectors {
        use crate::role_config::kv_router_config_for_role;
        use dynamo_kv_router::config::KvRouterConfig;
        use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;

        let base = KvRouterConfig::default();
        async fn build(
            cfg: &EppStandaloneConfig,
            base: &KvRouterConfig,
            role: WorkerRole,
        ) -> Arc<Selector> {
            Arc::new(
                Selector::new_with_kv_router_config(
                    cfg,
                    role,
                    kv_router_config_for_role(base, role),
                    WorkerSelectionPolicyRegistry::default(),
                )
                .await
                .expect("role selector should build"),
            )
        }
        RoleSelectors::Disaggregated {
            prefill: build(cfg, &base, WorkerRole::Prefill).await,
            decode: build(cfg, &base, WorkerRole::Decode).await,
        }
    }

    async fn await_counts(selectors: &RoleSelectors, model: &str, prefill: usize, decode: usize) {
        let RoleSelectors::Disaggregated {
            prefill: p,
            decode: d,
        } = selectors
        else {
            panic!("expected a disaggregated topology");
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            while p.schedulable_count(model) != prefill || d.schedulable_count(model) != decode {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "catalogs did not converge to prefill={prefill} decode={decode}; \
                 got prefill={} decode={}",
                p.schedulable_count(model),
                d.schedulable_count(model)
            )
        });
    }

    #[tokio::test]
    async fn disaggregated_reconcile_splits_add_flip_and_remove() {
        let cfg = disagg_config();
        let model = cfg.model_name.clone();
        let selectors = role_selectors(&cfg).await;

        let (discovery, changes_tx) = PodDiscovery::for_test(vec![]);
        let adapter = TopologyAdapter::spawn(discovery.clone(), selectors.clone(), &cfg);

        // Add: one worker of each role lands in its own instance.
        discovery.set_workers(vec![
            role_worker(1, "10.0.0.1", WorkerRole::Prefill),
            role_worker(2, "10.0.0.2", WorkerRole::Decode),
        ]);
        changes_tx.send(1).expect("adapter is listening");
        await_counts(&selectors, &model, 1, 1).await;

        let RoleSelectors::Disaggregated { prefill, decode } = &selectors else {
            unreachable!()
        };
        // The invariant, asserted directly: each instance is one role, so asking
        // decode what it holds cannot be answered by a role filter.
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([2]));
        assert_eq!(prefill.schedulable_worker_ids(&model), HashSet::from([1]));

        // Role flip: the same worker_id moves between instances.
        discovery.set_workers(vec![
            role_worker(1, "10.0.0.1", WorkerRole::Decode),
            role_worker(2, "10.0.0.2", WorkerRole::Decode),
        ]);
        changes_tx.send(2).expect("adapter is listening");
        await_counts(&selectors, &model, 0, 2).await;
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([1, 2]));

        // Remove: dropping one decode worker leaves the other untouched.
        discovery.set_workers(vec![role_worker(2, "10.0.0.2", WorkerRole::Decode)]);
        changes_tx.send(3).expect("adapter is listening");
        await_counts(&selectors, &model, 0, 1).await;
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([2]));

        drop(adapter);
    }

    #[tokio::test]
    async fn decode_workers_are_schedulable_without_kv_event_endpoints() {
        // The whole reason discovery leaves decode's endpoint `None`: the decode
        // instance runs with KV events off, so schedulability must not demand one.
        let cfg = disagg_config();
        let selectors = role_selectors(&cfg).await;
        let RoleSelectors::Disaggregated { decode, .. } = &selectors else {
            unreachable!()
        };

        let registration = build_registration(
            role_worker(9, "10.0.0.9", WorkerRole::Decode),
            &RegistrationDefaults::for_role(&cfg, WorkerRole::Decode),
        );
        assert!(
            registration.kv_events_endpoints.is_empty(),
            "a decode registration carries no kv-events endpoint"
        );

        decode
            .reconcile(&[registration])
            .await
            .expect("reconcile should succeed");
        assert_eq!(decode.schedulable_count(&cfg.model_name), 1);
    }

    #[test]
    fn per_role_defaults_take_that_role_capacity() {
        let cfg = EppStandaloneConfig {
            prefill_max_num_batched_tokens: Some(16384),
            decode_max_num_batched_tokens: Some(2048),
            ..disagg_config()
        };

        let prefill = RegistrationDefaults::for_role(&cfg, WorkerRole::Prefill);
        assert_eq!(prefill.max_num_batched_tokens, Some(16384));

        let decode = RegistrationDefaults::for_role(&cfg, WorkerRole::Decode);
        assert_eq!(decode.max_num_batched_tokens, Some(2048));
    }
}
