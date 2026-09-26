// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridges pod discovery to the selector catalogs. Every reflector change
//! becomes a desired-membership snapshot for the kv-router `CatalogReconciler`;
//! disaggregated topology runs one reconciler per role over the same reflector.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::DEFAULT_ROUTING_GROUP;
use dynamo_kv_router::services::selection::{
    CatalogReconciler, WorkerCatalogSource, WorkerRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::pod_discovery::{PodDiscovery, RawWorker};
use crate::selector::RoleSelectors;
use crate::worker_role::WorkerRole;

/// Per-role registration values that do not come from the pod itself.
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

    pub fn for_role(cfg: &EppStandaloneConfig, role: WorkerRole) -> Self {
        Self {
            model_name: cfg.model_name.clone(),
            block_size: cfg.block_size,
            total_kv_blocks: cfg.total_kv_blocks,
            max_num_batched_tokens: cfg.max_num_batched_tokens_for(role),
        }
    }
}

/// One role's view of the reflector, fed to that role's reconciler.
struct RoleReflectorSource {
    reflector: PodDiscovery,
    changes: watch::Receiver<u64>,
    role: WorkerRole,
    defaults: RegistrationDefaults,
    primed: bool,
    closed: bool,
}

#[async_trait]
impl WorkerCatalogSource for RoleReflectorSource {
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if self.closed {
            return None;
        }
        if self.primed && self.changes.changed().await.is_err() {
            tracing::warn!(
                role = %self.role,
                "Reflector change channel closed; clearing selector topology"
            );
            self.closed = true;
            return Some(Vec::new());
        }
        self.primed = true;
        Some(
            self.reflector
                .ready_workers_for(self.role)
                .into_iter()
                .map(|worker| worker_request(worker, &self.defaults))
                .collect(),
        )
    }
}

/// Owns the reconcile tasks; dropping it cancels them.
pub struct TopologyAdapter {
    cancel: CancellationToken,
}

impl TopologyAdapter {
    pub fn spawn(
        reflector: PodDiscovery,
        selectors: RoleSelectors,
        cfg: &EppStandaloneConfig,
    ) -> Self {
        let cancel = CancellationToken::new();
        for (role, selector) in selectors.each() {
            let source = RoleReflectorSource {
                changes: reflector.subscribe_changes(),
                reflector: reflector.clone(),
                role,
                defaults: RegistrationDefaults::for_role(cfg, role),
                primed: false,
                closed: false,
            };
            tokio::spawn(
                CatalogReconciler::new(Arc::clone(selector.service.core()))
                    .run(source, cancel.child_token()),
            );
        }
        Self { cancel }
    }
}

impl Drop for TopologyAdapter {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// A decode worker carries no KV-event endpoints; the core only demands one
/// when the instance consumes KV events, which the decode selector does not.
fn worker_request(w: RawWorker, defaults: &RegistrationDefaults) -> WorkerRequest {
    WorkerRequest {
        worker_id: w.worker_id,
        model_name: defaults.model_name.clone(),
        routing_group: DEFAULT_ROUTING_GROUP.to_string(),
        endpoint: Some(w.http_endpoint),
        block_size: Some(defaults.block_size),
        data_parallel_start_rank: Some(0),
        data_parallel_size: Some((w.kv_events_endpoints.len() as u32).max(1)),
        kv_events_endpoints: w.kv_events_endpoints,
        replay_endpoint: w.replay_endpoint,
        total_kv_blocks: defaults.total_kv_blocks,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;

    use super::*;
    use crate::epp_standalone_config::EppTopologyMode;
    use crate::role_config::kv_router_config_for_role;
    use crate::selector::Selector;

    fn config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            total_kv_blocks: Some(1000),
            ..EppStandaloneConfig::for_test()
        }
    }

    fn disagg_config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            topology_mode: EppTopologyMode::Disaggregated,
            model_name: "test-model".to_string(),
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

    fn role_worker(id: u64, ip: &str, role: WorkerRole) -> RawWorker {
        RawWorker {
            worker_id: id,
            pod_name: format!("w-{id}"),
            pod_ip: ip.to_string(),
            role,
            http_endpoint: format!("http://{ip}:8000"),
            kv_events_endpoints: if role == WorkerRole::Decode {
                HashMap::new()
            } else {
                HashMap::from([(0, format!("tcp://{ip}:5557"))])
            },
            replay_endpoint: None,
        }
    }

    fn worker(id: u64, ip: &str) -> RawWorker {
        role_worker(id, ip, WorkerRole::Aggregated)
    }

    #[test]
    fn registration_maps_env_and_endpoints() {
        let mut raw = worker(7, "10.0.0.1");
        raw.kv_events_endpoints
            .insert(1, "tcp://10.0.0.1:5558".to_string());
        let request = worker_request(raw, &defaults());
        assert_eq!(request.worker_id, 7);
        assert_eq!(request.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(request.endpoint.as_deref(), Some("http://10.0.0.1:8000"));
        assert_eq!(request.block_size, Some(16));
        assert_eq!(request.data_parallel_size, Some(2));
        assert_eq!(
            request.kv_events_endpoints.get(&0).unwrap(),
            "tcp://10.0.0.1:5557"
        );
        assert_eq!(request.total_kv_blocks, Some(1000));

        let cfg = EppStandaloneConfig {
            prefill_max_num_batched_tokens: Some(16384),
            decode_max_num_batched_tokens: Some(2048),
            ..disagg_config()
        };
        for (role, want) in [
            (WorkerRole::Prefill, Some(16384)),
            (WorkerRole::Decode, Some(2048)),
        ] {
            let got = RegistrationDefaults::for_role(&cfg, role).max_num_batched_tokens;
            assert_eq!(got, want, "{role} max_num_batched_tokens");
        }
    }

    #[tokio::test]
    async fn channel_close_clears_selector_topology() {
        let selector = Arc::new(
            Selector::new(&config(), WorkerSelectionPolicyRegistry::default())
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

    /// Two real `SelectionService`s configured exactly as production does.
    async fn role_selectors(cfg: &EppStandaloneConfig) -> RoleSelectors {
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
        let RoleSelectors::Disaggregated { prefill, decode } = &selectors else {
            unreachable!()
        };

        // Add: one of each.
        discovery.set_workers(vec![
            role_worker(1, "10.0.0.1", WorkerRole::Prefill),
            role_worker(2, "10.0.0.2", WorkerRole::Decode),
        ]);
        changes_tx.send(1).expect("adapter is listening");
        await_counts(&selectors, &model, 1, 1).await;
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([2]));
        assert_eq!(prefill.schedulable_worker_ids(&model), HashSet::from([1]));

        // Role flip: worker 1 moves catalogs in place.
        discovery.set_workers(vec![
            role_worker(1, "10.0.0.1", WorkerRole::Decode),
            role_worker(2, "10.0.0.2", WorkerRole::Decode),
        ]);
        changes_tx.send(2).expect("adapter is listening");
        await_counts(&selectors, &model, 0, 2).await;
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([1, 2]));

        // Remove.
        discovery.set_workers(vec![role_worker(2, "10.0.0.2", WorkerRole::Decode)]);
        changes_tx.send(3).expect("adapter is listening");
        await_counts(&selectors, &model, 0, 1).await;
        assert_eq!(decode.schedulable_worker_ids(&model), HashSet::from([2]));
        drop(adapter);
    }

    #[tokio::test]
    async fn decode_workers_are_schedulable_without_kv_event_endpoints() {
        let cfg = disagg_config();
        let selectors = role_selectors(&cfg).await;
        let RoleSelectors::Disaggregated { decode, .. } = &selectors else {
            unreachable!()
        };
        let request = worker_request(
            role_worker(9, "10.0.0.9", WorkerRole::Decode),
            &RegistrationDefaults::for_role(&cfg, WorkerRole::Decode),
        );
        assert!(request.kv_events_endpoints.is_empty());

        CatalogReconciler::new(Arc::clone(decode.service.core()))
            .apply(&[request])
            .await
            .expect("reconcile should succeed");
        assert_eq!(decode.schedulable_count(&cfg.model_name), 1);
    }
}
