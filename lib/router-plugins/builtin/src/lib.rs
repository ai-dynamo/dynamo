// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router plugins Dynamo ships.
//!
//! Routing hosts always link the default through `default_registry`. The optional custom
//! catalog adds the named default, two-tier, and ThunderAgent providers through `register`. The default
//! itself uses the same public candidate inputs and scorer/picker dispatch as external policies.
//! Sequence tracking, eligibility, and admission remain in dynamo-kv-router.

mod default;
mod thunderagent;
mod two_tier_cost_fn;
pub use default::{DefaultWorkerSelector, default_factory, default_policy};

/// Registry containing the required default only, without an optional policy catalog.
pub fn default_registry() -> RouterPluginRegistry {
    RouterPluginRegistry::default().with_default_factory(default_factory())
}

use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

/// Register the named providers Dynamo ships, without changing the host's default factory.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(registry: &mut RouterPluginRegistry) -> Result<(), RouterPluginRegistryError> {
    default::register(registry)?;
    two_tier_cost_fn::register(registry)?;
    thunderagent::register(registry)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::WorkerSelectionPolicyRegistryError;
    use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
    use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

    use super::*;

    /// Resolve router-policy YAML exactly as the Python bindings do at startup, so these cover the
    /// real configuration path rather than the registrars in isolation.
    fn resolve(
        yaml: &str,
    ) -> (
        KvRouterConfig,
        Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError>,
    ) {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy_file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let mut registry = default_registry();
        register(&mut registry).unwrap();
        let resolved = registry.resolve(&config);
        (config, resolved)
    }

    /// Catches a policy type name that drifts from its documentation, and proves the documented
    /// instance shape constructs for every stage it selects.
    #[test]
    fn resolves_documented_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  prefill: dynamo-two-tier-cost-fn
  decode: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");

        let partition = RoutingPartitionRef::new("model", "default");
        for worker_type in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            factory(&config, worker_type, partition);
        }
    }

    /// An unknown parameter key is a mistake, most often a misremembered threshold name. It must
    /// fail startup rather than silently leaving the default in place.
    #[test]
    fn rejects_an_unknown_parameter_key() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_affinity_threshold: 0.8
"#,
        );

        let Err(error) = resolved else {
            panic!("an unknown parameter must fail resolution");
        };
        assert!(
            matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type, .. }
                if policy_type == two_tier_cost_fn::POLICY_TYPE),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("cache_affinity_threshold"),
            "the error should name the offending key: {error}"
        );
    }

    /// An external host drives native cache-aware selection and affinity without AISimulate.
    #[tokio::test]
    async fn selection_core_consumes_physical_cache_with_manual_affinity() {
        use std::collections::{HashMap, HashSet};
        use std::sync::Arc;
        use std::time::Duration;

        use dynamo_kv_router::RoutingPartitionId;
        use dynamo_kv_router::indexer::KvIndexerInterface;
        use dynamo_kv_router::protocols::{
            BlockHashOptions, RoutingConstraints, WorkerWithDpRank, compute_block_hash_for_seq,
        };
        use dynamo_kv_router::services::indexer::{backend::Indexer, registry::WorkerRegistry};
        use dynamo_kv_router::services::selection::affinity::{
            AcquireStep, SessionAffinity, SessionAffinityConfig, subagent_group_affinity_id,
        };
        use dynamo_kv_router::services::selection::{
            HostCache, KvEventIngress, KvIndexSource, PromptRequest, SelectionAdmission,
            SelectionHost, SelectionOperation, SelectionOutcome, SelectionServiceBuilder,
            SessionBinding, WorkerRequest,
        };
        use tokio::time::Instant;

        struct PhysicalEvents;
        #[async_trait::async_trait]
        impl KvEventIngress for PhysicalEvents {
            fn open(
                &self,
                registry: &WorkerRegistry,
                key: &RoutingPartitionId,
                block_size: u32,
            ) -> Indexer {
                registry.get_or_create_indexer(key.clone(), block_size)
            }
        }

        let config = KvRouterConfig {
            use_kv_events: true,
            router_temperature: 0.0,
            router_queue_threshold: None,
            ..Default::default()
        };
        let service =
            SelectionServiceBuilder::new(config, WorkerType::Aggregated, default_registry())
                .indexer_threads(1)
                .host(SelectionHost {
                    cache: HostCache {
                        index: KvIndexSource::Owned(Arc::new(PhysicalEvents)),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .build()
                .await
                .unwrap();
        let core = service.core();
        let key = RoutingPartitionId::new("model", "default");
        for worker_id in [1, 2] {
            core.upsert_worker(WorkerRequest {
                worker_id,
                model_name: "model".into(),
                routing_group: "default".into(),
                endpoint: Some(format!("http://simulation-worker-{worker_id}")),
                kv_events_endpoint: None,
                kv_events_endpoints: HashMap::new(),
                replay_endpoint: None,
                block_size: Some(4),
                data_parallel_start_rank: Some(4),
                data_parallel_size: Some(2),
                max_num_batched_tokens: Some(1024),
                total_kv_blocks: Some(64),
                stable_routing_id: None,
                is_eagle: None,
                taints: HashSet::new(),
                topology_domains: HashMap::new(),
                kv_transfer_domain: None,
                kv_transfer_enforcement: None,
                kv_transfer_preferred_weight: None,
                router_hint_worker_type: None,
                router_hint_source_control_endpoints: HashMap::new(),
                kv_event_source_mode: None,
            })
            .await
            .unwrap();
        }
        let partition = core.partition(&key).unwrap();
        let tokens: Vec<_> = (1..=8).collect();
        let hashes: Vec<_> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let mut event = dynamo_kv_router::test_utils::make_store_event(1, &hashes);
        event.event.dp_rank = 5;
        partition.indexer().try_apply_event(event).await.unwrap();
        if let Indexer::Single { primary, .. } = partition.indexer() {
            primary.flush().await;
        }
        let prompt = PromptRequest {
            token_ids: Some(tokens),
            ..Default::default()
        };
        let epoch = Instant::now();
        let ttl = Duration::from_secs(10);
        let table =
            SessionAffinity::with_manual_clock(SessionAffinityConfig::new(ttl), epoch).unwrap();
        let group = subagent_group_affinity_id("run-a/parent");
        assert_ne!(group, subagent_group_affinity_id("run-b/parent"));
        for id in ["first-child", "sibling"] {
            let AcquireStep::Held(hold) = table.try_acquire(&group, None).unwrap() else {
                panic!("available group");
            };
            let selected = core
                .run_selection(SelectionOperation {
                    key: key.clone(),
                    prompt: prompt.view(),
                    router_config_override: None,
                    expected_output_tokens: Some(2),
                    priority_jump: 0.0,
                    strict_priority: 0,
                    policy_class: None,
                    session_context: None,
                    session: SessionBinding::None,
                    affinity_target: hold.target(),
                    pinned_worker: None,
                    allowed_worker_ids: None,
                    routing_constraints: RoutingConstraints::default(),
                    admission: SelectionAdmission::Book {
                        selection_id: id.into(),
                    },
                    track_active_blocks: true,
                    return_routing_hashes: false,
                    replay_id: None,
                })
                .await
                .result
                .unwrap();
            let SelectionOutcome::Selected(selected) = selected else {
                panic!("accepted native selection");
            };
            assert_eq!(selected.response.best_worker, WorkerWithDpRank::new(1, 5));
            assert_eq!(selected.response.target_cached_prefix_blocks, 2);
            assert_eq!(selected.response.cached_tokens, 8);
            // Booking is tentative until this host reports successful dispatch.
            let lease = table
                .commit(hold, selected.response.best_worker.into())
                .unwrap();
            table.advance_clock(epoch + ttl * 2).unwrap();
            assert_eq!(
                table.query_target(&group, None).unwrap(),
                Some(selected.response.best_worker.into())
            );
            assert_eq!(table.query_target("run-a/parent", None).unwrap(), None);
            core.prefill_complete(id).await.unwrap();
            core.free_reservation(id).await.unwrap();
            drop(lease);
        }
        table.advance_clock(epoch + ttl * 3).unwrap();
        assert_eq!(table.query_target(&group, None).unwrap(), None);
        service.shutdown().await;
    }
}
