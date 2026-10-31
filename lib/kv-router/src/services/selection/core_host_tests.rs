// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection-core tests that need the ZMQ ingress, the service builder, or the flat
//! configuration: the kernel owns the core, the composed router owns those, so they live here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use dynamo_kv_indexer::backend::test_util::store_event;
use dynamo_kv_indexer::backend::{Indexer, IndexerPolicy};
use dynamo_scheduler::config::RouterConfig;
use dynamo_scheduler::identity::RoutingPartitionId;
use dynamo_scheduler::protocols::{LocalBlockHash, RoutingConstraints, StorageTier, WorkerId};
use dynamo_scheduler::selection::{
    HostCache, KvIndexSource, PromptRequest, SelectRequest, SelectionCacheConfig, SelectionCore,
    SelectionError, SelectionHost, WorkerLifecycle, WorkerRequest,
};
use dynamo_scheduler::types::WorkerSelectionPolicyFactory;
use dynamo_session_affinity::{SessionAffinityConfig, SessionAffinityMode};
use std::collections::HashSet;

use crate::WorkerType;
use crate::tracking_hash::tracking_hash_context;

use crate::config::KvRouterConfig;
use crate::services::selection::{
    SelectionServiceBuilder, ZmqDirectIngress, local_selection_core_with_host,
};

fn test_config(use_kv_events: bool) -> KvRouterConfig {
    KvRouterConfig {
        use_kv_events,
        router_queue_threshold: None,
        ..Default::default()
    }
}

fn reference_factory() -> dynamo_scheduler::types::WorkerSelectionPolicyFactory {
    Arc::new(|config, role, _| {
        dynamo_worker_selection::WorkerSelectionPolicy::reference(
            config.clone(),
            role.default_selector_label(),
        )
    })
}

fn zmq_host_with_ingress() -> (SelectionHost, Arc<ZmqDirectIngress>) {
    let ingress = Arc::new(ZmqDirectIngress::default());
    let host = SelectionHost {
        cache: HostCache {
            index: KvIndexSource::Owned(ingress.clone()),
            ..HostCache::default()
        },
        ..SelectionHost::default()
    };
    (host, ingress)
}

fn local_core_with_ingress(
    config: KvRouterConfig,
    indexer_threads: usize,
    cancel_token: CancellationToken,
) -> (SelectionCore, Arc<ZmqDirectIngress>) {
    let (host, ingress) = zmq_host_with_ingress();
    let core = local_selection_core_with_host(
        config,
        indexer_threads,
        cancel_token,
        SelectionCacheConfig::default(),
        reference_factory(),
        host,
    )
    .expect("valid test config");
    (core, ingress)
}

fn default_key() -> RoutingPartitionId {
    RoutingPartitionId::new("model", "default")
}

fn worker(worker_id: WorkerId) -> WorkerRequest {
    WorkerRequest {
        worker_id,
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        endpoint: Some(format!("http://worker-{worker_id}:8000")),
        kv_events_endpoint: None,
        kv_events_endpoints: HashMap::new(),
        replay_endpoint: None,
        block_size: Some(4),
        data_parallel_start_rank: None,
        data_parallel_size: None,
        max_num_batched_tokens: Some(1024),
        total_kv_blocks: None,
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
    }
}

fn worker_with_kv_events(worker_id: WorkerId) -> WorkerRequest {
    WorkerRequest {
        kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
        ..worker(worker_id)
    }
}

/// The former in-crate fixture: a core over the flat configuration with an explicit host,
/// policy factory, worker type, and session-affinity settings.
fn core_with(
    config: KvRouterConfig,
    host: SelectionHost,
    policy_factory: Option<WorkerSelectionPolicyFactory>,
    worker_type: WorkerType,
    affinity: Option<SessionAffinityConfig>,
) -> SelectionCore {
    let tracking_hash =
        Arc::new(tracking_hash_context(&config).expect("valid tracking hash configuration"));
    let indexer_policy =
        IndexerPolicy::from_config(&dynamo_kv_indexer::config::IndexerConfig::from(&config))
            .expect("indexer policy");
    let config = RouterConfig::from(&config)
        .validate()
        .expect("valid router config");
    SelectionCore::new(
        config,
        1,
        CancellationToken::new(),
        None,
        policy_factory.unwrap_or_else(reference_factory),
        host,
        worker_type,
        true,
        SelectionCacheConfig::default(),
        tracking_hash,
        indexer_policy,
        affinity,
    )
}

fn core_with_session_affinity() -> SelectionCore {
    core_with_session_affinity_mode(SessionAffinityMode::Hard)
}

fn core_with_session_affinity_mode(mode: SessionAffinityMode) -> SelectionCore {
    core_with(
        test_config(false),
        SelectionHost::default(),
        None,
        WorkerType::Aggregated,
        Some(SessionAffinityConfig::new(Duration::from_secs(10)).with_mode(mode)),
    )
}

fn prompt() -> PromptRequest {
    PromptRequest {
        token_ids: Some(vec![1, 2, 3, 4]),
        mm_routing_info: None,
        block_mm_infos: None,
        block_hashes: None,
        sequence_hashes: None,
        isl_tokens: None,
        lora_name: None,
        cache_namespace: None,
        is_eagle: None,
    }
}

fn select_request() -> SelectRequest {
    SelectRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: None,
        prompt: prompt(),
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        advisory: false,
    }
}

#[tokio::test]
async fn reupsert_recreates_a_listener_lost_to_a_cancelled_update() {
    use dynamo_kv_indexer::KvIndexerInterface;

    let (core, ingress) = local_core_with_ingress(test_config(true), 1, CancellationToken::new());
    core.upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker upsert");
    let entry = core.partition(&default_key()).expect("partition");
    let hashes = [11u64, 12];
    entry
        .indexer()
        .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
        .await
        .unwrap();
    if let Indexer::Single { primary, .. } = &entry.indexer() {
        let _ = primary.flush().await;
    }
    // An endpoint update cancelled right after removing the listener leaves
    // the catalog record schedulable, the listener gone, and its blocks indexed.
    ingress.listeners().expect("attached").forget_listener(1, 0);

    let record = core
        .upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker re-upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable, "{record:?}");
    assert!(ingress.listeners().expect("attached").has_listener(1, 0));
    // The purge is queued to the indexer thread; wait for it to land.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let overlap = entry
                .indexer()
                .find_matches(hashes.iter().copied().map(LocalBlockHash).collect())
                .await
                .expect("find matches");
            if overlap.scores.is_empty() {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("stale blocks survived the re-upsert");
}

#[tokio::test]
async fn multi_rank_worker_with_replay_endpoint_is_incomplete() {
    let (core, ingress) = local_core_with_ingress(test_config(true), 1, CancellationToken::new());

    let record = core
        .upsert_worker(WorkerRequest {
            data_parallel_size: Some(2),
            kv_events_endpoints: HashMap::from([
                (0, "tcp://127.0.0.1:5557".to_string()),
                (1, "tcp://127.0.0.1:5558".to_string()),
            ]),
            replay_endpoint: Some("tcp://127.0.0.1:5600".to_string()),
            ..worker(1)
        })
        .await
        .expect("worker upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Incomplete, "{record:?}");
    assert!(!ingress.listeners().expect("attached").has_worker(1));
}

#[tokio::test]
async fn upsert_moves_global_worker_id_between_routing_groups() {
    let (core, ingress) = local_core_with_ingress(test_config(true), 1, CancellationToken::new());
    let mut group_a = worker_with_kv_events(1);
    group_a.routing_group = "group-a".to_string();
    core.upsert_worker(group_a).await.expect("group A upsert");
    let indexed = |group: &str| {
        ingress
            .listeners()
            .expect("attached")
            .list_filtered(Some("model"), Some(group))
            .len()
    };
    assert_eq!(indexed("group-a"), 1);

    let mut group_b = worker_with_kv_events(1);
    group_b.routing_group = "group-b".to_string();
    core.upsert_worker(group_b).await.expect("group B upsert");

    assert!(core.list_workers(Some("model"), Some("group-a")).is_empty());
    assert_eq!(core.list_workers(Some("model"), Some("group-b")).len(), 1);
    assert_eq!(indexed("group-a"), 0);
    assert_eq!(indexed("group-b"), 1);

    let mut select_a = select_request();
    select_a.routing_group = "group-a".to_string();
    assert!(matches!(
        core.select(select_a).await,
        Err(SelectionError::NotReady(_))
    ));
    let mut select_b = select_request();
    select_b.routing_group = "group-b".to_string();
    assert_eq!(core.select(select_b).await.unwrap().worker_id, 1);

    core.delete_worker(1).await.expect("delete group B worker");
    assert_eq!(indexed("group-b"), 0);
}

#[tokio::test]
async fn shutdown_cancels_listeners_but_keeps_parent_alive() {
    let parent = CancellationToken::new();
    let (core, ingress) = local_core_with_ingress(test_config(true), 1, parent.clone());

    let record = core
        .upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
    assert_eq!(
        ingress
            .listeners()
            .expect("attached")
            .listener_cancelled(1, 0),
        Some(false)
    );

    core.shutdown();
    assert!(core.cancel_token().is_cancelled());
    assert!(!parent.is_cancelled());
    assert_eq!(
        ingress
            .listeners()
            .expect("attached")
            .listener_cancelled(1, 0),
        Some(true)
    );
}

#[tokio::test]
async fn affinity_configuration_rejects_invalid_or_conflicting_config() {
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(dynamo_session_affinity::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let result = SelectionServiceBuilder::new(
            test_config(false),
            WorkerType::Aggregated,
            Default::default(),
        )
        .session_affinity(ttl)
        .build()
        .await;
        assert!(result.is_err());
    }
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.unwrap();
    let partition = core.partition(&default_key()).unwrap();
    assert!(matches!(
        partition.session_affinity(SessionAffinityConfig::new(Duration::from_secs(20))),
        Err(SelectionError::Conflict(_))
    ));
    assert!(matches!(
        partition.session_affinity(
            SessionAffinityConfig::new(Duration::from_secs(10))
                .with_mode(SessionAffinityMode::Soft)
        ),
        Err(SelectionError::Conflict(_))
    ));
}

/// The ZMQ ingress's detach drops a departed worker's blocks; a metadata update keeps them.
#[tokio::test]
async fn rejoined_worker_feeds_partition_index() {
    use dynamo_scheduler::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    for update_only in [false, true] {
        let (core, ingress) =
            local_core_with_ingress(test_config(true), 1, CancellationToken::new());
        let registry = ingress.listeners().expect("ingress attached");
        let request = worker_with_kv_events(1);
        core.upsert_worker(request.clone()).await.unwrap();
        let key = default_key();
        let partition = core.partition(&key).unwrap();
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let indexer = registry.get_indexer(&key).unwrap().indexer.clone();
        indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        indexer.dump_events().await.unwrap();
        if !update_only {
            core.delete_worker(1).await.unwrap();
        }
        let mut request = request;
        request.total_kv_blocks = Some(2048);
        core.upsert_worker(request).await.unwrap();
        let retained = partition.indexer().dump_events().await.unwrap();
        if update_only {
            assert!(
                !retained.is_empty(),
                "metadata update cleared cached blocks"
            );
        } else {
            assert!(retained.is_empty(), "removed worker retained cached blocks");
            let current = registry.get_indexer(&key).unwrap().indexer.clone();
            current
                .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
                .await
                .unwrap();
            current.dump_events().await.unwrap();
        }
        let mut request = select_request();
        request.prompt = PromptRequest {
            token_ids: Some(tokens),
            ..PromptRequest::default()
        };
        assert_eq!(
            core.select(request).await.unwrap().overlap.longest_matched,
            8,
            "update_only={update_only}"
        );
        assert_eq!(
            partition.indexer().dump_events().await.unwrap().len(),
            indexer.dump_events().await.unwrap().len()
        );
    }
}
